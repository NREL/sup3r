"""Base data handling classes.
@author: bbenton
"""

import logging
import os
import warnings
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime as dt

import numpy as np
from rex import Resource
from rex.utilities import log_mem

from sup3r.bias.bias_transforms import get_spatial_bc_factors, local_qdm_bc
from sup3r.containers.loaders.base import Loader
from sup3r.containers.wranglers.abstract import AbstractWrangler
from sup3r.preprocessing.feature_handling import (
    Feature,
)
from sup3r.utilities.utilities import (
    get_chunk_slices,
    get_raster_shape,
    nn_fill_array,
    spatial_coarsening,
)

np.random.seed(42)

logger = logging.getLogger(__name__)


class WranglerH5(AbstractWrangler):
    """Sup3r data extraction and processing in preparation for downstream
    containers like Sampler objects or BatchQueue objects."""

    def __init__(
        self,
        loader: Loader,
        target=None,
        shape=None,
        temporal_slice=slice(None, None, 1),
        max_delta=20,
        hr_spatial_coarsen=None,
        time_roll=0,
        raster_file=None,
        time_chunk_size=None,
        cache_pattern=None,
        overwrite_cache=False,
        load_cached=False,
        mask_nan=False,
        fill_nan=False,
        max_workers=None,
        res_kwargs=None,
    ):
        """
        Parameters
        ----------
        loader : Loader
            Loader object which just loads the data. This has been initialized
            with file_paths to the data and the features requested
        target : tuple
            (lat, lon) lower left corner of raster. Either need target+shape or
            raster_file.
        shape : tuple
            (rows, cols) grid size. Either need target+shape or raster_file.
        max_delta : int, optional
            Optional maximum limit on the raster shape that is retrieved at
            once. If shape is (20, 20) and max_delta=10, the full raster will
            be retrieved in four chunks of (10, 10). This helps adapt to
            non-regular grids that curve over large distances, by default 20
        temporal_slice : slice
            Slice specifying extent and step of temporal extraction. e.g.
            slice(start, stop, time_pruning). If equal to slice(None, None, 1)
            the full time dimension is selected.
        hr_spatial_coarsen : int | None
            Optional input to coarsen the high-resolution spatial field. This
            can be used if (for example) you have 2km source data, but you want
            the final high res prediction target to be 4km resolution, then
            hr_spatial_coarsen would be 2 so that the GAN is trained on
            aggregated 4km high-res data.
        time_roll : int
            The number of places by which elements are shifted in the time
            axis. Can be used to convert data to different timezones. This is
            passed to np.roll(a, time_roll, axis=2) and happens AFTER the
            temporal_slice operation.
        raster_file : str | None
            .txt file for raster_index array for the corresponding target and
            shape. If specified the raster_index will be loaded from the file
            if it exists or written to the file if it does not yet exist. If
            None and raster_index is not provided raster_index will be
            calculated directly. Either need target+shape, raster_file, or
            raster_index input.
        time_chunk_size : int
            Size of chunks to split time dimension into for parallel data
            extraction. If running in serial this can be set to the size of the
            full time index for best performance.
        cache_pattern : str | None
            Pattern for files for saving feature data. e.g.
            file_path_{feature}.pkl. Each feature will be saved to a file with
            the feature name replaced in cache_pattern. If not None
            feature arrays will be saved here and not stored in self.data until
            load_cached_data is called. The cache_pattern can also include
            {shape}, {target}, {times} which will help ensure unique cache
            files for complex problems.
        overwrite_cache : bool
            Whether to overwrite any previously saved cache files.
        load_cached : bool
            Whether to load data from cache files
        mask_nan : bool
            Flag to mask out (remove) any timesteps with NaN data from the
            source dataset. This is False by default because it can create
            discontinuities in the timeseries.
        fill_nan : bool
            Flag to gap-fill any NaN data from the source dataset using a
            nearest neighbor algorithm. This is False by default because it can
            hide bad datasets that should be identified by the user.
        max_workers : int | None
            Max number of workers to use for parallel processes involved in
            data extraction / loading.
        """
        super().__init__(
            target=target,
            shape=shape,
            raster_file=raster_file,
            temporal_slice=temporal_slice,
        )
        self.file_paths = loader.file_paths
        self.features = loader.features
        self.max_delta = max_delta
        self.hr_spatial_coarsen = hr_spatial_coarsen or 1
        self.time_roll = time_roll
        self.current_obs_index = None
        self.overwrite_cache = overwrite_cache
        self.load_cached = load_cached
        self.data = None
        self.res_kwargs = res_kwargs or {}
        self._time_chunk_size = time_chunk_size
        self._shape = None
        self._single_ts_files = None
        self._cache_pattern = cache_pattern
        self._cache_files = None
        self._handle_features = None
        self._extract_features = None
        self._noncached_features = None
        self._raster_index = None
        self._raw_features = None
        self._raw_data = {}
        self._time_chunks = None
        self.max_workers = max_workers

        self.preflight()

        overwrite = (
            self.overwrite_cache
            and self.cache_files is not None
            and all(os.path.exists(fp) for fp in self.cache_files)
        )

        if self.try_load and self.load_cached:
            logger.info(
                f'All {self.cache_files} exist. Loading from cache '
                f'instead of extracting from source files.'
            )
            self.load_cached_data()

        elif self.try_load and not self.load_cached:
            self.clear_data()
            logger.info(
                f'All {self.cache_files} exist. Call '
                'load_cached_data() or use load_cache=True to load '
                'this data from cache files.'
            )
        else:
            if overwrite:
                logger.info(
                    f'{self.cache_files} exists but overwrite_cache '
                    'is set to True. Proceeding with extraction.'
                )

            self._raster_size_check()
            self._run_data_init_if_needed()

            if self._cache_pattern is not None:
                self.cache_data(self.cache_files)

        if fill_nan and self.data is not None:
            self.run_nn_fill()
        elif mask_nan and self.data is not None:
            self.mask_nan()

        if (
            self.hr_spatial_coarsen > 1
            and self.lat_lon.shape == self.raw_lat_lon.shape
        ):
            self.lat_lon = spatial_coarsening(
                self.lat_lon, s_enhance=self.hr_spatial_coarsen, obs_axis=False
            )

        logger.info('Finished intializing DataHandler.')
        log_mem(logger, log_level='INFO')

    def __getitem__(self, key):
        """Interface for sampler objects."""
        return self.data[key]

    @property
    def try_load(self):
        """Check if we should try to load cache"""
        return self._should_load_cache(
            self._cache_pattern, self.cache_files, self.overwrite_cache
        )

    def check_clear_data(self):
        """Check if data is cached and clear data if not load_cached"""
        if self._cache_pattern is not None and not self.load_cached:
            self.data = None
            self.val_data = None

    def _run_data_init_if_needed(self):
        """Check if any features need to be extracted and proceed with data
        extraction"""
        if any(self.features):
            self.data = self.load()
            mask = np.isinf(self.data)
            self.data[mask] = np.nan
            nan_perc = 100 * np.isnan(self.data).sum() / self.data.size
            if nan_perc > 0:
                msg = 'Data has {:.3f}% NaN values!'.format(nan_perc)
                logger.warning(msg)
                warnings.warn(msg)

    @property
    def attrs(self):
        """Get atttributes of input data

        Returns
        -------
        dict
            Dictionary of attributes
        """
        return self.source_handler(self.file_paths).attrs

    @property
    def cache_files(self):
        """Cache files for storing extracted data"""
        if self._cache_files is None:
            self._cache_files = self.get_cache_file_names(self.cache_pattern)
        return self._cache_files

    @property
    def raster_index(self):
        """Raster index property"""
        if self._raster_index is None:
            self._raster_index = self.get_raster_index()
        return self._raster_index

    @raster_index.setter
    def raster_index(self, raster_index):
        """Update raster index property"""
        self._raster_index = raster_index

    @classmethod
    def get_handle_features(cls, file_paths):
        """Get all available features in input data

        Parameters
        ----------
        file_paths : list
            List of input file paths

        Returns
        -------
        handle_features : list
            List of available input features
        """
        handle_features = []
        for f in file_paths:
            handle = cls.source_handler([f])
            handle_features += [Feature.get_basename(r) for r in handle]
        return list(set(handle_features))

    @property
    def handle_features(self):
        """All features available in raw input"""
        if self._handle_features is None:
            self._handle_features = self.get_handle_features(self.file_paths)
        return self._handle_features

    @property
    def noncached_features(self):
        """Get list of features needing extraction or derivation"""
        if self._noncached_features is None:
            self._noncached_features = self.check_cached_features(
                self.features,
                cache_files=self.cache_files,
                overwrite_cache=self.overwrite_cache,
                load_cached=self.load_cached,
            )
        return self._noncached_features

    @property
    def extract_features(self):
        """Features to extract directly from the source handler"""
        lower_features = [f.lower() for f in self.handle_features]
        return [
            f
            for f in self.raw_features
            if self.lookup(f, 'compute') is None
            or Feature.get_basename(f.lower()) in lower_features
        ]

    @property
    def derive_features(self):
        """List of features which need to be derived from other features"""
        return [
            f
            for f in set(
                list(self.noncached_features) + list(self.extract_features)
            )
            if f not in self.extract_features
        ]

    @property
    def cached_features(self):
        """List of features which have been requested but have been determined
        not to need extraction. Thus they have been cached already."""
        return [f for f in self.features if f not in self.noncached_features]

    @property
    def raw_features(self):
        """Get list of features needed for computations"""
        if self._raw_features is None:
            self._raw_features = self.get_raw_feature_list(
                self.noncached_features, self.handle_features
            )

        return self._raw_features

    def preflight(self):
        """Run some preflight checks and verify that the inputs are valid"""

        self.cap_worker_args(self.max_workers)

        if len(self.sample_shape) == 2:
            logger.info(
                'Found 2D sample shape of {}. Adding temporal dim of 1'.format(
                    self.sample_shape
                )
            )
            self.sample_shape = (*self.sample_shape, 1)

        start = self.temporal_slice.start
        stop = self.temporal_slice.stop

        msg = (
            f'sample_shape[2] ({self.sample_shape[2]}) cannot be larger '
            'than the number of time steps in the raw data '
            f'({len(self.raw_time_index)}).'
        )
        if len(self.raw_time_index) < self.sample_shape[2]:
            logger.warning(msg)
            warnings.warn(msg)

        msg = (
            f'The requested time slice {self.temporal_slice} conflicts '
            f'with the number of time steps ({len(self.raw_time_index)}) '
            'in the raw data'
        )
        t_slice_is_subset = start is not None and stop is not None
        good_subset = (
            t_slice_is_subset
            and (stop - start <= len(self.raw_time_index))
            and stop <= len(self.raw_time_index)
            and start <= len(self.raw_time_index)
        )
        if t_slice_is_subset and not good_subset:
            logger.error(msg)
            raise RuntimeError(msg)

        msg = (
            f'Initializing DataHandler {self.input_file_info}. '
            f'Getting temporal range {self.time_index[0]!s} to '
            f'{self.time_index[-1]!s} (inclusive) '
            f'based on temporal_slice {self.temporal_slice}'
        )
        logger.info(msg)

        logger.info(
            f'Using max_workers={self.max_workers}, '
            f'norm_workers={self.norm_workers}, '
            f'extract_workers={self.extract_workers}, '
            f'compute_workers={self.compute_workers}, '
            f'load_workers={self.load_workers}'
        )

    @staticmethod
    def get_closest_row_col(lat_lon, target):
        """Get closest indices to target lat lon

        Parameters
        ----------
        lat_lon : ndarray
            Array of lat/lon
            (spatial_1, spatial_2, 2)
            Last dimension in order of (lat, lon)
        target : tuple
            (lat, lon) for target coordinate

        Returns
        -------
        row : int
            row index for closest lat/lon to target lat/lon
        col : int
            col index for closest lat/lon to target lat/lon
        """
        dist = np.hypot(
            lat_lon[..., 0] - target[0], lat_lon[..., 1] - target[1]
        )
        row, col = np.where(dist == np.min(dist))
        row = row[0]
        col = col[0]
        return row, col

    @classmethod
    def get_lat_lon(cls, file_paths, raster_index, invert_lat=False):
        """Get lat/lon grid for requested target and shape

        Parameters
        ----------
        file_paths : list
            path to data file
        raster_index : ndarray | list
            Raster index array or list of slices
        invert_lat : bool
            Flag to invert data along the latitude axis. Wrf data tends to use
            an increasing ordering for latitude while wtk uses a decreasing
            ordering.

        Returns
        -------
        ndarray
            (spatial_1, spatial_2, 2) Lat/Lon array with same ordering in last
            dimension
        """
        lat_lon = cls.lookup('lat_lon', 'compute')(file_paths, raster_index)
        if invert_lat:
            lat_lon = lat_lon[::-1]
        # put angle betwen -180 and 180
        lat_lon[..., 1] = (lat_lon[..., 1] + 180) % 360 - 180
        return lat_lon.astype(np.float32)

    @property
    def shape(self):
        """Full data shape

        Returns
        -------
        shape : tuple
            Full data shape
            (spatial_1, spatial_2, temporal, features)
        """
        if self._shape is None:
            self._shape = self.data.shape
        return self._shape

    @property
    def size(self):
        """Size of data array

        Returns
        -------
        size : int
            Number of total elements contained in data array
        """
        return np.prod(self.requested_shape)

    def cache_data(self, cache_file_paths):
        """Cache feature data to file and delete from memory

        Parameters
        ----------
        cache_file_paths : str | None
            Path to file for saving feature data
        """
        self._cache_data(
            self.data, self.features, cache_file_paths, self.overwrite_cache
        )

    @property
    def requested_shape(self):
        """Get requested shape for cached data"""
        shape = get_raster_shape(self.raster_index)
        return (
            shape[0] // self.hr_spatial_coarsen,
            shape[1] // self.hr_spatial_coarsen,
            len(self.raw_time_index[self.temporal_slice]),
            len(self.features),
        )

    def load_cached_data(self, with_split=True):
        """Load data from cache files and split into training and validation

        Parameters
        ----------
        with_split : bool
            Whether to split into training and validation data or not.
        """
        if self.data is not None:
            logger.info('Called load_cached_data() but self.data is not None')

        elif self.data is None:
            msg = (
                'Found {} cache files but need {} for features {}! '
                'These are the cache files that were found: {}'.format(
                    len(self.cache_files),
                    len(self.features),
                    self.features,
                    self.cache_files,
                )
            )
            assert len(self.cache_files) == len(self.features), msg

            self.data = np.full(
                shape=self.requested_shape, fill_value=np.nan, dtype=np.float32
            )

            logger.info(f'Loading cached data from: {self.cache_files}')
            max_workers = self.load_workers
            self._load_cached_data(
                data=self.data,
                cache_files=self.cache_files,
                features=self.features,
                max_workers=max_workers,
            )

            self.time_index = self.raw_time_index[self.temporal_slice]

            nan_perc = 100 * np.isnan(self.data).sum() / self.data.size
            if nan_perc > 0:
                msg = 'Data has {:.3f}% NaN values!'.format(nan_perc)
                logger.warning(msg)
                warnings.warn(msg)

            if with_split and self.val_split > 0:
                logger.debug(
                    'Splitting data into training / validation sets '
                    f'({1 - self.val_split}, {self.val_split}) '
                    f'for {self.input_file_info}'
                )

                self.data, self.val_data = self.split_data(
                    val_split=self.val_split, shuffle_time=self.shuffle_time
                )

    def load(self):
        """Build base 4D data array. Can handle multiple files but assumes
        each file has the same spatial domain

        Returns
        -------
        data : np.ndarray
            4D array of high res data
            (spatial_1, spatial_2, temporal, features)
        """
        now = dt.now()
        logger.debug(f'Loading data for raster of shape {self.grid_shape}')
        # get the file-native time index without pruning
        if self.is_time_independent:
            n_steps = 1
            shifted_time_chunks = [slice(None)]
        else:
            n_steps = len(self.raw_time_index[self.temporal_slice])
            shifted_time_chunks = get_chunk_slices(
                n_steps, self.time_chunk_size
            )

        self.run_data_extraction()
        self.run_data_compute()

        logger.info('Building final data array')
        self.data_fill(shifted_time_chunks, self.extract_workers)

        if self.invert_lat:
            self.data = self.data[::-1]

        if self.time_roll != 0:
            logger.debug('Applying time roll to data array')
            self.data = np.roll(self.data, self.time_roll, axis=2)

        if self.hr_spatial_coarsen > 1:
            logger.debug('Applying hr spatial coarsening to data array')
            self.data = spatial_coarsening(
                self.data, s_enhance=self.hr_spatial_coarsen, obs_axis=False
            )
        if self.load_cached:
            for f in self.cached_features:
                f_index = self.features.index(f)
                logger.info(f'Loading {f} from {self.cache_files[f_index]}')
                with open(self.cache_files[f_index], 'rb') as fh:
                    self.data[..., f_index] = pickle.load(fh)

        logger.info(
            f'Finished extracting data for {self.input_file_info} in '
            f'{dt.now() - now}'
        )

        return self.data.astype(np.float32)

    def run_nn_fill(self):
        """Run nn nan fill on full data array."""
        for i in range(self.data.shape[-1]):
            if np.isnan(self.data[..., i]).any():
                self.data[..., i] = nn_fill_array(self.data[..., i])

    def mask_nan(self):
        """Drop timesteps with NaN data"""
        nan_mask = np.isnan(self.data).any(axis=(0, 1, 3))
        logger.info(
            'Removing {} out of {} timesteps due to NaNs'.format(
                nan_mask.sum(), self.data.shape[2]
            )
        )
        self.data = self.data[:, :, ~nan_mask, :]

    def run_data_extraction(self):
        """Run the raw dataset extraction process from disk to raw
        un-manipulated datasets.
        """
        if self.extract_features:
            logger.info(
                f'Starting extraction of {self.extract_features} '
                f'using {len(self.time_chunks)} time_chunks.'
            )
            if self.extract_workers == 1:
                self._raw_data = self.serial_extract(
                    self.file_paths,
                    self.raster_index,
                    self.time_chunks,
                    self.extract_features,
                    **self.res_kwargs,
                )

            else:
                self._raw_data = self.parallel_extract(
                    self.file_paths,
                    self.raster_index,
                    self.time_chunks,
                    self.extract_features,
                    self.extract_workers,
                    **self.res_kwargs,
                )

            logger.info(
                f'Finished extracting {self.extract_features} for '
                f'{self.input_file_info}'
            )

    def run_data_compute(self):
        """Run the data computation / derivation from raw features to desired
        features.
        """
        if self.derive_features:
            logger.info(f'Starting computation of {self.derive_features}')

            if self.compute_workers == 1:
                self._raw_data = self.serial_compute(
                    self._raw_data,
                    self.file_paths,
                    self.raster_index,
                    self.time_chunks,
                    self.derive_features,
                    self.noncached_features,
                    self.handle_features,
                )

            elif self.compute_workers != 1:
                self._raw_data = self.parallel_compute(
                    self._raw_data,
                    self.file_paths,
                    self.raster_index,
                    self.time_chunks,
                    self.derive_features,
                    self.noncached_features,
                    self.handle_features,
                    self.compute_workers,
                )

            logger.info(
                f'Finished computing {self.derive_features} for '
                f'{self.input_file_info}'
            )

    def _single_data_fill(self, t, t_slice, f_index, f):
        """Place single extracted / computed chunk in final data array

        Parameters
        ----------
        t : int
            Index of time slice in extracted / computed raw data dictionary
        t_slice : slice
            Time slice corresponding to the location in the final data array
        f_index : int
            Index of feature in the final data array
        f : str
            Name of corresponding feature in the raw data dictionary
        """
        tmp = self._raw_data[t][f]
        if len(tmp.shape) == 2:
            tmp = tmp[..., np.newaxis]
        self.data[..., t_slice, f_index] = tmp

    def serial_data_fill(self, shifted_time_chunks):
        """Fill final data array in serial

        Parameters
        ----------
        shifted_time_chunks : list
            List of time slices corresponding to the appropriate location of
            extracted / computed chunks in the final data array
        """
        for t, ts in enumerate(shifted_time_chunks):
            for _, f in enumerate(self.noncached_features):
                f_index = self.features.index(f)
                self._single_data_fill(t, ts, f_index, f)
            logger.info(
                f'Added {t + 1} of {len(shifted_time_chunks)} '
                'chunks to final data array'
            )
            self._raw_data.pop(t)

    def data_fill(self, shifted_time_chunks, max_workers=None):
        """Fill final data array with extracted / computed chunks

        Parameters
        ----------
        shifted_time_chunks : list
            List of time slices corresponding to the appropriate location of
            extracted / computed chunks in the final data array
        max_workers : int | None
            Max number of workers to use for building final data array. If None
            max available workers will be used. If 1 cached data will be loaded
            in serial
        """
        self.data = np.zeros(
            (
                self.grid_shape[0],
                self.grid_shape[1],
                self.n_tsteps,
                len(self.features),
            ),
            dtype=np.float32,
        )

        if max_workers == 1:
            self.serial_data_fill(shifted_time_chunks)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                futures = {}
                now = dt.now()
                for t, ts in enumerate(shifted_time_chunks):
                    for _, f in enumerate(self.noncached_features):
                        f_index = self.features.index(f)
                        future = exe.submit(
                            self._single_data_fill, t, ts, f_index, f
                        )
                        futures[future] = {'t': t, 'fidx': f_index}

                logger.info(
                    f'Started adding {len(futures)} chunks '
                    f'to data array in {dt.now() - now}.'
                )

                for i, future in enumerate(as_completed(futures)):
                    try:
                        future.result()
                    except Exception as e:
                        msg = (
                            f'Error adding ({futures[future]["t"]}, '
                            f'{futures[future]["fidx"]}) chunk to '
                            'final data array.'
                        )
                        logger.exception(msg)
                        raise RuntimeError(msg) from e
                    logger.debug(
                        f'Added {i + 1} out of {len(futures)} '
                        'chunks to final data array'
                    )
        logger.info('Finished building data array')

    @abstractmethod
    def get_raster_index(self):
        """Get raster index for file data. Here we assume the list of paths in
        file_paths all have data with the same spatial domain. We use the first
        file in the list to compute the raster

        Returns
        -------
        raster_index : np.ndarray
            2D array of grid indices for H5 or list of
            slices for NETCDF
        """

    def lin_bc(self, bc_files, threshold=0.1):
        """Bias correct the data in this DataHandler using linear bias
        correction factors from files output by MonthlyLinearCorrection or
        LinearCorrection from sup3r.bias.bias_calc

        Parameters
        ----------
        bc_files : list | tuple | str
            One or more filepaths to .h5 files output by
            MonthlyLinearCorrection or LinearCorrection. These should contain
            datasets named "{feature}_scalar" and "{feature}_adder" where
            {feature} is one of the features contained by this DataHandler and
            the data is a 3D array of shape (lat, lon, time) where time is
            length 1 for annual correction or 12 for monthly correction.
        threshold : float
            Nearest neighbor euclidean distance threshold. If the DataHandler
            coordinates are more than this value away from the bias correction
            lat/lon, an error is raised.
        """

        if isinstance(bc_files, str):
            bc_files = [bc_files]

        completed = []
        for idf, feature in enumerate(self.features):
            for fp in bc_files:
                dset_scalar = f'{feature}_scalar'
                dset_adder = f'{feature}_adder'
                with Resource(fp) as res:
                    dsets = [dset.lower() for dset in res.dsets]
                    check = (
                        dset_scalar.lower() in dsets
                        and dset_adder.lower() in dsets
                    )
                if feature not in completed and check:
                    scalar, adder = get_spatial_bc_factors(
                        lat_lon=self.lat_lon,
                        feature_name=feature,
                        bias_fp=fp,
                        threshold=threshold,
                    )

                    if scalar.shape[-1] == 1:
                        scalar = np.repeat(scalar, self.shape[2], axis=2)
                        adder = np.repeat(adder, self.shape[2], axis=2)
                    elif scalar.shape[-1] == 12:
                        idm = self.time_index.month.values - 1
                        scalar = scalar[..., idm]
                        adder = adder[..., idm]
                    else:
                        msg = (
                            'Can only accept bias correction factors '
                            'with last dim equal to 1 or 12 but '
                            'received bias correction factors with '
                            'shape {}'.format(scalar.shape)
                        )
                        logger.error(msg)
                        raise RuntimeError(msg)

                    logger.info(
                        'Bias correcting "{}" with linear '
                        'correction from "{}"'.format(
                            feature, os.path.basename(fp)
                        )
                    )
                    self.data[..., idf] *= scalar
                    self.data[..., idf] += adder
                    completed.append(feature)

    def qdm_bc(
        self, bc_files, reference_feature, relative=True, threshold=0.1
    ):
        """Bias Correction using Quantile Delta Mapping

        Bias correct this DataHandler's data with Quantile Delta Mapping. The
        required statistical distributions should be pre-calculated using
        :class:`sup3r.bias.bias_calc.QuantileDeltaMappingCorrection`.

        Warning: There is no guarantee that the coefficients from ``bc_files``
        match the resource processed here. Be careful choosing ``bc_files``.

        Parameters
        ----------
        bc_files : list | tuple | str
            One or more filepaths to .h5 files output by
            :class:`bias_calc.QuantileDeltaMappingCorrection`. These should
            contain datasets named "base_{reference_feature}_params",
            "bias_{feature}_params", and "bias_fut_{feature}_params" where
            {feature} is one of the features contained by this DataHandler and
            the data is a 3D array of shape (lat, lon, time) where time.
        reference_feature : str
            Name of the feature used as (historical) reference. Dataset with
            name "base_{reference_feature}_params" will be retrieved from
            ``bc_files``.
        relative : bool, default=True
            Switcher to apply QDM as a relative (use True) or absolute (use
            False) correction value.
        threshold : float, default=0.1
            Nearest neighbor euclidean distance threshold. If the DataHandler
            coordinates are more than this value away from the bias correction
            lat/lon, an error is raised.
        """

        if isinstance(bc_files, str):
            bc_files = [bc_files]

        completed = []
        for idf, feature in enumerate(self.features):
            for fp in bc_files:
                logger.info(
                    'Bias correcting "{}" with QDM '
                    'correction from "{}"'.format(
                        feature, os.path.basename(fp)
                    )
                )
                self.data[..., idf] = local_qdm_bc(
                    self.data[..., idf],
                    self.lat_lon,
                    reference_feature,
                    feature,
                    bias_fp=fp,
                    threshold=threshold,
                    relative=relative,
                )
                completed.append(feature)
