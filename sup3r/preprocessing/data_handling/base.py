"""Base data handling classes.
@author: bbenton
"""
import copy
import logging
import os
import pickle
import warnings
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime as dt
from fnmatch import fnmatch
from typing import ClassVar

import numpy as np
import pandas as pd
from rex import Resource
from rex.utilities import log_mem
from rex.utilities.fun_utils import get_fun_call_str

from sup3r.bias.bias_transforms import get_spatial_bc_factors, local_qdm_bc
from sup3r.preprocessing.data_handling.mixin import (
    InputMixIn,
    TrainingPrepMixIn,
)
from sup3r.preprocessing.feature_handling import (
    BVFreqMon,
    BVFreqSquaredNC,
    Feature,
    FeatureHandler,
    InverseMonNC,
    LatLonNC,
    PotentialTempNC,
    PressureNC,
    Rews,
    Shear,
    TempNC,
    UWind,
    VWind,
    WinddirectionNC,
    WindspeedNC,
)
from sup3r.utilities import ModuleName
from sup3r.utilities.cli import BaseCLI
from sup3r.utilities.utilities import (
    estimate_max_workers,
    get_chunk_slices,
    get_raster_shape,
    nn_fill_array,
    spatial_coarsening,
    uniform_box_sampler,
    uniform_time_sampler,
    weighted_box_sampler,
    weighted_time_sampler,
)

np.random.seed(42)

logger = logging.getLogger(__name__)


class DataHandler(FeatureHandler, InputMixIn, TrainingPrepMixIn):
    """Sup3r data handling and extraction for low-res source data or for
    artificially coarsened high-res source data for training.

    The sup3r data handler class is based on a 4D numpy array of shape:
    (spatial_1, spatial_2, temporal, features)
    """

    def __init__(self,
                 file_paths,
                 features,
                 target=None,
                 shape=None,
                 max_delta=20,
                 temporal_slice=slice(None, None, 1),
                 hr_spatial_coarsen=None,
                 time_roll=0,
                 val_split=0.0,
                 sample_shape=(10, 10, 1),
                 raster_file=None,
                 raster_index=None,
                 shuffle_time=False,
                 time_chunk_size=None,
                 cache_pattern=None,
                 overwrite_cache=False,
                 overwrite_ti_cache=False,
                 load_cached=False,
                 lr_only_features=tuple(),
                 hr_exo_features=tuple(),
                 handle_features=None,
                 single_ts_files=None,
                 mask_nan=False,
                 fill_nan=False,
                 worker_kwargs=None,
                 res_kwargs=None):
        """
        Parameters
        ----------
        file_paths : str | list
            A single source h5 wind file to extract raster data from or a list
            of netcdf files with identical grid. The string can be a unix-style
            file path which will be passed through glob.glob
        features : list
            list of features to extract from the provided data
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
        val_split : float32
            Fraction of data to store for validation
        sample_shape : tuple
            Size of spatial and temporal domain used in a single high-res
            observation for batching
        raster_file : str | None
            .txt file for raster_index array for the corresponding target and
            shape. If specified the raster_index will be loaded from the file
            if it exists or written to the file if it does not yet exist. If
            None and raster_index is not provided raster_index will be
            calculated directly. Either need target+shape, raster_file, or
            raster_index input.
        raster_index : list
            List of tuples or slices. Used as an alternative to computing the
            raster index from target+shape or loading the raster index from
            file
        shuffle_time : bool
            Whether to shuffle time indices before validation split
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
        overwrite_ti_cache : bool
            Whether to overwrite any previously saved time index cache files.
        overwrite_ti_cache : bool
            Whether to overwrite saved time index cache files.
        load_cached : bool
            Whether to load data from cache files
        lr_only_features : list | tuple
            List of feature names or patt*erns that should only be included in
            the low-res training set and not the high-res observations.
        hr_exo_features : list | tuple
            List of feature names or patt*erns that should be included in the
            high-resolution observation but not expected to be output from the
            generative model. An example is high-res topography that is to be
            injected mid-network.
        handle_features : list | None
            Optional list of features which are available in the provided data.
            Providing this eliminates the need for an initial search of
            available features prior to data extraction.
        single_ts_files : bool | None
            Whether input files are single time steps or not. If they are this
            enables some reduced computation. If None then this will be
            determined from file_paths directly.
        mask_nan : bool
            Flag to mask out (remove) any timesteps with NaN data from the
            source dataset. This is False by default because it can create
            discontinuities in the timeseries.
        fill_nan : bool
            Flag to gap-fill any NaN data from the source dataset using a
            nearest neighbor algorithm. This is False by default because it can
            hide bad datasets that should be identified by the user.
        worker_kwargs : dict | None
            Dictionary of worker values. Can include max_workers,
            extract_workers, compute_workers, load_workers, norm_workers,
            and ti_workers. Each argument needs to be an integer or None.

            The value of `max workers` will set the value of all other worker
            args. If max_workers == 1 then all processes will be serialized. If
            max_workers == None then other worker args will use their own
            provided values.

            `extract_workers` is the max number of workers to use for
            extracting features from source data. If None it will be estimated
            based on memory limits. If 1 processes will be serialized.
            `compute_workers` is the max number of workers to use for computing
            derived features from raw features in source data. `load_workers`
            is the max number of workers to use for loading cached feature
            data. `norm_workers` is the max number of workers to use for
            normalizing feature data. `ti_workers` is the max number of
            workers to use to get full time index. Useful when there are many
            input files each with a single time step. If this is greater than
            one, time indices for input files will be extracted in parallel
            and then concatenated to get the full time index. If input files
            do not all have time indices or if there are few input files this
            should be set to one.
        res_kwargs : dict | None
            kwargs passed to source handler for data extraction. e.g. This
            could be {'parallel': True,
                      'concat_dim': 'Time',
                      'combine': 'nested',
                      'chunks': {'south_north': 120, 'west_east': 120}}
            which then gets passed to xr.open_mfdataset(file, **res_kwargs)
        """
        InputMixIn.__init__(self,
                            target=target,
                            shape=shape,
                            raster_file=raster_file,
                            raster_index=raster_index,
                            temporal_slice=temporal_slice)

        self.file_paths = file_paths
        self.features = (features if isinstance(features, (list, tuple))
                         else [features])
        self.features = copy.deepcopy(self.features)
        self.val_time_index = None
        self.max_delta = max_delta
        self.val_split = val_split
        self.sample_shape = sample_shape
        self.hr_spatial_coarsen = hr_spatial_coarsen or 1
        self.time_roll = time_roll
        self.shuffle_time = shuffle_time
        self.current_obs_index = None
        self.overwrite_cache = overwrite_cache
        self.overwrite_ti_cache = overwrite_ti_cache
        self.load_cached = load_cached
        self.data = None
        self.val_data = None
        self.res_kwargs = res_kwargs or {}
        self._single_ts_files = single_ts_files
        self._cache_pattern = cache_pattern
        self._lr_only_features = lr_only_features
        self._hr_exo_features = hr_exo_features
        self._time_chunk_size = time_chunk_size
        self._handle_features = handle_features
        self._cache_files = None
        self._extract_features = None
        self._noncached_features = None
        self._raw_features = None
        self._raw_data = {}
        self._time_chunks = None
        self._means = None
        self._stds = None
        self._is_normalized = False
        self.worker_kwargs = worker_kwargs or {}
        self.max_workers = self.worker_kwargs.get('max_workers', None)
        self._ti_workers = self.worker_kwargs.get('ti_workers', None)
        self._extract_workers = self.worker_kwargs.get('extract_workers', None)
        self._norm_workers = self.worker_kwargs.get('norm_workers', None)
        self._load_workers = self.worker_kwargs.get('load_workers', None)
        self._compute_workers = self.worker_kwargs.get('compute_workers', None)
        self._worker_attrs = [
            '_ti_workers',
            '_norm_workers',
            '_compute_workers',
            '_extract_workers',
            '_load_workers'
        ]

        self.preflight()

        overwrite = (self.overwrite_cache and self.cache_files is not None
                     and all(os.path.exists(fp) for fp in self.cache_files))

        if self.try_load and self.load_cached:
            logger.info(f'All {self.cache_files} exist. Loading from cache '
                        f'instead of extracting from source files.')
            self.load_cached_data()

        elif self.try_load and not self.load_cached:
            self.clear_data()
            logger.info(f'All {self.cache_files} exist. Call '
                        'load_cached_data() or use load_cache=True to load '
                        'this data from cache files.')
        else:
            if overwrite:
                logger.info(f'{self.cache_files} exists but overwrite_cache '
                            'is set to True. Proceeding with extraction.')

            self._raster_size_check()
            self._run_data_init_if_needed()

            if self._cache_pattern is not None:
                self.cache_data(self.cache_files)
                self.data = None if not self.load_cached else self.data

            self._val_split_check()

        if fill_nan and self.data is not None:
            self.run_nn_fill()
        elif mask_nan and self.data is not None:
            self.mask_nan()

        if (self.hr_spatial_coarsen > 1
                and self.lat_lon.shape == self.raw_lat_lon.shape):
            self.lat_lon = spatial_coarsening(
                self.lat_lon,
                s_enhance=self.hr_spatial_coarsen,
                obs_axis=False)

        logger.info('Finished intializing DataHandler.')
        log_mem(logger, log_level='INFO')

    @property
    def try_load(self):
        """Check if we should try to load cache"""
        return self._should_load_cache(self._cache_pattern,
                                       self.cache_files,
                                       self.overwrite_cache)

    def check_clear_data(self):
        """Check if data is cached and clear data if not load_cached"""
        if self._cache_pattern is not None and not self.load_cached:
            self.data = None
            self.val_data = None

    def _run_data_init_if_needed(self):
        """Check if any features need to be extracted and proceed with data
        extraction"""
        if any(self.features):
            self.data = self.run_all_data_init()
            mask = np.isinf(self.data)
            self.data[mask] = np.nan
            nan_perc = 100 * np.isnan(self.data).sum() / self.data.size
            if nan_perc > 0:
                msg = 'Data has {:.3f}% NaN values!'.format(nan_perc)
                logger.warning(msg)
                warnings.warn(msg)

    def _raster_size_check(self):
        """Check if the sample_shape is larger than the requested raster
        size"""
        bad_shape = (self.sample_shape[0] > self.grid_shape[0]
                     and self.sample_shape[1] > self.grid_shape[1])
        if bad_shape:
            msg = (f'spatial_sample_shape {self.sample_shape[:2]} is '
                   f'larger than the raster size {self.grid_shape}')
            logger.warning(msg)
            warnings.warn(msg)

    def _val_split_check(self):
        """Check if val_split > 0 and split data into validation and training.
        Make sure validation data is larger than sample_shape"""

        if self.data is not None and self.val_split > 0.0:
            self.data, self.val_data = self.split_data(
                val_split=self.val_split, shuffle_time=self.shuffle_time)
            msg = (f'Validation data has shape={self.val_data.shape} '
                   f'and sample_shape={self.sample_shape}. Use a smaller '
                   'sample_shape and/or larger val_split.')
            check = any(
                val_size < samp_size for val_size,
                samp_size in zip(self.val_data.shape, self.sample_shape))
            if check:
                logger.warning(msg)
                warnings.warn(msg)

    @classmethod
    @abstractmethod
    def get_full_domain(cls, file_paths):
        """Get target and shape for full domain"""

    def clear_data(self):
        """Free memory used for data arrays"""
        self.data = None
        self.val_data = None

    @classmethod
    @abstractmethod
    def source_handler(cls, file_paths, **kwargs):
        """Handle for source data. Uses xarray, ResourceX, etc.

        NOTE: that xarray appears to treat open file handlers as singletons
        within a threadpool, so its okay to open this source_handler without a
        context handler or a .close() statement.
        """

    @property
    def attrs(self):
        """Get atttributes of input data

        Returns
        -------
        dict
            Dictionary of attributes
        """
        handle = self.source_handler(self.file_paths)
        desc = handle.attrs
        return desc

    @property
    def extract_workers(self):
        """Get upper bound for extract workers based on memory limits. Used to
        extract data from source dataset. The max number of extract workers
        is number of time chunks * number of features"""
        proc_mem = 4 * self.grid_mem * len(self.time_index)
        proc_mem /= len(self.time_chunks)
        n_procs = len(self.time_chunks) * len(self.extract_features)
        n_procs = int(np.ceil(n_procs))
        extract_workers = estimate_max_workers(self._extract_workers,
                                               proc_mem,
                                               n_procs)
        return extract_workers

    @property
    def compute_workers(self):
        """Get upper bound for compute workers based on memory limits. Used to
        compute derived features from source dataset."""
        proc_mem = int(
            np.ceil(
                len(self.extract_features)
                / np.maximum(len(self.derive_features), 1)))
        proc_mem *= 4 * self.grid_mem * len(self.time_index)
        proc_mem /= len(self.time_chunks)
        n_procs = len(self.time_chunks) * len(self.derive_features)
        n_procs = int(np.ceil(n_procs))
        compute_workers = estimate_max_workers(self._compute_workers,
                                               proc_mem,
                                               n_procs)
        return compute_workers

    @property
    def load_workers(self):
        """Get upper bound on load workers based on memory limits. Used to load
        cached data."""
        proc_mem = 2 * self.feature_mem
        n_procs = 1
        if self.cache_files is not None:
            n_procs = len(self.cache_files)
        load_workers = estimate_max_workers(self._load_workers,
                                            proc_mem,
                                            n_procs)
        return load_workers

    @property
    def time_chunks(self):
        """Get time chunks which will be extracted from source data

        Returns
        -------
        _time_chunks : list
            List of time chunks used to split up source data time dimension
            so that each chunk can be extracted individually
        """
        if self._time_chunks is None:
            if self.is_time_independent:
                self._time_chunks = [slice(None)]
            else:
                self._time_chunks = get_chunk_slices(len(self.raw_time_index),
                                                     self.time_chunk_size,
                                                     self.temporal_slice)
        return self._time_chunks

    @property
    def is_time_independent(self):
        """Get whether source data files are time independent"""
        return self.raw_time_index[0] is None

    @property
    def n_tsteps(self):
        """Get number of time steps to extract"""
        if self.is_time_independent:
            return 1
        else:
            return len(self.raw_time_index[self.temporal_slice])

    @property
    def time_chunk_size(self):
        """Get upper bound on time chunk size based on memory limits"""
        if self._time_chunk_size is None:
            step_mem = self.feature_mem * len(self.extract_features)
            step_mem /= len(self.time_index)
            if step_mem == 0:
                self._time_chunk_size = self.n_tsteps
            else:
                self._time_chunk_size = np.min(
                    [int(1e9 / step_mem), self.n_tsteps])
            logger.info('time_chunk_size arg not specified. Using '
                        f'{self._time_chunk_size}.')
        return self._time_chunk_size

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
            f for f in self.raw_features if self.lookup(f, 'compute') is None
            or Feature.get_basename(f.lower()) in lower_features
        ]

    @property
    def derive_features(self):
        """List of features which need to be derived from other features"""
        derive_features = [
            f for f in set(
                list(self.noncached_features) + list(self.extract_features))
            if f not in self.extract_features
        ]
        return derive_features

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
                self.noncached_features, self.handle_features)

        return self._raw_features

    @property
    def lr_only_features(self):
        """List of feature names or patt*erns that should only be included in
        the low-res training set and not the high-res observations."""
        if isinstance(self._lr_only_features, str):
            self._lr_only_features = [self._lr_only_features]

        elif isinstance(self._lr_only_features, tuple):
            self._lr_only_features = list(self._lr_only_features)

        elif self._lr_only_features is None:
            self._lr_only_features = []

        return self._lr_only_features

    @property
    def lr_features(self):
        """Get a list of low-resolution features. It is assumed that all
        features are used in the low-resolution observations. If you want to
        use high-res-only features, use the DualDataHandler class."""
        return self.features

    @property
    def hr_exo_features(self):
        """Get a list of exogenous high-resolution features that are only used
        for training e.g., mid-network high-res topo injection. These must come
        at the end of the high-res feature set. These can also be input to the
        model as low-res features."""

        if isinstance(self._hr_exo_features, str):
            self._hr_exo_features = [self._hr_exo_features]

        elif isinstance(self._hr_exo_features, tuple):
            self._hr_exo_features = list(self._hr_exo_features)

        elif self._hr_exo_features is None:
            self._hr_exo_features = []

        if any('*' in fn for fn in self._hr_exo_features):
            hr_exo_features = []
            for feature in self.features:
                match = any(fnmatch(feature.lower(), pattern.lower())
                            for pattern in self._hr_exo_features)
                if match:
                    hr_exo_features.append(feature)
            self._hr_exo_features = hr_exo_features

        if len(self._hr_exo_features) > 0:
            msg = (f'High-res train-only features "{self._hr_exo_features}" '
                   f'do not come at the end of the full high-res feature set: '
                   f'{self.features}')
            last_feat = self.features[-len(self._hr_exo_features):]
            assert list(self._hr_exo_features) == list(last_feat), msg

        return self._hr_exo_features

    @property
    def hr_out_features(self):
        """Get a list of high-resolution features that are intended to be
        output by the GAN. Does not include high-resolution exogenous
        features"""

        out = []
        for feature in self.features:
            lr_only = any(fnmatch(feature.lower(), pattern.lower())
                          for pattern in self.lr_only_features)
            ignore = lr_only or feature in self.hr_exo_features
            if not ignore:
                out.append(feature)

        if len(out) == 0:
            msg = (f'It appears that all handler features "{self.features}" '
                   'were specified as `hr_exo_features` or `lr_only_features` '
                   'and therefore there are no output features!')
            logger.error(msg)
            raise RuntimeError(msg)

        return out

    @property
    def grid_mem(self):
        """Get memory used by a feature at a single time step

        Returns
        -------
        int
            Number of bytes for a single feature array at a single time step
        """
        grid_mem = np.prod(self.grid_shape)
        # assuming feature arrays are float32 (4 bytes)
        return 4 * grid_mem

    @property
    def feature_mem(self):
        """Number of bytes for a single feature array. Used to estimate
        max_workers.

        Returns
        -------
        int
            Number of bytes for a single feature array
        """
        feature_mem = self.grid_mem * len(self.time_index)
        return feature_mem

    def preflight(self):
        """Run some preflight checks and verify that the inputs are valid"""

        self.cap_worker_args(self.max_workers)

        if len(self.sample_shape) == 2:
            logger.info(
                'Found 2D sample shape of {}. Adding temporal dim of 1'.format(
                    self.sample_shape))
            self.sample_shape = (*self.sample_shape, 1)

        start = self.temporal_slice.start
        stop = self.temporal_slice.stop

        msg = (f'sample_shape[2] ({self.sample_shape[2]}) cannot be larger '
               'than the number of time steps in the raw data '
               f'({len(self.raw_time_index)}).')
        if len(self.raw_time_index) < self.sample_shape[2]:
            logger.warning(msg)
            warnings.warn(msg)

        msg = (f'The requested time slice {self.temporal_slice} conflicts '
               f'with the number of time steps ({len(self.raw_time_index)}) '
               'in the raw data')
        t_slice_is_subset = start is not None and stop is not None
        good_subset = (t_slice_is_subset
                       and (stop - start <= len(self.raw_time_index))
                       and stop <= len(self.raw_time_index)
                       and start <= len(self.raw_time_index))
        if t_slice_is_subset and not good_subset:
            logger.error(msg)
            raise RuntimeError(msg)

        msg = (f'Initializing DataHandler {self.input_file_info}. '
               f'Getting temporal range {self.time_index[0]!s} to '
               f'{self.time_index[-1]!s} (inclusive) '
               f'based on temporal_slice {self.temporal_slice}')
        logger.info(msg)

        logger.info(f'Using max_workers={self.max_workers}, '
                    f'norm_workers={self.norm_workers}, '
                    f'extract_workers={self.extract_workers}, '
                    f'compute_workers={self.compute_workers}, '
                    f'load_workers={self.load_workers}, '
                    f'ti_workers={self.ti_workers}')

    @staticmethod
    def get_closest_lat_lon(lat_lon, target):
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
        dist = np.hypot(lat_lon[..., 0] - target[0],
                        lat_lon[..., 1] - target[1])
        row, col = np.where(dist == np.min(dist))
        row = row[0]
        col = col[0]
        return row, col

    def get_lat_lon_df(self, target, features=None):
        """Get timeseries for given target

        Parameters
        ----------
        target : tuple
            (lat, lon) for target coordinate
        features : list | None
            Optional list of features to include in returned data. If None then
            all available features are returned.

        Returns
        -------
        df : pd.DataFrame
            Pandas dataframe with columns for each feature and timeindex for
            the given target
        """
        row, col = self.get_closest_lat_lon(self.lat_lon, target)
        df = pd.DataFrame()
        df['time'] = self.time_index
        if self.data is None:
            self.load_cached_data()
        data = self.data[row, col]
        features = features if features is not None else self.features
        for f in features:
            i = self.features.index(f)
            df[f] = data[:, i]
        return df

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

    @classmethod
    def get_node_cmd(cls, config):
        """Get a CLI call to initialize DataHandler and cache data.

        Parameters
        ----------
        config : dict
            sup3r data handler config with all necessary args and kwargs to
            initialize DataHandler and run data extraction.
        """
        import_str = ('from sup3r.preprocessing.data_handling '
                      f'import {cls.__name__};\n'
                      'import time;\n'
                      'from gaps import Status;\n'
                      'from rex import init_logger;\n')
        dh_init_str = get_fun_call_str(cls, config)

        log_file = config.get('log_file', None)
        log_level = config.get('log_level', 'INFO')
        log_arg_str = f'"sup3r", log_level="{log_level}"'
        if log_file is not None:
            log_arg_str += f', log_file="{log_file}"'

        cache_check = config.get('cache_pattern', False)

        msg = 'No cache file prefix provided.'
        if not cache_check:
            logger.warning(msg)
            warnings.warn(msg)

        cmd = (f"python -c \'{import_str}\n"
               "t0 = time.time();\n"
               f"logger = init_logger({log_arg_str});\n"
               f"data_handler = {dh_init_str};\n"
               "t_elap = time.time() - t0;\n")

        pipeline_step = config.get('pipeline_step') or ModuleName.DATA_EXTRACT
        cmd = BaseCLI.add_status_cmd(config, pipeline_step, cmd)
        cmd += ";\'\n"
        return cmd.replace('\\', '/')

    def get_cache_file_names(self,
                             cache_pattern,
                             grid_shape=None,
                             time_index=None,
                             target=None,
                             features=None):
        """Get names of cache files from cache_pattern and feature names

        Parameters
        ----------
        cache_pattern : str
            Pattern to use for cache file names
        grid_shape : tuple
            Shape of grid to use for cache file naming
        time_index : list | pd.DatetimeIndex
            Time index to use for cache file naming
        target : tuple
            Target to use for cache file naming
        features : list
            List of features to use for cache file naming

        Returns
        -------
        list
            List of cache file names
        """
        grid_shape = grid_shape if grid_shape is not None else self.grid_shape
        time_index = time_index if time_index is not None else self.time_index
        target = target if target is not None else self.target
        features = features if features is not None else self.features

        return self._get_cache_file_names(cache_pattern,
                                          grid_shape,
                                          time_index,
                                          target,
                                          features)

    def get_next(self):
        """Get data for observation using random observation index. Loops
        repeatedly over randomized time index

        Returns
        -------
        observation : np.ndarray
            4D array
            (spatial_1, spatial_2, temporal, features)
        """
        self.current_obs_index = self._get_observation_index(
            self.data, self.sample_shape)
        observation = self.data[self.current_obs_index]
        return observation

    def split_data(self, data=None, val_split=0.0, shuffle_time=False):
        """Split time dimension into set of training indices and validation
        indices

        Parameters
        ----------
        data : np.ndarray
            4D array of high res data
            (spatial_1, spatial_2, temporal, features)
        val_split : float
            Fraction of data to separate for validation.
        shuffle_time : bool
            Whether to shuffle time or not.

        Returns
        -------
        data : np.ndarray
            (spatial_1, spatial_2, temporal, features)
            Training data fraction of initial data array. Initial data array is
            overwritten by this new data array.
        val_data : np.ndarray
            (spatial_1, spatial_2, temporal, features)
            Validation data fraction of initial data array.
        """
        data = data if data is not None else self.data

        assert len(self.time_index) == self.data.shape[-2]

        train_indices, val_indices = self._split_data_indices(
            data, val_split=val_split, shuffle_time=shuffle_time)
        self.val_data = self.data[:, :, val_indices, :]
        self.data = self.data[:, :, train_indices, :]

        self.val_time_index = self.time_index[val_indices]
        self.time_index = self.time_index[train_indices]

        return self.data, self.val_data

    @property
    def shape(self):
        """Full data shape

        Returns
        -------
        shape : tuple
            Full data shape
            (spatial_1, spatial_2, temporal, features)
        """
        return self.data.shape

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
        self._cache_data(self.data,
                         self.features,
                         cache_file_paths,
                         self.overwrite_cache)

    @property
    def requested_shape(self):
        """Get requested shape for cached data"""
        shape = get_raster_shape(self.raster_index)
        requested_shape = (shape[0] // self.hr_spatial_coarsen,
                           shape[1] // self.hr_spatial_coarsen,
                           len(self.raw_time_index[self.temporal_slice]),
                           len(self.features))
        return requested_shape

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
            msg = ('Found {} cache files but need {} for features {}! '
                   'These are the cache files that were found: {}'.format(
                       len(self.cache_files),
                       len(self.features),
                       self.features,
                       self.cache_files))
            assert len(self.cache_files) == len(self.features), msg

            self.data = np.full(shape=self.requested_shape,
                                fill_value=np.nan,
                                dtype=np.float32)

            logger.info(f'Loading cached data from: {self.cache_files}')
            max_workers = self.load_workers
            self._load_cached_data(data=self.data,
                                   cache_files=self.cache_files,
                                   features=self.features,
                                   max_workers=max_workers)

            self.time_index = self.raw_time_index[self.temporal_slice]

            nan_perc = 100 * np.isnan(self.data).sum() / self.data.size
            if nan_perc > 0:
                msg = 'Data has {:.3f}% NaN values!'.format(nan_perc)
                logger.warning(msg)
                warnings.warn(msg)

            if with_split and self.val_split > 0:
                logger.debug('Splitting data into training / validation sets '
                             f'({1 - self.val_split}, {self.val_split}) '
                             f'for {self.input_file_info}')

                self.data, self.val_data = self.split_data(
                    val_split=self.val_split, shuffle_time=self.shuffle_time)

    def run_all_data_init(self):
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
            shifted_time_chunks = get_chunk_slices(n_steps,
                                                   self.time_chunk_size)

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
            self.data = spatial_coarsening(self.data,
                                           s_enhance=self.hr_spatial_coarsen,
                                           obs_axis=False)
        if self.load_cached:
            for f in self.cached_features:
                f_index = self.features.index(f)
                logger.info(f'Loading {f} from {self.cache_files[f_index]}')
                with open(self.cache_files[f_index], 'rb') as fh:
                    self.data[..., f_index] = pickle.load(fh)

        logger.info(f'Finished extracting data for {self.input_file_info} in '
                    f'{dt.now() - now}')

        return self.data.astype(np.float32)

    def run_nn_fill(self):
        """Run nn nan fill on full data array."""
        for i in range(self.data.shape[-1]):
            if np.isnan(self.data[..., i]).any():
                self.data[..., i] = nn_fill_array(self.data[..., i])

    def mask_nan(self):
        """Drop timesteps with NaN data"""
        nan_mask = np.isnan(self.data).any(axis=(0, 1, 3))
        logger.info('Removing {} out of {} timesteps due to NaNs'.format(
            nan_mask.sum(), self.data.shape[2]))
        self.data = self.data[:, :, ~nan_mask, :]

    def run_data_extraction(self):
        """Run the raw dataset extraction process from disk to raw
        un-manipulated datasets.
        """
        if self.extract_features:
            logger.info(f'Starting extraction of {self.extract_features} '
                        f'using {len(self.time_chunks)} time_chunks.')
            if self.extract_workers == 1:
                self._raw_data = self.serial_extract(self.file_paths,
                                                     self.raster_index,
                                                     self.time_chunks,
                                                     self.extract_features,
                                                     **self.res_kwargs)

            else:
                self._raw_data = self.parallel_extract(self.file_paths,
                                                       self.raster_index,
                                                       self.time_chunks,
                                                       self.extract_features,
                                                       self.extract_workers,
                                                       **self.res_kwargs)

            logger.info(f'Finished extracting {self.extract_features} for '
                        f'{self.input_file_info}')

    def run_data_compute(self):
        """Run the data computation / derivation from raw features to desired
        features.
        """
        if self.derive_features:
            logger.info(f'Starting computation of {self.derive_features}')

            if self.compute_workers == 1:
                self._raw_data = self.serial_compute(self._raw_data,
                                                     self.file_paths,
                                                     self.raster_index,
                                                     self.time_chunks,
                                                     self.derive_features,
                                                     self.noncached_features,
                                                     self.handle_features)

            elif self.compute_workers != 1:
                self._raw_data = self.parallel_compute(self._raw_data,
                                                       self.file_paths,
                                                       self.raster_index,
                                                       self.time_chunks,
                                                       self.derive_features,
                                                       self.noncached_features,
                                                       self.handle_features,
                                                       self.compute_workers)

            logger.info(f'Finished computing {self.derive_features} for '
                        f'{self.input_file_info}')

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
            logger.info(f'Added {t + 1} of {len(shifted_time_chunks)} '
                        'chunks to final data array')
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
        self.data = np.zeros((self.grid_shape[0],
                              self.grid_shape[1],
                              self.n_tsteps,
                              len(self.features)),
                             dtype=np.float32)

        if max_workers == 1:
            self.serial_data_fill(shifted_time_chunks)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                futures = {}
                now = dt.now()
                for t, ts in enumerate(shifted_time_chunks):
                    for _, f in enumerate(self.noncached_features):
                        f_index = self.features.index(f)
                        future = exe.submit(self._single_data_fill,
                                            t, ts, f_index, f)
                        futures[future] = {'t': t, 'fidx': f_index}

                logger.info(f'Started adding {len(futures)} chunks '
                            f'to data array in {dt.now() - now}.')

                for i, future in enumerate(as_completed(futures)):
                    try:
                        future.result()
                    except Exception as e:
                        msg = (f'Error adding ({futures[future]["t"]}, '
                               f'{futures[future]["fidx"]}) chunk to '
                               'final data array.')
                        logger.exception(msg)
                        raise RuntimeError(msg) from e
                    logger.debug(f'Added {i+1} out of {len(futures)} '
                                 'chunks to final data array')
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
                    check = (dset_scalar.lower() in dsets
                             and dset_adder.lower() in dsets)
                if feature not in completed and check:
                    scalar, adder = get_spatial_bc_factors(
                        lat_lon=self.lat_lon,
                        feature_name=feature,
                        bias_fp=fp,
                        threshold=threshold)

                    if scalar.shape[-1] == 1:
                        scalar = np.repeat(scalar, self.shape[2], axis=2)
                        adder = np.repeat(adder, self.shape[2], axis=2)
                    elif scalar.shape[-1] == 12:
                        idm = self.time_index.month.values - 1
                        scalar = scalar[..., idm]
                        adder = adder[..., idm]
                    else:
                        msg = ('Can only accept bias correction factors '
                               'with last dim equal to 1 or 12 but '
                               'received bias correction factors with '
                               'shape {}'.format(scalar.shape))
                        logger.error(msg)
                        raise RuntimeError(msg)

                    logger.info('Bias correcting "{}" with linear '
                                'correction from "{}"'.format(
                                    feature, os.path.basename(fp)))
                    self.data[..., idf] *= scalar
                    self.data[..., idf] += adder
                    completed.append(feature)

    def qdm_bc(self,
               bc_files,
               reference_feature,
               relative=True,
               threshold=0.1,
               no_trend=False):
        """Bias Correction using Quantile Delta Mapping

        Bias correct this DataHandler's data with Quantile Delta Mapping. The
        required statistical distributions should be pre-calculated using
        :class:`sup3r.bias.qdm.QuantileDeltaMappingCorrection`.

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
        no_trend: bool, default=False
            An option to ignore the trend component of the correction, thus
            resulting in an ordinary Quantile Mapping, i.e. corrects the bias
            by comparing the distributions of the biased dataset with a
            reference datasets. See ``params_mf`` of
            :class:`rex.utilities.bc_utils.QuantileDeltaMapping`.
            Note that this assumes that "bias_{feature}_params"
            (``params_mh``) is the data distribution representative for the
            target data.
        """

        if isinstance(bc_files, str):
            bc_files = [bc_files]

        completed = []
        for idf, feature in enumerate(self.features):
            for fp in bc_files:
                logger.info('Bias correcting "{}" with QDM '
                            'correction from "{}"'.format(
                                feature, os.path.basename(fp)))
                self.data[..., idf] = local_qdm_bc(data=self.data[..., idf],
                                                   lat_lon=self.lat_lon,
                                                   base_dset=reference_feature,
                                                   feature_name=feature,
                                                   bias_fp=fp,
                                                   time_index=self.time_index,
                                                   threshold=threshold,
                                                   relative=relative,
                                                   no_trend=no_trend)
                completed.append(feature)


# pylint: disable=W0223
class DataHandlerDC(DataHandler):
    """Data-centric data handler"""

    FEATURE_REGISTRY: ClassVar[dict] = {
        'BVF2_(.*)m': BVFreqSquaredNC,
        'BVF_MO_(.*)m': BVFreqMon,
        'RMOL': InverseMonNC,
        'U_(.*)': UWind,
        'V_(.*)': VWind,
        'Windspeed_(.*)m': WindspeedNC,
        'Winddirection_(.*)m': WinddirectionNC,
        'lat_lon': LatLonNC,
        'Shear_(.*)m': Shear,
        'REWS_(.*)m': Rews,
        'Temperature_(.*)m': TempNC,
        'Pressure_(.*)m': PressureNC,
        'PotentialTemp_(.*)m': PotentialTempNC,
        'PT_(.*)m': PotentialTempNC,
        'topography': ['HGT', 'orog']
    }

    def get_observation_index(self,
                              temporal_weights=None,
                              spatial_weights=None):
        """Randomly gets weighted spatial sample and time sample

        Parameters
        ----------
        temporal_weights : array
            Weights used to select time slice
            (n_time_chunks)
        spatial_weights : array
            Weights used to select spatial chunks
            (n_lat_chunks * n_lon_chunks)

        Returns
        -------
        observation_index : tuple
            Tuple of sampled spatial grid, time slice, and features indices.
            Used to get single observation like self.data[observation_index]
        """
        if spatial_weights is not None:
            spatial_slice = weighted_box_sampler(self.data,
                                                 self.sample_shape[:2],
                                                 weights=spatial_weights)
        else:
            spatial_slice = uniform_box_sampler(self.data,
                                                self.sample_shape[:2])
        if temporal_weights is not None:
            temporal_slice = weighted_time_sampler(self.data,
                                                   self.sample_shape[2],
                                                   weights=temporal_weights)
        else:
            temporal_slice = uniform_time_sampler(self.data,
                                                  self.sample_shape[2])

        return (*spatial_slice, temporal_slice, np.arange(len(self.features)))

    def get_next(self, temporal_weights=None, spatial_weights=None):
        """Get data for observation using weighted random observation index.
        Loops repeatedly over randomized time index.

        Parameters
        ----------
        temporal_weights : array
            Weights used to select time slice
            (n_time_chunks)
        spatial_weights : array
            Weights used to select spatial chunks
            (n_lat_chunks * n_lon_chunks)

        Returns
        -------
        observation : np.ndarray
            4D array
            (spatial_1, spatial_2, temporal, features)
        """
        self.current_obs_index = self.get_observation_index(
            temporal_weights=temporal_weights, spatial_weights=spatial_weights)
        observation = self.data[self.current_obs_index]
        return observation
