# -*- coding: utf-8 -*-
"""
Sup3r preprocessing module.
@author: bbenton
"""

from abc import abstractmethod
from fnmatch import fnmatch
import logging
import xarray as xr
import numpy as np
import os
from datetime import datetime as dt
import pickle
import warnings
import glob
from concurrent.futures import (as_completed, ThreadPoolExecutor)

from rex import MultiFileWindX, MultiFileNSRDBX
from rex.utilities import log_mem
from rex.utilities.fun_utils import get_fun_call_str

from sup3r.utilities.utilities import (estimate_max_workers, get_chunk_slices,
                                       interp_var_to_height,
                                       interp_var_to_pressure,
                                       uniform_box_sampler,
                                       uniform_time_sampler,
                                       weighted_time_sampler,
                                       get_raster_shape,
                                       ignore_case_path_fetch,
                                       get_source_type,
                                       daily_temporal_coarsening,
                                       spatial_coarsening)
from sup3r.preprocessing.feature_handling import (FeatureHandler,
                                                  Feature,
                                                  BVFreqMon,
                                                  BVFreqSquaredH5,
                                                  BVFreqSquaredNC,
                                                  InverseMonNC,
                                                  LatLonNC,
                                                  LatLonNCforCC,
                                                  TempNC,
                                                  UWindH5,
                                                  VWindH5,
                                                  UWindNsrdb,
                                                  VWindNsrdb,
                                                  LatLonH5,
                                                  ClearSkyRatioH5,
                                                  CloudMaskH5,
                                                  WindspeedNC,
                                                  WinddirectionNC,
                                                  Shear,
                                                  Rews
                                                  )

np.random.seed(42)

logger = logging.getLogger(__name__)


def get_handler_class(file_paths):
    """Method to get source type specific DataHandler class

    Parameters
    ----------
    file_paths : list
        list of file paths

    Returns
    -------
    DataHandler
        Either DataHandlerNC, DataHandlerH5, DataHandlerH5SolarCC
    """
    if get_source_type(file_paths) == 'h5':
        HandlerClass = DataHandlerH5
        if all('nsrdb' in os.path.basename(fp) for fp in file_paths):
            HandlerClass = DataHandlerH5SolarCC
    else:
        HandlerClass = DataHandlerNC
    return HandlerClass


class DataHandler(FeatureHandler):
    """Sup3r data handling and extraction"""

    # list of features / feature name patterns that are input to the generative
    # model but are not part of the synthetic output and are not sent to the
    # discriminator. These are case-insensitive and follow the Unix shell-style
    # wildcard format.
    TRAIN_ONLY_FEATURES = ('BVF*', 'inversemoninobukhovlength_*', 'RMOL')

    def __init__(self, file_paths, features, target=None, shape=None,
                 max_delta=50,
                 temporal_slice=slice(None),
                 hr_spatial_coarsen=None,
                 time_roll=0,
                 val_split=0.1,
                 sample_shape=(10, 10, 1),
                 raster_file=None,
                 shuffle_time=False,
                 extract_workers=None,
                 compute_workers=None,
                 time_chunk_size=100,
                 cache_file_prefix=None,
                 overwrite_cache=False,
                 load_cached=False,
                 train_only_features=None):
        """Data handling and extraction

        Parameters
        ----------
        file_paths : str | list
            A single source h5 wind file to extract raster data from or a list
            of netcdf files with identical grid. The string can be a unix-style
            file path which will be passed through glob.glob
        features : list
            list of features to extract
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
            File for raster_index array for the corresponding target and shape.
            If specified the raster_index will be loaded from the file if it
            exists or written to the file if it does not yet exist.  If None
            raster_index will be calculated directly. Either need target+shape
            or raster_file.
        shuffle_time : bool
            Whether to shuffle time indices before valiidation split
        compute_workers : int | None
            max number of workers to use for computing features. If
            compute_workers == 1 then extraction will be serialized.
        extract_workers : int | None
            max number of workers to use for data extraction. If
            extract_workers == 1 then extraction will be serialized.
        time_chunk_size : int
            Size of chunks to split time dimension into for parallel data
            extraction. If running in serial this can be set to the size of the
            full time index for best performance.
        cache_file_prefix : str | None
            Prefix of files for saving feature data. Each feature will be saved
            to a file with the feature name appended to the end of
            cache_file_prefix. If not None feature arrays will be saved here
            and not stored in self.data until load_cached_data is called.
        overwrite_cache : bool
            Whether to overwrite any previously saved cache files.
        load_cached : bool
            Whether to load data from cache files
        train_only_features : list | tuple | None
            List of feature names or patt*erns that should only be included in
            the training set and not the output. If None (default), this will
            default to the class TRAIN_ONLY_FEATURES attribute.
        """

        check = ((target is not None and shape is not None)
                 or (raster_file is not None and os.path.exists(raster_file)))
        msg = ('You must either provide the target+shape inputs or an existing'
               ' raster_file input.')
        assert check, msg

        super().__init__()

        if isinstance(file_paths, str):
            file_paths = glob.glob(file_paths)
        self.file_paths = sorted(file_paths)

        self.train_only_features = train_only_features
        if self.train_only_features is None:
            self.train_only_features = self.TRAIN_ONLY_FEATURES

        self.features = features
        self.grid_shape = shape
        self.val_time_index = None
        self.target = target
        self.max_delta = max_delta
        self.raster_file = raster_file
        self.raster_index = None
        self.val_split = val_split
        self.sample_shape = sample_shape
        self.temporal_slice = temporal_slice
        self.hr_spatial_coarsen = hr_spatial_coarsen or 1
        self.time_roll = time_roll
        self.raw_time_index = self.get_time_index(self.file_paths)
        self.time_index = self.raw_time_index[self.temporal_slice]
        self.shuffle_time = shuffle_time
        self.current_obs_index = None
        self.overwrite_cache = overwrite_cache
        self.load_cached = load_cached
        self.cache_files = self.get_cache_file_names(cache_file_prefix)
        self.data = None
        self.val_data = None
        self.lat_lon = None
        self.compute_workers = compute_workers
        self.extract_workers = extract_workers

        logger.info('Initializing DataHandler '
                    f'{self.file_info_logging(self.file_paths)}')

        self.preflight()

        try_load = (cache_file_prefix is not None
                    and not self.overwrite_cache
                    and all(os.path.exists(fp) for fp in self.cache_files))

        overwrite = (self.overwrite_cache
                     and self.cache_files is not None
                     and all(os.path.exists(fp) for fp in self.cache_files))

        if try_load and self.load_cached:
            logger.info(f'All {self.cache_files} exist. Loading from cache '
                        f'instead of extracting from {self.file_paths}')
            self.load_cached_data()

        elif try_load and not self.load_cached:
            self.data = None
            logger.info(f'All {self.cache_files} exist. Call '
                        'load_cached_data() or use load_cache=True to load '
                        'this data from cache files.')
        else:
            if overwrite:
                logger.info(f'{self.cache_files} exists but overwrite_cache '
                            'is set to True. Proceeding with extraction.')

            self.raster_index = self.get_raster_index(self.file_paths,
                                                      self.target,
                                                      self.grid_shape)

            raster_shape = get_raster_shape(self.raster_index)
            bad_shape = (sample_shape[0] > raster_shape[0]
                         and sample_shape[1] > raster_shape[1])
            if bad_shape:
                msg = (f'spatial_sample_shape {sample_shape[:2]} is larger '
                       f'than the raster size {raster_shape}')
                logger.warning(msg)
                warnings.warn(msg)

            self.data = self.extract_data(
                self.file_paths, self.raster_index, self.features,
                temporal_slice=self.temporal_slice,
                hr_spatial_coarsen=self.hr_spatial_coarsen,
                time_roll=self.time_roll,
                extract_workers=self.extract_workers,
                compute_workers=self.compute_workers,
                time_chunk_size=time_chunk_size,
                cache_files=self.cache_files,
                overwrite_cache=self.overwrite_cache)

            if cache_file_prefix is not None:
                self.cache_data(self.cache_files)
                self.data = None if not self.load_cached else self.data

            if self.data is not None:
                self.data, self.val_data = self.split_data()

        logger.info('Finished intializing DataHandler.')
        log_mem(logger, log_level='INFO')

    @classmethod
    @abstractmethod
    def source_handler(cls, file_paths):
        """Handler for source data. Can use xarray, ResourceX, etc."""

    @property
    def attrs(self):
        """Get atttributes of input data

        Returns
        -------
        dict
            Dictionary of attributes
        """
        with self.source_handler(self.file_paths) as handle:
            desc = handle.attrs
        return desc

    @classmethod
    def get_handle_features(cls, file_paths):
        """Lookup inputs needed to compute feature

        Parameters
        ----------
        file_paths : list
            List of data file paths

        Returns
        -------
        list
            List of available features in data
        """

        with cls.source_handler(file_paths) as handle:
            handle_features = [Feature.get_basename(r) for r in handle]
        return handle_features

    @property
    def feature_mem(self):
        """Number of bytes for a single feature array. Used to estimate
        max_workers.

        Returns
        -------
        int
            Number of bytes for a single feature array
        """
        if self.raster_index is None:
            self.raster_index = self.get_raster_index(self.file_paths)
        feature_mem = np.product(self.grid_shape) * len(self.time_index)
        return 4 * feature_mem

    def preflight(self):
        """Run some preflight checks and verify that the inputs are valid"""
        if len(self.sample_shape) == 2:
            logger.info('Found 2D sample shape of {}. Adding spatial dim of 1'
                        .format(self.sample_shape))
            self.sample_shape = self.sample_shape + (1,)

        start = self.temporal_slice.start
        stop = self.temporal_slice.stop
        n_steps = self.raw_time_index[start:stop]
        n_steps = len(n_steps)
        msg = (f'Temporal slice step ({self.temporal_slice.step}) does not '
               f'evenly divide the number of time steps ({n_steps})')
        check = self.temporal_slice.step is None
        check = check or n_steps % self.temporal_slice.step == 0
        if not check:
            logger.warning(msg)
            warnings.warn(msg)

        msg = (f'sample_shape[2] ({self.sample_shape[2]}) cannot be larger '
               'than the number of time steps in the raw data '
               f'({len(self.raw_time_index)}).')
        assert len(self.raw_time_index) >= self.sample_shape[2], msg

        msg = (f'The requested time slice {self.temporal_slice} conflicts '
               f'with the number of time steps ({len(self.raw_time_index)}) '
               'in the raw data')
        t_slice_is_subset = (start is not None and stop is not None)
        good_subset = (t_slice_is_subset
                       and (stop - start <= len(self.raw_time_index))
                       and stop <= len(self.raw_time_index)
                       and start <= len(self.raw_time_index))
        if t_slice_is_subset and not good_subset:
            logger.error(msg)
            raise RuntimeError(msg)

    @classmethod
    def get_lat_lon(cls, file_paths, raster_index, time_slice):
        """Store lat lon for future output

        Parameters
        ----------
        file_paths : list
            path to data file
        raster_index : ndarray | list
            Raster index array or list of slices
        time_slice : slice
            slice of time to extract

        Returns
        -------
        ndarray
            (spatial_1, spatial_2, 2) Lat/Lon array with same ordering in last
            dimension
        """
        return cls.extract_feature(file_paths, raster_index, 'lat_lon',
                                   time_slice)

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
                      f'import {cls.__name__}; from rex import init_logger')
        dh_init_str = get_fun_call_str(cls, config)
        log_file = config.get('log_file', None)
        log_level = config.get('log_level', 'INFO')
        log_arg_str = '\"sup3r\", '
        log_arg_str += f'log_file=\"{log_file}\", '
        log_arg_str += f'log_level=\"{log_level}\"'
        cache_check = config.get('cache_file_prefix', False)
        msg = ('No cache file prefix provided.')
        if not cache_check:
            logger.warning(msg)
            warnings.warn(msg)

        cmd = (f"python -c \'{import_str};\n"
               f"logger = init_logger({log_arg_str});\n"
               f"data_handler = {dh_init_str};\'\n").replace('\\', '/')

        return cmd

    def get_cache_file_names(self, cache_file_prefix):
        """Get names of cache files from cache_file_prefix and feature names

        Parameters
        ----------
        cache_file_prefix : str
            Prefix to use for cache file names

        Returns
        -------
        list
            List of cache file names
        """
        if cache_file_prefix is not None:
            basedir = os.path.dirname(cache_file_prefix)
            if not os.path.exists(basedir):
                os.makedirs(basedir)
            cache_files = [
                f'{cache_file_prefix}_{f.lower()}.pkl' for f in self.features]
            for i, fp in enumerate(cache_files):
                fp_check = ignore_case_path_fetch(fp)
                if fp_check is not None:
                    cache_files[i] = fp_check
        else:
            cache_files = None

        return cache_files

    @classmethod
    def file_info_logging(cls, file_paths):
        """Method to provide info about files in log output. Since NETCDF files
        have single time slices printing out all the file paths is just a text
        dump without much info.

        Parameters
        ----------
        file_paths : list
            List of file paths

        Returns
        -------
        str
            message to append to log output that does not include a huge info
            dump of file paths
        """

        msg = (f'source files: {file_paths}')
        return msg

    @property
    def output_features(self):
        """Get a list of features that should be output by the generative model
        corresponding to the features in the high res batch array."""
        out = []
        for feature in self.features:
            ignore = any(fnmatch(feature.lower(), pattern.lower())
                         for pattern in self.train_only_features)
            if not ignore:
                out.append(feature)
        return out

    def unnormalize(self, means, stds):
        """Remove normalization from stored means and stds"""
        for i in range(self.shape[-1]):
            self.val_data[..., i] = self.val_data[..., i] * stds[i] + means[i]
            self.data[..., i] = self.data[..., i] * stds[i] + means[i]

    def normalize(self, means, stds):
        """Normalize all data features

        Parameters
        ----------
        means : np.ndarray
            dimensions (features)
            array of means for all features with same ordering as data features
        stds : np.ndarray
            dimensions (features)
            array of means for all features with same ordering as data features
        """
        if self.extract_workers == 1:
            for i in range(self.shape[-1]):
                self._normalize_data(i, means[i], stds[i])
        else:
            self.parallel_normalization(means, stds)

    def parallel_normalization(self, means, stds):
        """Run normalization of features in parallel

        Parameters
        ----------
        means : np.ndarray
            dimensions (features)
            array of means for all features with same ordering as data features
        stds : np.ndarray
            dimensions (features)
            array of means for all features with same ordering as data features
        """
        proc_mem = 2 * self.feature_mem
        logger.info(f'Normalizing {self.shape[-1]} features.')
        max_workers = estimate_max_workers(self.extract_workers,
                                           proc_mem, self.shape[-1])
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = {}
            now = dt.now()
            for i in range(self.shape[-1]):
                future = exe.submit(self._normalize_data, i, means[i],
                                    stds[i])
                futures[future] = i

            logger.info(f'Started normalizing {self.shape[-1]} features '
                        f'in {dt.now() - now}.')

            for i, future in enumerate(as_completed(futures)):
                future.result()
                logger.debug(f'{i + 1} out of {self.shape[-1]} features '
                             'normalized.')

    def _normalize_data(self, feature_index, mean, std):
        """Normalize data with initialized mean and standard deviation for a
        specific feature

        Parameters
        ----------
        feature_index : int
            index of feature to be normalized
        mean : float32
            specified mean of associated feature
        std : float32
            specificed standard deviation for associated feature
        """

        self.val_data[..., feature_index] -= mean
        self.data[..., feature_index] -= mean

        if std > 0:
            self.val_data[..., feature_index] /= std
            self.data[..., feature_index] /= std
        else:
            msg = ('Standard Deviation is zero for '
                   f'{self.features[feature_index]}')
            logger.warning(msg)
            warnings.warn(msg)

    def get_observation_index(self):
        """Randomly gets spatial sample and time sample

        Returns
        -------
        observation_index : tuple
            Tuple of sampled spatial grid, time slice, and features indices.
            Used to get single observation like self.data[observation_index]
        """
        spatial_slice = uniform_box_sampler(self.data, self.sample_shape[:2])
        temporal_slice = uniform_time_sampler(self.data, self.sample_shape[2])
        return tuple(
            spatial_slice + [temporal_slice] + [np.arange(len(self.features))])

    def get_next(self):
        """Gets data for observation using random observation index. Loops
        repeatedly over randomized time index

        Returns
        -------
        observation : np.ndarray
            4D array
            (spatial_1, spatial_2, temporal, features)
        """
        self.current_obs_index = self.get_observation_index()
        observation = self.data[self.current_obs_index]
        return observation

    def split_data(self, data=None):
        """Splits time dimension into set of training indices and validation
        indices

        Parameters
        ----------
        data : np.ndarray
            4D array of high res data
            (spatial_1, spatial_2, temporal, features)

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

        if data is not None:
            self.data = data

        n_observations = self.data.shape[2]
        all_indices = np.arange(n_observations)
        n_val_obs = int(self.val_split * n_observations)

        if self.shuffle_time:
            np.random.shuffle(all_indices)

        val_indices = all_indices[:n_val_obs]
        training_indices = all_indices[n_val_obs:]

        if not self.shuffle_time:
            [self.val_data, self.data] = np.split(self.data, [n_val_obs],
                                                  axis=2)
        else:
            self.val_data = self.data[:, :, val_indices, :]
            self.data = self.data[:, :, training_indices, :]

        self.val_time_index = self.time_index[val_indices]
        self.time_index = self.time_index[training_indices]

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

    def cache_data(self, cache_file_paths):
        """Cache feature data to file and delete from memory

        Parameters
        ----------
        cache_file_paths : str | None
            Path to file for saving feature data
        """

        for i, fp in enumerate(cache_file_paths):
            if not os.path.exists(fp) or self.overwrite_cache:
                if self.overwrite_cache and os.path.exists(fp):
                    logger.info(f'Overwriting {self.features[i]} with shape '
                                f'{self.data[..., i].shape} to {fp}')
                else:
                    logger.info(f'Saving {self.features[i]} with shape '
                                f'{self.data[..., i].shape} to {fp}')

                with open(fp, 'wb') as fh:
                    pickle.dump(self.data[..., i], fh, protocol=4)
            else:
                msg = (f'Called cache_data but {fp} already exists. Set to '
                       'overwrite_cache to True to overwrite.')
                logger.warning(msg)
                warnings.warn(msg)

    def parallel_load(self, max_workers=None):
        """Load feature data in parallel

        Parameters
        ----------
        max_workers : int | None
            Max number of workers to use for parallel data loading. If None
            the max number of available workers will be used.
        """
        proc_mem = 2 * self.feature_mem
        logger.info(f'Loading {len(self.cache_files)} cache files.')
        max_workers = estimate_max_workers(max_workers, proc_mem,
                                           len(self.cache_files))
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = {}
            now = dt.now()
            for i, fp in enumerate(self.cache_files):
                future = exe.submit(self.load_single_cached_feature, fp=fp)
                futures[future] = {'idx': i, 'fp': os.path.basename(fp)}

            logger.info(f'Started loading all {len(self.cache_files)} cache '
                        f'files in {dt.now() - now}.')

            for i, future in enumerate(as_completed(futures)):
                future.result()
                logger.debug(f'{i + 1} out of {len(futures)} cache files '
                             f'loaded: {futures[future]["fp"]}')

    def load_single_cached_feature(self, fp):
        """Load single feature from given file

        Parameters
        ----------
        fp : string
            File path for feature cache file

        Raises
        ------
        RuntimeError
            Error raised if shape conflicts with requested shape
        """
        idx = self.cache_files.index(fp)
        assert self.features[idx].lower() in fp.lower()
        fp = ignore_case_path_fetch(fp)
        logger.info(f'Loading {self.features[idx]} from '
                    f'{os.path.basename(fp)}')

        with open(fp, 'rb') as fh:
            try:
                self.data[..., idx] = np.array(pickle.load(fh),
                                               dtype=np.float32)
            except Exception as e:
                msg = ('Data loaded from from cache file "{}" '
                       'could not be written to feature channel {} '
                       'of full data array of shape {}. '
                       'Make sure the cached data has the '
                       'appropriate shape.'
                       .format(fp, idx, self.data.shape))
                raise RuntimeError(msg) from e

    def load_cached_data(self):
        """Load data from cache files and split into training and validation

        Parameters
        ----------
        max_workers : int | None
            Max number of workers to use for loading cached features. If None
            max available workers will be used. If 1 cached data will be loaded
            in serial
        """

        if self.data is not None:
            msg = ('Called load_cached_data() but self.data is not None')
            logger.warning(msg)
            warnings.warn(msg)

        elif self.data is None:
            self.raster_index = getattr(self, 'raster_index', None)
            if self.raster_index is None:
                self.raster_index = self.get_raster_index(
                    self.file_paths, self.target, self.grid_shape)

            shape = get_raster_shape(self.raster_index)
            requested_shape = (shape[0] // self.hr_spatial_coarsen,
                               shape[1] // self.hr_spatial_coarsen,
                               len(self.time_index),
                               len(self.features))

            msg = ('Found {} cache files but need {} for features {}! '
                   'These are the cache files that were found: {}'
                   .format(len(self.cache_files), len(self.features),
                           self.features, self.cache_files))
            assert len(self.cache_files) == len(self.features), msg

            self.data = np.full(shape=requested_shape, fill_value=np.nan,
                                dtype=np.float32)

            if self.extract_workers == 1:
                for _, fp in enumerate(self.cache_files):
                    self.load_single_cached_feature(fp)
            else:
                self.parallel_load(max_workers=self.extract_workers)

            nan_perc = (100 * np.isnan(self.data).sum() / self.data.size)
            if nan_perc > 0:
                msg = ('Data has {:.2f}% NaN values!'.format(nan_perc))
                logger.warning(msg)
                warnings.warn(msg)

            logger.debug('Splitting data into training / validation sets '
                         f'({1 - self.val_split}, {self.val_split}) '
                         f'for {self.file_info_logging(self.file_paths)}')
            self.data, self.val_data = self.split_data()

    @classmethod
    def check_cached_features(cls, file_paths, features, cache_files=None,
                              overwrite_cache=False, load_cached=False):
        """Check which features have been cached and check flags to determine
        whether to load or extract this features again

        Parameters
        ----------
        file_paths : str | list
            path to data file
        features : list
            list of features to extract
        cache_files : list | None
            Path to files with saved feature data
        overwrite_cache : bool
            Whether to overwrite cached files
        load_cached : bool
            Whether to load data from cache files

        Returns
        -------
        list
            List of features to extract. Might not include features which have
            cache files.
        """
        extract_features = []
        # check if any features can be loaded from cache
        if cache_files is not None:
            for i, f in enumerate(features):
                check = (os.path.exists(cache_files[i])
                         and f.lower() in cache_files[i].lower())
                if check:
                    if not overwrite_cache:
                        if load_cached:
                            msg = (f'{f} found in cache file {cache_files[i]}.'
                                   ' Loading from cache instead of extracting '
                                   f'from {file_paths}')
                            logger.info(msg)
                        else:
                            msg = (f'{f} found in cache file {cache_files[i]}.'
                                   ' Call load_cached_data() or use '
                                   'load_cached=True to load this data.')
                            logger.info(msg)
                    else:
                        msg = (f'{cache_files[i]} exists but overwrite_cache '
                               'is set to True. Proceeding with extraction.')
                        logger.info(msg)
                        extract_features.append(f)
                else:
                    extract_features.append(f)
        else:
            extract_features = features

        return extract_features

    @classmethod
    def extract_data(cls, file_paths, raster_index, features,
                     temporal_slice=slice(None, None, 1),
                     hr_spatial_coarsen=None, time_roll=0,
                     extract_workers=None, compute_workers=None,
                     time_chunk_size=100, cache_files=None,
                     overwrite_cache=False, load_cached=False):
        """Building base 4D data array. Can handle multiple files but assumes
        each file has the same spatial domain

        Parameters
        ----------
        file_paths : str | list
            path to data file
        raster_index : np.ndarray
            2D array of grid indices for H5 or list of slices for NETCDF
        features : list
            list of features to extract
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
        compute_workers : int | None
            max number of workers to use for computing features.
            If compute_workers == 1 then extraction will be serialized.
        extract_workers : int | None
            max number of workers to use for data extraction.
            If extract_workers == 1 then extraction will be serialized.
        time_chunk_size : int
            Size of chunks to split time dimension into for smaller data
            extractions
        cache_files : list | None
            Path to files with saved feature data
        overwrite_cache : bool
            Whether to overwrite cached files
        load_cached : bool
            Whether to load data from cache files

        Returns
        -------
        data : np.ndarray
            4D array of high res data
            (spatial_1, spatial_2, temporal, features)
        """

        now = dt.now()
        if not isinstance(file_paths, list):
            file_paths = [file_paths]

        shape = get_raster_shape(raster_index)
        logger.debug(f'Loading data for raster of shape {shape}')

        # get the file-native time index without pruning
        time_index = cls.get_time_index(file_paths)
        n_steps = len(time_index[temporal_slice])

        data_array = np.zeros((shape[0], shape[1], n_steps, len(features)),
                              dtype=np.float32)

        # split time dimension into smaller slices which can be
        # extracted in parallel
        time_chunks = get_chunk_slices(len(time_index), time_chunk_size,
                                       temporal_slice)
        shifted_time_chunks = get_chunk_slices(n_steps, time_chunk_size)

        extract_features = cls.check_cached_features(
            file_paths, features, cache_files=cache_files,
            overwrite_cache=overwrite_cache, load_cached=load_cached)

        handle_features = cls.get_handle_features(file_paths)
        raw_features = cls.get_raw_feature_list(extract_features,
                                                handle_features)

        proc_mem = 8 * np.product(shape) * len(time_index[temporal_slice])
        extract_workers = estimate_max_workers(extract_workers, proc_mem,
                                               len(extract_features))
        raw_data = cls.parallel_extract(file_paths, raster_index, time_chunks,
                                        raw_features, extract_workers)

        logger.info(f'Finished extracting {raw_features} for '
                    f'{cls.file_info_logging(file_paths)}')

        mult = np.int(np.ceil(len(raw_features) / len(extract_features)))
        proc_mem = mult * proc_mem
        compute_workers = estimate_max_workers(compute_workers, proc_mem,
                                               len(extract_features))
        raw_data = cls.parallel_compute(raw_data, raster_index, time_chunks,
                                        raw_features, extract_features,
                                        handle_features, compute_workers)

        logger.info(f'Finished computing {extract_features} for '
                    f'{cls.file_info_logging(file_paths)}')

        for t, t_slice in enumerate(shifted_time_chunks):
            for _, f in enumerate(extract_features):
                f_index = features.index(f)
                data_array[..., t_slice, f_index] = raw_data[t][f]
            raw_data.pop(t)

        data_array = np.roll(data_array, time_roll, axis=2)

        hr_spatial_coarsen = hr_spatial_coarsen or 1
        if hr_spatial_coarsen > 1:
            data_array = spatial_coarsening(data_array,
                                            s_enhance=hr_spatial_coarsen,
                                            obs_axis=False)
        if load_cached:
            for f in [f for f in features if f not in extract_features]:
                f_index = features.index(f)
                with open(cache_files[f_index], 'rb') as fh:
                    data_array[..., f_index] = pickle.load(fh)

        logger.info('Finished extracting data for '
                    f'{cls.file_info_logging(file_paths)} in {dt.now() - now}')

        return data_array

    @abstractmethod
    def get_raster_index(self, file_paths, target, shape):
        """Get raster index for file data. Here we assume the list of paths in
        file_paths all have data with the same spatial domain. We use the first
        file in the list to compute the raster

        Parameters
        ----------
        file_paths : str | list
            path to data file
        target : tuple
            (lat, lon) for lower left corner
        shape : tuple
            (n_rows, n_cols) grid size

        Returns
        -------
        raster_index : np.ndarray
            2D array of grid indices for H5 or list of
            slices for NETCDF
        """

    @classmethod
    @abstractmethod
    def get_time_index(cls, file_paths):
        """Get time index from input files"""


class DataHandlerNC(DataHandler):
    """Data Handler for NETCDF data"""

    @classmethod
    def source_handler(cls, file_paths):
        """Xarray data handler

        Parameters
        ----------
        file_paths : str | list
            paths to data files

        Returns
        -------
        data : xarray.Dataset
        """
        return xr.open_mfdataset(file_paths, combine='nested',
                                 concat_dim='Time')

    @classmethod
    def file_info_logging(cls, file_paths):
        """Method to provide info about files in log output. Since NETCDF files
        have single time slices printing out all the file paths is just a text
        dump without much info.

        Parameters
        ----------
        file_paths : list
            List of file paths

        Returns
        -------
        str
            message to append to log output that does not include a huge info
            dump of file paths
        """
        ti = cls.get_time_index(file_paths)
        msg = (f'source files for dates from {ti[0]} to {ti[-1]}')
        return msg

    @classmethod
    def get_time_index(cls, file_paths):
        """Get time index from data files

        Parameters
        ----------
        file_paths : list
            path to data file

        Returns
        -------
        time_index : np.ndarray
            Time index from nc source file(s)
        """
        with cls.source_handler(file_paths) as handle:
            if hasattr(handle, 'XTIME'):
                time_index = handle.XTIME.values
            elif hasattr(handle, 'time'):
                time_index = handle.indexes['time']
                time_index = [dt.strptime(str(t), '%Y-%m-%d %H:%M:%S')
                              for t in time_index]
                time_index = [np.datetime64(t) for t in time_index]
                time_index = np.array(time_index)
        return time_index

    @classmethod
    def feature_registry(cls):
        """Registry of methods for computing features

        Returns
        -------
        dict
            Method registry
        """
        registry = {
            'BVF2_(.*)m': BVFreqSquaredNC,
            'BVF_MO_(.*)m': BVFreqMon,
            'RMOL': InverseMonNC,
            'Windspeed_(.*)m': WindspeedNC,
            'Winddirection_(.*)m': WinddirectionNC,
            'lat_lon': LatLonNC,
            'Shear_(.*)m': Shear,
            'REWS_(.*)m': Rews,
            'Temperature_(.*)m': TempNC,
            'Pressure_(.*)m': 'P_(.*)m'}
        return registry

    @classmethod
    def get_handle_features(cls, file_paths):
        """Lookup inputs needed to compute feature

        Parameters
        ----------
        file_paths : list
            List of data file paths

        Returns
        -------
        list
            List of available features in data
        """

        with cls.source_handler(file_paths) as handle:
            handle_features = [Feature.get_basename(r) for r in handle]
        return handle_features

    @classmethod
    def extract_feature(cls, file_paths, raster_index, feature,
                        time_slice=slice(None)) -> np.dtype(np.float32):
        """Extract single feature from data source

        Parameters
        ----------
        file_paths : list
            path to data file
        raster_index : ndarray
            Raster index array
        feature : str
            Feature to extract from data
        time_slice : slice
            slice of time to extract

        Returns
        -------
        ndarray
            Data array for extracted feature
            (spatial_1, spatial_2, temporal)
        """

        with cls.source_handler(file_paths) as handle:
            f_info = Feature(feature, handle)
            interp_height = f_info.height
            interp_pressure = f_info.pressure
            basename = f_info.basename

            method = cls.lookup(feature, 'compute')
            if method is not None and basename not in handle:
                return method(file_paths, raster_index)

            elif feature in handle:
                idx = tuple([time_slice] + raster_index)
                fdata = np.array(handle[feature][idx], dtype=np.float32)
            elif basename in handle:
                if interp_height is not None:
                    fdata = interp_var_to_height(handle, basename,
                                                 raster_index,
                                                 np.float32(interp_height),
                                                 time_slice)
                elif interp_pressure is not None:
                    fdata = interp_var_to_pressure(handle, basename,
                                                   raster_index,
                                                   np.float32(interp_pressure),
                                                   time_slice)
            else:
                msg = f'{feature} cannot be extracted from source data.'
                logger.exception(msg)
                raise ValueError(msg)

        fdata = np.transpose(fdata, (1, 2, 0))
        return fdata.astype(np.float32)

    @staticmethod
    def get_closest_lat_lon(lat_lon, target):
        """Get closest indices to target lat lon to use for lower left corner
        of raster index

        Parameters
        ----------
        lat_lon : ndarray
            Array of lat/lon
            (spatial_1, spatial_2, 2)
            Last dimension in order of (lat, lon)
        target : tuple
            (lat, lon) for lower left corner

        Returns
        -------
        row : int
            row index for closest lat/lon to target lat/lon
        col : int
            col index for closest lat/lon to target lat/lon
        """
        lat_diff = lon_diff = np.inf
        row = col = -1

        for i in range(lat_lon.shape[0]):
            for j in range(lat_lon.shape[1]):
                lat = lat_lon[i, j, 0]
                lon = lat_lon[i, j, 1]
                tmp_lat_diff = lat - target[0]
                tmp_lon_diff = lon - target[1]
                check = (0 <= tmp_lat_diff < lat_diff
                         and 0 <= tmp_lon_diff < lon_diff)
                if check:
                    lat_diff = np.abs(lat - target[0])
                    lon_diff = np.abs(lon - target[1])
                    row = i
                    col = j
        return row, col

    def get_raster_index(self, file_paths, target=None, shape=None):
        """Get raster index for file data. Here we assume the list of paths in
        file_paths all have data with the same spatial domain. We use the first
        file in the list to compute the raster.

        Parameters
        ----------
        file_paths : list
            path to data files
        target : tuple
            (lat, lon) for lower left corner
        shape : tuple
            (n_rows, n_cols) grid size

        Returns
        -------
        raster_index : np.ndarray
            2D array of grid indices
        """

        check = (self.raster_file is not None
                 and os.path.exists(self.raster_file))
        check = check or (shape is not None and target is not None)
        msg = ('Must provide raster file or shape + target to get '
               'raster index')
        assert check, msg
        if self.raster_file is not None and os.path.exists(self.raster_file):
            logger.debug(f'Loading raster index: {self.raster_file} '
                         f'for {self.file_info_logging(self.file_paths)}')
            raster_index = np.load(self.raster_file.replace('.txt', '.npy'),
                                   allow_pickle=True)
            raster_index = list(raster_index)
            self.grid_shape = get_raster_shape(raster_index)
        else:
            logger.debug('Calculating raster index from WRF file '
                         f'for shape {shape} and target {target}')
            lat_lon = self.get_lat_lon(file_paths, [slice(None), slice(None)],
                                       self.temporal_slice)
            min_lat = np.min(lat_lon[..., 0])
            min_lon = np.min(lat_lon[..., 1])
            max_lat = np.max(lat_lon[..., 0])
            max_lon = np.max(lat_lon[..., 1])
            msg = (f'target {target} out of bounds with min lat/lon {min_lat}/'
                   f'{min_lon} and max lat/lon {max_lat}/{max_lon}')
            assert (min_lat <= target[0] <= max_lat
                    and min_lon <= target[1] <= max_lon), msg

            row, col = self.get_closest_lat_lon(lat_lon, target)
            raster_index = [slice(row, row + shape[0]),
                            slice(col, col + shape[1])]

            if (raster_index[0].stop > lat_lon.shape[0]
               or raster_index[1].stop > lat_lon.shape[1]):
                msg = (f'Invalid target {target}, shape {shape}, and raster '
                       f'{raster_index} for data domain of size '
                       f'{lat_lon.shape[:-1]} with lower left corner '
                       f'({np.min(lat_lon[..., 0])}, '
                       f'{np.min(lat_lon[..., 1])}).')
                raise ValueError(msg)

            self.lat_lon = lat_lon[tuple(raster_index + [slice(None)])]

            mask = ((self.lat_lon[..., 0] >= target[0])
                    & (self.lat_lon[..., 1] >= target[1]))
            if mask.sum() != np.product(shape):
                msg = (f'Found {mask.sum()} coordinates but should have found '
                       f'{shape[0]} by {shape[1]}')
                logger.warning(msg)
                warnings.warn(msg)

            if self.raster_file is not None:
                logger.debug(f'Saving raster index: {self.raster_file}')
                np.save(self.raster_file.replace('.txt', '.npy'), raster_index)
        return raster_index


class DataHandlerNCforCC(DataHandlerNC):
    """Data Handler for NETCDF climate change data"""

    @classmethod
    def feature_registry(cls):
        """Registry of methods for computing features or extracting renamed
        features

        Returns
        -------
        dict
            Method registry
        """
        registry = {
            'U_(.*)': 'ua_(.*)',
            'V_(.*)': 'va_(.*)',
            'lat_lon': LatLonNCforCC}
        return registry

    @classmethod
    def source_handler(cls, file_paths):
        """Xarray data handler

        Parameters
        ----------
        file_paths : str | list
            paths to data files

        Returns
        -------
        data : xarray.Dataset
        """
        return xr.open_mfdataset(file_paths)


class DataHandlerH5(DataHandler):
    """DataHandler for H5 Data"""

    # the handler from rex to open h5 data.
    REX_HANDLER = MultiFileWindX

    @classmethod
    def source_handler(cls, file_paths):
        """rex data handler

        Parameters
        ----------
        file_paths : str | list
            paths to data files

        Returns
        -------
        data : ResourceX
        """
        return cls.REX_HANDLER(file_paths)

    @classmethod
    def get_time_index(cls, file_paths):
        """Get time index from data files

        Parameters
        ----------
        file_paths : list
            path to data file

        Returns
        -------
        time_index : pd.DateTimeIndex
            Time index from h5 source file(s)
        """
        with cls.source_handler(file_paths) as handle:
            time_index = handle.time_index
        return time_index

    @classmethod
    def feature_registry(cls):
        """Registry of methods for computing features or extracting renamed
        features

        Returns
        -------
        dict
            Method registry
        """
        registry = {
            'BVF2_(.*)m': BVFreqSquaredH5,
            'BVF_MO_(.*)m': BVFreqMon,
            'U_(.*)m': UWindH5,
            'V_(.*)m': VWindH5,
            'lat_lon': LatLonH5,
            'REWS_(.*)m': Rews,
            'RMOL': 'inversemoninobukhovlength_2m',
            'P_(.*)m': 'pressure_(.*)m'}
        return registry

    @classmethod
    def extract_feature(cls, file_paths, raster_index, feature,
                        time_slice=slice(None)) -> np.dtype(np.float32):
        """Extract single feature from data source

        Parameters
        ----------
        file_paths : list
            path to data file
        raster_index : ndarray
            Raster index array
        feature : str
            Feature to extract from data
        time_slice : slice
            slice of time to extract

        Returns
        -------
        ndarray
            Data array for extracted feature
            (spatial_1, spatial_2, temporal)
        """

        with cls.source_handler(file_paths) as handle:
            method = cls.lookup(feature, 'compute')
            if method is not None and feature not in handle:
                return method(file_paths, raster_index)
            else:
                try:
                    fdata = handle[(feature, time_slice,)
                                   + tuple([raster_index.flatten()])]
                except ValueError as e:
                    msg = f'{feature} cannot be extracted from source data'
                    logger.exception(msg)
                    raise ValueError(msg) from e

        fdata = fdata.reshape((-1, raster_index.shape[0],
                               raster_index.shape[1]))
        fdata = np.transpose(fdata, (1, 2, 0))
        return fdata.astype(np.float32)

    def get_raster_index(self, file_paths, target=None, shape=None):
        """Get raster index for file data. Here we assume the list of paths in
        file_paths all have data with the same spatial domain. We use the first
        file in the list to compute the raster.

        Parameters
        ----------
        file_paths : list
            path to data file
        target : tuple
            (lat, lon) for lower left corner
        shape : tuple
            (n_rows, n_cols) grid size

        Returns
        -------
        raster_index : np.ndarray
            2D array of grid indices
        """
        check = (self.raster_file is not None
                 and os.path.exists(self.raster_file))
        check = check or (shape is not None and target is not None)
        msg = ('Must provide raster file or shape + target to get '
               'raster index')
        assert check, msg

        if self.raster_file is not None and os.path.exists(self.raster_file):
            logger.debug(f'Loading raster index: {self.raster_file} '
                         f'for {self.file_info_logging(self.file_paths)}')
            raster_index = np.loadtxt(self.raster_file).astype(np.uint32)
            self.grid_shape = get_raster_shape(raster_index)
        else:
            logger.debug('Calculating raster index from WTK file '
                         f'for shape {shape} and target {target}')
            with self.source_handler(file_paths[0]) as handle:
                raster_index = handle.get_raster_index(
                    target, shape, max_delta=self.max_delta)
            self.lat_lon = self.get_lat_lon(file_paths, raster_index,
                                            self.temporal_slice)
            if self.raster_file is not None:
                logger.debug(f'Saving raster index: {self.raster_file}')
                np.savetxt(self.raster_file, raster_index)
        return raster_index


class DataHandlerH5WindCC(DataHandlerH5):
    """Special data handling and batch sampling for h5 wtk or nsrdb data for
    climate change applications"""

    # the handler from rex to open h5 data.
    REX_HANDLER = MultiFileWindX

    # list of features / feature name patterns that are input to the generative
    # model but are not part of the synthetic output and are not sent to the
    # discriminator. These are case-insensitive and follow the Unix shell-style
    # wildcard format.
    TRAIN_ONLY_FEATURES = tuple()

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args : list
            Same positional args as DataHandlerH5
        **kwargs : dict
            Same keyword args as DataHandlerH5
        """
        sample_shape = kwargs.get('sample_shape', (10, 10, 24))
        t_shape = sample_shape[-1]

        if len(sample_shape) == 2:
            logger.info('Found 2D sample shape of {}. Adding spatial dim of 24'
                        .format(sample_shape))
            sample_shape = sample_shape + (24,)
            t_shape = sample_shape[-1]
            kwargs['sample_shape'] = sample_shape

        if t_shape < 24 or t_shape % 24 != 0:
            msg = ('Climate Change DataHandler can only work with temporal '
                   'sample shapes that are one or more days of hourly data '
                   '(e.g. 24, 48, 72...). The requested temporal sample '
                   'shape was: {}'.format(t_shape))
            logger.error(msg)
            raise RuntimeError(msg)

        # validation splits not enabled for solar CC model.
        kwargs['val_split'] = 0.0

        super().__init__(*args, **kwargs)

        self.daily_data = None
        self.daily_data_slices = None
        self.run_daily_averages()

    def run_daily_averages(self):
        """Calculate daily average data and store as attribute."""
        msg = ('Data needs to be hourly with at least 24 hours, but data '
               'shape is {}.'.format(self.data.shape))
        assert self.data.shape[2] % 24 == 0, msg
        assert self.data.shape[2] > 24, msg

        n_data_days = int(self.data.shape[2] / 24)
        daily_data_shape = (self.data.shape[0:2] + (n_data_days,)
                            + (self.data.shape[3],))

        logger.info('Calculating daily average datasets for {} training '
                    'data days.'.format(n_data_days))

        self.daily_data = np.zeros(daily_data_shape, dtype=np.float32)

        self.daily_data_slices = np.array_split(np.arange(self.data.shape[2]),
                                                n_data_days)
        self.daily_data_slices = [slice(x[0], x[-1] + 1)
                                  for x in self.daily_data_slices]
        for d, t_slice in enumerate(self.daily_data_slices):
            self.daily_data[:, :, d, :] = daily_temporal_coarsening(
                self.data[:, :, t_slice, :], temporal_axis=2)[:, :, 0, :]

        logger.info('Finished calculating daily average datasets for {} '
                    'training data days.'.format(n_data_days))

    def _normalize_data(self, feature_index, mean, std):
        """Normalize data with initialized mean and standard deviation for a
        specific feature

        Parameters
        ----------
        feature_index : int
            index of feature to be normalized
        mean : float32
            specified mean of associated feature
        std : float32
            specificed standard deviation for associated feature
        """
        super()._normalize_data(feature_index, mean, std)
        self.daily_data[..., feature_index] -= mean
        self.daily_data[..., feature_index] /= std

    @classmethod
    def feature_registry(cls):
        """Registry of methods for computing features

        Returns
        -------
        dict
            Method registry
        """
        registry = {'U_(.*)m': UWindH5,
                    'V_(.*)m': VWindH5,
                    'lat_lon': LatLonH5}
        return registry

    def get_observation_index(self):
        """Randomly gets spatial sample and time sample

        Returns
        -------
        obs_ind_hourly : tuple
            Tuple of sampled spatial grid, time slice, and features indices.
            Used to get single observation like self.data[observation_index].
            This is for hourly high-res data slicing.
        obs_ind_daily : tuple
            Same as obs_ind_hourly but the temporal index (i=2) is a slice of
            the daily data (self.daily_data) with day integers.
        """
        spatial_slice = uniform_box_sampler(self.data, self.sample_shape[:2])

        n_days = int(self.sample_shape[2] / 24)
        rand_day_ind = np.random.choice(len(self.daily_data_slices) - n_days)
        t_slice_0 = self.daily_data_slices[rand_day_ind]
        t_slice_1 = self.daily_data_slices[rand_day_ind + n_days - 1]
        t_slice_hourly = slice(t_slice_0.start, t_slice_1.stop)
        t_slice_daily = slice(rand_day_ind, rand_day_ind + n_days)

        obs_ind_hourly = tuple(spatial_slice
                               + [t_slice_hourly]
                               + [np.arange(len(self.features))])

        obs_ind_daily = tuple(spatial_slice
                              + [t_slice_daily]
                              + [np.arange(len(self.features))])

        return obs_ind_hourly, obs_ind_daily

    def get_next(self):
        """Gets data for observation using random observation index. Loops
        repeatedly over randomized time index

        Returns
        -------
        obs_hourly : np.ndarray
            4D array
            (spatial_1, spatial_2, temporal_hourly, features)
        obs_daily_avg : np.ndarray
            4D array but the temporal axis is temporal_hourly//24
            (spatial_1, spatial_2, temporal_daily, features)
        """
        obs_ind_hourly, obs_ind_daily = self.get_observation_index()
        self.current_obs_index = obs_ind_hourly
        obs_hourly = self.data[obs_ind_hourly]
        obs_daily_avg = self.daily_data[obs_ind_daily]
        return obs_hourly, obs_daily_avg

    def split_data(self, data=None):
        """Splits time dimension into set of training indices and validation
        indices. For NSRDB it makes sure that the splits happen at midnight.

        Parameters
        ----------
        data : np.ndarray
            4D array of high res data
            (spatial_1, spatial_2, temporal, features)

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

        if data is not None:
            self.data = data

        midnight_ilocs = np.where((self.time_index.hour == 0)
                                  & (self.time_index.minute == 0)
                                  & (self.time_index.second == 0))[0]

        n_val_obs = int(np.ceil(self.val_split * len(midnight_ilocs)))
        val_split_index = midnight_ilocs[n_val_obs]

        self.val_data = self.data[:, :, slice(None, val_split_index), :]
        self.data = self.data[:, :, slice(val_split_index, None), :]

        self.val_time_index = self.time_index[slice(None, val_split_index)]
        self.time_index = self.time_index[slice(val_split_index, None)]

        return self.data, self.val_data


class DataHandlerH5SolarCC(DataHandlerH5WindCC):
    """Special data handling and batch sampling for h5 NSRDB solar data for
    climate change applications"""

    # the handler from rex to open h5 data.
    REX_HANDLER = MultiFileNSRDBX

    # list of features / feature name patterns that are input to the generative
    # model but are not part of the synthetic output and are not sent to the
    # discriminator. These are case-insensitive and follow the Unix shell-style
    # wildcard format.
    TRAIN_ONLY_FEATURES = ('U', 'V', 'air_temperature')

    @classmethod
    def feature_registry(cls):
        """Registry of methods for computing features

        Returns
        -------
        dict
            Method registry
        """
        registry = {
            'U': UWindNsrdb,
            'V': VWindNsrdb,
            'lat_lon': LatLonH5,
            'cloud_mask': CloudMaskH5,
            'clearsky_ratio': ClearSkyRatioH5}
        return registry


# pylint: disable=W0223
class DataHandlerDC(DataHandler):
    """Data-centric data handler"""

    def get_observation_index(self, temporal_weights):
        """Randomly gets spatial sample and time sample

        Returns
        -------
        observation_index : tuple
            Tuple of sampled spatial grid, time slice, and features indices.
            Used to get single observation like self.data[observation_index]
        temporal_focus : slice
            Slice used to select prefered temporal range from full extent
        """
        spatial_slice = uniform_box_sampler(self.data, self.sample_shape[:2])
        temporal_slice = weighted_time_sampler(self.data, self.sample_shape[2],
                                               weights=temporal_weights)
        return tuple(
            spatial_slice + [temporal_slice] + [np.arange(len(self.features))])

    def get_next(self, temporal_weights):
        """Gets data for observation using random observation index. Loops
        repeatedly over randomized time index

        Returns
        -------
        observation : np.ndarray
            4D array
            (spatial_1, spatial_2, temporal, features)
        temporal_focus : slice
            Slice used to select prefered temporal range from full extent
        """
        self.current_obs_index = self.get_observation_index(temporal_weights)
        observation = self.data[self.current_obs_index]
        return observation


class DataHandlerDCforNC(DataHandlerNC, DataHandlerDC):
    """Data centric data handler for NETCDF files"""


class DataHandlerDCforH5(DataHandlerH5, DataHandlerDC):
    """Data centric data handler for H5 files"""
