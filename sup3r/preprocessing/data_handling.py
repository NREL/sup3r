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

from rex import MultiFileWindX, MultiFileNSRDBX
from rex.utilities import log_mem

from sup3r.utilities.utilities import (get_chunk_slices,
                                       uniform_box_sampler,
                                       uniform_time_sampler,
                                       weighted_time_sampler,
                                       daily_time_sampler,
                                       interp_var,
                                       get_raster_shape,
                                       ignore_case_path_fetch,
                                       get_time_index,
                                       get_source_type,
                                       get_wrf_date_range
                                       )
from sup3r.preprocessing.feature_handling import (FeatureHandler,
                                                  Feature,
                                                  BVFreqMonH5,
                                                  BVFreqMonNC,
                                                  BVFreqSquaredH5,
                                                  BVFreqSquaredNC,
                                                  UWindH5,
                                                  VWindH5,
                                                  UWindNsrdb,
                                                  VWindNsrdb,
                                                  LatLonH5,
                                                  ClearSkyRatioH5,
                                                  CloudMaskH5)

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
        Either DataHandlerNC, DataHandlerH5, DataHandlerNsrdb

    """
    if get_source_type(file_paths) == 'h5':
        HandlerClass = DataHandlerH5
        if all('nsrdb' in os.path.basename(fp) for fp in file_paths):
            HandlerClass = DataHandlerNsrdb
    else:
        HandlerClass = DataHandlerNC
    return HandlerClass


class DataHandler(FeatureHandler):
    """Sup3r data handling and extraction"""

    # list of features / feature name patterns that are input to the generative
    # model but are not part of the synthetic output and are not sent to the
    # discriminator. These are case-insensitive and follow the Unix shell-style
    # wildcard format.
    TRAIN_ONLY_FEATURES = ('BVF_*', 'inversemoninobukhovlength_*')

    def __init__(self, file_path, features, target=None, shape=None,
                 max_delta=20, temporal_slice=slice(None),
                 time_roll=0, val_split=0.1,
                 sample_shape=(10, 10, 1),
                 raster_file=None, shuffle_time=False,
                 max_extract_workers=None,
                 max_compute_workers=None,
                 time_chunk_size=100,
                 cache_file_prefix=None,
                 overwrite_cache=False,
                 load_cached=False):
        """Data handling and extraction

        Parameters
        ----------
        file_path : str
            A single source h5 wind file to extract raster data from or a list
            of netcdf files with identical grid
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
        time_roll : int
            The number of places by which elements are shifted in the time
            axis. Can be used to convert data to different timezones. This is
            passed to np.roll(a, time_roll, axis=2) and happens AFTER the
            time_pruning operation.
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
        max_compute_workers : int | None
            max number of workers to use for computing features. If
            max_compute_workers == 1 then extraction will be serialized.
        max_extract_workers : int | None
            max number of workers to use for data extraction. If
            max_extract_workers == 1 then extraction will be serialized.
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
        """

        check = ((target is not None and shape is not None)
                 or (raster_file is not None and os.path.exists(raster_file)))
        msg = ('You must either provide the target+shape inputs or an existing'
               ' raster_file input.')
        assert check, msg

        super().__init__()
        self.file_path = file_path
        if not isinstance(self.file_path, list):
            self.file_path = [self.file_path]
        self.file_path = sorted(self.file_path)

        logger.info(
            'Initializing DataHandler '
            f'{self.file_info_logging(self.file_path)}')

        self.features = features
        self.grid_shape = shape
        self.val_time_index = None
        self.target = target
        self.max_delta = max_delta
        self.raster_file = raster_file
        self.val_split = val_split
        self.sample_shape = sample_shape
        self.temporal_slice = temporal_slice
        self.time_roll = time_roll
        self.raw_time_index = get_time_index(self.file_path)
        self.time_index = self.raw_time_index[temporal_slice]
        self.shuffle_time = shuffle_time
        self.current_obs_index = None
        self.overwrite_cache = overwrite_cache
        self.load_cached = load_cached
        self.cache_files = self.get_cache_file_names(cache_file_prefix)
        self.data = None
        self.val_data = None

        n_steps = self.raw_time_index[temporal_slice.start:temporal_slice.stop]
        n_steps = len(n_steps)
        msg = (f'Temporal slice step ({temporal_slice.step}) does not evenly '
               f'divide the number of time steps ({n_steps})')
        check = temporal_slice.step is None
        check = check or n_steps % temporal_slice.step == 0
        if not check:
            logger.warning(msg)
            warnings.warn(msg)

        msg = (f'sample_shape[2] ({self.sample_shape[2]}) cannot be larger '
               'than the number of time steps in the raw data '
               f'({len(self.raw_time_index)}).')
        assert len(self.raw_time_index) >= self.sample_shape[2], msg

        msg = (f'The requested time slice {temporal_slice} conflicts with the '
               f'number of time steps ({len(self.raw_time_index)}) '
               'in the raw data')
        if (temporal_slice.start is not None
                and temporal_slice.stop is not None):
            assert ((temporal_slice.stop - temporal_slice.start
                     <= len(self.raw_time_index))
                    and temporal_slice.stop <= len(self.raw_time_index)
                    and temporal_slice.start <= len(self.raw_time_index)), msg

        if cache_file_prefix is not None and not self.overwrite_cache and all(
                os.path.exists(fp) for fp in self.cache_files):
            if self.load_cached:
                logger.info(
                    f'All {self.cache_files} exist. Loading from cache instead'
                    f' of extracting from {self.file_path}')
                self.load_cached_data()
            else:
                logger.info(
                    f'All {self.cache_files} exist. Call the '
                    'load_cached_data() method or use load_cache=True to load '
                    'this data.')
                self.data = None

        else:
            if self.overwrite_cache and self.cache_files is not None and all(
                    os.path.exists(fp) for fp in self.cache_files):
                logger.info(
                    f'{self.cache_files} exists but overwrite_cache is set to '
                    'True. Proceeding with extraction.')

            self.raster_index = self.get_raster_index(self.file_path,
                                                      self.target,
                                                      self.grid_shape)
            msg = ('sample_shape[0] / sample_shape[1] is larger than '
                   'the raster size')
            raster_shape = get_raster_shape(self.raster_index)
            if (sample_shape[0] <= raster_shape[0]
                    and sample_shape[1] <= raster_shape[1]):
                logger.warning(msg)
                warnings.warn(msg)

            self.data = self.extract_data(
                self.file_path, self.raster_index, self.features,
                temporal_slice=self.temporal_slice, time_roll=self.time_roll,
                max_extract_workers=max_extract_workers,
                max_compute_workers=max_compute_workers,
                time_chunk_size=time_chunk_size, cache_files=self.cache_files,
                overwrite_cache=self.overwrite_cache)

            if cache_file_prefix is None:
                self.data, self.val_data = self.split_data()
            else:
                self.cache_data(self.cache_files)
                self.data = None

        logger.info('Finished intializing DataHandler.')
        log_mem(logger, log_level='INFO')

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
                f'{cache_file_prefix}_{f.lower()}.pkl'
                for f in self.features]
            for i, fp in enumerate(cache_files):
                fp_check = ignore_case_path_fetch(fp)
                if fp_check is not None:
                    cache_files[i] = fp_check
        else:
            cache_files = None

        return cache_files

    @classmethod
    def file_info_logging(cls, file_path):
        """Method to provide info about files in log output. Since NETCDF files
        have single time slices printing out all the file paths is just a text
        dump without much info.

        Parameters
        ----------
        file_path : list
            List of file paths

        Returns
        -------
        str
            message to append to log output that does not include a huge info
            dump of file paths
        """

        msg = (f'source files: {file_path}')
        return msg

    @property
    def output_features(self):
        """Get a list of features that should be output by the generative model
        corresponding to the features in the high res batch array."""
        out = []
        for feature in self.features:
            ignore = any(fnmatch(feature.lower(), pattern.lower())
                         for pattern in self.TRAIN_ONLY_FEATURES)
            if not ignore:
                out.append(feature)
        return out

    def unnormalize(self, means, stds):
        """Remove normalization from stored means and stds"""
        for i in range(self.shape[-1]):
            self.val_data[..., i] = self.val_data[..., i] * stds[i] + means[i]
            self.data[..., i] = self.data[..., i] * stds[i] + means[i]

    def normalize(self, means, stds):
        """Normalize all data features Parameters
        ----------
        means : np.ndarray
            dimensions (features)
            array of means for all features with same ordering as data features
        stds : np.ndarray
            dimensions (features)
            array of means for all features with same ordering as data features
        """

        logger.debug(
            f'Normalizing data for {self.file_info_logging(self.file_path)}')
        for i in range(self.shape[-1]):
            self._normalize_data(i, means[i], stds[i])

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

    def load_cached_data(self):
        """Load data from cache files and split into training and validation
        """

        if self.data is not None:
            msg = ('Called load_cached_data() but self.data is not None')
            logger.warning(msg)
            warnings.warn(msg)

        elif self.data is None:
            self.raster_index = getattr(self, 'raster_index', None)
            if self.raster_index is None:
                self.raster_index = self.get_raster_index(
                    self.file_path, self.target, self.grid_shape)

            shape = get_raster_shape(self.raster_index)
            requested_shape = (shape[0], shape[1], len(self.time_index),
                               len(self.features))

            msg = ('Found {} cache files but need {} for features {}! '
                   'These are the cache files that were found: {}'
                   .format(len(self.cache_files), len(self.features),
                           self.features, self.cache_files))
            assert len(self.cache_files) == len(self.features), msg

            self.data = np.full(shape=requested_shape, fill_value=np.nan,
                                dtype=np.float32)

            for i, fp in enumerate(self.cache_files):

                assert self.features[i].lower() in fp.lower()
                fp = ignore_case_path_fetch(fp)
                logger.info(f'Loading {self.features[i]} from {fp}')

                with open(fp, 'rb') as fh:
                    log_mem(logger)

                    try:
                        self.data[..., i] = np.array(pickle.load(fh),
                                                     dtype=np.float32)
                    except Exception as e:
                        msg = ('Data loaded from from cache file "{}" '
                               'could not be written to feature channel {} '
                               'of full data array of shape {}. '
                               'Make sure the cached data has the '
                               'appropriate shape.'
                               .format(fp, i, self.data.shape))
                        raise RuntimeError(msg) from e

            nan_perc = (100 * np.isnan(self.data).sum() / self.data.size)
            if nan_perc > 0:
                msg = ('Data has {:.2f}% NaN values!'.format(nan_perc))
                logger.warning(msg)
                warnings.warn(msg)

            logger.debug('Splitting data into training / validation sets '
                         f'({1 - self.val_split}, {self.val_split}) '
                         f'for {self.file_info_logging(self.file_path)}')
            self.data, self.val_data = self.split_data()

    @classmethod
    def check_cached_features(cls, file_path, features, cache_files=None,
                              overwrite_cache=False, load_cached=False):
        """Check which features have been cached and check flags to determine
        whether to load or extract this features again

        Parameters
        ----------
        file_path : str | list
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
                if (os.path.exists(cache_files[i])
                        and f.lower() in cache_files[i].lower()):
                    if not overwrite_cache:
                        if load_cached:
                            logger.info(
                                f'{f} found in cache file {cache_files[i]}. '
                                'Loading from cache instead of extracting '
                                f'from {file_path}')
                        else:
                            logger.info(
                                f'{f} found in cache file {cache_files[i]}. '
                                'Call load_cached_data() or use '
                                'load_cached=True to load this data.')
                    else:
                        logger.info(
                            f'{cache_files[i]} exists but overwrite_cache is '
                            'set to True. Proceeding with extraction.')
                        extract_features.append(f)
                else:
                    extract_features.append(f)
        else:
            extract_features = features

        return extract_features

    @classmethod
    def extract_data(cls, file_path, raster_index, features,
                     temporal_slice=slice(None, None, 1),
                     time_roll=0,
                     max_extract_workers=None,
                     max_compute_workers=None,
                     time_chunk_size=100,
                     cache_files=None,
                     overwrite_cache=False,
                     load_cached=False):
        """Building base 4D data array. Can handle multiple files but assumes
        each file has the same spatial domain

        Parameters
        ----------
        file_path : str | list
            path to data file
        raster_index : np.ndarray
            2D array of grid indices for H5 or list of
            slices for NETCDF
        features : list
            list of features to extract
        temporal_slice : slice
            Slice specifying extent and step of temporal extraction. e.g.
            slice(start, stop, time_pruning). If equal to slice(None, None, 1)
            the full time dimension is selected.
        time_roll : int
            The number of places by which elements are shifted in the time
            axis. Can be used to convert data to different timezones. This is
            passed to np.roll(a, time_roll, axis=2) and happens AFTER the
            time_pruning operation.
        max_compute_workers : int | None
            max number of workers to use for computing features.
            If max_compute_workers == 1 then extraction will be serialized.
        max_extract_workers : int | None
            max number of workers to use for data extraction.
            If max_extract_workers == 1 then extraction will be serialized.
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
        if not isinstance(file_path, list):
            file_path = [file_path]

        shape = get_raster_shape(raster_index)
        logger.debug(
            f'Loading data for raster of shape {shape}')

        # get the file-native time index without pruning
        time_index = get_time_index(file_path)
        n_steps = len(time_index[temporal_slice])

        data_array = np.zeros(
            (shape[0], shape[1], n_steps, len(features)), dtype=np.float32)

        # split time dimension into smaller slices which can be
        # extracted in parallel
        time_chunks = get_chunk_slices(
            len(time_index), time_chunk_size, temporal_slice)
        shifted_time_chunks = get_chunk_slices(n_steps, time_chunk_size)

        extract_features = cls.check_cached_features(
            file_path, features, cache_files=cache_files,
            overwrite_cache=overwrite_cache, load_cached=load_cached)

        raw_features = cls.get_raw_feature_list(file_path, extract_features)

        logger.info(
            f'Starting {extract_features} extraction for '
            f'{cls.file_info_logging(file_path)}')

        raw_data = cls.parallel_extract(file_path, raster_index, time_chunks,
                                        raw_features, max_extract_workers)

        logger.info(f'Finished extracting {extract_features} for '
                    f'{cls.file_info_logging(file_path)}')

        raw_data = cls.parallel_compute(raw_data, raster_index, time_chunks,
                                        raw_features, extract_features,
                                        max_compute_workers)

        logger.info(f'Finished computing {extract_features} for '
                    f'{cls.file_info_logging(file_path)}')

        for t, t_slice in enumerate(shifted_time_chunks):
            for _, f in enumerate(extract_features):
                f_index = features.index(f)
                data_array[..., t_slice, f_index] = raw_data[t][f]
            raw_data.pop(t)

        data_array = np.roll(data_array, time_roll, axis=2)

        if load_cached:
            for f in [f for f in features if f not in extract_features]:
                f_index = features.index(f)
                with open(cache_files[f_index], 'rb') as fh:
                    data_array[..., f_index] = pickle.load(fh)

        logger.info('Finished extracting data for '
                    f'{cls.file_info_logging(file_path)} in {dt.now() - now}')

        return data_array

    @abstractmethod
    def get_raster_index(self, file_path, target, shape):
        """Get raster index for file data. Here we assume the list of paths in
        file_path all have data with the same spatial domain. We use the first
        file in the list to compute the raster

        Parameters
        ----------
        file_path : str | list
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


class DataHandlerNC(DataHandler):
    """Data Handler for NETCDF data"""

    @classmethod
    def file_info_logging(cls, file_path):
        """More concise file info about NETCDF files

        Parameters
        ----------
        file_path : list
            List of file paths

        Returns
        -------
        str
            message to append to log output that does not include a huge info
            dump of file paths
        """

        dirname = os.path.dirname(file_path[0])
        date_start, date_end = get_wrf_date_range(file_path)
        msg = (f'{len(file_path)} files from {dirname} '
               f'with date range: {date_start} - {date_end}')
        return msg

    @classmethod
    def feature_registry(cls):
        """Registry of methods for computing features

        Returns
        -------
        dict
            Method registry
        """
        registry = {
            'BVF_squared_(.*)': BVFreqSquaredNC,
            'BVF_MO_(.*)': BVFreqMonNC}
        return registry

    @classmethod
    def get_raw_feature_list(cls, file_path, features):
        """Lookup inputs needed to compute feature

        Parameters
        ----------
        file_path : list
            List of data file paths
        feature : str
            Feature to lookup in registry

        Returns
        -------
        list
            List of input features
        """

        with xr.open_mfdataset(file_path, combine='nested',
                               concat_dim='Time') as handle:
            input_features = cls.get_raw_feature_list_from_handle(
                features, handle)
        return input_features

    @classmethod
    def extract_feature(cls, file_path, raster_index, feature,
                        time_slice=slice(None)) -> np.dtype(np.float32):
        """Extract single feature from data source

        Parameters
        ----------
        file_path : list
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

        with xr.open_mfdataset(file_path, combine='nested',
                               concat_dim='Time') as handle:

            f_info = Feature(feature, handle)
            interp_height = f_info.height
            basename = f_info.basename

            method = cls.lookup(feature, 'compute')
            if method is not None and basename not in handle:
                return method(file_path, raster_index)

            else:
                try:
                    if len(handle[basename].shape) > 3:
                        if interp_height is None:
                            fdata = np.array(
                                handle[feature][
                                    tuple([time_slice] + [0] + raster_index)],
                                dtype=np.float32)
                        else:
                            logger.debug(
                                f'Interpolating {basename}'
                                f' at height {interp_height}m')
                            fdata = interp_var(
                                handle, basename, raster_index,
                                np.float32(interp_height),
                                time_slice)
                    else:
                        fdata = np.array(
                            handle[feature][
                                tuple([time_slice] + raster_index)],
                            dtype=np.float32)

                except ValueError as e:
                    msg = f'{feature} cannot be extracted from source data'
                    logger.exception(msg)
                    raise ValueError(msg) from e

        fdata = np.transpose(fdata, (1, 2, 0))
        return fdata.astype(np.float32)

    def get_raster_index(self, file_path, target, shape):
        """Get raster index for file data. Here we assume the list of paths in
        file_path all have data with the same spatial domain. We use the first
        file in the list to compute the raster.

        Parameters
        ----------
        file_path : list
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

        if self.raster_file is not None and os.path.exists(self.raster_file):
            logger.debug(f'Loading raster index: {self.raster_file} '
                         f'for {self.file_info_logging(self.file_path)}')
            raster_index = np.load(self.raster_file)
        else:
            logger.debug('Calculating raster index from WRF file '
                         f'for shape {shape} and target {target}')
            with xr.open_mfdataset(file_path, combine='nested',
                                   concat_dim='Time') as handle:
                lats = handle['XLAT'].values[0, :, 0]
                lons = handle['XLONG'].values[0, 0, :]
                lat_diff = list(lats - target[0])
                lat_idx = np.argmin(np.abs(lat_diff))
                lon_diff = list(lons - target[1])
                lon_idx = np.argmin(np.abs(lon_diff))
                raster_index = [slice(lat_idx, lat_idx + shape[0]),
                                slice(lon_idx, lon_idx + shape[1])]

                if (raster_index[1].stop >= len(lat_diff)
                   or raster_index[1].stop >= len(lon_diff)):
                    raise ValueError(
                        f'Invalid target {target} and shape {shape} for '
                        f'data domain of size ({len(lat_diff)}, '
                        f'{len(lon_diff)}) with lower left corner '
                        f'({np.min(lats)}, {np.min(lons)})')

                if self.raster_file is not None:
                    logger.debug(f'Saving raster index: {self.raster_file}')
                    np.save(self.raster_file, raster_index)
        return raster_index


class DataHandlerH5(DataHandler):
    """DataHandler for H5 Data"""

    # the handler from rex to open h5 data.
    REX_HANDLER = MultiFileWindX

    @classmethod
    def feature_registry(cls):
        """Registry of methods for computing features

        Returns
        -------
        dict
            Method registry
        """
        registry = {
            'BVF_squared_(.*)': BVFreqSquaredH5,
            'BVF_MO_(.*)': BVFreqMonH5,
            'U_(.*)m': UWindH5,
            'V_(.*)m': VWindH5,
            'lat_lon': LatLonH5}
        return registry

    @classmethod
    def get_raw_feature_list(cls, file_path, features):
        """Lookup inputs needed to compute feature

        Parameters
        ----------
        feature : str
            Feature to lookup in registry

        Returns
        -------
        list
            List of input features
        """

        with cls.REX_HANDLER(file_path) as handle:
            input_features = cls.get_raw_feature_list_from_handle(
                features, handle)
        return input_features

    @classmethod
    def extract_feature(cls, file_path, raster_index, feature,
                        time_slice=slice(None)) -> np.dtype(np.float32):
        """Extract single feature from data source

        Parameters
        ----------
        file_path : list
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

        with cls.REX_HANDLER(file_path) as handle:

            method = cls.lookup(feature, 'compute')
            if method is not None and feature not in handle:
                return method(file_path, raster_index)

            else:
                try:
                    fdata = handle[
                        tuple([feature] + [time_slice]
                              + [raster_index.flatten()])]

                except ValueError as e:
                    msg = f'{feature} cannot be extracted from source data'
                    logger.exception(msg)
                    raise ValueError(msg) from e

        fdata = fdata.reshape(
            (-1, raster_index.shape[0], raster_index.shape[1]))
        fdata = np.transpose(fdata, (1, 2, 0))

        return fdata.astype(np.float32)

    def get_raster_index(self, file_path, target, shape):
        """Get raster index for file data. Here we assume the list of paths in
        file_path all have data with the same spatial domain. We use the first
        file in the list to compute the raster.

        Parameters
        ----------
        file_path : list
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

        if self.raster_file is not None and os.path.exists(self.raster_file):
            logger.debug(f'Loading raster index: {self.raster_file} '
                         f'for {self.file_info_logging(self.file_path)}')
            raster_index = np.loadtxt(self.raster_file).astype(np.uint32)
        else:
            logger.debug('Calculating raster index from WTK file '
                         f'for shape {shape} and target {target}')
            with self.REX_HANDLER(file_path[0]) as handle:
                raster_index = handle.get_raster_index(
                    target, shape, max_delta=self.max_delta)
            if self.raster_file is not None:
                logger.debug(
                    f'Saving raster index: {self.raster_file}')
                np.savetxt(self.raster_file, raster_index)
        return raster_index


class DataHandlerNsrdb(DataHandlerH5):
    """Special data handling and batch sampling for NSRDB solar data"""

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

    def get_observation_index(self):
        """Randomly gets spatial sample and time sample

        Returns
        -------
        observation_index : tuple
            Tuple of sampled spatial grid, time slice, and features indices.
            Used to get single observation like self.data[observation_index]
        """
        spatial_slice = uniform_box_sampler(self.data,
                                            self.sample_shape[:2])
        temporal_slice = daily_time_sampler(self.data,
                                            self.sample_shape[2],
                                            self.time_index)
        obs_index = tuple(spatial_slice
                          + [temporal_slice]
                          + [np.arange(len(self.features))])
        return obs_index

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

        n_val_obs = int(np.round(self.val_split * len(midnight_ilocs)))
        val_split_index = midnight_ilocs[n_val_obs]

        self.val_data = self.data[:, :, slice(None, val_split_index), :]
        self.data = self.data[:, :, slice(val_split_index, None), :]

        self.val_time_index = self.time_index[slice(None, val_split_index)]
        self.time_index = self.time_index[slice(val_split_index, None)]

        return self.data, self.val_data


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
