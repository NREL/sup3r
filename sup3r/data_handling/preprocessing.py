# -*- coding: utf-8 -*-
"""
Sup3r preprocessing module.
"""
from abc import abstractmethod
from fnmatch import fnmatch
from concurrent.futures import as_completed
import logging
import xarray as xr
import numpy as np
import os
import re
from datetime import datetime as dt
from collections import defaultdict
import pickle

from rex import WindX
from rex.utilities import log_mem
from rex.utilities.execution import SpawnProcessPool
from sup3r.utilities.utilities import (uniform_box_sampler,
                                       uniform_time_sampler,
                                       interp_var,
                                       transform_rotate_wind,
                                       BVF_squared)

from sup3r import __version__

np.random.seed(42)

logger = logging.getLogger(__name__)


def get_source_type(file_paths):
    """Get data source type
    ----------
    file_paths : list
        path to data file
    Returns
    -------
    source_type : str
        Either h5 or nc
    """
    if not isinstance(file_paths, list):
        file_paths = [file_paths]

    _, source_type = os.path.splitext(file_paths[0])
    if source_type == '.h5':
        return 'h5'
    else:
        return 'nc'


def get_time_index(file_paths):
    """Get data file handle
    based on file type
    ----------
    file_paths : list
        path to data file
    Returns
    -------
    handle : xarray | WindX
        data file extension
    """
    if get_source_type(file_paths) == 'h5':
        with WindX(file_paths[0], hsds=False) as handle:
            time_index = handle.time_index
    else:
        with xr.open_mfdataset(file_paths, combine='nested',
                               concat_dim='Time') as handle:
            time_index = handle['Times']
    return time_index


def get_raster_shape(raster_index):
    """method to get shape of raster_index"""

    if any(isinstance(r, slice) for r in raster_index):
        shape = (raster_index[0].stop - raster_index[0].start,
                 raster_index[1].stop - raster_index[1].start)
    else:
        shape = raster_index.shape
    return shape


def get_handler_class(file_paths):
    """Method to get source type specific
    DataHandler class

    Parameters
    ----------
    file_paths : list
        list of file paths

    Returns
    -------
    DataHandler
        Either DataHandlerNC or DataHandlerH5

    """
    if get_source_type(file_paths) == 'h5':
        HandlerClass = DataHandlerH5
    else:
        HandlerClass = DataHandlerNC
    return HandlerClass


class Feature:
    """Class to simplify feature
    computations. Stores alternative names,
    feature height, feature basename, name of
    feature in handle"""

    def __init__(self, feature, handle):
        """Takes a feature (e.g. U_100m) and
        gets the height (100), basename (U) and
        determines whether the feature is found in
        the data handle

        Parameters
        ----------
        feature : str
            Raw feature name e.g. U_100m
        handle : WindX | xarray
            handle for data file
        """
        self.alternative_names = {
            'temperature': 'T',
            'pressure': 'P',
            'T': 'temperature',
            'P': 'pressure'
        }
        self.raw_name = feature
        self.height = self.get_feature_height(feature)
        self.basename = self.get_feature_basename(feature)
        self.alt_name = self.check_renamed_feature(handle, feature)
        if self.raw_name in handle:
            self.handle_input = self.raw_name
        elif self.basename in handle:
            self.handle_input = self.basename
        else:
            self.handle_input = None

    @staticmethod
    def get_feature_basename(feature):
        """Get basename of feature. e.g.
        temperature from temperature_100m

        Parameters
        ----------
        feature : str
            Name of feature. e.g. U_100m

        Returns
        -------
        str
            feature basename
        """

        height = Feature.get_feature_height(feature)
        if height is not None:
            suffix = feature.split('_')[-1]
            basename = feature.strip(f'_{suffix}')
        else:
            basename = feature
        return basename

    @staticmethod
    def get_feature_height(feature):
        """Get height from feature name
        to use in height interpolation

        Parameters
        ----------
        feature : str
            Name of feature. e.g. U_100m

        Returns
        -------
        float | None
            height to use for interpolation
            in meters
        """
        height = feature.split('_')[-1].strip('m')
        if not height.isdigit():
            height = None
        return height

    def check_renamed_feature(self, handle, feature):
        """Method to account for possible alternative
        feature names. e.g. T for temperature

        Parameters
        ----------
        handle : WindX | xarray
            handle pointing to file data
        feature : str
            Feature name. e.g. temperature_100m

        Returns
        -------
        renamed_feature : str
            New feature name. e.g. T_100m
        """

        for k, v in self.alternative_names.items():
            if k in feature:
                renamed_feature = feature.replace(k, v)
                handle_basenames = [
                    self.get_feature_basename(f) for f in handle]
                if v in handle_basenames:
                    return renamed_feature
        return None


class FeatureHandler:
    """Feature Handler with cache
    for previously loaded features used
    in other calculations
    """

    TIME_IND_FEATURES = ('lat_lon',)

    @classmethod
    def pop_old_data(cls, data, chunk_number, all_features):
        """Remove input feature data if no longer needed for
        requested features

        Parameters
        ----------
        data : dict
            dictionary of feature arrays with integer keys
            for chunks and str keys for features.
            e.g. data[chunk_number][feature] = array.
            (spatial_1, spatial_2, temporal)
        chunk_number : int
            time chunk index to check
        all_features : list
            list of all requested features including those
            requiring derivation from input features

        """
        old_keys = [f for f in data[chunk_number]
                    if f not in all_features]
        for k in old_keys:
            data[chunk_number].pop(k)

    @classmethod
    def serial_extract(cls, file_path, data, raster_index, time_chunks,
                       input_features):

        """Extract features in series

        Parameters
        ----------
        file_path : list
            list of file paths
        data : dict
            dictionary of feature arrays with integer keys
            for chunks and str keys for features. Empty unless
            data has been stored for future computations.
            e.g. data[chunk_number][feature] = array.
            (spatial_1, spatial_2, temporal)
        raster_index : ndarray
            raster index for spatial domain
        time_chunks : list
            List of slices to chunk data feature extraction
            along time dimension
        input_features : list
            list of input feature strings

        Returns
        -------
        dict
            dictionary of feature arrays with integer keys
            for chunks and str keys for features.
            e.g. data[chunk_number][feature] = array.
            (spatial_1, spatial_2, temporal)
        """

        time_dep_features = [f for f in input_features
                             if f not in cls.TIME_IND_FEATURES]
        time_ind_features = [f for f in input_features
                             if f in cls.TIME_IND_FEATURES]

        for t, t_slice in enumerate(time_chunks):
            for f in time_dep_features:
                data[t][f] = cls.extract_feature(
                    file_path, raster_index, f, t_slice)
        for f in time_ind_features:
            data[-1][f] = cls.extract_feature(
                file_path, raster_index, f)

        return data

    @classmethod
    def parallel_extract(cls, file_path, data, raster_index, time_chunks,
                         input_features, max_workers=None):

        """Extract features using parallel subprocesses

        Parameters
        ----------
        file_path : list
            list of file paths
        data : dict
            dictionary of feature arrays with integer keys
            for chunks and str keys for features. Empty unless
            data has been stored for future computations.
            e.g. data[chunk_number][feature] = array.
            (spatial_1, spatial_2, temporal)
        raster_index : ndarray
            raster index for spatial domain
        time_chunks : list
            List of slices to chunk data feature extraction
            along time dimension
        input_features : list
            list of input feature strings
        max_workers : int | None
            Number of max workers to use for extraction.
            If equal to 1 then method is run in serial

        Returns
        -------
        dict
            dictionary of feature arrays with integer keys
            for chunks and str keys for features.
            e.g. data[chunk_number][feature] = array.
            (spatial_1, spatial_2, temporal)
        """

        logger.info(f'Extracting {input_features}')

        futures = {}
        now = dt.now()

        time_dep_features = [f for f in input_features
                             if f not in cls.TIME_IND_FEATURES]
        time_ind_features = [f for f in input_features
                             if f in cls.TIME_IND_FEATURES]

        if max_workers == 1:
            return cls.serial_extract(
                file_path, data, raster_index, time_chunks, input_features)
        else:
            with SpawnProcessPool(max_workers=max_workers) as exe:
                for t, t_slice in enumerate(time_chunks):
                    for f in time_dep_features:
                        future = exe.submit(cls.extract_feature,
                                            file_path=file_path,
                                            raster_index=raster_index,
                                            feature=f,
                                            time_slice=t_slice)
                        meta = {'feature': f,
                                'chunk': t}
                        futures[future] = meta

                for f in time_ind_features:
                    future = exe.submit(cls.extract_feature,
                                        file_path=file_path,
                                        raster_index=raster_index,
                                        feature=f)
                    meta = {'feature': f,
                            'chunk': -1}
                    futures[future] = meta

                shape = get_raster_shape(raster_index)
                logger.info(
                    f'Started extracting {input_features}'
                    f' in {dt.now() - now}. Using {len(time_chunks)}'
                    f' time chunks of shape ({shape[0]}, {shape[1]}, '
                    f'{time_chunks[0].stop - time_chunks[0].start}) '
                    f'for {len(input_features)} features')

                for i, future in enumerate(as_completed(futures)):
                    v = futures[future]
                    data[v['chunk']][v['feature']] = future.result()
                    if i % (len(futures) // 10 + 1) == 0:
                        logger.debug(f'{i+1} out of {len(futures)} feature '
                                     'chunks extracted.')

        return data

    @classmethod
    def serial_compute(cls, data, time_chunks,
                       input_features, all_features):

        """Compute features in series

        Parameters
        ----------
        data : dict
            dictionary of feature arrays with integer keys
            for chunks and str keys for features.
            e.g. data[chunk_number][feature] = array.
            (spatial_1, spatial_2, temporal)
        raster_index : ndarray
            raster index for spatial domain
        time_chunks : list
            List of slices to chunk data feature extraction
            along time dimension
        input_features : list
            list of input feature strings
        all_features : list
            list of all features including those requiring
            derivation from input features
        max_workers : int | None
            Number of max workers to use for extraction.
            If equal to 1 then method is run in serial

        Returns
        -------
        data : dict
            dictionary of feature arrays, including computed
            features, with integer keys for chunks and str
            keys for features. Includes
            e.g. data[chunk_number][feature] = array.
            (spatial_1, spatial_2, temporal)
        """

        derived_features = [f for f in all_features if f not in input_features]

        for t, _ in enumerate(time_chunks):
            for _, f in enumerate(derived_features):
                method = cls.lookup_method(f)
                height = Feature.get_feature_height(f)
                tmp = cls.get_input_arrays(data, t, f)
                data[t][f] = method(tmp, height)
            cls.pop_old_data(data, t, all_features)

        return data

    @classmethod
    def parallel_compute(cls, data, raster_index, time_chunks,
                         input_features, all_features,
                         max_workers=None):

        """Compute features using parallel subprocesses

        Parameters
        ----------
        data : dict
            dictionary of feature arrays with integer keys
            for chunks and str keys for features.
            e.g. data[chunk_number][feature] = array.
            (spatial_1, spatial_2, temporal)
        data_array : ndarray
            Array to fill with feature data
            (spatial_1, spatial_2, temporal, features)
        raster_index : ndarray
            raster index for spatial domain
        time_chunks : list
            List of slices to chunk data feature extraction
            along time dimension
        input_features : list
            list of input feature strings
        all_features : list
            list of all features including those requiring
            derivation from input features
        max_workers : int | None
            Number of max workers to use for computation.
            If equal to 1 then method is run in serial

        Returns
        -------
        data : dict
            dictionary of feature arrays, including computed
            features, with integer keys for chunks and str
            keys for features. Includes
            e.g. data[chunk_number][feature] = array.
            (spatial_1, spatial_2, temporal)
        """

        derived_features = [f for f in all_features if f not in input_features]
        logger.info(f'Computing {derived_features}')

        futures = {}
        now = dt.now()
        if max_workers == 1:
            return cls.serial_compute(
                data, time_chunks, input_features, all_features)
        else:
            with SpawnProcessPool(max_workers=max_workers) as exe:
                for t, _ in enumerate(time_chunks):
                    for f in derived_features:
                        method = cls.lookup_method(f)
                        height = Feature.get_feature_height(f)
                        tmp = cls.get_input_arrays(data, t, f)
                        future = exe.submit(
                            method, data=tmp, height=height)

                        meta = {'feature': f,
                                'chunk': t}

                        futures[future] = meta

                    cls.pop_old_data(
                        data, t, all_features)

                shape = get_raster_shape(raster_index)
                logger.info(
                    f'Started computing {derived_features}'
                    f' in {dt.now() - now}. Using {len(time_chunks)}'
                    f' time chunks of shape ({shape[0]}, {shape[1]}, '
                    f'{time_chunks[0].stop - time_chunks[0].start}) '
                    f'for {len(derived_features)} features')

                for i, future in enumerate(as_completed(futures)):
                    v = futures[future]
                    data[v['chunk']][v['feature']] = future.result()
                    if i % (len(futures) // 10 + 1) == 0:
                        logger.debug(f'{i+1} out of {len(futures)} feature '
                                     'chunks computed')

        return data

    @classmethod
    def get_input_arrays(cls, data, chunk_number, f):
        """Get only arrays needed for computations

        Parameters
        ----------
        data : dict
            Dictionary of feature arrays
        chunk_number :
            time chunk for which to get input arrays
        f : str
            feature to compute using input arrays

        Returns
        -------
        dict
            Dictionary of arrays with only needed features
        """
        inputs = cls.lookup_inputs(f)
        tmp = {}
        for r in inputs(f):
            if r in data[chunk_number]:
                tmp[r] = data[chunk_number][r]
            else:
                tmp[r] = data[-1][r]
        return tmp

    @classmethod
    def method_registry(cls):
        """Registry of methods for computing features

        Returns
        -------
        dict
            Method registry
        """
        registry = {
            'BVF_squared_(.*)': cls.get_bvf_squared,
            'BVF_MO_(.*)': cls.get_bvf_mo,
            'U_(.*)m': cls.get_u,
            'V_(.*)m': cls.get_v,
            'lat_lon': cls.get_lat_lon}
        return registry

    @classmethod
    def input_registry(cls):
        """Registry of inputs for computing features

        Returns
        -------
        dict
            Input registry
        """
        registry = {
            'BVF_squared_(.*)': cls.get_bvf_inputs,
            'BVF_MO_(.*)': cls.get_bvf_mo_inputs,
            'U_(.*)m': cls.get_u_inputs,
            'V_(.*)m': cls.get_v_inputs}
        return registry

    @classmethod
    @abstractmethod
    def get_bvf_inputs(cls, feature):
        """Get list of raw features used
        in bvf calculation

        Parameters
        ----------
        feature : str
            name of feature. e.g. BVF_squared_100m

        Returns
        -------
        list
            list of features used to compute bvf_squared
        """

    @classmethod
    @abstractmethod
    def get_bvf_mo_inputs(cls, feature):
        """Get list of raw features used
        in bvf_mo calculation

        Parameters
        ----------
        feature : str
            name of feature. e.g. BVF_MO_100m

        Returns
        -------
        list
            list of features used to compute bvf_mo
        """

    @classmethod
    @abstractmethod
    def get_u_inputs(cls, feature):
        """Get list of raw features used
        in u calculation

        Parameters
        ----------
        feature : str
            name of feature. e.g. U_100m

        Returns
        -------
        list
            list of features used to compute U
        """

    @classmethod
    @abstractmethod
    def get_v_inputs(cls, feature):
        """Get list of raw features used
        in v calculation

        Parameters
        ----------
        feature : str
            name of feature. e.g. V_100m

        Returns
        -------
        list
            list of features used to compute V
        """

    @classmethod
    def lookup_inputs(cls, feature):
        """Lookup feature in feature registry

        Parameters
        ----------
        feature : str
            Feature to lookup in registry

        Returns
        -------
        method | None
            Method to use for computing feature
        """

        input_registry = cls.input_registry()
        for k, v in input_registry.items():
            if re.match(k, feature):
                return v
        return None

    @classmethod
    def lookup_method(cls, feature):
        """Lookup method to compute feature

        Parameters
        ----------
        feature : str
            Feature to lookup in registry

        Returns
        -------
        method | None
            Method to use for computing feature
        """

        method_registry = cls.method_registry()
        for k, v in method_registry.items():
            if re.match(k, feature):
                return v
        return None

    @classmethod
    def get_raw_feature_list(cls, features):
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
        raw_features = []
        for f in features:
            method = cls.lookup_inputs(f)
            if method is not None:
                for r in method(f):
                    if r not in raw_features:
                        raw_features.append(r)
            else:
                if f not in raw_features:
                    raw_features.append(f)

        return raw_features

    @classmethod
    @abstractmethod
    def extract_feature(
            cls, file_path, raster_index,
            feature, time_slice=slice(None)) -> np.dtype(np.float32):
        """Extract single feature from data source

        Parameters
        ----------
        file_path : list
            path to data file
        raster_index : ndarray
            Raster index array
        time_slice : slice
            slice of time to extract
        feature : str
            Feature to extract from data

        Returns
        -------
        ndarray
            Data array for extracted feature
            (spatial_1, spatial_2, temporal)
        """

    @classmethod
    def get_u(cls, data, height) -> np.dtype(np.float32):
        """Compute U wind component

        Parameters
        ----------
        data : dict
            dictionary of feature arrays used for this compuation
        height : str | int
            Height of U/V to extract in meters.
            e.g. 100

        Returns
        -------
        U : ndarray
            array of U wind component
        """

        u, _ = cls.get_uv(data, height)
        return u

    @classmethod
    def get_v(cls, data, height) -> np.dtype(np.float32):
        """Compute V wind component

        Parameters
        ----------
        data : dict
            dictionary of feature arrays used for this compuation
        height : str | int
            Height of U/V to extract in meters.
            e.g. 100

        Returns
        -------
        V : ndarray
            array of V wind component
        """

        _, v = cls.get_uv(data, height)
        return v

    @classmethod
    @abstractmethod
    def get_bvf_squared(
            cls, data, height) -> np.dtype(np.float32):
        """Compute BVF squared

        Parameters
        ----------
        data : dict
            dictionary of feature arrays used for this compuation
        height : str
            Height of top level in meters

        Returns
        -------
        ndarray
            BVF squared array
        """

    @classmethod
    @abstractmethod
    def get_bvf_mo(
            cls, data, height) -> np.dtype(np.float32):
        """Compute BVF_squared times monin obukhov length

        Parameters
        ----------
        data : dict
            dictionary of feature arrays used for this compuation
        height : str
            Height of top level in meters

        Returns
        -------
        ndarray
            BVF_MO array
        """

    @classmethod
    @abstractmethod
    def get_uv(cls, data, height):
        """Compute U and V wind components

        Parameters
        ----------
        data : dict
            dictionary of feature arrays used for this compuation
        height : str | int
            Height of U/V to extract in meters.
            e.g. 100

        Returns
        -------
        U : ndarray
            array of U wind component
        V : ndarray
            array of V wind component
        """

    @classmethod
    @abstractmethod
    def get_lat_lon(cls, file_path, raster_index):
        """Get lats and lons corresponding to raster
        for use in windspeed/direction -> u/v mapping

        Parameters
        ----------
        file_path : list
            path to data file
        raster_index : ndarray
            Raster index array

        Returns
        -------
        ndarray
            lat lon array
            (spatial_1, spatial_2, 2)
        """


class DataHandler(FeatureHandler):
    """Sup3r data handling and extraction"""

    # list of features / feature name patterns that are input to the generative
    # model but are not part of the synthetic output and are not sent to the
    # discriminator. These are case-insensitive and follow the Unix shell-style
    # wildcard format.
    TRAIN_ONLY_FEATURES = ('BVF_*', 'inversemoninbukhovlength_*')

    def __init__(self, file_path, features, target=None, shape=None,
                 max_delta=20, time_pruning=1, val_split=0.1,
                 temporal_sample_shape=1, spatial_sample_shape=(10, 10),
                 raster_file=None, shuffle_time=False,
                 max_extract_workers=None, max_compute_workers=None,
                 time_chunk_size=100,
                 cache_file_prefix=None,
                 overwrite_cache=False):

        """Data handling and extraction

        Parameters
        ----------
        file_path : str
            A single source h5 wind file to extract raster data from
            or a list of netcdf files with identical grid
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
        raster_file : str | None
            File for raster_index array for the corresponding target and
            shape. If specified the raster_index will be loaded from the file
            if it exists or written to the file if it does not yet exist.
            If None raster_index will be calculated directly. Either need
            target+shape or raster_file.
        val_split : float32
            Fraction of data to store for validation
        spatial_sample_shape : tuple
            Size of spatial slice used in a single high-res observation for
            spatial batching
        temporal_sample_shape : int
            Number of time slices used in a single high-res observation for
            temporal batching
        time_pruning : int
            Number of timesteps to downsample. If time_pruning=1 no time
            steps will be skipped.
        shuffle_time : bool
            Whether to shuffle time indices before valiidation split
        max_compute_workers : int | None
            max number of workers to use for computing features.
            If max_compute_workers == 1 then extraction will be serialized.
        max_extract_workers : int | None
            max number of workers to use for data extraction.
            If max_extract_workers == 1 then extraction will be serialized.
        time_chunk_size : int
            Size of chunks to split time dimension into for parallel data
            extraction. If running in serial this can be set to the size
            of the full time index for best performance.
        cache_file_prefix : str | None
            Prefix of files for saving feature data. Each feature will be saved
            to a file with the feature name appended to the end of
            cache_file_prefix. If not None feature arrays will be saved here
            and not stored in self.data until load_cached_data is called.
        overwrite_cache : bool
            Whether to overwrite any previously saved cache files.
        """
        logger.info(
            f'Initializing DataHandler from source files: {file_path}')

        check = ((target is not None and shape is not None)
                 or (raster_file is not None and os.path.exists(raster_file)))
        msg = ('You must either provide the target+shape inputs '
               'or an existing raster_file input.')
        assert check, msg

        super().__init__()
        self.file_path = file_path
        if not isinstance(self.file_path, list):
            self.file_path = [self.file_path]
        self.file_path = sorted(self.file_path)
        self.features = features
        self.grid_shape = shape
        self.time_index = get_time_index(self.file_path)
        self.target = target
        self.max_delta = max_delta
        self.raster_file = raster_file
        self.val_split = val_split
        self.spatial_sample_shape = spatial_sample_shape
        self.temporal_sample_shape = temporal_sample_shape
        self.time_pruning = time_pruning
        self.shuffle_time = shuffle_time
        self.current_obs_index = None
        self.raster_index = self.get_raster_index(
            self.file_path, self.target, self.grid_shape)
        self.cache_files = [f'{cache_file_prefix}_{f}.npy' for f in features]
        self.overwrite_cache = overwrite_cache

        if cache_file_prefix is not None and not self.overwrite_cache and all(
                os.path.exists(fp) for fp in self.cache_files):
            logger.info(f'{self.cache_files} exists. Loading data from cache '
                        f'instead of extracting from {file_path}')
            self.load_cached_data()

        else:
            if self.overwrite_cache:
                logger.info(
                    f'{self.cache_files} exists but overwrite_cache is '
                    'set to True. Proceeding with extraction.')

            self.data = self.extract_data(
                self.file_path, self.raster_index, self.time_index,
                self.features, self.time_pruning,
                max_extract_workers=max_extract_workers,
                max_compute_workers=max_compute_workers,
                time_chunk_size=time_chunk_size)

            if cache_file_prefix is None:
                self.data, self.val_data = self.split_data(self.data)
            else:
                self.cache_data(self.cache_files)
                self.data = None

        msg = ('The temporal_sample_shape cannot '
               'be larger than the number of time steps in the raw data.')
        assert len(self.time_index) >= temporal_sample_shape, msg

        logger.info('Finished intializing DataHandler.')
        log_mem(logger, log_level='INFO')

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
            self.val_data[:, :, :, i] = \
                (self.val_data[:, :, :, i]) * stds[i] + means[i]

            self.data[:, :, :, i] = \
                (self.data[:, :, :, i]) * stds[i] + means[i]

    def normalize(self, means, stds):
        """Normalize all data features
        Parameters
        ----------
        means : np.ndarray
            dimensions (features)
            array of means for all features
            with same ordering as data features
        stds : np.ndarray
            dimensions (features)
            array of means for all features
            with same ordering as data features
        """

        for i in range(self.shape[-1]):
            self._normalize_data(i, means[i], stds[i])

    def _normalize_data(self, feature_index, mean, std):
        """Normalize data with initialized
        mean and standard deviation for
        a specific feature

        Parameters
        ----------
        feature_index : int
            index of feature to be normalized
        mean : float32
            specified mean of associated feature
        std : float32
            specificed standard deviation for associated feature
        """

        if std == 0:
            std = 1
            logger.warning(
                'Standard Deviation is zero for '
                f'{self.features[feature_index]}')

        self.val_data[:, :, :, feature_index] = \
            (self.val_data[:, :, :, feature_index] - mean) / std

        self.data[:, :, :, feature_index] = \
            (self.data[:, :, :, feature_index] - mean) / std

    def get_observation_index(self):
        """Randomly gets spatial sample and time sample

        Returns
        -------
        observation_index : tuple
            Tuple of sampled spatial grid, time slice,
            and features indices. Used to get single observation
            like self.data[observation_index]
        """
        spatial_slice = uniform_box_sampler(
            self.data, self.spatial_sample_shape)
        temporal_slice = uniform_time_sampler(
            self.data, self.temporal_sample_shape)
        return tuple(
            spatial_slice + [temporal_slice] + [np.arange(len(self.features))])

    def get_next(self):
        """Gets data for observation using
        random observation index. Loops repeatedly
        over randomized time index

        Returns
        -------
        observation : np.ndarray
            4D array
            (spatial_1, spatial_2, temporal, features)
        """
        self.current_obs_index = self.get_observation_index()
        observation = self.data[self.current_obs_index]
        return observation

    def split_data(self, data):
        """Splits time dimension into set of training indices
        and validation indices

        Parameters
        ----------
        data : np.ndarray
            4D array of high res data
            (spatial_1, spatial_2, temporal, features)

        Returns
        -------
        data : np.ndarray
            (spatial_1, spatial_2, temporal, features)
            Training data fraction of initial data array. Initial
            data array is overwritten by this new data array.
        val_data : np.ndarray
            (spatial_1, spatial_2, temporal, features)
            Validation data fraction of initial data array.
        """

        n_observations = self.shape[2]
        all_indices = np.arange(n_observations)
        if self.shuffle_time:
            np.random.shuffle(all_indices)

        n_val_obs = int(self.val_split * n_observations)
        val_indices = all_indices[:n_val_obs]
        training_indices = all_indices[n_val_obs:]
        self.val_data = data[:, :, val_indices, :]
        self.data = data[:, :, training_indices, :]
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

    @classmethod
    def computed_features(cls, f_list, previously_computed=None):
        """Keep track of computed features for deleting
        old data

        Parameters
        ----------
        f_list : list
            list of features that have been requested
        previously_computed : list, optional
            previous list of computed features, by default []

        Returns
        -------
        list
            new list of computed features
        """

        if previously_computed is None:
            previously_computed = []
        for f in f_list:
            if f not in previously_computed:
                previously_computed.append(f)
        return previously_computed

    @classmethod
    def needed_features(cls, f_list, computed_features):
        """Keep track of needed features for deleting
        old data

        Parameters
        ----------
        f_list : list
            list of features that have been requested
        computed_features : list, optional
            list of computed features

        Returns
        -------
        list
            new list of needed features
        """
        return cls.get_raw_feature_list(
            set(f_list) - set(computed_features))

    def cache_data(self, cache_file_paths):
        """Cache feature data to file and delete from memory

        Parameters
        ----------
        cache_file_path : str | None
            Path to file for saving feature data
        """

        for i, fp in enumerate(cache_file_paths):
            if not os.path.exists(fp) or self.overwrite_cache:
                if self.overwrite_cache:
                    logger.info(
                        f'Overwriting. Saving {self.features[i]} to {fp}')
                else:
                    logger.info(f'Saving {self.features[i]} to {fp}')

                with open(fp, 'wb') as fh:
                    pickle.dump(self.data[:, :, :, i], fh, protocol=4)
            else:
                logger.warning(
                    f'Called cache_data but {fp} '
                    'already exists. Set to overwrite_cache to True to '
                    'overwrite.')

    def load_cached_data(self):
        """Load data from cache files and split into
        training and validation
        """

        feature_arrays = []
        for i, fp in enumerate(self.cache_files):
            assert self.features[i] in fp
            logger.info(f'Loading {self.features[i]} from {fp}')

            with open(fp, 'rb') as fh:
                feature_arrays.append(
                    np.array(pickle.load(fh)[:, :, :, np.newaxis],
                             dtype=np.float32))

        self.data = np.concatenate(feature_arrays, axis=-1)

        shape = get_raster_shape(self.raster_index)
        requested_shape = (shape[0], shape[1],
                           len(self.time_index[::self.time_pruning]),
                           len(self.features))
        msg = (f'Data loaded from cache {self.data.shape} '
               f'does not match the requested shape {requested_shape}')
        assert self.data.shape == requested_shape, msg

        del feature_arrays
        self.data, self.val_data = self.split_data(self.data)

    @classmethod
    def extract_data(cls, file_path, raster_index,
                     time_index, features, time_pruning,
                     max_extract_workers=None,
                     max_compute_workers=None,
                     time_chunk_size=100):
        """Building base 4D data array. Can
        handle multiple files but assumes each
        file has the same spatial domain

        Parameters
        ----------
        file_path : str | list
            path to data file
        raster_index : np.ndarray
            2D array of grid indices for H5 or list of
            slices for NETCDF
        time_index : list
            List of time indices specifying selection
            along the time dimensions
        features : list
            list of features to extract
        time_pruning : int
            Number of timesteps to downsample. If time_pruning=1 no time
            steps will be skipped.
        max_compute_workers : int | None
            max number of workers to use for computing features.
            If max_compute_workers == 1 then extraction will be serialized.
        max_extract_workers : int | None
            max number of workers to use for data extraction.
            If max_extract_workers == 1 then extraction will be serialized.
        time_chunk_size : int
            Size of chunks to split time dimension into for smaller
            data extractions

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

        data_array = np.zeros(
            (shape[0], shape[1], len(time_index), len(features)),
            dtype=np.float32)

        # split time dimension into smaller slices which can be
        # extracted in parallel
        n_chunks = int(np.ceil(len(time_index) / time_chunk_size))
        time_chunks = np.array_split(np.arange(0, len(time_index)), n_chunks)
        time_chunks = [slice(t[0], t[-1] + 1) for t in time_chunks]

        raw_data = defaultdict(dict)
        raw_features = cls.get_raw_feature_list(features)

        log_mem(logger)
        logger.info(f'Starting {features} extraction from {file_path}')

        raw_data = cls.parallel_extract(
            file_path, raw_data, raster_index, time_chunks,
            raw_features, max_extract_workers)

        log_mem(logger)
        logger.info(f'Finished extracting {features}')

        raw_data = cls.parallel_compute(
            raw_data, raster_index, time_chunks,
            raw_features, features, max_compute_workers)

        log_mem(logger)
        logger.info(f'Finished computing {features}')

        for t, t_slice in enumerate(time_chunks):
            for i, f in enumerate(features):
                data_array[:, :, t_slice, i] = raw_data[t][f]
            raw_data.pop(t)

        data_array = data_array[:, :, ::time_pruning, :]
        logger.info('Finished extracting data from '
                    f'{file_path} in {dt.now() - now}')

        return data_array

    @abstractmethod
    def get_raster_index(self, file_path, target, shape):
        """Get raster index for file data. Here we
        assume the list of paths in file_path all have data
        with the same spatial domain. We use the first
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
    def extract_feature(
            cls, file_path, raster_index,
            feature, time_slice=slice(None)) -> np.dtype(np.float32):
        """Extract single feature from data source

        Parameters
        ----------
        file_path : list
            path to data file
        raster_index : ndarray
            Raster index array
        time_slice : slice
            slice of time to extract
        feature : str
            Feature to extract from data

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

            method = cls.lookup_method(feature)
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
                                handle, basename, float(interp_height))
                            fdata = fdata[
                                tuple([time_slice] + raster_index)]
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
        """Get raster index for file data. Here we
        assume the list of paths in file_path all have data
        with the same spatial domain. We use the first
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
            logger.debug(f'Loading raster index: {self.raster_file}')
            raster_index = np.load(self.raster_file)
        else:
            logger.debug('Calculating raster index from WRF file '
                         f'for shape {shape} and target {target}')
            with xr.open_mfdataset(file_path, combine='nested',
                                   concat_dim='Time') as handle:
                lat_diff = list(handle['XLAT'][0, :, 0] - target[0])
                lat_idx = np.argmin(np.abs(lat_diff))
                lon_diff = list(handle['XLONG'][0, 0, :] - target[1])
                lon_idx = np.argmin(np.abs(lon_diff))
                raster_index = [slice(lat_idx, lat_idx + shape[0]),
                                slice(lon_idx, lon_idx + shape[1])]

                if (raster_index[1].stop >= len(lat_diff)
                   or raster_index[1].stop >= len(lon_diff)):
                    raise ValueError(
                        f'Invalid target {target} and shape {shape} for '
                        f'data domain of size ({len(lat_diff)}, '
                        f'{len(lon_diff)}) with lower left corner '
                        f'({np.min(handle["XLAT"][0, :, 0].values)}, '
                        f'{np.min(handle["XLONG"][0, 0, :].values)})')

                if self.raster_file is not None:
                    logger.debug(f'Saving raster index: {self.raster_file}')
                    np.save(self.raster_file, raster_index)
        return raster_index

    @classmethod
    def get_bvf_mo_inputs(cls, feature):
        """Get list of raw features used
        in bvf_mo calculation

        Parameters
        ----------
        feature : str
            name of feature. e.g. BVF_MO_100m

        Returns
        -------
        list
            list of features used to compute bvf_mo
        """

        height = Feature.get_feature_height(feature)
        return [f'T_{height}m', f'T_{int(height) - 100}m',
                f'RMOL_{height}m']

    @classmethod
    def get_bvf_inputs(cls, feature):
        """Get list of raw features used
        in bvf calculation

        Parameters
        ----------
        feature : str
            name of feature. e.g. BVF_squared_100m

        Returns
        -------
        list
            list of features used to compute bvf_squared
        """

        height = Feature.get_feature_height(feature)
        return [f'T_{height}m', f'T_{int(height) - 100}m']

    @classmethod
    @abstractmethod
    def get_u_inputs(cls, feature):
        """Get list of raw features used
        in u calculation

        Parameters
        ----------
        feature : str
            name of feature. e.g. U_100m

        Returns
        -------
        list
            list of features used to compute U
        """

        height = Feature.get_feature_height(feature)
        return [f'U_{height}m']

    @classmethod
    @abstractmethod
    def get_v_inputs(cls, feature):
        """Get list of raw features used
        in v calculation

        Parameters
        ----------
        feature : str
            name of feature. e.g. V_100m

        Returns
        -------
        list
            list of features used to compute V
        """

        height = Feature.get_feature_height(feature)
        return [f'V_{height}m']

    @classmethod
    def get_bvf_squared(
            cls, data, height) -> np.dtype(np.float32):
        """Compute BVF squared

        Parameters
        ----------
        data : dict
            dictionary of feature arrays used for this compuation
        height : str
            Height of top level in meters

        Returns
        -------
        ndarray
            BVF squared array
        """

        if height is None:
            height = 200

        # T is perturbation potential temperature for wrf and the
        # base potential temperature is 300K
        bvf_squared = np.float32(9.81 / 100)
        bvf_squared *= (data[f'T_{height}m']
                        - data[f'T_{int(height) - 100}m'])
        bvf_squared /= (data[f'T_{height}m']
                        + data[f'T_{int(height) - 100}m'])
        bvf_squared /= np.float32(2)
        return bvf_squared

    @classmethod
    def get_bvf_mo(
            cls, data, height) -> np.dtype(np.float32):
        """Compute BVF squared times monin obukhov length

        Parameters
        ----------
        data : dict
            dictionary of feature arrays used for this compuation
        height : str
            Height of top level in meters

        Returns
        -------
        ndarray
            BVF_MO array
        """

        if height is None:
            height = 200

        # T is perturbation potential temperature for wrf and the
        # base potential temperature is 300K
        bvf_mo = np.float32(9.81 / 100)
        bvf_mo *= (data[f'T_{height}m']
                   - data[f'T_{int(height) - 100}m'])
        bvf_mo /= (data[f'T_{height}m']
                   + data[f'T_{int(height) - 100}m'])
        bvf_mo /= np.float32(2)
        bvf_mo /= data[f'RMOL_{height}m']

        # making this zero when not both bvf and mo are negative
        bvf_mo[data[f'RMOL_{height}m'] >= 0] = 0
        bvf_mo[bvf_mo < 0] = 0

        return bvf_mo

    @classmethod
    def get_uv(cls, data, height):
        """Compute U and V wind components

        Parameters
        ----------
        data : dict
            dictionary of feature arrays used for this compuation
        height : str | int
            Height of U/V to extract in meters.
            e.g. 100

        Returns
        -------
        U : ndarray
            array of U wind component
        V : ndarray
            array of V wind component
        """

        return data[f'U_{height}m'], data[f'V_{height}m']

    @classmethod
    def get_lat_lon(cls, file_path, raster_index):
        """Get lats and lons corresponding to raster
        for use in windspeed/direction -> u/v mapping

        Parameters
        ----------
        file_path : list
            path to data files
        raster_index : ndarray
            Raster index array

        Returns
        -------
        ndarray
            lat lon array
            (spatial_1, spatial_2, 2)
        """

        lat = cls.extract_feature(
            file_path, raster_index, 'XLAT', time_slice=slice(0, 1))
        lon = cls.extract_feature(
            file_path, raster_index, 'XLONG', time_slice=slice(0, 1))
        lat_lon = np.concatenate([lat, lon], axis=2)

        return lat_lon


class DataHandlerH5(DataHandler):
    """DataHandler for H5 Data"""

    @classmethod
    def extract_feature(
            cls, file_path, raster_index,
            feature, time_slice=slice(None)) -> np.dtype(np.float32):
        """Extract single feature from data source

        Parameters
        ----------
        file_path : list
            path to data file
        raster_index : ndarray
            Raster index array
        time_slice : slice
            slice of time to extract
        feature : str
            Feature to extract from data

        Returns
        -------
        ndarray
            Data array for extracted feature
            (spatial_1, spatial_2, temporal)
        """

        with WindX(file_path[0], hsds=False) as handle:

            method = cls.lookup_method(feature)
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
        """Get raster index for file data. Here we
        assume the list of paths in file_path all have data
        with the same spatial domain. We use the first
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
            logger.debug(f'Loading raster index: {self.raster_file}')
            raster_index = np.loadtxt(self.raster_file).astype(np.uint32)
        else:
            logger.debug('Calculating raster index from WTK file '
                         f'for shape {shape} and target {target}')
            with WindX(file_path[0], hsds=False) as handle:
                raster_index = handle.get_raster_index(
                    target, shape, max_delta=self.max_delta)
            if self.raster_file is not None:
                logger.debug(
                    f'Saving raster index: {self.raster_file}')
                np.savetxt(self.raster_file, raster_index)
        return raster_index

    @classmethod
    def get_bvf_inputs(cls, feature):
        """Get list of raw features used
        in bvf calculation

        Parameters
        ----------
        feature : str
            name of feature. e.g. BVF_squared_100m

        Returns
        -------
        list
            list of features used to compute bvf_squared
        """
        height = Feature.get_feature_height(feature)
        features = [f'temperature_{height}m',
                    f'temperature_{int(height) - 100}m',
                    f'pressure_{height}m',
                    f'pressure_{int(height) - 100}m']

        return features

    @classmethod
    def get_bvf_mo_inputs(cls, feature):
        """Get list of raw features used
        in bvf_mo calculation

        Parameters
        ----------
        feature : str
            name of feature. e.g. BVF_MO_100m

        Returns
        -------
        list
            list of features used to compute bvf_mo
        """
        height = Feature.get_feature_height(feature)
        features = [f'temperature_{height}m',
                    f'temperature_{int(height) - 100}m',
                    f'pressure_{height}m',
                    f'pressure_{int(height) - 100}m',
                    'inversemoninobukhovlength_2m']

        return features

    @classmethod
    def get_u_inputs(cls, feature):
        """Get list of raw features used
        in u calculation

        Parameters
        ----------
        feature : str
            name of feature. e.g. U_100m

        Returns
        -------
        list
            list of features used to compute U
        """

        height = Feature.get_feature_height(feature)
        features = [f'windspeed_{height}m',
                    f'winddirection_{height}m',
                    'lat_lon']
        return features

    @classmethod
    def get_v_inputs(cls, feature):
        """Get list of raw features used
        in v calculation

        Parameters
        ----------
        feature : str
            name of feature. e.g. V_100m

        Returns
        -------
        list
            list of features used to compute V
        """

        height = Feature.get_feature_height(feature)
        features = [f'windspeed_{height}m',
                    f'winddirection_{height}m',
                    'lat_lon']
        return features

    @classmethod
    def get_bvf_squared(
            cls, data, height) -> np.dtype(np.float32):
        """Compute BVF squared

        Parameters
        ----------
        data : dict
            dictionary of feature arrays used for this compuation
        height : str
            Height of top level in meters

        Returns
        -------
        ndarray
            BVF squared array
        """

        if height is None:
            height = 200

        return BVF_squared(
            data[f'temperature_{height}m'],
            data[f'temperature_{int(height) - 100}m'],
            data[f'pressure_{height}m'],
            data[f'pressure_{int(height) - 100}m'],
            100)

    @classmethod
    def get_bvf_mo(
            cls, data, height) -> np.dtype(np.float32):
        """Compute BVF squared times monin obukhov length

        Parameters
        ----------
        data : dict
            dictionary of feature arrays used for this compuation
        height : str
            Height of top level in meters

        Returns
        -------
        ndarray
            BVF_MO array
        """

        if height is None:
            height = 200

        bvf_mo = BVF_squared(
            data[f'temperature_{height}m'],
            data[f'temperature_{int(height) - 100}m'],
            data[f'pressure_{height}m'],
            data[f'pressure_{int(height) - 100}m'],
            100) / data['inversemoninobukhovlength_2m']

        # making this zero when not both bvf and mo are negative
        bvf_mo[data['inversemoninobukhovlength_2m'] >= 0] = 0
        bvf_mo[bvf_mo < 0] = 0
        return bvf_mo

    @classmethod
    def get_uv(cls, data, height):
        """Compute U and V wind components

        Parameters
        ----------
        data : dict
            dictionary of feature arrays used for this compuation
        height : str | int
            Height of U/V to extract in meters.
            e.g. 100

        Returns
        -------
        U : ndarray
            array of U wind component
        V : ndarray
            array of V wind component
        """

        return transform_rotate_wind(
            data[f'windspeed_{height}m'],
            data[f'winddirection_{height}m'],
            data['lat_lon'])

    @classmethod
    def get_lat_lon(cls, file_path, raster_index):
        """Get lats and lons corresponding to raster
        for use in windspeed/direction -> u/v mapping

        Parameters
        ----------
        file_path : list
            path to data file
        raster_index : ndarray
            Raster index array

        Returns
        -------
        ndarray
            lat lon array
            (spatial_1, spatial_2, 2)
        """

        with WindX(file_path[0], hsds=False) as handle:
            lat_lon = handle.lat_lon[tuple([raster_index.flatten()])]
            lat_lon = lat_lon.reshape(
                (raster_index.shape[0],
                 raster_index.shape[1], 2))

        return lat_lon
