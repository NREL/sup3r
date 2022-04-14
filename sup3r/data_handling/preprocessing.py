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
import psutil
from datetime import datetime as dt
from collections import defaultdict

from rex import WindX
from rex.utilities import log_mem
from rex.utilities.execution import SpawnProcessPool
from sup3r.utilities.utilities import (spatial_coarsening,
                                       uniform_box_sampler,
                                       temporal_coarsening,
                                       uniform_time_sampler,
                                       interp_var,
                                       transform_rotate_wind,
                                       BVF_squared)
from sup3r import __version__

np.random.seed(42)

logger = logging.getLogger(__name__)


def get_file_handle(file_paths):
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
    if not isinstance(file_paths, list):
        file_paths = [file_paths]

    _, source_type = os.path.splitext(file_paths[0])
    if source_type == '.h5':
        handle = WindX(file_paths[0], hsds=False)
    else:
        handle = xr.open_mfdataset(
            file_paths, combine='nested', concat_dim='Time')
        handle['time_index'] = handle['Times']
    return handle


class RasterIndex(list):
    """RasterIndex class to add
    shape method to NC raster_index"""

    def __init__(self, raster_obj):
        if any(isinstance(r, slice) for r in raster_obj):
            super().__init__(raster_obj)
            self.shape = (self[0].stop - self[0].start,
                          self[1].stop - self[1].start)
        else:
            super().__init__([raster_obj.flatten()])
            self.shape = raster_obj.shape


class Feature:
    """Class to simplify feature
    computations. Stores alternative names,
    feature height, feature basename, name of
    feature in handle"""

    def __init__(self, feature, handle):
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
        old_keys = [f for f in data[chunk_number].keys()
                    if f not in all_features]
        for k in old_keys:
            data[chunk_number].pop(k)

    @classmethod
    def serial_extract(cls, file_path, raster_index, time_chunks,
                       input_features):

        """Extract features in series

        Parameters
        ----------
        file_path : list
            list of file paths
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

        logger.info(f'Extracting {input_features}')
        mem = psutil.virtual_memory()
        logger.debug(
            f'Current memory usage is {mem.used / 1e9 :.3f} GB'
            f' out of {mem.total / 1e9 :.3f} GB total.')

        data = defaultdict(dict)

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

        logger.info(f'Finished extracting {input_features}')
        return data

    @classmethod
    def parallel_extract(cls, file_path, raster_index, time_chunks,
                         input_features, max_workers=None):

        """Extract features using parallel subprocesses

        Parameters
        ----------
        file_path : list
            list of file paths
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
        mem = psutil.virtual_memory()
        logger.debug(
            f'Current memory usage is {mem.used / 1e9 :.3f} GB'
            f' out of {mem.total / 1e9 :.3f} GB total.')

        futures = {}
        data = defaultdict(dict)
        now = dt.now()

        time_dep_features = [f for f in input_features
                             if f not in cls.TIME_IND_FEATURES]
        time_ind_features = [f for f in input_features
                             if f in cls.TIME_IND_FEATURES]

        if max_workers == 1:
            cls.serial_extract(
                file_path, raster_index, time_chunks, input_features)
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

                logger.info(
                    f'Started extracting {input_features}'
                    f' in {dt.now() - now}. Using {len(time_chunks)}'
                    f' time chunks of shape ({raster_index.shape[0]}, '
                    f'{raster_index.shape[1]}, '
                    f'{time_chunks[0].stop - time_chunks[0].start}) '
                    f'for {len(input_features)} features')

                for i, future in enumerate(as_completed(futures)):
                    if i % (len(futures) // 10 + 1) == 0:
                        logger.debug(f'{i+1} out of {len(futures)} feature '
                                     'chunks extracted.')
                        mem = psutil.virtual_memory()
                        logger.debug(
                            f'Current memory usage is {mem.used / 1e9 :.3f} GB'
                            f' out of {mem.total / 1e9 :.3f} GB total.')

            logger.info('Building input feature dictionary of '
                        f'{len(input_features)} features and '
                        f'{len(time_chunks)} time_chunks')
            for k, v in futures.items():
                data[v['chunk']][v['feature']] = k.result()

        logger.info(f'Finished extracting {input_features}')
        return data

    @classmethod
    def serial_compute(cls, data, data_array, time_chunks,
                       input_features, all_features):

        """Compute features in series

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
            Number of max workers to use for extraction.
            If equal to 1 then method is run in serial

        Returns
        -------
        data_array : ndarray
            Final array with feature data to be used for training
            (spatial_1, spatial_2, temporal, features)
        """

        derived_features = [f for f in all_features if f not in input_features]
        non_derived_features = [f for f in all_features if f in input_features]
        logger.info(f'Computing {derived_features}')

        mem = psutil.virtual_memory()
        logger.debug(
            f'Current memory usage is {mem.used / 1e9 :.3f} GB'
            f' out of {mem.total / 1e9 :.3f} GB total.')

        for t, t_slice in enumerate(time_chunks):
            for i, f in enumerate(derived_features):
                method = cls.lookup_method(f)
                height = Feature.get_feature_height(f)
                tmp = cls.get_input_arrays(data, t, f)
                data_array[:, :, t_slice, i] = method(tmp, height)

        for t, t_slice in enumerate(time_chunks):
            for i, f in enumerate(non_derived_features):
                data_array[:, :, t_slice, i] = data[t][f]
                data[t].pop(f)
            data.pop(t)

        logger.info(f'Finished computing {derived_features}')
        return data_array

    @classmethod
    def parallel_compute(cls, data, data_array, raster_index, time_chunks,
                         input_features, all_features, max_workers=None):

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
            Number of max workers to use for extraction.
            If equal to 1 then method is run in serial

        Returns
        -------
        data_array : ndarray
            Final array with feature data to be used for training
            (spatial_1, spatial_2, temporal, features)
        """

        derived_features = [f for f in all_features if f not in input_features]
        non_derived_features = [f for f in all_features if f in input_features]
        logger.info(f'Computing {derived_features}')

        mem = psutil.virtual_memory()
        logger.debug(
            f'Current memory usage is {mem.used / 1e9 :.3f} GB'
            f' out of {mem.total / 1e9 :.3f} GB total.')

        futures = {}
        now = dt.now()
        if max_workers == 1:
            cls.serial_compute(
                data, data_array, time_chunks, input_features, all_features)
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

                logger.info(
                    f'Started computing {derived_features}'
                    f' in {dt.now() - now}. Using {len(time_chunks)}'
                    f' time chunks of shape ({raster_index.shape[0]}, '
                    f'{raster_index.shape[1]}, '
                    f'{time_chunks[0].stop - time_chunks[0].start}) '
                    f'for {len(derived_features)} features')

                for i, future in enumerate(as_completed(futures)):
                    if i % (len(futures) // 10 + 1) == 0:
                        logger.debug(f'{i+1} out of {len(futures)} feature '
                                     'chunks computed')
                        mem = psutil.virtual_memory()
                        logger.debug(
                            f'Current memory usage is {mem.used / 1e9 :.3f} GB'
                            f' out of {mem.total / 1e9 :.3f} GB total.')

            logger.info(f'Finished computing {derived_features}')

            logger.info('Building final data array')
            for k, v in futures.items():
                t = v['chunk']
                t_slice = time_chunks[t]
                f = v['feature']
                f_index = all_features.index(f)
                data_array[:, :, t_slice, f_index] = k.result()

            for t, t_slice in enumerate(time_chunks):
                for i, f in enumerate(non_derived_features):
                    data_array[:, :, t_slice, i] = data[t][f]
                    data[t].pop(f)
                data.pop(t)

        return data_array

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
    TRAIN_ONLY_FEATURES = ('BVF_*',)

    def __init__(self, file_path, features, target=None, shape=None,
                 max_delta=20, time_pruning=1, val_split=0.1,
                 temporal_sample_shape=1, spatial_sample_shape=(10, 10),
                 raster_file=None, shuffle_time=False, max_workers=None,
                 time_chunk_size=100):

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
        max_workers : int | None
            max number of workers to use for data extraction.
            If max_workers == 1 then extraction will be serialized.
        time_chunk_size : int
            Size of chunks to split time dimension into for data extraction
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
        self.time_index = get_file_handle(self.file_path).time_index
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
        self.data = self.extract_data(
            self.file_path, self.raster_index, self.time_index,
            self.features, self.time_pruning, max_workers,
            time_chunk_size)
        self.data, self.val_data = self.split_data(self.data)

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
    def extract_data(cls, file_path, raster_index,
                     time_index, features, time_pruning,
                     max_workers=None, time_chunk_size=100):
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
        max_workers : int | None
            max number of workers to use for data extraction.
            If max_workers == 1 then extraction will be serialized.
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

        logger.debug(
            f'Loading data for raster of shape {raster_index.shape}')

        data_array = np.zeros(
            (raster_index.shape[0], raster_index.shape[1],
             len(time_index), len(features)),
            dtype=np.float32)

        n_chunks = len(time_index) // time_chunk_size + 1
        time_chunks = np.array_split(np.arange(0, len(time_index)), n_chunks)
        time_chunks = [slice(t[0], t[-1] + 1) for t in time_chunks]

        raw_features = cls.get_raw_feature_list(features)

        raw_data = cls.parallel_extract(
            file_path, raster_index, time_chunks,
            raw_features, max_workers)

        data_array = cls.parallel_compute(
            raw_data, data_array, raster_index, time_chunks,
            raw_features, features, max_workers)

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

    def __init__(self, file_path, features, target=None, shape=None,
                 max_delta=20, time_pruning=1, val_split=0.1,
                 temporal_sample_shape=1, spatial_sample_shape=(10, 10),
                 raster_file=None, shuffle_time=False, max_workers=None,
                 time_chunk_size=100):

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
            Whether to shuffle time indices before validation split
        max_workers : int | None
            max number of workers to use for data extraction.
            If max_workers == 1 then extraction will be serialized.
        time_chunk_size : int
            Size of chunks to split time dimension into for data extraction
        """

        super().__init__(
            file_path, features, target, shape, max_delta,
            time_pruning, val_split, temporal_sample_shape,
            spatial_sample_shape, raster_file, shuffle_time,
            max_workers, time_chunk_size)

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

        logger.debug(f'Extracting {feature}.')

        handle = get_file_handle(file_path)

        mem = psutil.virtual_memory()
        logger.debug(
            f'Current memory usage is {mem.used / 1e9 :.3f} GB'
            f' out of {mem.total / 1e9 :.3f} GB total.')

        f_info = Feature(feature, handle)
        interp_height = f_info.height
        basename = f_info.basename

        method = cls.lookup_method(feature)
        if method is not None and basename not in handle:
            return method(file_path, raster_index)

        try:
            if len(handle[basename].shape) > 3:
                if interp_height is None:
                    fdata = np.array(
                        handle[feature][
                            tuple([time_slice] + [0] + raster_index)],
                        dtype=np.float32)
                else:
                    logger.debug(
                        f'Interpolating {basename} at height {interp_height}m')
                    fdata = interp_var(
                        handle, basename, float(interp_height))
                    fdata = fdata[
                        tuple([time_slice] + raster_index)]
            else:
                fdata = np.array(
                    handle[feature][tuple([time_slice] + raster_index)],
                    dtype=np.float32)

            fdata = fdata.reshape(
                (raster_index.shape[0], raster_index.shape[1], -1))

        except ValueError:
            logger.error(
                f'{feature} cannot be extracted from source data')

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
            nc_file = xr.open_mfdataset(
                file_path, combine='nested', concat_dim='Time')
            lat_diff = list(nc_file['XLAT'][0, :, 0] - target[0])
            lat_idx = np.argmin(np.abs(lat_diff))
            lon_diff = list(nc_file['XLONG'][0, 0, :] - target[1])
            lon_idx = np.argmin(np.abs(lon_diff))
            raster_index = [slice(lat_idx, lat_idx + shape[0]),
                            slice(lon_idx, lon_idx + shape[1])]

            if (raster_index[1].stop >= len(lat_diff)
               or raster_index[1].stop >= len(lon_diff)):
                raise ValueError(
                    f'Invalid target {target} and shape {shape} for '
                    f'data domain of size ({len(lat_diff)}, '
                    f'{len(lon_diff)}) with lower left corner '
                    f'({np.min(nc_file["XLAT"][0, :, 0].values)}, '
                    f'{np.min(nc_file["XLONG"][0, 0, :].values)})')

            if self.raster_file is not None:
                logger.debug(f'Saving raster index: {self.raster_file}')
                np.save(self.raster_file, raster_index)
        return RasterIndex(raster_index)

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

    def __init__(self, file_path, features, target=None, shape=None,
                 max_delta=20, time_pruning=1, val_split=0.1,
                 temporal_sample_shape=1, spatial_sample_shape=(10, 10),
                 raster_file=None, shuffle_time=False, max_workers=None,
                 time_chunk_size=100):

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
        max_workers : int | None
            max number of workers to use for data extraction.
            If max_workers == 1 then extraction will be serialized.
        time_chunk_size : int
            Size of chunks to split time dimension into for data extraction
        """

        super().__init__(
            file_path, features, target, shape, max_delta,
            time_pruning, val_split, temporal_sample_shape,
            spatial_sample_shape, raster_file, shuffle_time,
            max_workers, time_chunk_size)

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

        logger.debug(f'Extracting {feature}.')

        handle = get_file_handle(file_path)

        mem = psutil.virtual_memory()
        logger.debug(
            f'Current memory usage is {mem.used / 1e9 :.3f} GB'
            f' out of {mem.total / 1e9 :.3f} GB total.')

        method = cls.lookup_method(feature)
        if method is not None and feature not in handle:
            return method(file_path, raster_index)

        try:
            fdata = handle[
                tuple([feature] + [time_slice] + raster_index)]

            fdata = fdata.reshape(
                (raster_index.shape[0], raster_index.shape[1], -1))

        except ValueError:
            logger.error(
                f'{feature} cannot be extracted from source data')

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
            with WindX(file_path[0]) as res:
                raster_index = res.get_raster_index(
                    target, shape, max_delta=self.max_delta)
            if self.raster_file is not None:
                logger.debug(
                    f'Saving raster index: {self.raster_file}')
                np.savetxt(self.raster_file, raster_index)
        return RasterIndex(raster_index)

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
        features = [f'windspeed_{height}m',
                    f'winddirection_{height}m',
                    'lat_lon']
        return features

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

        handle = get_file_handle(file_path)
        lat_lon = handle.lat_lon[tuple(raster_index)]
        lat_lon = lat_lon.reshape(
            (raster_index.shape[0],
             raster_index.shape[1], 2))

        return lat_lon


class ValidationData:
    """Iterator for validation data"""

    def __init__(self, data_handlers, batch_size=8,
                 spatial_res=3, temporal_res=1,
                 temporal_coarsening_method='subsample',
                 output_features_ind=None):
        """
        Parameters
        ----------
        handlers : list[DataHandler]
            List of DataHandler instances
        batch_size : int
            Size of validation data batches
        temporal_res : int
            Factor by which to coarsen temporal dimension
        spatial_res : int
            Factor by which to coarsen spatial dimensions
        temporal_coarsening_method : str
            [subsample, average, total]
            Subsample will take every temporal_res-th time step,
            average will average over temporal_res time steps,
            total will sum over temporal_res time steps
        output_features_ind : list | np.ndarray | None
            List/array of feature channel indices that are used for generative
            output, without any feature indices used only for training.
        """

        spatial_shapes = np.array(
            [d.spatial_sample_shape for d in data_handlers])
        temporal_shapes = np.array(
            [d.temporal_sample_shape for d in data_handlers])
        assert np.all(spatial_shapes[0] == spatial_shapes)
        assert np.all(temporal_shapes[0] == temporal_shapes)

        self.handlers = data_handlers
        self.spatial_sample_shape = spatial_shapes[0]
        self.temporal_sample_shape = temporal_shapes[0]
        self.val_indices = self._get_val_indices()
        self.max = np.ceil(
            len(self.val_indices) / (batch_size))
        self.batch_size = batch_size
        self.spatial_res = spatial_res
        self.temporal_res = temporal_res
        self._remaining_observations = len(self.val_indices)
        self.temporal_coarsening_method = temporal_coarsening_method
        self._i = 0
        self.output_features_ind = output_features_ind

    def _get_val_indices(self):
        """List of dicts to index each validation data
        observation across all handlers

        Returns
        -------
        val_indices : list[dict]
            List of dicts with handler_index and tuple_index.
            The tuple index is used to get validation data observation
            with data[tuple_index]"""

        val_indices = []
        for i, h in enumerate(self.handlers):
            for _ in range(h.val_data.shape[2]):
                spatial_slice = uniform_box_sampler(
                    h.val_data, self.spatial_sample_shape)
                temporal_slice = uniform_time_sampler(
                    h.val_data, self.temporal_sample_shape)
                tuple_index = tuple(
                    spatial_slice + [temporal_slice]
                    + [np.arange(h.val_data.shape[-1])])
                val_indices.append(
                    {'handler_index': i,
                     'tuple_index': tuple_index})
        return val_indices

    @property
    def shape(self):
        """Shape of full validation dataset across all handlers

        Returns
        -------
        shape : tuple
            (spatial_1, spatial_2, temporal, features)
            With temporal extent equal to the sum across
            all data handlers time dimension
        """
        time_steps = 0
        for h in self.handlers:
            time_steps += h.val_data.shape[2]
        return (self.handlers[0].val_data.shape[0],
                self.handlers[0].val_data.shape[1],
                time_steps,
                self.handlers[0].val_data.shape[3])

    def __iter__(self):
        self._i = 0
        self._remaining_observations = len(self.val_indices)
        return self

    def __len__(self):
        """
        Returns
        -------
        len : int
            Number of total batches
        """
        return int(self.max)

    def __next__(self):
        """Get validation data batch

        Returns
        -------
        batch : Batch
            validation data batch with low and high res data
            each with n_observations = batch_size
        """
        if self._remaining_observations > 0:
            if self._remaining_observations > self.batch_size:
                high_res = np.zeros((
                    self.batch_size,
                    self.spatial_sample_shape[0],
                    self.spatial_sample_shape[1],
                    self.temporal_sample_shape,
                    self.handlers[0].shape[-1]),
                    dtype=np.float32)
            else:
                high_res = np.zeros((
                    self._remaining_observations,
                    self.spatial_sample_shape[0],
                    self.spatial_sample_shape[1],
                    self.temporal_sample_shape,
                    self.handlers[0].shape[-1]),
                    dtype=np.float32)
            for i in range(high_res.shape[0]):
                val_index = self.val_indices[self._i + i]
                high_res[i, :, :, :, :] = self.handlers[
                    val_index['handler_index']].val_data[
                        val_index['tuple_index']]
                self._remaining_observations -= 1

            if self.temporal_sample_shape == 1:
                high_res = high_res[:, :, :, 0, :]
            batch = Batch.get_coarse_batch(
                high_res, self.spatial_res,
                temporal_res=self.temporal_res,
                temporal_coarsening_method=self.temporal_coarsening_method,
                output_features_ind=self.output_features_ind)
            self._i += 1
            return batch
        else:
            raise StopIteration


class Batch:
    """Batch of low_res and high_res data"""

    def __init__(self, low_res, high_res):
        """Stores low and high res data

        Parameters
        ----------
        low_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        high_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        """
        self._low_res = low_res
        self._high_res = high_res

    def __len__(self):
        """Get the number of observations in this batch."""
        return len(self._low_res)

    @property
    def shape(self):
        """Get the (low_res_shape, high_res_shape) shapes."""
        return (self._low_res.shape, self._high_res.shape)

    @property
    def low_res(self):
        """Get the low-resolution data for the batch."""
        return self._low_res

    @property
    def high_res(self):
        """Get the high-resolution data for the batch."""
        return self._high_res

    @staticmethod
    def reduce_features(high_res, output_features_ind=None):
        """Remove any feature channels that are only intended for the low-res
        training input.

        Parameters
        ----------
        high_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        output_features_ind : list | np.ndarray | None
            List/array of feature channel indices that are used for generative
            output, without any feature indices used only for training.
        """
        if output_features_ind is None:
            return high_res
        else:
            return high_res[..., output_features_ind]

    @classmethod
    def get_coarse_batch(cls, high_res,
                         spatial_res, temporal_res=1,
                         temporal_coarsening_method='subsample',
                         output_features_ind=None):
        """Coarsen high res data and return Batch with
        high res and low res data

        Parameters
        ----------
        high_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        spatial_res : int
            factor by which to coarsen spatial dimensions
        temporal_res : int
            factor by which to coarsen temporal dimension
        temporal_coarsening_method : str
            method to use for temporal coarsening.
            can be subsample, average, or total
        output_features_ind : list | np.ndarray | None
            List/array of feature channel indices that are used for generative
            output, without any feature indices used only for training.

        Returns
        -------
        Batch
            Batch instance with low and high res data
        """
        low_res = spatial_coarsening(
            high_res, spatial_res)

        if temporal_res != 1:
            low_res = temporal_coarsening(
                low_res, temporal_res,
                temporal_coarsening_method)

        high_res = cls.reduce_features(high_res, output_features_ind)

        batch = cls(low_res, high_res)
        return batch


class BatchHandler:
    """Sup3r base batch handling class"""

    def __init__(self, data_handlers, batch_size=8,
                 spatial_res=3, temporal_res=2,
                 means=None, stds=None,
                 norm=True, n_batches=10,
                 temporal_coarsening_method='subsample'):
        """
        Parameters
        ----------
        data_handlers : list[DataHandler]
            List of DataHandler instances
        batch_size : int
            Number of observations in a batch
        spatial_res : int
            Factor by which to coarsen spatial dimensions to generate
            low res data
        temporal_res : int
            Factor by which to coarsen temporal dimension to generate
            low res data
        norm : bool
            Whether to normalize the data or not
        means : np.ndarray
            dimensions (features)
            array of means for all features
            with same ordering as data features. If not None
            and norm is True these will be used for normalization
        stds : np.ndarray
            dimensions (features)
            array of means for all features
            with same ordering as data features. If not None
            and norm is True these will be used form normalization
        temporal_coarsening_method : str
            [subsample, average, total]
            Subsample will take every temporal_res-th time step,
            average will average over temporal_res time steps,
            total will sum over temporal_res time steps
        """

        spatial_shapes = np.array(
            [d.spatial_sample_shape for d in data_handlers])
        temporal_shapes = np.array(
            [d.temporal_sample_shape for d in data_handlers])
        assert np.all(spatial_shapes[0] == spatial_shapes)
        assert np.all(temporal_shapes[0] == temporal_shapes)

        self.data_handlers = data_handlers
        self._i = 0
        self.low_res = None
        self.high_res = None
        self.data_handler = None
        self.batch_size = batch_size
        self._val_data = None
        self.spatial_res = spatial_res
        self.temporal_res = temporal_res
        self.spatial_sample_shape = spatial_shapes[0]
        self.temporal_sample_shape = temporal_shapes[0]
        self.means = np.zeros((self.shape[-1]))
        self.stds = np.zeros((self.shape[-1]))
        self.n_batches = n_batches
        self.temporal_coarsening_method = temporal_coarsening_method
        self.current_batch_indices = None
        self.current_handler_index = None

        if norm:
            self.normalize(means, stds)

        self.val_data = ValidationData(
            data_handlers, batch_size=batch_size,
            spatial_res=spatial_res, temporal_res=temporal_res,
            temporal_coarsening_method=temporal_coarsening_method,
            output_features_ind=self.output_features_ind)

    def __len__(self):
        """Use user input of n_batches to specify length

        Returns
        -------
        self.n_batches : int
            Number of batches possible to iterate over
        """
        return self.n_batches

    @property
    def training_features(self):
        """Get the ordered list of feature names held in this object's
        data handlers"""
        return self.data_handlers[0].features

    @property
    def output_features(self):
        """Get the ordered list of feature names held in this object's
        data handlers"""
        return self.data_handlers[0].output_features

    @property
    def output_features_ind(self):
        """Get the feature channel indices that should be used for the
        generated output features"""
        if self.training_features == self.output_features:
            return None
        else:
            out = [i for i, feature in enumerate(self.training_features)
                   if feature in self.output_features]
            return out

    @staticmethod
    def get_source_type(file_paths):
        """Get data file type to use
        in source_type checking
        Parameters
        ----------
        file_paths : list
            path to data file
        Returns
        -------
        source_type : str
            data file extension
        """
        if not isinstance(file_paths, list):
            file_paths = [file_paths]

        _, source_type = os.path.splitext(file_paths[0])
        if source_type == '.h5':
            return 'h5'
        else:
            return 'nc'

    @staticmethod
    def chunk_file_paths(file_paths, list_chunk_size=None):
        """Split list of file paths into chunks
        of size list_chunk_size

        Parameters
        ----------
        file_paths : list
            List of file paths
        list_chunk_size : int, optional
            Size of file path liist chunk, by default None

        Returns
        -------
        list
            List of file path chunks
        """

        if isinstance(file_paths, list) and list_chunk_size is not None:
            file_paths = sorted(file_paths)
            n_chunks = len(file_paths) // list_chunk_size + 1
            file_paths = list(np.array_split(file_paths, n_chunks))
            file_paths = [list(fps) for fps in file_paths]
        return file_paths

    @staticmethod
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
        if BatchHandler.get_source_type(file_paths) == 'h5':
            HandlerClass = DataHandlerH5
        else:
            HandlerClass = DataHandlerNC
        return HandlerClass

    @classmethod
    def make(cls, file_paths, features,
             targets=None, shape=None, val_split=0.2,
             spatial_sample_shape=(10, 10),
             temporal_sample_shape=10,
             spatial_res=3, temporal_res=2,
             max_delta=20, norm=True,
             raster_files=None, time_pruning=1,
             batch_size=8, n_batches=10,
             means=None, stds=None,
             temporal_coarsening_method='subsample',
             list_chunk_size=None,
             max_workers=None,
             time_chunk_size=100):

        """Method to initialize both
        data and batch handlers

        Parameters
        ----------
        file_paths : list
            list of file paths
        targets : tuple
            List of several (lat, lon) lower left corner of raster. Either need
            target+shape or raster_file.
        shape : tuple
            (rows, cols) grid size
        features : list
            list of features to extract
        val_split : float32
            fraction of data to reserve for validation
        batch_size : int
            number of observations in a batch
        spatial_sample_shape : tuple
            size of spatial slices used for spatial batching
        temporal_sample_shape : int
            size of time slices used for temporal batching
        spatial_res: int
            factor by which to coarsen spatial dimensions
        temporal_res: int
            factor by which to coarsen temporal dimension
        max_delta : int, optional
            Optional maximum limit on the raster shape that is retrieved at
            once. If shape is (20, 20) and max_delta=10, the full raster will
            be retrieved in four chunks of (10, 10). This helps adapt to
            non-regular grids that curve over large distances, by default 20
        raster_files : list | str | None
            Files for raster_index array for the corresponding targets and
            shape. If a list these can be different files for different
            targets. If a string the same file will be used for all
            targets. If None raster_index will be calculated directly.
        norm : bool
            Wether to normalize data using means/stds calulcated across
            all handlers
        time_pruning : int
            Number of timesteps to downsample. If time_pruning=1 no time
            steps will be skipped.
        means : np.ndarray
            dimensions (features)
            array of means for all features
            with same ordering as data features
        stds : np.ndarray
            dimensions (features)
            array of means for all features
            with same ordering as data features
        n_batches : int
            Number of batches to iterate through
        temporal_coarsening_method : str
            [subsample, average, total]
            Subsample will take every temporal_res-th time step,
            average will average over temporal_res time steps,
            total will sum over temporal_res time steps
        list_chunk_size : int
            Size of chunks to split file_paths into if a list of files
            is passed. If None no splitting will be performed.
        max_workers : int | None
            max number of workers to use for data extraction.
            If max_workers == 1 then extraction will be serialized.
        time_chunk_size : int
            Size of chunks to split time dimension into for data extraction

        Returns
        -------
        batchHandler : BatchHandler
            batchHandler with dataHandler attribute
        """

        check = ((targets is not None and shape is not None)
                 or raster_files is not None)
        msg = ('You must either provide the targets+shape inputs '
               'or the raster_files input.')
        assert check, msg

        HandlerClass = cls.get_handler_class(file_paths)
        file_paths = cls.chunk_file_paths(file_paths, list_chunk_size)

        data_handlers = []
        for i, f in enumerate(file_paths):
            if raster_files is None:
                raster_file = None
            else:
                if not isinstance(raster_files, list):
                    raster_file = raster_files
                else:
                    raster_file = raster_files[i]
            if not isinstance(targets, list):
                target = targets
            else:
                target = targets[i]
            data_handlers.append(
                HandlerClass(
                    f, features, target=target,
                    shape=shape, max_delta=max_delta,
                    raster_file=raster_file, val_split=val_split,
                    spatial_sample_shape=spatial_sample_shape,
                    temporal_sample_shape=temporal_sample_shape,
                    time_pruning=time_pruning,
                    max_workers=max_workers,
                    time_chunk_size=time_chunk_size))
        batch_handler = BatchHandler(
            data_handlers, spatial_res=spatial_res,
            temporal_res=temporal_res, batch_size=batch_size,
            norm=norm, means=means, stds=stds, n_batches=n_batches,
            temporal_coarsening_method=temporal_coarsening_method)
        return batch_handler

    @property
    def shape(self):
        """Shape of full dataset across all handlers

        Returns
        -------
        shape : tuple
            (spatial_1, spatial_2, temporal, features)
            With temporal extent equal to the sum across
            all data handlers time dimension
        """
        time_steps = 0
        for h in self.data_handlers:
            time_steps += h.shape[2]
        return (self.data_handlers[0].shape[0],
                self.data_handlers[0].shape[1],
                time_steps,
                self.data_handlers[0].shape[3])

    def _get_stats(self):
        """Get standard deviations and means
        for all data features

        Returns
        -------
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
            n_elems = 0
            for data_handler in self.data_handlers:
                self.means[i] += np.sum(data_handler.data[:, :, :, i])
                n_elems += \
                    data_handler.shape[0] \
                    * data_handler.shape[1] \
                    * data_handler.shape[2]
            self.means[i] = self.means[i] / n_elems
            for data_handler in self.data_handlers:
                self.stds[i] += \
                    np.sum((data_handler.data[:, :, :, i] - self.means[i])**2)
            self.stds[i] = np.sqrt(self.stds[i] / n_elems)

    def normalize(self, means=None, stds=None):
        """Compute means and stds for each feature
        across all datasets and normalize each data handler dataset.
        Checks if input means and stds are different from stored
        means and stds and renormalizes if they are new
        """
        if means is None or stds is None:
            self._get_stats()
        elif means is not None and stds is not None:
            if (not np.array_equal(means, self.means)
                    or not np.array_equal(stds, self.stds)):
                self.unnormalize()
            self.means = means
            self.stds = stds
        for d in self.data_handlers:
            d.normalize(self.means, self.stds)

    def unnormalize(self):
        """Remove normalization from stored means and stds"""
        for d in self.data_handlers:
            d.unnormalize(self.means, self.stds)

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        self.current_batch_indices = []
        if self._i <= self.n_batches:
            handler_index = np.random.randint(
                0, len(self.data_handlers))
            self.current_handler_index = handler_index
            handler = self.data_handlers[handler_index]
            high_res = np.zeros((self.batch_size,
                                 self.spatial_sample_shape[0],
                                 self.spatial_sample_shape[1],
                                 self.temporal_sample_shape,
                                 self.shape[-1]))
            for i in range(self.batch_size):
                high_res[i, :, :, :, :] = handler.get_next()
                self.current_batch_indices.append(handler.current_obs_index)

            batch = Batch.get_coarse_batch(
                high_res, self.spatial_res,
                temporal_res=self.temporal_res,
                temporal_coarsening_method=self.temporal_coarsening_method,
                output_features_ind=self.output_features_ind)

            self._i += 1
            return batch
        else:
            raise StopIteration


class SpatialBatchHandler(BatchHandler):
    """Sup3r spatial batch handling class"""

    def __init__(self, data_handlers,
                 batch_size=8, spatial_res=3,
                 means=None, stds=None,
                 norm=True, n_batches=10):
        """
        Parameters
        ----------
        data_handlers : list[DataHandler]
            List of DataHandler instances
        batch_size : int
            Number of observations in a batch
        spatial_res : int
            Factor by which to coarsen spatial dimensions to generate
            low res data
        norm : bool
            Whether to normalize the data or not
        means : np.ndarray
            dimensions (features)
            array of means for all features
            with same ordering as data features. If not None
            and norm is True these will be used for normalization
        stds : np.ndarray
            dimensions (features)
            array of means for all features
            with same ordering as data features. If not None
            and norm is True these will be used form normalization
        """
        super().__init__(data_handlers, batch_size=batch_size,
                         spatial_res=spatial_res, temporal_res=1,
                         norm=norm, n_batches=n_batches,
                         means=means, stds=stds)

    @classmethod
    def make(cls, file_paths, features,
             targets=None, shape=None,
             val_split=0.2, batch_size=8,
             spatial_sample_shape=(10, 10),
             spatial_res=3, max_delta=20,
             norm=True, raster_files=None,
             time_pruning=1, means=None,
             n_batches=10,
             stds=None,
             list_chunk_size=None,
             max_workers=None,
             time_chunk_size=100):

        """Method to initialize both
        data and batch handlers

        Parameters
        ----------
        file_paths : list
            list of file paths to wind data files
        features : list
            list of features to extract
        targets : tuple
            List of several (lat, lon) lower left corner of raster. Either need
            target+shape or raster_file.
        shape : tuple
            (rows, cols) grid size. Either need target+shape or raster_file.
        val_split : float32
            fraction of data to reserve for validation
        batch_size : int
            number of observations in a batch
        spatial_sample_shape : tuple
            size of spatial slices used for spatial batching
        spatial_res: int
            factor by which to coarsen spatial dimensions
        max_delta : int, optional
            Optional maximum limit on the raster shape that is retrieved at
            once. If shape is (20, 20) and max_delta=10, the full raster will
            be retrieved in four chunks of (10, 10). This helps adapt to
            non-regular grids that curve over large distances, by default 20
        raster_files : list | str | None
            Files for raster_index array for the corresponding targets and
            shape. If a list these can be different files for different
            targets. If a string the same file will be used for all
            targets. If None raster_index will be calculated directly. Either
            need target+shape or raster_file.
        norm : bool
            Wether to normalize data using means/stds calulcated across
            all handlers
        time_pruning : int
            Number of timesteps to downsample. If time_pruning=1 no time
            steps will be skipped.
        means : np.ndarray
            dimensions (features)
            array of means for all features
            with same ordering as data features
        stds : np.ndarray
            dimensions (features)
            array of means for all features
            with same ordering as data features
        n_batches : int
            Number of batches to iterate through
        list_chunk_size : int
            Size of chunks to split file_paths into if a list of files
            is passed. If None no splitting will be performed.
        max_workers : int | None
            max number of workers to use for data extraction.
            If max_workers == 1 then extraction will be serialized.
        time_chunk_size : int
            Size of chunks to split time dimension into for data extraction

        Returns
        -------
        batchHandler : SpatialBatchHandler
            batchHandler with dataHandler attribute
        """

        check = ((targets is not None and shape is not None)
                 or raster_files is not None)
        msg = ('You must either provide the targets+shape inputs '
               'or the raster_files input.')
        assert check, msg

        HandlerClass = cls.get_handler_class(file_paths)
        file_paths = cls.chunk_file_paths(file_paths, list_chunk_size)

        data_handlers = []
        for i, f in enumerate(file_paths):
            if raster_files is None:
                raster_file = None
            if not isinstance(raster_files, list):
                raster_file = raster_files
            else:
                raster_file = raster_files[i]
            if not isinstance(targets, list):
                target = targets
            else:
                target = targets[i]
            data_handlers.append(
                HandlerClass(
                    f, features,
                    target=target, shape=shape,
                    max_delta=max_delta,
                    raster_file=raster_file,
                    val_split=val_split,
                    spatial_sample_shape=spatial_sample_shape,
                    temporal_sample_shape=1,
                    time_pruning=time_pruning,
                    max_workers=max_workers,
                    time_chunk_size=time_chunk_size))
        batch_handler = SpatialBatchHandler(
            data_handlers, spatial_res=spatial_res,
            batch_size=batch_size, norm=norm, means=means,
            stds=stds, n_batches=n_batches)
        return batch_handler

    def __next__(self):
        if self._i <= self.n_batches:
            handler_index = np.random.randint(
                0, len(self.data_handlers))
            handler = self.data_handlers[handler_index]
            high_res = np.zeros((self.batch_size,
                                 self.spatial_sample_shape[0],
                                 self.spatial_sample_shape[1],
                                 self.shape[-1]), dtype=np.float32)
            for i in range(self.batch_size):
                high_res[i, :, :, :] = handler.get_next()[:, :, 0, :]

            batch = Batch.get_coarse_batch(
                high_res, self.spatial_res,
                output_features_ind=self.output_features_ind)

            self._i += 1
            return batch
        else:
            raise StopIteration
