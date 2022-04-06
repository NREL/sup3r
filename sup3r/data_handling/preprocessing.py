# -*- coding: utf-8 -*-
"""
Sup3r preprocessing module.
"""
from abc import abstractmethod
import logging
import xarray as xr
import numpy as np
import os
import re

from rex import WindX
from rex.utilities import log_mem
from sup3r.utilities.utilities import (spatial_coarsening,
                                       uniform_box_sampler,
                                       temporal_coarsening,
                                       uniform_time_sampler,
                                       interp_var,
                                       transform_rotate_wind,
                                       gradient_richardson_number,
                                       BVF_squared)
from sup3r import __version__

np.random.seed(42)

logger = logging.getLogger(__name__)


class RasterIndex(list):
    """RasterIndex class to add
    shape method to NC raster_index"""

    def __new__(cls, raster_obj):
        obj = list(raster_obj).__new__(cls)
        cls._shape = None
        if any(isinstance(r, slice) for r in raster_obj):
            return obj
        else:
            return np.array(raster_obj)

    @property
    def shape(self):
        """Shape method for both
        H5 and NC raster_index objects

        Returns
        -------
        tuple
            Shape of raster index
        """

        if self._shape is None:
            if any(isinstance(r, slice) for r in self):
                self._shape = (
                    self[0].stop - self[0].start,
                    self[1].stop - self[1].start)
            else:
                self._shape = np.array(self).shape
        return self._shape


class FeatureHandler:
    """Feature Handler with cache
    for previously loaded features used
    in other calculations
    """

    def __init__(self):
        self.feature_cache = {}
        self.registry = {
            'BVF_squared': self.get_bvf_squared,
            'Ri': self.get_richardson_number,
            'U_(.*)m': self.get_u,
            'V_(.*)m': self.get_v}

    def lookup(self, feature):
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
        for k, v in self.registry.items():
            if re.match(k, feature):
                return v
        return None

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
        height = re.search(r'\d*m', feature)
        if height is not None:
            height = float(height[0].strip('m'))
        return height

    def compute_feature(self, handle, raster_index,
                        feature):
        """Compute single feature by extracting
        needed features from data source

        Parameters
        ----------
        handle : WindX | xarray
            Data Handle for either WTK data
            or WRF data
        raster_index : ndarray
            Raster index array
        feature : str
            Feature to extract from data

        Returns
        -------
        ndarray
            Data array for computed feature

        """

        if feature not in handle:
            logger.info(f'Computing {feature}.')
        if feature in self.feature_cache:
            logger.info(
                f'{feature} already computed. '
                'Loading from cache.')
            return self.feature_cache[feature]

        method = self.lookup(feature)
        height = self.get_feature_height(feature)
        if method is not None:
            if height is not None:
                fdata = method(
                    handle, raster_index, height)
            else:
                fdata = method(
                    handle, raster_index)
        else:
            if feature in handle:
                fdata = self.extract_feature(
                    handle, raster_index, feature)
            else:
                try:
                    fdata = self.extract_feature(
                        handle, raster_index, feature,
                        height)
                except ValueError:
                    logger.error(
                        f'{feature} not found in source data.')

        self.feature_cache[feature] = fdata
        return fdata

    @abstractmethod
    def extract_feature(self, handle, raster_index,
                        feature, interp_height=None):
        """Extract single feature from data source

        Parameters
        ----------
        handle : WindX | xarray
            Data Handle for either WTK data
            or WRF data
        raster_index : ndarray
            Raster index array
        feature : str
            Feature to extract from data
        interp_height : float | None
            Interpolation height for wrf data

        Returns
        -------
        ndarray
            Data array for extracted feature
        """

    def get_u(self, handle, raster_index, height):
        """Compute U wind component

        Parameters
        ----------
        handle : WindX | xarray
            Data Handle for either WTK data
            or WRF data
        raster_index : ndarray
            Raster index array
        height : str | int
            Height of U/V to extract in meters.
            e.g. 100

        Returns
        -------
        U : ndarray
            array of U wind component
        """

        u, _ = self.get_uv(handle, raster_index, height)
        return u

    def get_v(self, handle, raster_index, height):
        """Compute V wind component

        Parameters
        ----------
        handle : WindX | xarray
            Data Handle for either WTK data
            or WRF data
        raster_index : ndarray
            Raster index array
        height : str | int
            Height of U/V to extract in meters.
            e.g. 100

        Returns
        -------
        V : ndarray
            array of V wind component
        """

        _, v = self.get_uv(handle, raster_index, height)
        return v

    @abstractmethod
    def get_uv(self, handle, raster_index, height):
        """Compute U and V wind components

        Parameters
        ----------
        handle : WindX | xarray
            Data Handle for either WTK data
            or WRF data
        raster_index : ndarray
            Raster index array
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

    @abstractmethod
    def get_richardson_number(self, handle, raster_index):
        """Compute Bulk Richardson Number

        Parameters
        ----------
        handle : WindX | xarray
            Data Handle for either WTK data
            or WRF data
        raster_index : ndarray
            Raster index array

        Returns
        -------
        ndarray
            Bulk Richardson Number array
        """

    @abstractmethod
    def get_bvf_squared(self, handle, raster_index):
        """Compute BVF squared

        Parameters
        ----------
        handle : WindX | xarray
            Data Handle for either WTK data
            or WRF data
        raster_index : ndarray
            Raster index array

        Returns
        -------
        ndarray
            BVF squared array
        """

    @abstractmethod
    def get_lat_lon(self, handle, raster_index):
        """Get lats and lons corresponding to raster
        for use in windspeed/direction -> u/v mapping

        Parameters
        ----------
        handle : WindX | xarray
            Data Handle for either WTK data
            or WRF data
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

    def __init__(self, file_path, features, target=None, shape=None,
                 max_delta=20, time_pruning=1, val_split=0.1,
                 temporal_sample_shape=1, spatial_sample_shape=(10, 10),
                 raster_file=None, shuffle_time=False):

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
        """
        logger.info('Initializing DataHandler from source files: {}'
                    .format(file_path))

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

        self.data = self.extract_data()
        self.data, self.val_data = self._split_data()

        logger.info('Finished intializing DataHandler.')
        log_mem(logger, log_level='INFO')

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
            logger.error(
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

    def _get_file_data(self, file_path, raster_index,
                       features):
        """Extract fields from file for region
        given by target and shape

        Parameters
        ----------
        file_path : str | list
            File path
        target : tuple
            (lat, lon) for lower left corner of region
        shape : tuple
            (n_lat, n_lon) grid size for region
        features : list
            list of fields to extract from file

        Returns
        -------
        y : np.ndarray
            4D array of extracted data
            (spatial_1, spatial_2, temporal, features)
        """

        logger.debug('Loading data for raster of shape {}'
                     .format(raster_index.shape))

        handle = self._get_file_handle(file_path)

        data = np.zeros((raster_index.shape[0],
                         raster_index.shape[1],
                         len(handle.time_index),
                         len(features)),
                        dtype=np.float32)
        for j, f in enumerate(features):
            data[:, :, :, j] = self.compute_feature(
                handle, raster_index, f)

        return data

    def _split_data(self):
        """Splits time dimension into set of training indices
        and validation indices

        Parameters
        ----------
        val_split : float32, optional
            Fraction of full data array to
            reserve for validation, by default 0.2

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
        self.val_data = self.data[:, :, val_indices, :]
        self.data = self.data[:, :, training_indices, :]
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

    def extract_data(self):
        """Building base 4D data array. Can
        handle multiple files but assumes each
        file has the same spatial domain

        Returns
        -------
        data : np.ndarray
            4D array of high res data
            (spatial_1, spatial_2, temporal, features)
        """

        if not isinstance(self.file_path, list):
            self.file_path = [self.file_path]

        y = np.concatenate(
            [self._get_file_data(f, self.raster_index, self.features)
                for f in self.file_path], axis=2)

        self.data = y[:, :, ::self.time_pruning, :]

        return self.data

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

    @abstractmethod
    def _get_file_handle(self, file_path):
        """Get file type specific file handle

        Parameters
        ----------
        file_path : str
            Path to file for which to return a handle

        Returns
        -------
        handle : xarray | WindX
            File handle for corresponding file path
        """


class DataHandlerNC(DataHandler):
    """Data Handler for NETCDF data"""

    def __init__(self, file_path, features, target=None, shape=None,
                 max_delta=20, time_pruning=1, val_split=0.1,
                 temporal_sample_shape=1, spatial_sample_shape=(10, 10),
                 raster_file=None, shuffle_time=False):

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
        """

        super().__init__(
            file_path, features, target, shape, max_delta,
            time_pruning, val_split, temporal_sample_shape,
            spatial_sample_shape, raster_file, shuffle_time)

    def _get_file_handle(self, file_path):
        """Get file type specific file handle

        Parameters
        ----------
        file_path : list
            Path to files for which to return a handle

        Returns
        -------
        handle : WindX
            File handle for corresponding file path
        """

        handle = xr.open_mfdataset(
            file_path, combine='nested', concat_dim='Time')
        handle['time_index'] = handle['Times']
        return handle

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
            raster_index = RasterIndex(
                [slice(lat_idx, lat_idx + shape[0]),
                 slice(lon_idx, lon_idx + shape[1])])

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
        return raster_index

    def extract_feature(self, handle, raster_index,
                        feature, interp_height=None):
        """Extract single feature from data source

        Parameters
        ----------
        handle : WindX | xarray
            Data Handle for either WTK data
            or WRF data
        raster_index : ndarray
            Raster index array
        feature : str
            Feature to extract from data
        interp_height : float | None
            Interpolation height for wrf data

        Returns
        -------
        ndarray
            Data array for extracted feature

        """

        logger.debug(f'Extracting {feature}.')

        if feature in self.feature_cache:
            logger.debug(
                f'{feature} already extracted. Loading from cache. ')
            return self.feature_cache[feature]

        if len(handle[feature].shape) > 3:
            if interp_height is None:
                fdata = \
                    handle[feature][tuple([slice(None)] + [0] + raster_index)]
            else:
                fdata = interp_var(handle, feature, float(interp_height))
                fdata = fdata[tuple([slice(None)] + raster_index)]
        else:
            fdata = np.array(
                handle[feature][tuple([slice(None)] + raster_index)])

        fdata = np.transpose(fdata, (1, 2, 0))
        self.feature_cache[feature] = fdata
        return fdata

    def get_uv(self, handle, raster_index, height):
        """Compute U and V wind components

        Parameters
        ----------
        handle : WindX | xarray
            Data Handle for either WTK data
            or WRF data
        raster_index : ndarray
            Raster index array
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

        u = self.extract_feature(
            handle, raster_index, 'U', height)
        v = self.extract_feature(
            handle, raster_index, 'V', height)
        return u, v

    def get_richardson_number(self, handle, raster_index):
        """Compute Bulk Richardson Number

        Parameters
        ----------
        handle : WindX | xarray
            Data Handle for either WTK data
            or WRF data
        raster_index : ndarray
            Raster index array

        Returns
        -------
        ndarray
            Bulk Richardson Number array

        """

        T_top = self.extract_feature(
            handle, raster_index, 'T', 200) - 273.15
        T_bottom = self.extract_feature(
            handle, raster_index, 'T', 100) - 273.15
        P_top = self.extract_feature(
            handle, raster_index, 'P', 200)
        P_bottom = self.extract_feature(
            handle, raster_index, 'P', 100)
        U_top = self.extract_feature(
            handle, raster_index, 'U', 200)
        U_bottom = self.extract_feature(
            handle, raster_index, 'U', 100)
        V_top = self.extract_feature(
            handle, raster_index, 'V', 200)
        V_bottom = self.extract_feature(
            handle, raster_index, 'V', 100)

        return gradient_richardson_number(T_top, T_bottom, P_top, P_bottom,
                                          U_top, U_bottom, V_top, V_bottom,
                                          100)

    def get_bvf_squared(self, handle, raster_index):
        """Compute BVF squared

        Parameters
        ----------
        handle : WindX | xarray
            Data Handle for either WTK data
            or WRF data
        raster_index : ndarray
            Raster index array

        Returns
        -------
        ndarray
            BVF squared array

        """

        T_top = self.extract_feature(
            handle, raster_index, 'T', 200) - 273.15
        T_bottom = self.extract_feature(
            handle, raster_index, 'T', 100) - 273.15
        P_top = self.extract_feature(
            handle, raster_index, 'P', 200)
        P_bottom = self.extract_feature(
            handle, raster_index, 'P', 100)

        return BVF_squared(T_top, T_bottom,
                           P_top, P_bottom, 100)

    def get_lat_lon(self, handle, raster_index):
        """Get lats and lons corresponding to raster
        for use in windspeed/direction -> u/v mapping

        Parameters
        ----------
        handle : WindX | xarray
            Data Handle for either WTK data
            or WRF data
        raster_index : ndarray
            Raster index array

        Returns
        -------
        ndarray
            lat lon array
            (spatial_1, spatial_2, 2)
        """

        if 'lat_lon' in self.feature_cache:
            return self.feature_cache['lat_lon']

        lat_lon = np.zeros(
            (raster_index[0][1] - raster_index[0][0],
             raster_index[1][1] - raster_index[1][0], 2),
            dtype=np.float32)
        lat_lon[:, :, 0] = \
            handle['XLAT'][tuple([0] + raster_index)]
        lat_lon[:, :, 1] = \
            handle['XLONG'][tuple([0] + raster_index)]

        self.feature_cache['lat_lon'] = lat_lon
        return lat_lon


class DataHandlerH5(DataHandler):
    """DataHandler for H5 Data"""

    def __init__(self, file_path, features, target=None, shape=None,
                 max_delta=20, time_pruning=1, val_split=0.1,
                 temporal_sample_shape=1, spatial_sample_shape=(10, 10),
                 raster_file=None, shuffle_time=False):

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
        """

        super().__init__(
            file_path, features, target, shape, max_delta,
            time_pruning, val_split, temporal_sample_shape,
            spatial_sample_shape, raster_file, shuffle_time)

    def _get_file_handle(self, file_path):
        """Get file type specific file handle

        Parameters
        ----------
        file_path : str
            Path to file for which to return a handle

        Returns
        -------
        handle : WindX
            File handle for corresponding file path
        """

        return WindX(file_path, hsds=False)

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
            logger.debug('Loading raster index: {}'.format(self.raster_file))
            raster_index = np.loadtxt(self.raster_file).astype(np.uint32)
        else:
            logger.debug('Calculating raster index from WTK file '
                         f'for shape {shape} and target {target}')
            with WindX(file_path[0]) as res:
                raster_index = RasterIndex(
                    res.get_raster_index(
                        target, shape, max_delta=self.max_delta))
            if self.raster_file is not None:
                logger.debug('Saving raster index: {}'
                             .format(self.raster_file))
                np.savetxt(self.raster_file, raster_index)
        return raster_index

    def extract_feature(self, handle, raster_index,
                        feature):
        """Extract single feature from data source

        Parameters
        ----------
        handle : WindX | xarray
            Data Handle for either WTK data
            or WRF data
        raster_index : ndarray
            Raster index array
        feature : str
            Feature to extract from data
        interp_height : float | None
            Interpolation height for wrf data

        Returns
        -------
        ndarray
            Data array for extracted feature
        """

        logger.debug(f'Extracting {feature}.')

        if feature in self.feature_cache:
            logger.debug(
                f'{feature} already extracted. Loading from cache. ')
            return self.feature_cache[feature]

        fdata = handle[feature, :, raster_index.flatten()]
        fdata = fdata.reshape((len(fdata),
                               raster_index.shape[0],
                               raster_index.shape[1]))

        fdata = np.transpose(fdata, (1, 2, 0))
        self.feature_cache[feature] = fdata
        return fdata

    def get_uv(self, handle, raster_index, height):
        """Compute U and V wind components

        Parameters
        ----------
        handle : WindX | xarray
            Data Handle for either WTK data
            or WRF data
        raster_index : ndarray
            Raster index array
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

        required_inputs = [f'windspeed_{height}m',
                           f'winddirection_{height}m']
        ws = self.extract_feature(
            handle, raster_index, required_inputs[0])
        wd = self.extract_feature(
            handle, raster_index, required_inputs[1])
        lat_lon = self.get_lat_lon(handle, raster_index)
        logger.info(f'Transforming {required_inputs}'
                    ' to U/V and aligning with grid.')
        return transform_rotate_wind(ws, wd, lat_lon)

    def get_richardson_number(self, handle, raster_index):
        """Compute Bulk Richardson Number

        Parameters
        ----------
        handle : WindX | xarray
            Data Handle for either WTK data
            or WRF data
        raster_index : ndarray
            Raster index array

        Returns
        -------
        ndarray
            Bulk Richardson Number array

        """

        T_top = self.extract_feature(
            handle, raster_index, 'temperature_200m')
        T_bottom = self.extract_feature(
            handle, raster_index, 'temperature_100m')
        P_top = self.extract_feature(
            handle, raster_index, 'pressure_200m')
        P_bottom = self.extract_feature(
            handle, raster_index, 'pressure_100m')
        U_top, V_top = self.get_uv(
            handle, raster_index, 200)
        U_bottom, V_bottom = self.get_uv(
            handle, raster_index, 100)

        return gradient_richardson_number(T_top, T_bottom, P_top, P_bottom,
                                          U_top, U_bottom, V_top, V_bottom,
                                          100)

    def get_bvf_squared(self, handle, raster_index):
        """Compute BVF squared

        Parameters
        ----------
        handle : WindX | xarray
            Data Handle for either WTK data
            or WRF data
        raster_index : ndarray
            Raster index array

        Returns
        -------
        ndarray
            BVF squared array

        """

        T_top = self.extract_feature(
            handle, raster_index, 'temperature_200m')
        T_bottom = self.extract_feature(
            handle, raster_index, 'temperature_100m')
        P_top = self.extract_feature(
            handle, raster_index, 'pressure_200m')
        P_bottom = self.extract_feature(
            handle, raster_index, 'pressure_100m')

        return BVF_squared(T_top, T_bottom,
                           P_top, P_bottom, 100)

    def get_lat_lon(self, handle, raster_index):
        """Get lats and lons corresponding to raster
        for use in windspeed/direction -> u/v mapping

        Parameters
        ----------
        handle : WindX | xarray
            Data Handle for either WTK data
            or WRF data
        raster_index : ndarray
            Raster index array

        Returns
        -------
        ndarray
            lat lon array
            (spatial_1, spatial_2, 2)
        """

        if 'lat_lon' in self.feature_cache:
            return self.feature_cache['lat_lon']

        lat_lon = np.zeros((raster_index.shape[0],
                            raster_index.shape[1], 2),
                           dtype=np.float32)
        lat_lon = handle.lat_lon[raster_index]

        self.feature_cache['lat_lon'] = lat_lon
        return lat_lon


class ValidationData:
    """Iterator for validation data"""

    def __init__(self, data_handlers, batch_size=8,
                 spatial_res=3, temporal_res=1,
                 temporal_coarsening_method='subsample'):
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
                high_res, self.spatial_res, self.temporal_res,
                self.temporal_coarsening_method)
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

    @classmethod
    def get_coarse_batch(cls, high_res,
                         spatial_res, temporal_res,
                         temporal_coarsening_method):
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

        Returns
        -------
        Batch
            Batch instance with low and high res data
        """
        low_res = spatial_coarsening(
            high_res, spatial_res)
        low_res = temporal_coarsening(
            low_res, temporal_res,
            temporal_coarsening_method)
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
            temporal_coarsening_method=temporal_coarsening_method)

    def __len__(self):
        """Use user input of n_batches to specify length

        Returns
        -------
        self.n_batches : int
            Number of batches possible to iterate over
        """
        return self.n_batches

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
             list_chunk_size=None):

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
                    time_pruning=time_pruning))
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
                high_res, self.spatial_res, self.temporal_res,
                self.temporal_coarsening_method)
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
             list_chunk_size=None):

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
                    time_pruning=time_pruning))
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
            low_res = spatial_coarsening(
                high_res, self.spatial_res)
            batch = Batch(low_res, high_res)
            self._i += 1
            return batch
        else:
            raise StopIteration
