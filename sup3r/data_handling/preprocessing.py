# -*- coding: utf-8 -*-
"""
Sup3r preprocessing module.
"""
import xarray as xr
import numpy as np
import os

from rex import WindX
from reV.utilities.exceptions import ConfigError
from sup3r.utilities import utilities
from sup3r import __version__


class DataHandler:
    """Sup3r data handling and extraction"""

    def __init__(self, data_files, target,
                 shape, features, max_delta=20):
        """Data handling and extraction

        Parameters
        ----------
        data_files : list
            list of file paths
        target : tuple
            (lat, lon) lower left corner of raster
        shape : tuple
            (rows, cols) grid size
        features : list
            list of features to extract
        max_delta : int, optional
            Optional maximum limit on the raster shape that is retrieved at
            once. If shape is (20, 20) and max_delta=10, the full raseter will
            be retrieved in four chunks of (10, 10). This helps adapt to
            non-regular grids that curve over large distances, by default 20
        """

        self.data_files = data_files
        self.features = features
        self.shape = shape
        self.target = target
        self.raster_index = None
        self.max_delta = max_delta
        self.data, self.lat_lon = self.extract_data()

    def normalize_data(self, feature, mean, std):
        """Normalize data with initialized
        mean and standard deviation for
        a specific feature

        Returns
        -------
        data : np.ndarray
            normalized data array
        """
        self.data[:, :, :, self.features.index(feature)] = \
            (self.data[:, :, :, self.features.index(feature)] - mean) / std
        return self.data

    def extract_data(self):
        """Building base 4D data array

        Parameters
        ----------
        data_files : list
            list of strings of file paths

        Returns
        -------
        y : np.ndarray
            4D array of high res data
            (spatial_1, spatial_2, temporal, features)

        lat_lon : np.ndarray
            3D array of lat lon
            (spatial_1, spatial_2, 2)
            lat (lon) first channel (second channel)
        """

        raster_index = self.get_raster_index(self.data_files[0],
                                             self.target,
                                             self.shape)

        y, lat_lon = self._get_file_data(self.data_files[0],
                                         raster_index,
                                         self.features)
        if len(self.data_files) > 1:
            for f in self.data_files[1:]:
                tmp = self._get_file_data(f, raster_index,
                                          self.features,
                                          get_coords=False)
                y = np.concatenate((y, tmp), axis=2)

        y = utilities.transform_rotate_wind(y, lat_lon, self.features)

        self.data = y
        self.lat_lon = lat_lon
        self.raster_index = raster_index

        return y, lat_lon

    def get_raster_index(self, file_path, target, shape):
        """Get raster index for file data

        Parameters
        ----------
        file_path : str
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

        _, file_ext = os.path.splitext(file_path)
        if file_ext == '.h5':
            with WindX(file_path) as res:
                raster_index = res.get_raster_index(target, shape,
                                                    max_delta=self.max_delta)

        elif file_ext == '.nc':
            nc_file = xr.open_dataset(file_path)
            lat_diff = list(nc_file['XLAT'][0, :, 0] - target[0])
            lat_idx = np.argmin(np.abs(lat_diff))
            lon_diff = list(nc_file['XLON'][0, 0, :] - target[1])
            lon_idx = np.argmin(np.abs(lon_diff))
            raster_index = [[lat_idx, lat_idx + shape[0]],
                            [lon_idx, lon_idx + shape[1]]]
        else:
            raise ConfigError('Data must be either h5 or netcdf '
                              f'but received file extension: {file_ext}')
        return raster_index

    @classmethod
    def _get_file_data(cls, file_path,
                       raster_index,
                       features,
                       get_coords=True):
        """Extract fields from file for region
        given by target and shape

        Parameters
        ----------
        file_path : str
            File path
        target : tuple
            (lat, lon) for lower left corner of region
        shape : tuple
            (n_lat, n_lon) grid size for region
        features : list
            list of fields to extract from file
        get_coords : bool
            get coordinates

        Returns
        -------
        y : np.ndarray
            4D array of extracted data
            (spatial_1, spatial_2, temporal, features)
        lat_lon : np.ndarray
            3D array of (spatial_1, spatial_2, 2)
            with 2 channels as lat/lon in that order.
            Only returned if get_coords=True
        """

        _, file_ext = os.path.splitext(file_path)
        if file_ext == '.h5':
            return cls._get_h5_data(file_path, raster_index,
                                    features,
                                    get_coords=get_coords)
        elif file_ext == '.nc':
            return cls._get_nc_data(file_path, raster_index,
                                    features,
                                    get_coords=get_coords)
        else:
            raise ConfigError('Data must be either h5 or netcdf '
                              f'but received file extension: {file_ext}')

    @staticmethod
    def _get_h5_data(file_path, raster_index,
                     features, get_coords=True):
        """Get chunk of h5 data based on raster_indices
        and features

        Parameters
        ----------
        file_path : str
            h5 file path.
        raster_index : np.ndarray
            2D array of grid indices
        features : str list
            List of fields to extract from dataset
        get_coords : bool
            get coordinates

        Returns
        -------
        data : np.ndarray
            Real high-resolution data in a 4D array:
            (spatial_1, spatial_2, temporal, features)
        lat_lon : np.ndarray
            3D array of (spatial_1, spatial_2, 2)
            with 2 channels as lat/lon in that order.
            Only returned if get_coords=True
        """

        with WindX(file_path, hsds=False) as handle:

            data = np.zeros((raster_index.shape[0],
                             raster_index.shape[1],
                             len(handle.time_index),
                             len(features)), dtype=np.float32)

            for j, f in enumerate(features):
                data[:, :, :, j] = np.transpose(handle[f, :, raster_index],
                                                (1, 2, 0))
            if get_coords:
                lat_lon = np.zeros((raster_index.shape[0],
                                    raster_index.shape[1], 2))
                lat_lon = handle.lat_lon[raster_index]

        if get_coords:
            return data, lat_lon
        else:
            return data

    @staticmethod
    def _get_nc_data(file_path, raster_index,
                     features, get_coords=True,
                     level_index=None):
        """Get chunk of netcdf data based on raster_indices
        and features

        Parameters
        ----------
        file_path : str
            netcdf file path.
        raster_index : np.ndarray
            2D array of grid indices
        features : str list
            List of fields to extract from dataset
        get_coords : bool
            get coordinates

        Returns
        -------
        data : np.ndarray
            Real high-resolution data in a 4D array:
            (spatial_1, spatial_2, temporal, features)

        lat_lon : np.ndarray
            3D array of (spatial_1, spatial_2, 2)
            with 2 channels as lat/lon in that order.
            Only returned if get_coords=True
        """

        handle = xr.open_dataset(file_path)

        data = np.zeros((raster_index[0][1] - raster_index[0][0],
                         raster_index[1][1] - raster_index[1][0],
                         handle['Times'].shape[0],
                         len(features)), dtype=np.float32)

        for j, f in enumerate(features):
            if len(handle[f].shape) > 3:
                if level_index is None:
                    level_index = 0
                data[:, :, :, j] = \
                    np.transpose(
                        handle[f][:, level_index,
                                  raster_index[0][0]:raster_index[0][1],
                                  raster_index[1][0]:raster_index[1][1]],
                        (1, 2, 0))
            else:
                data[:, :, :, j] = \
                    np.transpose(
                        handle[f][:, level_index,
                                  raster_index[0][0]:raster_index[0][1],
                                  raster_index[1][0]:raster_index[1][1]],
                        (1, 2, 0))

            if get_coords:
                lat_lon = np.zeros((raster_index.shape[0],
                                    raster_index.shape[1], 2))
                lat_lon[:, :, 0] = \
                    handle['XLAT'][0, raster_index[0][0]:raster_index[0][1], 0]
                lat_lon[:, :, 1] = \
                    handle['XLONG'][0, :,
                                    raster_index[1][0]:raster_index[1][1]]

        if get_coords:
            return data, lat_lon
        else:
            return data


class Batch:
    """Batch of low_res and high_res data"""

    def __init__(self, low_res, high_res):
        """Stores low and high res data

        Parameters
        ----------
        low_res : np.ndarray
            4D array (batch_size, spatial_1, spatial_2, features)
        high_res : np.ndarray
            4D array (batch_size, spatial_1, spatial_2, features)
        """
        self.low_res = low_res
        self.high_res = high_res


class SpatialBatchHandler:
    """Sup3r spatial batch handling class"""

    def __init__(self, data, batch_size, val_split=0.2,
                 spatial_res=2):
        """
        Parameters
        ----------
        data : np.ndarray
            4D array (spatial_1, spatial_2, temporal, features)
        batch_size : int
            size of batches along temporal dimension
        spatial_res: int
            factor by which to coarsen spatial dimensions
        """

        self.data = data
        self.training_indices, self.val_indices = self._split_data(val_split)
        self.batch_size = batch_size
        self.spatial_res = spatial_res
        self.batch_indices = self._get_batch_indices()
        self.max = len(self.batch_indices)
        self._i = 0
        self.low_res = None
        self.high_res = None
        self.data_handler = None
        self.mean = None
        self.std = None

    def _split_data(self, val_split=0.2):
        """Splits time dimension into set of training indices
        and validation indices

        Parameters
        ----------
        val_split : float32, optional
            Fraction of full data array to
            reserve for validation, by default 0.2

        Returns
        -------
        training_indices : np.ndarray
            array of indices for training data slice
        val_indices : np.ndarray
            array of indices for validation data slice
        """

        n_observations = self.data.shape[2]
        n_training_obs = int(val_split * n_observations)
        self.training_indices = np.arange(n_training_obs)
        self.val_indices = np.arange(n_training_obs, n_observations)
        return self.training_indices, self.val_indices

    @property
    def val_data(self):
        """Validation data property

        Returns
        -------
        batch : Batch
            validation data batch. includes
            batch.low_res and batch.high_res
        """
        low_res, high_res = self._reshape_data(self.data[self.val_indices])
        batch = Batch(low_res, high_res)
        return batch

    @classmethod
    def make(cls, data_files, target,
             shape, features, val_split=0.2,
             batch_size=8, spatial_res=3,
             max_delta=20, norm=False):
        """Method to initialize both
        data and batch handlers

        Parameters
        ----------
        data_files : list
            list of file paths
        target : tuple
            (lat, lon) lower left corner of raster
        shape : tuple
            (rows, cols) grid size
        features : list
            list of features to extract
        val_split : float32
            fraction of data to reserve for validation
        batch_size : int
            size of batches along temporal dimension
        spatial_res: int
            factor by which to coarsen spatial dimensions
        max_delta : int, optional
            Optional maximum limit on the raster shape that is retrieved at
            once. If shape is (20, 20) and max_delta=10, the full raseter will
            be retrieved in four chunks of (10, 10). This helps adapt to
            non-regular grids that curve over large distances, by default 20

        Returns
        -------
        batchHandler : SpatialBatchHandler
            batchHandler with dataHandler attribute
        """
        data_handler = DataHandler(data_files, target,
                                   shape, features,
                                   max_delta=max_delta)
        batch_handler = SpatialBatchHandler(data_handler.data,
                                            batch_size,
                                            val_split,
                                            spatial_res)
        batch_handler.data_handler = data_handler
        return batch_handler

    def _get_batch_indices(self):
        """Get batches of data along temporal dimension
        """
        shuffled = self.training_indices.copy()
        np.random.shuffle(shuffled)
        n_batches = int(np.ceil(len(shuffled) / self.batch_size))
        self.batch_indices = np.array_split(shuffled, n_batches)

        return self.batch_indices

    def _reshape_data(self, high_res):
        """Coarsens high res data and reshapes data arrays
        to use time slices as observations

        Parameters
        ----------
        high_res : np.ndarray
            4D array (spatial_1, spatial_2, temporal, features)

        Returns
        -------
        low_res : np.ndarray
            4D array (temporal, spatial_1, spatial_2, features)
        high_res : np.ndarray
            4D array (temporal, spatial_1, spatial_2, features)
        """

        low_res = utilities.spatial_coarsening(high_res,
                                               self.spatial_res)
        low_res = low_res.transpose((2, 0, 1, 3))
        high_res = high_res.transpose((2, 0, 1, 3))
        return low_res, high_res

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i < self.max:
            self.high_res = \
                self.data[:, :,
                          self.batch_indices[self._i], :]
            self.low_res, self.high_res = \
                self._reshape_data(self.high_res)
            batch = Batch(self.low_res, self.high_res)
            self._i += 1
            return batch
        else:
            raise StopIteration
