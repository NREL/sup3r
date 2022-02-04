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


class MultiDataHandler:
    """Class for handling multiple instances
    of DataHandler
    """

    def __init__(self, file_paths, targets, shape,
                 features, max_delta=20):
        """
        Parameters
        ----------
        file_paths : list
            list of file paths
        targets : list of tuples | tuple
            list of lower left corner coordinates
            with same ordering as file_paths or single
            tuple for all files
        shape : tuple
            (n_rows, n_cols)
            grid size
        features : list
            list of features to extract from each
            data file
        """

        if not isinstance(targets, list):
            targets = [targets] * len(file_paths)
        if not isinstance(file_paths, list):
            file_paths = [file_paths]

        data_handlers = []
        for i, f in enumerate(file_paths):
            data_handlers.append(DataHandler(f, targets[i],
                                             shape, features,
                                             max_delta=max_delta))
        self.data_handlers = data_handlers
        self.current_handler = None
        self.max = len(data_handlers)
        self.grid_shape = shape
        self.features = features
        self._i = 0

    @property
    def shape(self):
        """Shape property

        Returns
        -------
        shape : tuple
            Full data shape with time dimension stacked
        """
        return (self.grid_shape[0], self.grid_shape[1],
                sum([d.shape[2] for d in self.data_handlers]),
                len(self.features))

    def __len__(self):
        """
        Returns
        -------
        len()
            number of data handlers
        """
        return len(self.data_handlers)

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i < self.max:
            self.current_handler = self.data_handlers[self._i]
            self._i += 1
            return self.current_handler
        else:
            raise StopIteration

    def handler_and_time_index(self, stacked_time_index):
        """Gets handler_index and time_index of that
        handler from a stacked time index. e.g. If stacked_time_index
        is 50 and self.data_handler[0].shape[2] = 45 then this will
        return handler_index = 1, time_index = 5

        Parameters
        ----------
        stacked_time_index : int
            time_index which treats the data handlers as having
            a stacked time dimension

        Returns
        -------
        handler_index : int
            index of the data handler associated with the given
            stacked_time_index
        time_index : int
            time_index associated with the stacked_time_index
            and the corresponding data handler
        """

        handler_index = 0
        time_index = stacked_time_index
        while time_index >= self.data_handlers[handler_index].shape[2]:
            time_index -= self.data_handlers[handler_index].shape[2]
            handler_index += 1
        return handler_index, time_index

    @property
    def stacked_data(self):
        """Stack data along time dimension

        Returns
        -------
        data : np.ndarray
            4D data array with time dimension concatenated
            from all data handler data
            (spatial_1, spatial_2, temporal, features)
        """
        return np.concatenate([d.data for d in self.data_handlers], axis=2)

    def data(self, time_steps):
        """Data method

        Parameters
        ----------
        time_steps : list
            list of integer time indices

        Returns
        -------
        data : np.ndarray
            4D data array with time dimension size
            equal to len(time_steps)
            (spatial_1, spatial_2, temporal, features)
        """
        data = np.zeros((self.shape[0], self.shape[1],
                         len(time_steps), self.shape[3]),
                        dtype=np.float32)
        for i, t in enumerate(time_steps):
            handler_index, time_index = self.handler_and_time_index(t)
            data[:, :, i, :] = \
                self.data_handlers[handler_index].data[:, :, time_index, :]
        return data


class DataHandler:
    """Sup3r data handling and extraction"""

    def __init__(self, file_path, target,
                 shape, features, max_delta=20,
                 raster_file=None):
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
            once. If shape is (20, 20) and max_delta=10, the full raster will
            be retrieved in four chunks of (10, 10). This helps adapt to
            non-regular grids that curve over large distances, by default 20
        """

        self.file_path = file_path
        self.features = features
        self.grid_shape = shape
        self.target = target
        self.raster_index = None
        self.max_delta = max_delta
        self.raster_file = raster_file
        self.data, self.lat_lon = self.extract_data()

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

        raster_index = self.get_raster_index(self.file_path,
                                             self.target,
                                             self.grid_shape)

        y, lat_lon = self._get_file_data(self.file_path,
                                         raster_index,
                                         self.features)

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

        if self.raster_file is not None and os.path.exists(self.raster_file):
            raster_index = np.loadtxt(self.raster_file)
        else:
            _, file_ext = os.path.splitext(file_path)
            if file_ext == '.h5':
                with WindX(file_path) as res:
                    raster_index = \
                        res.get_raster_index(target, shape,
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
            if self.raster_file is not None:
                np.savetxt(self.raster_file, raster_index)
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
                             len(features)),
                            dtype=np.float32)

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
                                    raster_index.shape[1], 2),
                                   dtype=np.float32)
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

    def __init__(self, multi_data_handler, batch_size=8, val_split=0.2,
                 spatial_res=2, norm=True):
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

        self.multi_data_handler = multi_data_handler
        self.val_split = val_split
        self.training_indices, self.val_indices = self._split_data()
        self.batch_size = batch_size
        self.spatial_res = spatial_res
        self.batch_indices = self._get_batch_indices()
        self.max = len(self.batch_indices)
        self._i = 0
        self.low_res = None
        self.high_res = None
        self.data_handler = None
        self.means = np.zeros((multi_data_handler.shape[3]), dtype=np.float32)
        self.stds = np.zeros((multi_data_handler.shape[3]), dtype=np.float32)

        if norm:
            self.means, self.stds = self._get_stats()
            self._normalize_data_all(self.means, self.stds)

    def data(self, time_steps):
        """Returns MultiDataHandler data method

        Parameters
        ----------
        time_steps : list
            list of integer time indices

        Returns
        -------
        data : np.ndarray
            4D data array with time dimension size
            equal to len(time_steps)
            (spatial_1, spatial_2, temporal, features)
        """
        return self.multi_data_handler.data(time_steps)

    @property
    def stacked_data(self):
        """MultiDataHandler stacked_data method

        Returns
        -------
        data : np.ndarray
            4D data array with time dimension concatenated
            from all data handler data
            (spatial_1, spatial_2, temporal, features)
        """
        return self.multi_data_handler.stacked_data

    def __len__(self):
        """Length method

        Returns
        -------
        n_batches : int
            Number of batches in handler instance
        """
        return len(self.batch_indices)

    def _normalize_data_all(self, means, stds):
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

        for i in range(self.multi_data_handler.shape[3]):
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

        Returns
        -------
        data : np.ndarray
            normalized data array
        """

        for d in self.multi_data_handler:
            d.data[:, :, :, feature_index] = \
                (d.data[:, :, :, feature_index] - mean) / std

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

        for i in range(self.multi_data_handler.shape[3]):
            self.means[i] = np.mean(self.stacked_data[:, :, :, i])
            self.stds[i] = np.std(self.stacked_data[:, :, :, i])
        return self.means, self.stds

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
        training_indices : np.ndarray
            array of indices for training data slice
        val_indices : np.ndarray
            array of indices for validation data slice
        """

        n_observations = self.multi_data_handler.shape[2]
        all_indices = np.arange(n_observations)
        shuffled = all_indices.copy()
        np.random.shuffle(shuffled)
        n_val_obs = int(self.val_split * n_observations)
        self.val_indices = shuffled[:n_val_obs]
        self.training_indices = shuffled[n_val_obs:]
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
        low_res, high_res = \
            self._reshape_data(self.data(self.val_indices))
        batch = Batch(low_res, high_res)
        return batch

    @classmethod
    def make(cls, file_paths, targets,
             shape, features, val_split=0.2,
             batch_size=8, spatial_res=3,
             max_delta=20, norm=True):
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
        multi_data_handler = MultiDataHandler(file_paths, targets,
                                              shape, features,
                                              max_delta=max_delta)
        batch_handler = SpatialBatchHandler(multi_data_handler,
                                            batch_size=batch_size,
                                            val_split=val_split,
                                            spatial_res=spatial_res,
                                            norm=norm)
        batch_handler.multi_data_handler = multi_data_handler
        return batch_handler

    def _get_batch_indices(self):
        """Get set of indices for data batches

        Returns
        -------
        batch_indices : np.ndarray
            array of indices for data batches
        """

        n_batches = int(np.ceil(len(self.training_indices) / self.batch_size))
        self.batch_indices = np.array_split(self.training_indices, n_batches)

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
                self.data(self.batch_indices[self._i])
            self.low_res, self.high_res = \
                self._reshape_data(self.high_res)
            batch = Batch(self.low_res, self.high_res)
            self._i += 1
            return batch
        else:
            raise StopIteration
