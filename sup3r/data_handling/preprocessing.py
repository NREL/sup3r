# -*- coding: utf-8 -*-
"""
Sup3r preprocessing module.
"""
import xarray as xr
import numpy as np
import os

from rex import WindX
from rex.utilities import log_mem, loggers
from reV.utilities.exceptions import ConfigError
from sup3r.utilities import utilities
from sup3r import __version__


logger = loggers.init_logger(__name__)


class DataHandler:
    """Sup3r data handling and extraction"""

    def __init__(self, file_path, target,
                 shape, features, max_delta=20,
                 raster_file=None, val_split=0.1,
                 spatial_sample_shape=(10, 10),
                 time_step=1):

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
        raster_file : str | None
            File for raster_index array for the corresponding target and
            shape. If specified the raster_index will be loaded from the file
            if it exists or written to the file if it does not yet exist.
            If None raster_index will be calculated directly.
        val_split : float32
            Fraction of data to store for validation
        spatial_sample_shape : tuple
            size of spatial slices used for spatial batching
        time_step : int
            Number of timesteps to downsample. If time_step=1 no time
            steps will be skipped.
        """

        self.file_path = file_path
        self.features = features
        self.grid_shape = shape
        self.target = target
        self.raster_index = None
        self.max_delta = max_delta
        self.raster_file = raster_file
        self.val_split = val_split
        self.spatial_sample_shape = spatial_sample_shape
        self.time_step = time_step
        self.data, self.lat_lon = self.extract_data()
        self.data, self.val_data = self._split_data()
        self._i = 0

        log_mem(logger, log_level='INFO')

    @property
    def random_time_index(self):
        """Array of time indices shuffled

        Returns
        -------
        np.ndarray
        """
        time_indices = np.arange(self.data.shape[2])
        np.random.shuffle(time_indices)
        return time_indices

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

    def get_observation_index(self):
        """Randomly gets spatial sample and time index

        Returns
        -------
        observation_index : tuple
            Tuple of sampled spatial grid, time_index,
            and features indices. Used to get single observation
            like self.data[observation_index]
        """
        spatial_slice = utilities.uniform_box_sampler(
            self.data, self.spatial_sample_shape)
        temporal_step = self.random_time_index[self._i]
        return tuple(
            spatial_slice + [temporal_step] + [np.arange(len(self.features))])

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        """Gets data for observation using
        random observation index

        Returns
        -------
        observation : np.ndarray
            3D array (spatial_1, spatial_2, features)
        """
        observation = self.data[self.get_observation_index()]
        self._i += 1
        if self._i >= len(self.random_time_index):
            self._i = 0
        return observation

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
        shuffled = all_indices.copy()
        np.random.shuffle(shuffled)
        n_val_obs = int(self.val_split * n_observations)
        val_indices = shuffled[:n_val_obs]
        training_indices = shuffled[n_val_obs:]
        self.val_data = self.data[:, :, val_indices, :]
        self.data = self.data[:, :, training_indices, :]
        return self.data, self.val_data

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

        self.val_data[:, :, :, feature_index] = \
            (self.val_data[:, :, :, feature_index] - mean) / std

        self.data[:, :, :, feature_index] = \
            (self.data[:, :, :, feature_index] - mean) / std

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

        y = utilities.transform_rotate_wind(
            y[:, :, ::self.time_step, :], lat_lon, self.features)

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
            raster_index = np.loadtxt(self.raster_file).astype(np.uint32)
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

    def __init__(self, data_handlers, batch_size=8,
                 spatial_res=3, norm=True,
                 spatial_sample_shape=(10, 10),
                 means=None, stds=None,
                 n_batches=10):
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
        spatial_sample_shape : tuple
            Shape of spatial sample to extract from full spatial domain
        """

        self.data_handlers = data_handlers
        self._i = 0
        self.low_res = None
        self.high_res = None
        self.data_handler = None
        self.batch_size = batch_size
        self._val_data = None
        self.spatial_res = spatial_res
        self.spatial_sample_shape = spatial_sample_shape
        self.means = np.zeros((self.shape[-1]))
        self.stds = np.zeros((self.shape[-1]))
        self.n_batches = n_batches

        if norm:
            self.normalize(means, stds)

        self._val_data = self.val_data

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

        low_res = high_res.reshape(-1, self.spatial_res,
                                   high_res.shape[1] // self.spatial_res,
                                   self.spatial_res,
                                   high_res.shape[2],
                                   high_res.shape[3]).sum((1, 3)) \
            / (self.spatial_res * self.spatial_res)
        low_res = low_res.transpose((2, 0, 1, 3))
        high_res = high_res.transpose((2, 0, 1, 3))
        return low_res, high_res

    @property
    def val_data(self):
        """Validation data property

        Returns
        -------
        batch : Batch
            validation data batch. includes
            batch.low_res and batch.high_res
        """
        if self._val_data is None:
            high_res = np.concatenate(
                [d.val_data for d in self.data_handlers],
                axis=2)
            low_res, high_res = \
                self._reshape_data(high_res)
            self._val_data = Batch(low_res, high_res)
        return self._val_data

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

    @classmethod
    def make(cls, file_paths, targets,
             shape, features, val_split=0.2,
             batch_size=8,
             spatial_sample_shape=(10, 10),
             spatial_res=3, max_delta=20,
             norm=True, raster_files=None,
             time_step=1, means=None,
             n_batches=10,
             stds=None):

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
            targets. If None raster_index will be calculated directly.
        norm : bool
            Wether to normalize data using means/stds calulcated across
            all handlers
        time_step : int
            Number of timesteps to downsample. If time_step=1 no time
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

        Returns
        -------
        batchHandler : SpatialBatchHandler
            batchHandler with dataHandler attribute
        """
        data_handlers = []
        for i, f in enumerate(file_paths):
            if raster_files is None:
                raster_file = None
            else:
                raster_file = raster_files[i]
            if not isinstance(targets, list):
                target = targets
            else:
                target = targets[i]
            data_handlers.append(
                DataHandler(f, target, shape, features,
                            max_delta=max_delta,
                            raster_file=raster_file,
                            val_split=val_split,
                            spatial_sample_shape=spatial_sample_shape,
                            time_step=time_step))
        batch_handler = SpatialBatchHandler(
            data_handlers, spatial_res=spatial_res,
            spatial_sample_shape=spatial_sample_shape,
            batch_size=batch_size, norm=norm, means=means,
            stds=stds, n_batches=n_batches)
        return batch_handler

    def normalize(self, means=None, stds=None):
        """Compute means and stds for each feature
        across all datasets and normalize each data handler dataset
        """

        if means is None or stds is None:
            self._get_stats()
        if means is not None:
            self.means = means
        if stds is not None:
            self.stds = stds
        for d in self.data_handlers:
            d.normalize(self.means, self.stds)

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i <= self.n_batches:
            handler_index = int(np.random.uniform(
                0, len(self.data_handlers)))
            handler = self.data_handlers[handler_index]
            high_res = np.zeros((self.batch_size,
                                 self.spatial_sample_shape[0],
                                 self.spatial_sample_shape[1],
                                 self.shape[-1]))
            for i in range(self.batch_size):
                high_res[i, :, :, :] = handler.__next__()
            low_res = \
                utilities.spatial_coarsening(high_res,
                                             self.spatial_res)
            batch = Batch(low_res, high_res)
            self._i += 1
            return batch
        else:
            raise StopIteration
