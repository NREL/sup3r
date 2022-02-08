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
                 features, max_delta=20, raster_files=None,
                 n_temporal_slices=8, n_spatial_slices=5,
                 spatial_sample_shape=(10, 10),
                 val_split=0.1, time_step=5):
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
        val_split : float32
            Fraction of data to store for validation
        n_temporal_slices : int
            number of temporal slices to use for batching
            across time dimension
        n_spatial_slices : int
            number of spatial slices to use for batching
            across spatial dimensions
        spatial_sample_shape : tuple
            size of spatial slices used for spatial batching
        spatial_res: int
            factor by which to coarsen spatial dimensions
        time_step : int
            Number of timesteps to downsample. If time_step=1 no time
            steps will be skipped.
        """

        if not isinstance(targets, list):
            targets = [targets] * len(file_paths)
        if not isinstance(file_paths, list):
            file_paths = [file_paths]
        if not isinstance(raster_files, list):
            raster_files = [raster_files] * len(file_paths)

        data_handlers = []
        for i, f in enumerate(file_paths):
            data_handlers.append(
                DataHandler(f, targets[i], shape, features,
                            max_delta=max_delta,
                            n_temporal_slices=n_temporal_slices,
                            n_spatial_slices=n_spatial_slices,
                            spatial_sample_shape=spatial_sample_shape,
                            val_split=val_split,
                            raster_file=raster_files[i],
                            time_step=time_step))
        self.data_handlers = data_handlers
        self.current_handler = None
        self.max = len(data_handlers)
        self.grid_shape = shape
        self.features = features
        self._i = 0
        self.means = np.zeros((self.shape[-1]), dtype=np.float32)
        self.stds = np.zeros((self.shape[-1]), dtype=np.float32)
        self.batch_indices = self.get_batch_indices()

    def get_batch_indices(self):
        """Aggregate batch indices from each data handler
        into single array of all batch indices, with an index
        for the data handler the batch came from.

        Returns
        -------
        batch_indices : np.ndarray
            list of batch indices across all data handlers
        """
        self.batch_indices = []
        for i, h in enumerate(self.data_handlers):
            for t in h.time_slices:
                for s in h.spatial_slices:
                    self.batch_indices.append(
                        {'handler_index': i,
                         'batch_indices':
                         tuple(s + [t] + [slice(0, len(self.features) + 1)])})
        np.random.shuffle(self.batch_indices)
        return self.batch_indices

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

        return self.means, self.stds

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


class DataHandler:
    """Sup3r data handling and extraction"""

    def __init__(self, file_path, target,
                 shape, features, max_delta=20,
                 raster_file=None, val_split=0.1,
                 n_temporal_slices=8, n_spatial_slices=5,
                 spatial_sample_shape=(10, 10), time_step=1):

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
        n_temporal_slices : int
            number of temporal slices to use for batching
            across time dimension
        n_spatial_slices : int
            number of spatial slices to use for batching
            across spatial dimensions
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
        self.n_temporal_slices = n_temporal_slices
        self.spatial_sample_shape = spatial_sample_shape
        self.n_spatial_slices = n_spatial_slices
        self.time_step = time_step
        self.data, self.lat_lon = self.extract_data()
        self.data = self.data[slice(None, None, time_step, None)]
        self.data, self.val_data = self._split_data()
        self.temporal_slices = self.get_temporal_slices()
        self.spatial_slices = self.get_spatial_slices()

    def get_spatial_slices(self):
        """Get spatial subsamples from full training domain

        Returns
        -------
        spatial_slices : list[tuples]
            List of n_spatial_slices slices corresponding to
            spatial subsamples of full training domain
        """

        self.spatial_slices = []
        for _ in range(self.n_spatial_slices):
            self.spatial_slices.append(
                utilities.uniform_box_sampler(
                    self.data, self.spatial_sample_shape))
        return self.spatial_slices

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

    def get_temporal_slices(self):
        """Get indices for temporal slices for data
        from this specific DataHandler instance

        Returns
        -------
        temporal_slices : np.ndarray
            Array of index arrays along time dimension
            from this handler's dataset
        """

        all_indices = np.arange(self.data.shape[2])
        self.time_slices = np.array_split(all_indices, self.n_temporal_slices)
        return self.time_slices

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

    def __init__(self, multi_data_handler, spatial_res, norm=True):
        """
        Parameters
        ----------
        multi_data_handler : MultiDataHandler
            Instance of MultiDataHandler. Includes set of
            DataHandler instances
        spatial_res : int
            Factor by which to coarsen spatial dimensions to generate
            low res data
        norm : bool
            Whether to normalize the data or not
        """

        self.multi_data_handler = multi_data_handler
        self._i = 0
        self.low_res = None
        self.high_res = None
        self.data_handler = None
        self.batch_indices = multi_data_handler.batch_indices
        self.max = len(self.batch_indices)
        self._val_data = None
        self.spatial_res = spatial_res

        if norm:
            self.normalize()

        self._val_data = self.val_data

    def __len__(self):
        """Length method

        Returns
        -------
        n_batches : int
            Number of batches in handler instance
        """
        return len(self.batch_indices)

    def normalize(self, means=None, stds=None):
        """Normalize data with calculated means/stds if
        these are passed as None or if not None then with
        arguments

        Parameters
        ----------
        means : list[float32] | None
            means for each feature to use for normalization
            or None
        stds : list[float32] | None
            stds for each feature to use for normalization
            or None
        """
        self.multi_data_handler.normalize(means=means, stds=stds)

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
                [d.val_data for d in self.multi_data_handler.data_handlers],
                axis=2)
            low_res, high_res = \
                self._reshape_data(high_res)
            self._val_data = Batch(low_res, high_res)
        return self._val_data

    @classmethod
    def make(cls, file_paths, targets,
             shape, features, val_split=0.2,
             n_temporal_slices=8, n_spatial_slices=5,
             spatial_sample_shape=(10, 10),
             spatial_res=3, max_delta=20,
             norm=True, raster_files=None,
             time_step=1):
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
        n_temporal_slices : int
            number of temporal slices to use for batching
            across time dimension
        n_spatial_slices : int
            number of spatial slices to use for batching
            across spatial dimensions
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

        Returns
        -------
        batchHandler : SpatialBatchHandler
            batchHandler with dataHandler attribute
        """
        multi_data_handler = MultiDataHandler(
            file_paths, targets, shape, features,
            max_delta=max_delta, val_split=val_split,
            raster_files=raster_files,
            n_temporal_slices=n_temporal_slices,
            n_spatial_slices=n_spatial_slices,
            spatial_sample_shape=spatial_sample_shape,
            time_step=time_step)
        batch_handler = SpatialBatchHandler(multi_data_handler,
                                            spatial_res=spatial_res,
                                            norm=norm)
        batch_handler.multi_data_handler = multi_data_handler
        return batch_handler

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
            batch = self.batch_indices[self._i]
            high_res = \
                self.multi_data_handler.data_handlers[
                    batch['handler_index']].data[batch['batch_indices']]
            low_res, high_res = self._reshape_data(high_res)
            batch = Batch(low_res, high_res)
            self._i += 1
            return batch
        else:
            raise StopIteration
