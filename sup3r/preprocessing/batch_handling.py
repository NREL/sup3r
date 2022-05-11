# -*- coding: utf-8 -*-
"""
Sup3r batch_handling module.

@author: bbenton
"""
import logging
import numpy as np
from datetime import datetime as dt
import threading

from rex.utilities import log_mem

from sup3r.utilities.utilities import (daily_time_sampler, spatial_coarsening,
                                       temporal_coarsening,
                                       uniform_box_sampler,
                                       uniform_time_sampler,
                                       nn_fill_array)
from sup3r.preprocessing.data_handling import get_handler_class
from sup3r import __version__

np.random.seed(42)

logger = logging.getLogger(__name__)


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

    # pylint: disable=W0613
    @classmethod
    def get_coarse_batch(cls, high_res,
                         s_enhance, t_enhance=1,
                         temporal_coarsening_method='subsample',
                         output_features_ind=None,
                         output_features=None):
        """Coarsen high res data and return Batch with
        high res and low res data

        Parameters
        ----------
        high_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        s_enhance : int
            Factor by which to coarsen spatial dimensions of the high
            resolution data
        t_enhance : int
            Factor by which to coarsen temporal dimension of the high
            resolution data
        temporal_coarsening_method : str
            Method to use for temporal coarsening. Can be subsample, average,
            or total
        output_features_ind : list | np.ndarray | None
            List/array of feature channel indices that are used for generative
            output, without any feature indices used only for training.
        output_features : list
            List of Generative model output feature names

        Returns
        -------
        Batch
            Batch instance with low and high res data
        """
        low_res = spatial_coarsening(
            high_res, s_enhance)

        if t_enhance != 1:
            low_res = temporal_coarsening(
                low_res, t_enhance,
                temporal_coarsening_method)

        high_res = cls.reduce_features(high_res, output_features_ind)
        batch = cls(low_res, high_res)

        return batch


class NsrdbBatch(Batch):
    """Special batch handler for NSRDB data"""

    @classmethod
    def get_coarse_batch(cls, high_res,
                         s_enhance, t_enhance=1,
                         temporal_coarsening_method='subsample',
                         output_features_ind=None,
                         output_features=None):
        """Coarsen high res data and return Batch with high res and low res
        data

        Parameters
        ----------
        high_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        s_enhance : int
            Factor by which to coarsen spatial dimensions of the high
            resolution data
        t_enhance : int
            Factor by which to coarsen temporal dimension of the high
            resolution data
        temporal_coarsening_method : str
            Method to use for temporal coarsening. Can be subsample, average,
            or total
        output_features_ind : list | np.ndarray | None
            List/array of feature channel indices that are used for generative
            output, without any feature indices used only for training.
        output_features : list
            List of Generative model output feature names

        Returns
        -------
        Batch
            Batch instance with low and high res data
        """

        # for nsrdb, do temporal avg first so you dont have to do spatial agg
        # across NaNs
        low_res = temporal_coarsening(
            high_res, t_enhance,
            temporal_coarsening_method)

        low_res = spatial_coarsening(
            low_res, s_enhance)

        high_res = cls.reduce_features(high_res, output_features_ind)

        if (output_features is not None
                and 'clearsky_ratio' in output_features):
            cs_ind = output_features.index('clearsky_ratio')
            if np.isnan(high_res[..., cs_ind]).any():
                high_res[..., cs_ind] = nn_fill_array(high_res[..., cs_ind])

        batch = cls(low_res, high_res)

        return batch


class ValidationData:
    """Iterator for validation data"""

    # Classes to use for handling an individual batch obj.
    BATCH_CLASS = Batch

    def __init__(self, data_handlers, batch_size=8, s_enhance=3, t_enhance=1,
                 temporal_coarsening_method='subsample',
                 output_features_ind=None,
                 output_features=None):
        """
        Parameters
        ----------
        handlers : list[DataHandler]
            List of DataHandler instances
        batch_size : int
            Size of validation data batches
        s_enhance : int
            Factor by which to coarsen spatial dimensions of the high
            resolution data
        t_enhance : int
            Factor by which to coarsen temporal dimension of the high
            resolution data
        temporal_coarsening_method : str
            [subsample, average, total]
            Subsample will take every t_enhance-th time step, average will
            average over t_enhance time steps, total will sum over t_enhance
            time steps
        output_features_ind : list | np.ndarray | None
            List/array of feature channel indices that are used for generative
            output, without any feature indices used only for training.
        output_features : list
            List of Generative model output feature names
        """

        handler_shapes = np.array(
            [d.sample_shape for d in data_handlers])
        assert np.all(handler_shapes[0] == handler_shapes)

        self.handlers = data_handlers
        self.sample_shape = handler_shapes[0]
        self.val_indices = self._get_val_indices()
        self.max = np.ceil(
            len(self.val_indices) / (batch_size))
        self.batch_size = batch_size
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        self._remaining_observations = len(self.val_indices)
        self.temporal_coarsening_method = temporal_coarsening_method
        self._i = 0
        self.output_features_ind = output_features_ind
        self.output_features = output_features

    def _get_val_indices(self):
        """List of dicts to index each validation data observation across all
        handlers

        Returns
        -------
        val_indices : list[dict]
            List of dicts with handler_index and tuple_index. The tuple index
            is used to get validation data observation with
            data[tuple_index]"""

        val_indices = []
        for i, h in enumerate(self.handlers):
            for _ in range(h.val_data.shape[2]):
                spatial_slice = uniform_box_sampler(
                    h.val_data, self.sample_shape[:2])
                temporal_slice = uniform_time_sampler(
                    h.val_data, self.sample_shape[2])
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
            With temporal extent equal to the sum across all data handlers time
            dimension
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
            validation data batch with low and high res data each with
            n_observations = batch_size
        """
        if self._remaining_observations > 0:
            if self._remaining_observations > self.batch_size:
                high_res = np.zeros((self.batch_size, self.sample_shape[0],
                                     self.sample_shape[1],
                                     self.sample_shape[2],
                                     self.handlers[0].shape[-1]),
                                    dtype=np.float32)
            else:
                high_res = np.zeros((self._remaining_observations,
                                     self.sample_shape[0],
                                     self.sample_shape[1],
                                     self.sample_shape[2],
                                     self.handlers[0].shape[-1]),
                                    dtype=np.float32)
            for i in range(high_res.shape[0]):
                val_index = self.val_indices[self._i + i]
                high_res[i, ...] = self.handlers[
                    val_index['handler_index']].val_data[
                    val_index['tuple_index']]
                self._remaining_observations -= 1

            if self.sample_shape[2] == 1:
                high_res = high_res[..., 0, :]
            batch = self.BATCH_CLASS.get_coarse_batch(
                high_res, self.s_enhance,
                t_enhance=self.t_enhance,
                temporal_coarsening_method=self.temporal_coarsening_method,
                output_features_ind=self.output_features_ind,
                output_features=self.output_features)
            self._i += 1
            return batch
        else:
            raise StopIteration


class NsrdbValidationData(ValidationData):
    """Iterator for daily NSRDB validation data"""

    # Classes to use for handling an individual batch obj.
    BATCH_CLASS = NsrdbBatch

    def _get_val_indices(self):
        """List of dicts to index each validation data observation across all
        handlers

        Returns
        -------
        val_indices : list[dict]
            List of dicts with handler_index and tuple_index. The tuple index
            is used to get validation data observation with
            data[tuple_index]"""

        val_indices = []
        for i, h in enumerate(self.handlers):
            for _ in range(h.val_data.shape[2]):

                spatial_slice = uniform_box_sampler(h.val_data,
                                                    self.sample_shape[:2])

                temporal_slice = daily_time_sampler(h.val_data,
                                                    self.sample_shape[2],
                                                    h.val_time_index)
                tuple_index = tuple(spatial_slice
                                    + [temporal_slice]
                                    + [np.arange(h.val_data.shape[-1])])

                val_indices.append({'handler_index': i,
                                    'tuple_index': tuple_index})

        return val_indices


class BatchHandler:
    """Sup3r base batch handling class"""

    # Classes to use for handling an individual batch obj.
    VAL_CLASS = ValidationData
    BATCH_CLASS = Batch

    def __init__(self, data_handlers, batch_size=8, s_enhance=3, t_enhance=2,
                 means=None, stds=None, norm=True, n_batches=10,
                 temporal_coarsening_method='subsample'):
        """
        Parameters
        ----------
        data_handlers : list[DataHandler]
            List of DataHandler instances
        batch_size : int
            Number of observations in a batch
        s_enhance : int
            Factor by which to coarsen spatial dimensions of the high
            resolution data to generate low res data
        t_enhance : int
            Factor by which to coarsen temporal dimension of the high
            resolution data to generate low res data
        means : np.ndarray
            dimensions (features)
            array of means for all features with same ordering as data
            features.  If not None and norm is True these will be used for
            normalization
        stds : np.ndarray
            dimensions (features)
            array of means for all features with same ordering as data
            features.  If not None and norm is True these will be used form
            normalization
        norm : bool
            Whether to normalize the data or not
        n_batches : int
            Number of batches in an epoch, this sets the iteration limit for
            this object.
        temporal_coarsening_method : str
            [subsample, average, total]
            Subsample will take every t_enhance-th time step, average will
            average over t_enhance time steps, total will sum over t_enhance
            time steps
        """

        handler_shapes = np.array(
            [d.sample_shape for d in data_handlers])
        assert np.all(handler_shapes[0] == handler_shapes)

        n_feature_arrays = len(data_handlers[0].features) * len(data_handlers)
        n_chunks = int(np.ceil(n_feature_arrays / 10))
        handler_chunks = np.array_split(data_handlers, n_chunks)

        for j, handler_chunk in enumerate(handler_chunks):
            futures = {}
            now = dt.now()
            for i, d in enumerate(handler_chunk):
                if d.data is None:
                    future = threading.Thread(target=d.load_cached_data)
                    futures[future] = i
                    future.start()

            logger.info(
                'Started loading all data handlers'
                f' for handler_chunk {j} in {dt.now() - now}. ')

            for i, future in enumerate(futures.keys()):
                future.join()
                logger.debug(
                    f'{i+1} out of {len(futures)} handlers for handler_chunk '
                    f'{j} loaded.')

        logger.debug('Finished loading data for BatchHandler')
        log_mem(logger)

        self.data_handlers = data_handlers
        self._i = 0
        self.low_res = None
        self.high_res = None
        self.data_handler = None
        self.batch_size = batch_size
        self._val_data = None
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        self.sample_shape = handler_shapes[0]
        self.means = np.zeros((self.shape[-1]), dtype=np.float32)
        self.stds = np.zeros((self.shape[-1]), dtype=np.float32)
        self.n_batches = n_batches
        self.temporal_coarsening_method = temporal_coarsening_method
        self.current_batch_indices = None
        self.current_handler_index = None

        if norm:
            logger.debug('Normalizing data for BatchHandler')
            self.normalize(means, stds)

        logger.debug('Getting validation data for BatchHandler')
        self.val_data = self.VAL_CLASS(
            data_handlers, batch_size=batch_size,
            s_enhance=s_enhance, t_enhance=t_enhance,
            temporal_coarsening_method=temporal_coarsening_method,
            output_features_ind=self.output_features_ind,
            output_features=self.output_features)

        logger.info('Finished initializing BatchHandler')

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
    def chunk_file_paths(file_paths, list_chunk_size=None):
        """Split list of file paths into chunks of size list_chunk_size

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
            n_chunks = int(np.ceil(len(file_paths) / list_chunk_size))
            file_paths = list(np.array_split(file_paths, n_chunks))
            file_paths = [list(fps) for fps in file_paths]
        return file_paths

    @classmethod
    def init_data_handlers(cls, file_paths, features, targets=None,
                           shape=None, val_split=0.2,
                           sample_shape=(10, 10, 10),
                           max_delta=20,
                           raster_files=None,
                           temporal_slice=slice(None),
                           time_roll=0,
                           list_chunk_size=None,
                           max_extract_workers=None,
                           max_compute_workers=None,
                           time_chunk_size=100,
                           cache_file_prefixes=None,
                           overwrite_cache=False):
        """
        Initialize set of data handlers for input to make method

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
        sample_shape : tuple
            size of spatial and temporal domain used for batching
        max_delta : int, optional
            Optional maximum limit on the raster shape that is retrieved at
            once. If shape is (20, 20) and max_delta=10, the full raster will
            be retrieved in four chunks of (10, 10). This helps adapt to
            non-regular grids that curve over large distances, by default 20
        raster_files : list | str | None
            Files for raster_index array for the corresponding targets and
            shape. If a list these can be different files for different
            targets. If a string the same file will be used for all targets. If
            None raster_index will be calculated directly.
        temporal_slice : slice
            Slice specifying extent and step of temporal extraction. e.g.
            slice(start, stop, time_pruning). If equal to slice(None, None, 1)
            the full time dimension is selected.
        time_roll : int
            The number of places by which elements are shifted in the time
            axis. Can be used to convert data to different timezones. This is
            passed to np.roll(a, time_roll, axis=2) and happens AFTER the
            time_pruning operation.
        list_chunk_size : int
            Size of chunks to split file_paths into if a list of files
            is passed. If None no splitting will be performed.
        max_compute_workers : int | None
            max number of workers to use for computing features.
            If max_compute_workers == 1 then extraction will be serialized.
        max_extract_workers : int | None
            max number of workers to use for data extraction.
            If max_extract_workers == 1 then extraction will be serialized.
        time_chunk_size : int
            Size of chunks to split time dimension into for data extraction
        cache_file_prefixes : list | None
            File prefixes for cached feature data. If None then feature data
            will be stored in memory while other features are being
            computed/extracted.
        overwrite_cache : bool
            Whether to overwrite any previously saved cache files.

        Returns
        -------
        list
            List of DataHandler objects used to initialize BatchHandler object
        """

        check = ((targets is not None and shape is not None)
                 or raster_files is not None)
        msg = ('You must either provide the targets+shape inputs '
               'or the raster_files input.')
        assert check, msg

        HandlerClass = get_handler_class(file_paths)
        file_paths = cls.chunk_file_paths(file_paths, list_chunk_size)

        data_handlers = []
        if not isinstance(file_paths, list):
            file_paths = [file_paths]

        for i, f in enumerate(file_paths):
            cache_file_prefix, raster_file, target = cls.make_inputs(
                cache_file_prefixes, raster_files, targets, i)
            data_handlers.append(
                HandlerClass(
                    f, features, target=target,
                    shape=shape, max_delta=max_delta,
                    raster_file=raster_file, val_split=val_split,
                    sample_shape=sample_shape,
                    temporal_slice=temporal_slice,
                    time_roll=time_roll,
                    max_extract_workers=max_extract_workers,
                    max_compute_workers=max_compute_workers,
                    time_chunk_size=time_chunk_size,
                    cache_file_prefix=cache_file_prefix,
                    overwrite_cache=overwrite_cache))
        return data_handlers

    @classmethod
    def make_inputs(cls, cache_file_prefixes, raster_files,
                    targets, handler_index):
        """Sanitize some of the inputs to the make method

        Parameters
        ----------
        cache_file_prefixes : list | None
            File prefixes for cached feature data. If None then feature data
            will be stored in memory while other features are being
            computed/extracted.
        raster_files : list | str | None
            Files for raster_index array for the corresponding targets and
            shape. If a list these can be different files for different
            targets. If a string the same file will be used for all targets. If
            None raster_index will be calculated directly.
        targets : list | tuple
            List of several (lat, lon) lower left corner of raster. Either need
            target+shape or raster_file.
        index : int
            Handler index to select corresponding element of cache_file_paths,
            raster_files, and targets

        Returns
        -------
        cache_file_path : str
            Path to cache file for the handler selected by the handler_index
        raster_file : str
            Path to raster file for the handler selected by the handler_index
        target : tuple
            (lat, lon) for the lower left corner of the raster for the handler
            selected by the handler_index
        """

        if cache_file_prefixes is None or cache_file_prefixes is False:
            cache_file_prefix = None
        else:
            if not isinstance(cache_file_prefixes, list):
                cache_file_prefixes = [cache_file_prefixes]
            cache_file_prefix = cache_file_prefixes[handler_index]
        if raster_files is None:
            raster_file = None
        else:
            if not isinstance(raster_files, list):
                raster_file = raster_files
            else:
                raster_file = raster_files[handler_index]
        if not isinstance(targets, list):
            target = targets
        else:
            target = targets[handler_index]

        return cache_file_prefix, raster_file, target

    @classmethod
    def make(cls, file_paths, features,
             targets=None, shape=None, val_split=0.2,
             sample_shape=(10, 10, 10),
             s_enhance=3, t_enhance=2,
             max_delta=20, norm=True,
             raster_files=None,
             temporal_slice=slice(None),
             time_roll=0,
             batch_size=8, n_batches=10,
             means=None, stds=None,
             temporal_coarsening_method='subsample',
             list_chunk_size=None,
             max_extract_workers=None,
             max_compute_workers=None,
             time_chunk_size=100,
             cache_file_prefixes=None,
             overwrite_cache=False):
        """Method to initialize both data and batch handlers

        Parameters
        ----------
        file_paths : list
            list of file paths
        features : list
            list of features to extract
        targets : tuple
            List of several (lat, lon) lower left corner of raster. Either need
            target+shape or raster_file.
        shape : tuple
            (rows, cols) grid size
        val_split : float32
            fraction of data to reserve for validation
        sample_shape : tuple
            size of spatial and temporal domain used for batching
        s_enhance: int
            factor by which to coarsen spatial dimensions of the high
            resolution data
        t_enhance: int
            factor by which to coarsen temporal dimension of the high
            resolution data
        max_delta : int, optional
            Optional maximum limit on the raster shape that is retrieved at
            once. If shape is (20, 20) and max_delta=10, the full raster will
            be retrieved in four chunks of (10, 10). This helps adapt to
            non-regular grids that curve over large distances, by default 20
        norm : bool
            Wether to normalize data using means/stds calulcated across all
            handlers
        raster_files : list | str | None
            Files for raster_index array for the corresponding targets and
            shape. If a list these can be different files for different
            targets. If a string the same file will be used for all targets. If
            None raster_index will be calculated directly.
        temporal_slice : slice
            Slice specifying extent and step of temporal extraction. e.g.
            slice(start, stop, time_pruning). If equal to slice(None, None, 1)
            the full time dimension is selected.
        time_roll : int
            The number of places by which elements are shifted in the time
            axis. Can be used to convert data to different timezones. This is
            passed to np.roll(a, time_roll, axis=2) and happens AFTER the
            time_pruning operation.
        batch_size : int
            number of observations in a batch
        n_batches : int
            Number of batches to iterate through
        means : np.ndarray
            dimensions (features)
            array of means for all features with same ordering as data features
        stds : np.ndarray
            dimensions (features)
            array of means for all features
            with same ordering as data features
        temporal_coarsening_method : str
            [subsample, average, total]
            Subsample will take every t_enhance-th time step, average will
            average over t_enhance time steps, total will sum over t_enhance
            time steps
        list_chunk_size : int
            Size of chunks to split file_paths into if a list of files is
            passed. If None no splitting will be performed.
        max_compute_workers : int | None
            max number of workers to use for computing features.
            If max_compute_workers == 1 then extraction will be serialized.
        max_extract_workers : int | None
            max number of workers to use for data extraction.
            If max_extract_workers == 1 then extraction will be serialized.
        time_chunk_size : int
            Size of chunks to split time dimension into for data extraction
        cache_file_prefixes : list | None | bool
            File prefixes for cached feature data. If None then feature data
            will be stored in memory while other features are being
            computed/extracted.
        overwrite_cache : bool
            Whether to overwrite any previously saved cache files.

        Returns
        -------
        batchHandler : BatchHandler
            batchHandler with dataHandler attribute
        """

        data_handlers = cls.init_data_handlers(
            file_paths, features,
            targets=targets, shape=shape, val_split=val_split,
            sample_shape=sample_shape,
            max_delta=max_delta,
            raster_files=raster_files,
            temporal_slice=temporal_slice,
            time_roll=time_roll,
            list_chunk_size=list_chunk_size,
            max_extract_workers=max_extract_workers,
            max_compute_workers=max_compute_workers,
            time_chunk_size=time_chunk_size,
            cache_file_prefixes=cache_file_prefixes,
            overwrite_cache=overwrite_cache)

        batch_handler = BatchHandler(
            data_handlers, s_enhance=s_enhance,
            t_enhance=t_enhance, batch_size=batch_size,
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
            With temporal extent equal to the sum across all data handlers time
            dimension
        """
        time_steps = 0
        for h in self.data_handlers:
            time_steps += h.shape[2]
        return (self.data_handlers[0].shape[0], self.data_handlers[0].shape[1],
                time_steps, self.data_handlers[0].shape[3])

    def _get_stats(self):
        """Get standard deviations and means for all data features

        Returns
        -------
        means : np.ndarray
            dimensions (features)
            array of means for all features with same ordering as data features
        stds : np.ndarray
            dimensions (features)
            array of means for all features with same ordering as data features
        """

        for i in range(self.shape[-1]):
            n_elems = 0
            for data_handler in self.data_handlers:
                self.means[i] += np.nansum(data_handler.data[..., i])
                n_elems += np.product(data_handler.shape[:-1])
            self.means[i] = self.means[i] / n_elems
            for data_handler in self.data_handlers:
                self.stds[i] += np.nansum(
                    (data_handler.data[..., i] - self.means[i])**2)
            self.stds[i] = np.sqrt(self.stds[i] / n_elems)

    def normalize(self, means=None, stds=None):
        """Compute means and stds for each feature across all datasets and
        normalize each data handler dataset.  Checks if input means and stds
        are different from stored means and stds and renormalizes if they are
        new """
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
        if self._i < self.n_batches:
            handler_index = np.random.randint(0, len(self.data_handlers))
            self.current_handler_index = handler_index
            handler = self.data_handlers[handler_index]
            high_res = np.zeros((self.batch_size, self.sample_shape[0],
                                 self.sample_shape[1], self.sample_shape[2],
                                 self.shape[-1]), dtype=np.float32)

            for i in range(self.batch_size):
                high_res[i, ...] = handler.get_next()
                self.current_batch_indices.append(handler.current_obs_index)

            batch = self.BATCH_CLASS.get_coarse_batch(
                high_res, self.s_enhance, t_enhance=self.t_enhance,
                temporal_coarsening_method=self.temporal_coarsening_method,
                output_features_ind=self.output_features_ind,
                output_features=self.output_features)

            self._i += 1
            return batch
        else:
            raise StopIteration


class NsrdbBatchHandler(BatchHandler):
    """Sup3r base batch handling class"""

    # Classes to use for handling an individual batch obj.
    VAL_CLASS = NsrdbValidationData
    BATCH_CLASS = NsrdbBatch


class SpatialBatchHandler(BatchHandler):
    """Sup3r spatial batch handling class"""

    @classmethod
    def make(cls, file_paths, features, targets=None, shape=None,
             val_split=0.2, batch_size=8, sample_shape=(10, 10), s_enhance=3,
             max_delta=20, norm=True, raster_files=None,
             temporal_slice=slice(None), time_roll=0, means=None, stds=None,
             n_batches=10, list_chunk_size=None, max_extract_workers=None,
             max_compute_workers=None, time_chunk_size=100,
             cache_file_prefixes=None):
        """Method to initialize both data and batch handlers

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
        sample_shape : tuple
            size of spatial slices used for spatial batching
        s_enhance: int
            factor by which to coarsen spatial dimensions
        max_delta : int, optional
            Optional maximum limit on the raster shape that is retrieved at
            once. If shape is (20, 20) and max_delta=10, the full raster will
            be retrieved in four chunks of (10, 10). This helps adapt to
            non-regular grids that curve over large distances, by default 20
        norm : bool
            Wether to normalize data using means/stds calulcated across
            all handlers
        raster_files : list | str | None
            Files for raster_index array for the corresponding targets and
            shape. If a list these can be different files for different
            targets. If a string the same file will be used for all
            targets. If None raster_index will be calculated directly. Either
            need target+shape or raster_file.
        temporal_slice : slice
            Slice specifying extent and step of temporal extraction. e.g.
            slice(start, stop, time_pruning). If equal to
            slice(None, None, 1) the full time dimension is selected.
        time_roll : int
            The number of places by which elements are shifted in the time
            axis. Can be used to convert data to different timezones. This is
            passed to np.roll(a, time_roll, axis=2) and happens AFTER the
            time_pruning operation.
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
        max_compute_workers : int | None
            max number of workers to use for computing features.
            If max_compute_workers == 1 then extraction will be serialized.
        max_extract_workers : int | None
            max number of workers to use for data extraction.
            If max_extract_workers == 1 then extraction will be serialized.
        time_chunk_size : int
            Size of chunks to split time dimension into for data extraction
        cache_file_prefixes : list | None | bool
            File prefixes for cached feature data. If None then feature data
            will be stored in memory while other features are being
            computed/extracted. If True then features will be cached using
            default file names.

        Returns
        -------
        batchHandler : SpatialBatchHandler
            batchHandler with dataHandler attribute
        """

        data_handlers = cls.init_data_handlers(
            file_paths, features,
            targets=targets, shape=shape, val_split=val_split,
            sample_shape=(sample_shape[0], sample_shape[1], 1),
            max_delta=max_delta,
            raster_files=raster_files,
            temporal_slice=temporal_slice,
            time_roll=time_roll,
            list_chunk_size=list_chunk_size,
            max_extract_workers=max_extract_workers,
            max_compute_workers=max_compute_workers,
            time_chunk_size=time_chunk_size,
            cache_file_prefixes=cache_file_prefixes)

        batch_handler = SpatialBatchHandler(
            data_handlers, s_enhance=s_enhance,
            t_enhance=1, batch_size=batch_size,
            norm=norm, means=means,
            stds=stds, n_batches=n_batches)
        return batch_handler

    def __next__(self):
        if self._i < self.n_batches:
            handler_index = np.random.randint(
                0, len(self.data_handlers))
            handler = self.data_handlers[handler_index]
            high_res = np.zeros((self.batch_size, self.sample_shape[0],
                                 self.sample_shape[1], self.shape[-1]),
                                dtype=np.float32)
            for i in range(self.batch_size):
                high_res[i, ...] = handler.get_next()[..., 0, :]

            batch = self.BATCH_CLASS.get_coarse_batch(
                high_res, self.s_enhance,
                output_features_ind=self.output_features_ind)

            self._i += 1
            return batch
        else:
            raise StopIteration
