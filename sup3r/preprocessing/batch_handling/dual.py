"""Batch handling classes for dual data handlers"""
import logging
import threading
import time

import numpy as np
import tensorflow as tf

from sup3r.preprocessing.batch_handling.base import (
    Batch,
    BatchBuilder,
    BatchHandler,
    ValidationData,
)
from sup3r.preprocessing.mixin import FeatureSets, MultiHandlerStats
from sup3r.utilities.utilities import uniform_box_sampler, uniform_time_sampler

logger = logging.getLogger(__name__)


class DualValidationData(ValidationData):
    """Iterator for validation data for training with dual data handler"""

    # Classes to use for handling an individual batch obj.
    BATCH_CLASS = Batch

    def _get_val_indices(self):
        """List of dicts to index each validation data observation across all
        handlers

        Returns
        -------
        val_indices : list[dict]
            List of dicts with handler_index and tuple_index. The tuple index
            is used to get validation data observation with
        data[tuple_index]
        """

        val_indices = []
        for i, h in enumerate(self.data_handlers):
            if h.hr_val_data is not None:
                for _ in range(h.hr_val_data.shape[2]):
                    spatial_slice = uniform_box_sampler(
                        h.lr_val_data.shape, self.lr_sample_shape[:2])
                    temporal_slice = uniform_time_sampler(
                        h.lr_val_data.shape, self.lr_sample_shape[2])
                    lr_index = (*spatial_slice, temporal_slice,
                                np.arange(h.lr_val_data.shape[-1]))
                    hr_index = [slice(s.start * self.s_enhance,
                                      s.stop * self.s_enhance)
                                for s in lr_index[:2]]
                    hr_index += [slice(s.start * self.t_enhance,
                                       s.stop * self.t_enhance)
                                 for s in lr_index[2:-1]]
                    hr_index.append(lr_index[-1])
                    hr_index = tuple(hr_index)
                    val_indices.append({
                        'handler_index': i,
                        'hr_index': hr_index,
                        'lr_index': lr_index
                    })
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
        for h in self.data_handlers:
            time_steps += h.hr_val_data.shape[2]
        return (self.data_handlers[0].hr_val_data.shape[0],
                self.data_handlers[0].hr_val_data.shape[1], time_steps,
                self.data_handlers[0].hr_val_data.shape[3])

    @property
    def hr_sample_shape(self):
        """Get sample shape for high_res data"""
        return self.data_handlers[0].hr_dh.sample_shape

    @property
    def lr_sample_shape(self):
        """Get sample shape for low_res data"""
        return self.data_handlers[0].lr_dh.sample_shape

    def __next__(self):
        """Get validation data batch

        Returns
        -------
        batch : Batch
            validation data batch with low and high res data each with
            n_observations = batch_size
        """
        self.current_batch_indices = []
        if self._remaining_observations > 0:
            if self._remaining_observations > self.batch_size:
                n_obs = self.batch_size
            else:
                n_obs = self._remaining_observations

            high_res = np.zeros(
                (n_obs, self.hr_sample_shape[0], self.hr_sample_shape[1],
                 self.hr_sample_shape[2], self.data_handlers[0].shape[-1]),
                dtype=np.float32,
            )
            low_res = np.zeros(
                (n_obs, self.lr_sample_shape[0], self.lr_sample_shape[1],
                 self.lr_sample_shape[2], self.data_handlers[0].shape[-1]),
                dtype=np.float32,
            )
            for i in range(high_res.shape[0]):
                val_index = self.val_indices[self._i + i]
                high_res[i, ...] = self.data_handlers[val_index[
                    'handler_index']].hr_val_data[val_index['hr_index']]
                low_res[i, ...] = self.data_handlers[val_index[
                    'handler_index']].lr_val_data[val_index['lr_index']]
                self._remaining_observations -= 1
                self.current_batch_indices.append(val_index)

            # This checks if there is only a single timestep. If so this means
            # we are using a spatial batch handler which uses 4D batches.
            if self.sample_shape[2] == 1:
                high_res = high_res[..., 0, :]
                low_res = low_res[..., 0, :]

            high_res = high_res[..., self.hr_features_ind]
            batch = self.BATCH_CLASS(low_res=low_res, high_res=high_res)
            self._i += 1
            return batch
        else:
            raise StopIteration


class DualBatchHandler(BatchHandler, FeatureSets):
    """Batch handling class for dual data handlers"""

    BATCH_CLASS = Batch
    VAL_CLASS = DualValidationData

    @property
    def lr_sample_shape(self):
        """Spatiotemporal shape of low res samples. (lats, lons, time)"""
        return self.data_handlers[0].lr_dh.sample_shape

    @property
    def hr_sample_shape(self):
        """Spatiotemporal shape of high res samples. (lats, lons, time)"""
        return self.data_handlers[0].hr_dh.sample_shape

    def __next__(self):
        """Get the next batch of observations.

        Returns
        -------
        batch : Batch
            Batch object with batch.low_res and batch.high_res attributes
            with the appropriate subsampling of interpolated ERA.
        """
        self.current_batch_indices = []
        if self._i < self.n_batches:
            handler = self.get_rand_handler()
            hr_list = []
            lr_list = []
            for _ in range(self.batch_size):
                lr_sample, hr_sample = handler.get_next()
                lr_list.append(tf.expand_dims(lr_sample, axis=0))
                hr_list.append(tf.expand_dims(hr_sample, axis=0))

            batch = self.BATCH_CLASS(
                low_res=tf.concat(lr_list, axis=0),
                high_res=tf.concat(hr_list, axis=0))

            self._i += 1
            return batch
        else:
            raise StopIteration


class LazyDualBatchHandler(MultiHandlerStats, FeatureSets):
    """Dual batch handler which uses lazy data handlers to load data as
    needed rather than all in memory at once.

    NOTE: This can be initialized from data extracted and written to netcdf
    from "non-lazy" data handlers.

    Example
    -------
    >>> for lr_handler, hr_handler in zip(lr_handlers, hr_handlers):
    >>>     dh = DualDataHandler(lr_handler, hr_handler)
    >>>     dh.to_netcdf(lr_file, hr_file)
    >>> lazy_dual_handlers = []
    >>> for lr_file, hr_file in zip(lr_files, hr_files):
    >>>     lazy_lr = LazyDataHandler(lr_file, lr_features, lr_sample_shape)
    >>>     lazy_hr = LazyDataHandler(hr_file, hr_features, hr_sample_shape)
    >>>     lazy_dual_handlers.append(LazyDualDataHandler(lazy_lr, lazy_hr))
    >>> lazy_batch_handler = LazyDualBatchHandler(lazy_dual_handlers)
    """

    BATCH_CLASS = Batch
    VAL_CLASS = DualValidationData

    def __init__(self, data_handlers, means_file, stdevs_file,
                 batch_size=32, n_batches=100, max_workers=None):
        self.data_handlers = data_handlers
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.queue_capacity = n_batches
        lr_shape = (
            self.batch_size, *self.lr_sample_shape, len(self.lr_features))
        hr_shape = (
            self.batch_size, *self.hr_sample_shape, len(self.hr_features))
        self.queue = tf.queue.FIFOQueue(self.queue_capacity,
                                        dtypes=[tf.float32, tf.float32],
                                        shapes=[lr_shape, hr_shape])
        self.val_data = []
        self._batch_counter = 0
        self._queue = None
        self._is_training = False
        self._enqueue_thread = None
        self.batch_pool = BatchBuilder(data_handlers,
                                       batch_size=batch_size,
                                       buffer_size=(n_batches * batch_size),
                                       max_workers=max_workers)
        MultiHandlerStats.__init__(
            self, data_handlers, means_file=means_file,
            stdevs_file=stdevs_file)
        FeatureSets.__init__(self, data_handlers)
        logger.info(f'Initialized {self.__class__.__name__} with '
                    f'{len(self.data_handlers)} data_handlers, '
                    f'means_file = {means_file}, stdevs_file = {stdevs_file}, '
                    f'batch_size = {batch_size}, n_batches = {n_batches}, '
                    f'max_workers = {max_workers}.')

    @property
    def lr_sample_shape(self):
        """Spatiotemporal shape of low res samples. (lats, lons, time)"""
        return self.data_handlers[0].lr_dh.sample_shape

    @property
    def hr_sample_shape(self):
        """Spatiotemporal shape of high res samples. (lats, lons, time)"""
        return self.data_handlers[0].hr_dh.sample_shape

    @property
    def s_enhance(self):
        """Get spatial enhancement factor of first (and all) data handlers."""
        return self.data_handlers[0].s_enhance

    @property
    def t_enhance(self):
        """Get temporal enhancement factor of first (and all) data handlers."""
        return self.data_handlers[0].t_enhance

    def start(self):
        """Start thread to keep sample queue full for batches."""
        logger.info(
            f'Running {self.__class__.__name__}.enqueue_thread.start()')
        self._is_training = True
        self._enqueue_thread = threading.Thread(target=self.enqueue_batches)
        self._enqueue_thread.start()

    def join(self):
        """Join thread to exit gracefully."""
        logger.info(
            f'Running {self.__class__.__name__}.enqueue_thread.join()')
        self._enqueue_thread.join()

    def stop(self):
        """Stop loading batches."""
        self._is_training = False
        self.join()

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self._batch_counter = 0
        return self

    def enqueue_batches(self):
        """Callback function for enqueue thread."""
        while self._is_training:
            queue_size = self.queue.size().numpy()
            if queue_size < self.queue_capacity:
                logger.info(f'{queue_size} batches in queue.')
                self.queue.enqueue(next(self.batch_pool))

    def __next__(self):
        """Get the next batch of observations.

        Returns
        -------
        batch : Batch
            Batch object with batch.low_res and batch.high_res attributes
            with the appropriate subsampling of interpolated ERA.
        """
        if self._batch_counter < self.n_batches:
            logger.info(f'Getting next batch: {self._batch_counter + 1} / '
                        f'{self.n_batches}')
            start = time.time()
            lr, hr = self.queue.dequeue()
            batch = self.BATCH_CLASS(low_res=lr, high_res=hr)
            logger.info(f'Built batch in {time.time() - start}.')
            self._batch_counter += 1
        else:
            raise StopIteration

        return batch


class SpatialDualBatchHandler(DualBatchHandler):
    """Batch handling class for h5 data as high res (usually WTK) and ERA5 as
    low res"""

    BATCH_CLASS = Batch
    VAL_CLASS = DualValidationData

    def __next__(self):
        """Get the next iterator output.

        Returns
        -------
        batch : Batch
            Batch object with batch.low_res and batch.high_res attributes
            with the appropriate subsampling of interpolated ERA.
        """
        self.current_batch_indices = []
        if self._i < self.n_batches:
            handler = self.get_rand_handler()
            hr_list = []
            lr_list = []
            for i in range(self.batch_size):
                logger.debug(f'Making batch, observation: {i + 1} / '
                             f'{self.batch_size}.')
                hr_sample, lr_sample = handler.get_next()
                hr_list.append(np.expand_dims(hr_sample[..., 0, :], axis=0))
                lr_list.append(np.expand_dims(lr_sample[..., 0, :], axis=0))
                self.current_batch_indices.append(handler.current_obs_index)

            batch = self.BATCH_CLASS(
                low_res=np.concatenate(lr_list, axis=0, dtype=np.float32),
                high_res=np.concatenate(hr_list, axis=0, dtype=np.float32))

            self._i += 1
            return batch
        else:
            raise StopIteration
