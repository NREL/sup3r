"""Batch handling classes for queued batch loads"""
import logging
import threading

import numpy as np
import tensorflow as tf
import xarray as xr
from rex import safe_json_load

from sup3r.preprocessing.data_handling import DualDataHandler
from sup3r.preprocessing.data_handling.base import DataHandler
from sup3r.preprocessing.dual_batch_handling import DualBatchHandler
from sup3r.utilities.utilities import (
    Timer,
)

logger = logging.getLogger(__name__)


class LazyDataHandler(DataHandler):
    """Lazy loading data handler. Uses precomputed netcdf files (usually from
    a DataHandler.to_netcdf() call after populating DataHandler.data) to create
    batches on the fly during training without previously loading to memory."""

    def __init__(
        self, files, features, sample_shape, lr_only_features=(),
        hr_exo_features=(), chunk_kwargs=None, mode='lazy'
    ):
        self.features = features
        self.sample_shape = sample_shape
        self._lr_only_features = lr_only_features
        self._hr_exo_features = hr_exo_features
        self.chunk_kwargs = (
            chunk_kwargs if chunk_kwargs is not None
            else {'south_north': 10, 'west_east': 10, 'time': 3})
        self.data = xr.open_mfdataset(files, chunks=chunk_kwargs)
        self._shape = (*self.data["latitude"].shape, len(self.data["time"]))
        self.mode = mode
        if mode == 'eager':
            logger.info(f'Loading {files} in eager mode.')
            self.data = self.data.compute()

        logger.info(f'Initialized {self.__class__.__name__} with '
                    f'files = {files}, features = {features}, '
                    f'sample_shape = {sample_shape}.')

    def _get_observation(self, obs_index):
        out = self.data[self.features].isel(
            south_north=obs_index[0],
            west_east=obs_index[1],
            time=obs_index[2],
        )
        if self.mode == 'lazy':
            out = out.compute()

        out = out.to_dataarray().values
        out = np.transpose(out, axes=(2, 3, 1, 0))
        #out = tf.convert_to_tensor(out)
        return out

    def get_next(self):
        """Get next observation sample."""
        obs_index = self.get_observation_index(self.shape, self.sample_shape)
        return self._get_observation(obs_index)


class LazyDualDataHandler(DualDataHandler):
    """Lazy loading dual data handler. Matches sample regions for low res and
    high res lazy data handlers."""

    def __init__(self, lr_dh, hr_dh, s_enhance=1, t_enhance=1):
        self.lr_dh = lr_dh
        self.hr_dh = hr_dh
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        self._means = None
        self._stds = None
        self.check_shapes()

        logger.info(f'Finished initializing {self.__class__.__name__}.')

    @property
    def means(self):
        """Get dictionary of means for all features available in low-res and
        high-res handlers."""
        if self._means is None:
            lr_features = self.lr_dh.features
            hr_only_features = [f for f in self.hr_dh.features
                                if f not in lr_features]
            self._means = dict(zip(
                lr_features,
                self.lr_dh.data[lr_features].mean(axis=0)))
            hr_means = dict(zip(
                hr_only_features,
                self.hr_dh.data[hr_only_features].mean(axis=0)))
            self._means.update(hr_means)
        return self._means

    @property
    def stds(self):
        """Get dictionary of standard deviations for all features available in
        low-res and high-res handlers."""
        if self._stds is None:
            lr_features = self.lr_dh.features
            hr_only_features = [f for f in self.hr_dh.features
                                if f not in lr_features]
            self._stds = dict(zip(lr_features,
                              self.lr_dh.data[lr_features].std(axis=0)))
            hr_stds = dict(zip(hr_only_features,
                               self.hr_dh.data[hr_only_features].std(axis=0)))
            self._stds.update(hr_stds)
        return self._stds

    @property
    def size(self):
        """'Size' of data handler. Used to compute handler weights for batch
        sampling."""
        return np.prod(self.lr_dh.shape)

    def check_shapes(self):
        """Make sure data handler shapes are compatible with enhancement
        factors."""
        hr_shape = self.hr_dh.shape
        lr_shape = self.lr_dh.shape
        enhanced_shape = (lr_shape[0] * self.s_enhance,
                          lr_shape[1] * self.s_enhance,
                          lr_shape[2] * self.t_enhance)
        msg = (f'hr_dh.shape {hr_shape} and enhanced lr_dh.shape '
               f'{enhanced_shape} are not compatible')
        assert hr_shape == enhanced_shape, msg

    def get_next(self):
        """Get next pair of low-res / high-res samples ensuring that low-res
        and high-res sampling regions match.

        Returns
        -------
        tuple
            (low_res, high_res) pair
        """
        lr_obs_idx = self.lr_dh.get_observation_index(self.lr_dh.shape,
                                                      self.lr_dh.sample_shape)
        lr_obs_idx = lr_obs_idx[:-1]
        hr_obs_idx = [slice(s.start * self.s_enhance, s.stop * self.s_enhance)
                      for s in lr_obs_idx[:2]]
        hr_obs_idx += [slice(s.start * self.t_enhance, s.stop * self.t_enhance)
                       for s in lr_obs_idx[2:]]
        out = (self.lr_dh._get_observation(lr_obs_idx),
               self.hr_dh._get_observation(hr_obs_idx))
        return out


class BatchBuilder:
    """Class to create dataset generator and build batches using samples from
    multiple DataHandler instances. The main requirement for the DataHandler
    instances is that they have a get_next() method which returns a tuple
    (low_res, high_res) of arrays."""

    def __init__(self, data_handlers, batch_size, buffer_size=None,
                 max_workers=None):
        self.data_handlers = data_handlers
        self.batch_size = batch_size
        self.buffer_size = buffer_size or 10 * batch_size
        self.handler_index = self.get_handler_index()
        self.max_workers = max_workers or batch_size
        self.sample_counter = 0
        self.batches = None
        self.prefetch()

    @property
    def handler_weights(self):
        """Get weights used to sample from different data handlers based on
        relative sizes"""
        sizes = [dh.size for dh in self.data_handlers]
        weights = sizes / np.sum(sizes)
        weights = weights.astype(np.float32)
        return weights

    def get_handler_index(self):
        """Get random handler index based on handler weights"""
        indices = np.arange(0, len(self.data_handlers))
        return np.random.choice(indices, p=self.handler_weights)

    def get_rand_handler(self):
        """Get random handler based on handler weights"""
        if self.sample_counter % self.batch_size == 0:
            self.handler_index = self.get_handler_index()
        return self.data_handlers[self.handler_index]

    @property
    def data(self):
        """Return tensorflow dataset generator."""
        lr_sample_shape = self.data_handlers[0].lr_sample_shape
        hr_sample_shape = self.data_handlers[0].hr_sample_shape
        lr_features = self.data_handlers[0].lr_features
        hr_features = (self.data_handlers[0].hr_out_features
                       + self.data_handlers[0].hr_exo_features)
        lr_shape = (*lr_sample_shape, len(lr_features))
        hr_shape = (*hr_sample_shape, len(hr_features))
        data = tf.data.Dataset.from_generator(
            self.gen,
            output_signature=(tf.TensorSpec(lr_shape, tf.float32,
                                            name='low_resolution'),
                              tf.TensorSpec(hr_shape, tf.float32,
                                            name='high_resolution')))
        data = data.map(lambda x,y : (x,y),
                        num_parallel_calls=self.max_workers)
        return data

    def __next__(self):
        if self.sample_counter % self.buffer_size == 0:
            self.prefetch()
        return next(self.batches)

    def __getitem__(self, index):
        """Get single sample. Batches are built from self.batch_size
        samples."""
        return self.get_rand_handler().get_next()

    def gen(self):
        """Generator method to enable Dataset.from_generator() call."""
        while True:
            idx = self.sample_counter
            self.sample_counter += 1
            yield self[idx]

    def prefetch(self):
        """Prefetch set of batches for an epoch."""
        data = self.data.prefetch(buffer_size=self.buffer_size)
        self.batches = iter(data.batch(self.batch_size))


class LazyDualBatchHandler(DualBatchHandler):
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

    def __init__(self, data_handlers, means_file=None, stdevs_file=None,
                 batch_size=32, n_batches=100, queue_size=100,
                 max_workers=None):
        self.data_handlers = data_handlers
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.queue_capacity = queue_size
        self._means = (None if means_file is None
                       else safe_json_load(means_file))
        self._stds = (None if stdevs_file is None
                      else safe_json_load(stdevs_file))
        self._i = 0
        self.val_data = []
        self.timer = Timer()
        self._queue = None
        self.enqueue_thread = threading.Thread(target=self.callback)
        self.batch_pool = BatchBuilder(data_handlers,
                                       batch_size=batch_size,
                                       max_workers=max_workers)
        logger.info(f'Initialized {self.__class__.__name__} with '
                    f'{len(self.data_handlers)} data_handlers, '
                    f'means_file = {means_file}, stdevs_file = {stdevs_file}, '
                    f'batch_size = {batch_size}, max_workers = {max_workers}.')

    @property
    def s_enhance(self):
        """Get spatial enhancement factor of first (and all) data handlers."""
        return self.data_handlers[0].s_enhance

    @property
    def t_enhance(self):
        """Get temporal enhancement factor of first (and all) data handlers."""
        return self.data_handlers[0].t_enhance

    @property
    def means(self):
        """Dictionary of means for each feature, computed across all data
        handlers."""
        if self._means is None:
            self._means = {}
            for k in self.data_handlers[0].features:
                self._means[k] = np.sum(
                    [dh.means[k] * wgt for (wgt, dh)
                     in zip(self.handler_weights, self.data_handlers)])
        return self._means

    @property
    def stds(self):
        """Dictionary of standard deviations for each feature, computed across
        all data handlers."""
        if self._stds is None:
            self._stds = {}
            for k in self.data_handlers[0].features:
                self._stds[k] = np.sqrt(np.sum(
                    [dh.stds[k]**2 * wgt for (wgt, dh)
                     in zip(self.handler_weights, self.data_handlers)]))
        return self._stds

    def start(self):
        """Start thread to keep sample queue full for batches."""
        self._is_training = True
        logger.info(
            f'Running {self.__class__.__name__}.enqueue_thread.start()')
        self.enqueue_thread.start()

    def join(self):
        """Join thread to exit gracefully."""
        logger.info(
            f'Running {self.__class__.__name__}.enqueue_thread.join()')
        self.enqueue_thread.join()

    def stop(self):
        """Stop loading batches."""
        self._is_training = False
        self.join()

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self.batch_counter = 0
        return self

    @property
    def queue(self):
        """Queue of (lr, hr) batches."""
        if self._queue is None:
            lr_shape = (
                self.batch_size, *self.lr_sample_shape, len(self.lr_features))
            hr_shape = (
                self.batch_size, *self.hr_sample_shape, len(self.hr_features))
            self._queue = tf.queue.FIFOQueue(
                self.queue_capacity,
                dtypes=[tf.float32, tf.float32],
                shapes=[lr_shape, hr_shape])
        return self._queue

    @property
    def queue_size(self):
        """Get number of batches in queue."""
        return self.queue.size().numpy()

    def callback(self):
        """Callback function for enqueue thread."""
        while self._is_training:
            while self.queue_size < self.queue_capacity:
                logger.info(f'{self.queue_size} batches in queue.')
                self.queue.enqueue(next(self.batch_pool))

    @property
    def is_empty(self):
        """Check if queue is empty."""
        return self.queue_size == 0

    def take_batch(self):
        """Take batch from queue."""
        if self.is_empty:
            return next(self.batch_pool)
        else:
            return self.queue.dequeue()

    def get_next_batch(self):
        """Take batch from queue and build batch class."""
        lr, hr = self.take_batch()
        batch = self.BATCH_CLASS(low_res=lr, high_res=hr)
        return batch


    def __next__(self):
        """Get the next batch of observations.

        Returns
        -------
        batch : Batch
            Batch object with batch.low_res and batch.high_res attributes
            with the appropriate subsampling of interpolated ERA.
        """
        if self.batch_counter < self.n_batches:
            logger.info(f'Getting next batch: {self.batch_counter + 1} / '
                        f'{self.n_batches}')
            batch = self.timer(self.get_next_batch)
            logger.info(
                f'Built batch in {self.timer.log["elapsed:get_next_batch"]}')
            self.batch_counter += 1
        else:
            raise StopIteration

        return batch


class TrainingSession:
    """Simple wrapper around batch handler and model to enable threads for
    batching and training separately."""

    def __init__(self, batch_handler, model, kwargs):
        self.model = model
        self.batch_handler = batch_handler
        self.kwargs = kwargs
        self.train_thread = threading.Thread(
            target=model.train, args=(batch_handler,), kwargs=kwargs)

        self.batch_handler.start()
        self.train_thread.start()

        self.train_thread.join()
        self.batch_handler.stop()
