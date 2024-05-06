"""Batch handling classes for queued batch loads"""
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tensorflow as tf
import xarray as xr
from rex import safe_json_load
from tqdm import tqdm

from sup3r.preprocessing.data_handling import DualDataHandler
from sup3r.preprocessing.data_handling.base import DataHandler
from sup3r.preprocessing.dual_batch_handling import DualBatchHandler
from sup3r.utilities.utilities import (
    Timer,
    uniform_box_sampler,
    uniform_time_sampler,
)

logger = logging.getLogger(__name__)


class LazyDataHandler(DataHandler):
    """Lazy loading data handler. Uses precomputed netcdf files (usually from
    a DataHandler.to_netcdf() call after populating DataHandler.data) to create
    batches on the fly during training without previously loading to memory."""

    def __init__(
        self, files, features, sample_shape, lr_only_features=(),
        hr_exo_features=(), chunk_kwargs=None
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
        self._i = 0

        logger.info(f'Initialized {self.__class__.__name__} with '
                    f'files = {files}, features = {features}, '
                    f'sample_shape = {sample_shape}.')

    def _get_observation_index(self):
        spatial_slice = uniform_box_sampler(
            self.shape, self.sample_shape[:2]
        )
        temporal_slice = uniform_time_sampler(
            self.shape, self.sample_shape[2]
        )
        return (*spatial_slice, temporal_slice)

    def _get_observation(self, obs_index):
        out = self.data[self.features].isel(
            south_north=obs_index[0],
            west_east=obs_index[1],
            time=obs_index[2],
        )
        out = tf.convert_to_tensor(out.to_dataarray())
        return tf.transpose(out, perm=[2, 3, 1, 0])

    def get_next(self):
        """Get next observation sample."""
        obs_index = self._get_observation_index()
        return self._get_observation(obs_index)

    def __getitem__(self, index):
        return self.get_next()

    def __next__(self):
        if self._i < self.epoch_samples:
            out = self.get_next()
            self._i += 1
            return out
        else:
            raise StopIteration


class LazyDualDataHandler(DualDataHandler):
    """Lazy loading dual data handler. Matches sample regions for low res and
    high res lazy data handlers."""

    def __init__(self, lr_dh, hr_dh, s_enhance=1, t_enhance=1,
                 epoch_samples=1024):
        self.lr_dh = lr_dh
        self.hr_dh = hr_dh
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        self.current_obs_index = None
        self._means = None
        self._stds = None
        self.epoch_samples = epoch_samples
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
            self._means = dict(zip(lr_features,
                                   self.lr_dh.data[lr_features].mean(axis=0)))
            hr_means = dict(zip(hr_only_features,
                                self.hr_dh[hr_only_features].mean(axis=0)))
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
                               self.hr_dh[hr_only_features].std(axis=0)))
            self._stds.update(hr_stds)
        return self._stds

    def __iter__(self):
        self._i = 0
        return self

    def __len__(self):
        return self.epoch_samples

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
            (high_res, low_res) pair
        """
        lr_obs_idx = self.lr_dh._get_observation_index()
        hr_obs_idx = [slice(s.start * self.s_enhance, s.stop * self.s_enhance)
                      for s in lr_obs_idx[:2]]
        hr_obs_idx += [slice(s.start * self.t_enhance, s.stop * self.t_enhance)
                       for s in lr_obs_idx[2:]]
        out = (self.hr_dh._get_observation(hr_obs_idx).numpy(),
               self.lr_dh._get_observation(lr_obs_idx).numpy())
        return out

    def __getitem__(self, index):
        return self.get_next()

    def __next__(self):
        if self._i < self.epoch_samples:
            out = self.get_next()
            self._i += 1
            return out
        else:
            raise StopIteration

    def __call__(self):
        """Call method to enable Dataset.from_generator() call."""
        for _ in range(self.epoch_samples):
            yield self.get_next()

    @property
    def data(self):
        """Return tensorflow dataset generator."""
        lr_shape = (*self.lr_dh.sample_shape, len(self.lr_dh.features))
        hr_shape = (*self.hr_dh.sample_shape, len(self.hr_dh.features))
        return tf.data.Dataset.from_generator(
            self.__call__,
            output_signature=(tf.TensorSpec(hr_shape, tf.float32),
                              tf.TensorSpec(lr_shape, tf.float32)))


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
                 batch_size=32, n_batches=100, n_epochs=100, max_workers=1):
        self.data_handlers = data_handlers
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_batches = n_batches
        self.epoch_samples = batch_size * n_batches
        self.queue_samples = self.epoch_samples * n_epochs
        self.total_obs = self.epoch_samples * self.n_epochs
        self._means = (None if means_file is None
                       else safe_json_load(means_file))
        self._stds = (None if stdevs_file is None
                      else safe_json_load(stdevs_file))
        self._i = 0
        self.val_data = []
        self.timer = Timer()
        self._queue = None
        self.enqueue_thread = None
        self.max_workers = max_workers

        logger.info(f'Initialized {self.__class__.__name__} with '
                    f'{len(self.data_handlers)} data_handlers, '
                    f'means_file = {means_file}, stdevs_file = {stdevs_file}, '
                    f'batch_size = {batch_size}, n_batches = {n_batches}, '
                    f'epoch_samples = {self.epoch_samples}')

        self.preflight(n_samples=(self.batch_size),
                       max_workers=max_workers)

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

    def preflight(self, n_samples, max_workers=1):
        """Load samples for first epoch."""
        logger.info(f'Loading {n_samples} samples to initialize queue.')
        self.enqueue_samples(n_samples, max_workers=max_workers)
        self.enqueue_thread = threading.Thread(
            target=self.callback, args=(self.max_workers))
        self.start()

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
        self._i = 0
        return self

    @property
    def queue(self):
        """Queue of (hr, lr) samples to use for building batches."""
        if self._queue is None:
            lr_shape = (*self.lr_sample_shape, len(self.lr_features))
            hr_shape = (*self.hr_sample_shape, len(self.hr_features))
            self._queue = tf.queue.FIFOQueue(
                self.queue_samples,
                dtypes=[tf.float32, tf.float32],
                shapes=[hr_shape, lr_shape])
        return self._queue

    def enqueue_samples(self, n_samples, max_workers=None):
        """Fill queue with enough samples for an epoch."""
        empty = self.queue_samples - self.queue.size()
        msg = (f'Requested number of samples {n_samples} exceeds the number '
               f'of empty spots in the queue {empty}')
        assert n_samples <= empty, msg
        logger.info(f'Loading {n_samples} samples into queue.')
        if max_workers == 1:
            for _ in tqdm(range(n_samples)):
                hr, lr = self.get_next()
                self.queue.enqueue((hr, lr))
        else:
            futures = []
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                for i in range(n_samples):
                    futures.append(exe.submit(self.get_next))
                    logger.info(f'Submitted {i + 1} futures.')
            for i, future in enumerate(as_completed(futures)):
                hr, lr = future.result()
                self.queue.enqueue((hr, lr))
                logger.info(f'Completed {i + 1} / {len(futures)} futures.')

    def callback(self, max_workers=None):
        """Callback function for enqueue thread."""
        while self._is_training:
            logger.info(f'{self.queue_size} samples in queue.')
            while self.queue_size < (self.queue_samples - self.batch_size):
                self.queue_next_batch(max_workers=max_workers)

    def queue_next_batch(self, max_workers=None):
        """Add N = batch_size samples to queue."""
        self.enqueue_samples(n_samples=self.batch_size,
                             max_workers=max_workers)

    @property
    def queue_size(self):
        """Get number of samples in queue."""
        return self.queue.size().numpy()

    @property
    def missing_samples(self):
        """Get number of empty spots in queue."""
        return self.queue_samples - self.queue_size

    @property
    def is_empty(self):
        """Check if queue is empty."""
        return self.queue_size == 0

    def take(self, n):
        """Take n samples from queue to build a batch."""
        logger.info(f'{self.queue.size().numpy()} samples in queue.')
        logger.info(f'Taking {n} samples.')
        return self.queue.dequeue_many(n)

    def _get_next_batch(self):
        """Take samples from queue and build batch class."""
        samples = self.take(self.batch_size)
        batch = self.BATCH_CLASS(
            high_res=samples[0], low_res=samples[1])
        return batch

    def get_next(self):
        """Get next pair of low-res / high-res samples from randomly selected
        data handler

        Returns
        -------
        tuple
            (high_res, low_res) pair
        """
        handler = self.get_rand_handler()
        return handler.get_next()

    def __getitem__(self, index):
        return self.get_next()

    def __call__(self):
        """Call method to enable Dataset.from_generator() call."""
        for _ in range(self.total_obs):
            yield self.get_next()

    def prefetch(self):
        """Return tensorflow dataset generator."""
        lr_shape = (*self.lr_sample_shape, len(self.lr_features))
        hr_shape = (*self.hr_sample_shape, len(self.hr_features))
        data = tf.data.Dataset.from_generator(
            self.__call__,
            output_signature=(tf.TensorSpec(hr_shape, tf.float32),
                              tf.TensorSpec(lr_shape, tf.float32)))
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return data

    def __next__(self):
        """Get the next batch of observations.

        Returns
        -------
        batch : Batch
            Batch object with batch.low_res and batch.high_res attributes
            with the appropriate subsampling of interpolated ERA.
        """
        if self._i < self.n_batches:
            logger.info(
                f'Getting next batch: {self._i + 1} / {self.n_batches}')
            batch = self.timer(self._get_next_batch)
            logger.info(
                f'Built batch in {self.timer.log["elapsed:_get_next_batch"]}')
            self._i += 1
        else:
            raise StopIteration

        return batch
