"""Base objects which generate, build, and operate on batches. Also can
interface with models."""

import logging
from typing import Tuple, Union

import numpy as np
import tensorflow as tf
from rex import safe_json_load

from sup3r.containers.batchers.abstract import (
    AbstractNormedBatchQueue,
)
from sup3r.utilities.utilities import (
    smooth_data,
    spatial_coarsening,
    temporal_coarsening,
)

logger = logging.getLogger(__name__)


AUTO = tf.data.experimental.AUTOTUNE
option_no_order = tf.data.Options()
option_no_order.experimental_deterministic = False

option_no_order.experimental_optimization.noop_elimination = True
option_no_order.experimental_optimization.apply_default_optimizations = True


class SingleBatch:
    """Single Batch of low_res and high_res data"""

    def __init__(self, low_res, high_res):
        """Store low and high res data

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
        self.low_res = low_res
        self.high_res = high_res
        self.shape = (low_res.shape, high_res.shape)

    def __len__(self):
        """Get the number of samples in this batch."""
        return len(self.low_res)

    # pylint: disable=W0613
    @classmethod
    def get_coarse_batch(
        cls,
        high_res,
        s_enhance,
        t_enhance=1,
        temporal_coarsening_method='subsample',
        hr_features_ind=None,
        features=None,
        smoothing=None,
        smoothing_ignore=None,
    ):
        """Coarsen high res data and return Batch with high res and
        low res data

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
            min, max, or total
        hr_features_ind : list | np.ndarray | None
            List/array of feature channel indices that are used for generative
            output, without any feature indices used only for training.
        features : list | None
            Ordered list of training features input to the generative model
        smoothing : float | None
            Standard deviation to use for gaussian filtering of the coarse
            data. This can be tuned by matching the kinetic energy of a low
            resolution simulation with the kinetic energy of a coarsened and
            smoothed high resolution simulation. If None no smoothing is
            performed.
        smoothing_ignore : list | None
            List of features to ignore for the smoothing filter. None will
            smooth all features if smoothing kwarg is not None

        Returns
        -------
        Batch
            Batch instance with low and high res data
        """
        low_res = spatial_coarsening(high_res, s_enhance)

        features = (
            features if features is not None else [None] * low_res.shape[-1]
        )

        hr_features_ind = (
            hr_features_ind
            if hr_features_ind is not None
            else np.arange(high_res.shape[-1])
        )

        smoothing_ignore = (
            smoothing_ignore if smoothing_ignore is not None else []
        )

        low_res = (
            low_res
            if t_enhance == 1
            else temporal_coarsening(
                low_res, t_enhance, temporal_coarsening_method
            )
        )

        low_res = smooth_data(low_res, features, smoothing_ignore, smoothing)
        high_res = high_res[..., hr_features_ind]
        batch = cls(low_res, high_res)

        return batch


class BatchQueue(AbstractNormedBatchQueue):
    """Base BatchQueue class."""

    BATCH_CLASS = SingleBatch

    def __init__(self, containers, batch_size, n_batches, queue_cap,
                 means_file, stdevs_file, max_workers=None):
        super().__init__(containers, batch_size, n_batches, queue_cap)
        self.means = safe_json_load(means_file)
        self.stds = safe_json_load(stdevs_file)
        self.container_index = self.get_container_index()
        self.container_weights = self.get_container_weights()
        self.max_workers = max_workers or self.batch_size

    @property
    def batches(self):
        """Return iterable of batches prefetched from the data generator."""
        if self._batches is None:
            self._batches = self.prefetch()
        return self._batches

    def get_output_signature(self):
        """Get tensorflow dataset output signature. If we are sampling from
        container pairs then this is a tuple for low / high res batches.
        Otherwise we are just getting high res batches and coarsening to get
        the corresponding low res batches."""

        if self.all_container_pairs:
            output_signature = (
                tf.TensorSpec(self.lr_shape, tf.float32, name='low_res'),
                tf.TensorSpec(self.hr_shape, tf.float32, name='high_res'),
            )
        else:
            output_signature = tf.TensorSpec(
                (*self.sample_shape, len(self.features)), tf.float32,
                 name='high_res')

        return output_signature

    def prefetch(self):
        """Prefetch set of batches from dataset generator."""
        logger.info(
            f'Prefetching batches with batch_size = {self.batch_size}.'
        )
        data = self.data.map(lambda x, y: (x, y),
                             num_parallel_calls=self.max_workers)
        data = self.data.prefetch(tf.data.experimental.AUTOTUNE)
        batches = data.batch(self.batch_size)
        return batches.as_numpy_iterator()

    def _get_batch_shape(self, sample_shape, features):
        """Get shape of full batch array. (n_obs, spatial_1, spatial_2,
        temporal, n_features)"""
        return (self.batch_size, *sample_shape, len(features))

    def get_queue(self):
        """Initialize FIFO queue for storing batches."""
        if self.all_container_pairs:
            shapes = [
                self._get_batch_shape(self.lr_sample_shape, self.lr_features),
                self._get_batch_shape(self.hr_sample_shape, self.hr_features),
            ]
            queue = tf.queue.FIFOQueue(
                self.queue_cap,
                dtypes=[tf.float32, tf.float32],
                shapes=shapes,
            )
        else:
            shapes = [self._get_batch_shape(self.sample_shape, self.features)]
            queue = tf.queue.FIFOQueue(
                self.queue_cap, dtypes=[tf.float32], shapes=shapes
            )
        return queue

    def batch_next(self, samples, **kwargs):
        """Returns wrapped collection of samples / observations."""
        if self.all_container_pairs:
            low_res, high_res = samples
            batch = self.BATCH_CLASS(low_res=low_res, high_res=high_res)
        else:
            batch = self.BATCH_CLASS.get_coarse_batch(
                high_res=samples, **kwargs
            )
        return batch

    def enqueue_batches(self):
        """Callback function for enqueue thread."""
        while self._is_training:
            queue_size = self.queue.size().numpy()
            if queue_size < self.queue_cap:
                logger.info(f'{queue_size} batches in queue.')
                self.queue.enqueue(next(self.batches))

    @ staticmethod
    def _normalize(array, means, stds):
        """Normalize an array with given means and stds."""
        return (array - means) / stds

    def normalize(
        self, samples
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Normalize a low-res / high-res pair with the stored means and
        stdevs."""
        means, stds = self.get_means(), self.get_stds()
        if self.all_container_pairs:
            lr, hr = samples
            lr_means, hr_means = means
            lr_stds, hr_stds = stds
            out = (
                self._normalize(lr, lr_means, lr_stds),
                self._normalize(hr, hr_means, hr_stds),
            )

        else:
            out = self._normalize(samples, means, stds)

        return out

    def get_next(self, **kwargs):
        """Get next batch of observations."""
        samples = self.queue.dequeue()
        samples = self.normalize(samples)
        batch = self.batch_next(samples, **kwargs)
        return batch

    def get_means(self) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Get array of means or a tuple of arrays, if containers are
        ContainerPairs."""
        if self.all_container_pairs:
            lr_means = np.array([self.means[k] for k in self.lr_features])
            hr_means = np.array([self.means[k] for k in self.hr_features])
            means = (lr_means, hr_means)
        else:
            means = np.array([self.means[k] for k in self.features])
        return means

    def get_stds(self) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Get array of stdevs or a tuple of arrays, if containers are
        ContainerPairs."""
        if self.all_container_pairs:
            lr_stds = np.array([self.stds[k] for k in self.lr_features])
            hr_stds = np.array([self.stds[k] for k in self.hr_features])
            stds = (lr_stds, hr_stds)
        else:
            stds = np.array([self.stds[k] for k in self.features])
        return stds
