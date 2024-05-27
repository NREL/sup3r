"""Abstract Batcher class used to generate batches for training."""

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from rex import safe_json_load

from sup3r.containers.collections.samplers import SamplerCollection
from sup3r.containers.samplers import DualSampler, Sampler

logger = logging.getLogger(__name__)


class Batch:
    """Basic single batch object, containing low_res and high_res data"""

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


class AbstractBatchQueue(SamplerCollection, ABC):
    """Abstract BatchQueue class. This class gets batches from a dataset
    generator and maintains a queue of normalized batches in a dedicated thread
    so the training routine can proceed as soon as batches as available.

    Notes
    -----
    If using a batch queue directly, rather than a :class:`BatchHandler` you
    will need to manually start the queue thread with self.start()
    """

    BATCH_CLASS = Batch

    def __init__(
        self,
        samplers: Union[List[Sampler], List[DualSampler]],
        batch_size,
        n_batches,
        s_enhance,
        t_enhance,
        means: Union[Dict, str],
        stds: Union[Dict, str],
        queue_cap: Optional[int] = None,
        max_workers: Optional[int] = None,
        default_device: Optional[str] = None,
        thread_name: Optional[str] = 'training',
    ):
        """
        Parameters
        ----------
        samplers : List[Sampler]
            List of Sampler instances
        batch_size : int
            Number of observations / samples in a batch
        n_batches : int
            Number of batches in an epoch, this sets the iteration limit for
            this object.
        s_enhance : int
            Integer factor by which the spatial axes is to be enhanced.
        t_enhance : int
            Integer factor by which the temporal axes is to be enhanced.
        means : Union[Dict, str]
            Either a .json path containing a dictionary or a dictionary of
            means which will be used to normalize batches as they are built.
            Provide a dictionary of zeros to run without normalization.
        stds : Union[Dict, str]
            Either a .json path containing a dictionary or a dictionary of
            standard deviations which will be used to normalize batches as they
            are built. Provide a dictionary of ones to run without
            normalization.
        queue_cap : int
            Maximum number of batches the batch queue can store.
        max_workers : int
            Number of workers / threads to use for getting samples used to
            build batches.
        default_device : str
            Default device to use for batch queue (e.g. /cpu:0, /gpu:0). If
            None this will use the first GPU if GPUs are available otherwise
            the CPU.
        thread_name : str
            Name of the queue thread. Default is 'training'. Used to set name
            to 'validation' for :class:`BatchQueue`, which has a training and
            validation queue.
        """
        super().__init__(
            samplers=samplers, s_enhance=s_enhance, t_enhance=t_enhance
        )
        self._sample_counter = 0
        self._batch_counter = 0
        self._batches = None
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.queue_cap = queue_cap or n_batches
        self.max_workers = max_workers or batch_size
        self.run_queue = threading.Event()
        self.means = (
            means if isinstance(means, dict) else safe_json_load(means)
        )
        self.stds = stds if isinstance(stds, dict) else safe_json_load(stds)
        self.container_index = self.get_container_index()
        self.queue_thread = threading.Thread(
            target=self.enqueue_batches,
            args=(self.run_queue,),
            name=thread_name,
        )
        self.queue = self.get_queue()
        self.gpu_list = tf.config.list_physical_devices('GPU')
        self.default_device = default_device or (
            '/cpu:0' if len(self.gpu_list) == 0 else '/gpu:0'
        )
        self.preflight()

    def preflight(self):
        """Get data generator and run checks before kicking off the queue."""
        self.data = self.get_data_generator()
        self.check_stats()
        self.check_features()
        self.check_enhancement_factors()

    def check_features(self):
        """Make sure all samplers have the same sets of features."""
        features = [c.features for c in self.containers]
        msg = 'Received samplers with different sets of features.'
        assert all(feats == features[0] for feats in features), msg

    def check_stats(self):
        """Make sure the provided stats cover the contained features."""
        msg = (
            f'Received means = {self.means} with self.features = '
            f'{self.features}.'
        )
        assert len(self.means) == len(self.features), msg
        msg = (
            f'Received stds = {self.stds} with self.features = '
            f'{self.features}.'
        )
        assert len(self.stds) == len(self.features), msg

    def check_enhancement_factors(self):
        """Make sure the enhancement factors evenly divide the sample_shape."""
        msg = (
            f'The sample_shape {self.sample_shape} is not consistent with '
            f'the enhancement factors {self.s_enhance, self.t_enhance}.'
        )
        assert all(
            samp % enhance == 0
            for samp, enhance in zip(
                self.sample_shape,
                [self.s_enhance, self.s_enhance, self.t_enhance],
            )
        ), msg

    @property
    def batches(self):
        """Return iterable of batches prefetched from the data generator."""
        if self._batches is None:
            self._batches = self.prefetch()
        return self._batches

    def generator(self):
        """Generator over batches, which are composed of data samples."""
        while True and self.run_queue.is_set():
            idx = self._sample_counter
            self._sample_counter += 1
            yield self[idx]

    @abstractmethod
    def get_output_signature(
        self,
    ) -> Union[Tuple[tf.TensorSpec, tf.TensorSpec], tf.TensorSpec]:
        """Get tensorflow dataset output signature. If we are sampling from
        container pairs then this is a tuple for low / high res batches.
        Otherwise we are just getting high res batches and coarsening to get
        the corresponding low res batches."""

    def get_data_generator(self):
        """Tensorflow dataset."""
        return tf.data.Dataset.from_generator(
            self.generator, output_signature=self.get_output_signature()
        )

    @abstractmethod
    def _parallel_map(self):
        """Perform call to map function to enable parallel sampling."""

    def prefetch(self):
        """Prefetch set of batches from dataset generator."""
        logger.debug(
            f'Prefetching {self.queue_thread.name} batches with '
            f'batch_size = {self.batch_size}.'
        )
        with tf.device(self.default_device):
            data = self._parallel_map()
            data = data.prefetch(tf.data.AUTOTUNE)
            batches = data.batch(
                self.batch_size,
                drop_remainder=True,
                deterministic=False,
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        return batches.as_numpy_iterator()

    @abstractmethod
    def _get_queue_shape(self) -> List[tuple]:
        """Get shape for queue. For DualSampler containers shape is a list of
        length = 2. Otherwise its a list of length = 1.  In both cases the list
        elements are of shape (batch_size,
        *sample_shape, len(features))"""

    def get_queue(self):
        """Initialize FIFO queue for storing batches.

        Returns
        -------
        tensorflow.queue.FIFOQueue
            First in first out queue with `size = self.queue_cap`
        """
        shapes = self._get_queue_shape()
        dtypes = [tf.float32] * len(shapes)
        out = tf.queue.FIFOQueue(
            self.queue_cap, dtypes=dtypes, shapes=self._get_queue_shape()
        )
        return out

    @abstractmethod
    def batch_next(self, samples):
        """Returns normalized collection of samples / observations. Performs
        coarsening on high-res data if Collection objects are Samplers and not
        DualSamplers

        Returns
        -------
        Batch
            Simple Batch object with `low_res` and `high_res` attributes
        """

    def start(self) -> None:
        """Start thread to keep sample queue full for batches."""
        logger.info(f'Starting {self.queue_thread.name} queue.')
        self.run_queue.set()
        self.queue_thread.start()

    def join(self) -> None:
        """Join thread to exit gracefully."""
        logger.info(
            f'Joining {self.queue_thread.name} queue thread to main ' 'thread.'
        )
        self.queue_thread.join()

    def stop(self) -> None:
        """Stop loading batches."""
        logger.info(f'Stopping {self.queue_thread.name} queue.')
        self.run_queue.clear()
        self.join()

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self._batch_counter = 0
        return self

    def enqueue_batches(self, run_queue: threading.Event) -> None:
        """Callback function for queue thread. While training the queue is
        checked for empty spots and filled. In the training thread, batches are
        removed from the queue."""
        try:
            while run_queue.is_set():
                queue_size = self.queue.size().numpy()
                if queue_size < self.queue_cap:
                    if queue_size == 1:
                        msg = f'1 batch in {self.queue_thread.name} queue'
                    else:
                        msg = (
                            f'{queue_size} batches in '
                            f'{self.queue_thread.name} queue.'
                        )
                    logger.debug(msg)

                    batch = next(self.batches, None)
                    if batch is not None:
                        self.queue.enqueue(batch)
        except KeyboardInterrupt:
            logger.info(
                f'Attempting to stop {self.queue.thread.name} ' 'batch queue.'
            )
            self.stop()

    def get_next(self) -> Batch:
        """Get next batch. This removes sets of samples from the queue and
        wraps them in the simple Batch class. This also removes the time
        dimension from samples for batches for spatial models

        Returns
        -------
        batch : Batch
            Batch object with batch.low_res and batch.high_res attributes
        """
        samples = self.queue.dequeue()
        if self.sample_shape[2] == 1:
            if isinstance(samples, (list, tuple)):
                samples = tuple([s[..., 0, :] for s in samples])
            else:
                samples = samples[..., 0, :]
        return self.batch_next(samples)

    def __next__(self) -> Batch:
        """
        Returns
        -------
        batch : Batch
            Batch object with batch.low_res and batch.high_res attributes
        """
        if self._batch_counter < self.n_batches:
            logger.debug(
                f'Getting next {self.queue_thread.name} batch: '
                f'{self._batch_counter + 1} / {self.n_batches}.'
            )
            start = time.time()
            batch = self.get_next()
            logger.debug(
                f'Built {self.queue_thread.name} batch in '
                f'{time.time() - start}.'
            )
            self._batch_counter += 1
        else:
            raise StopIteration

        return batch

    @property
    def lr_means(self):
        """Means specific to the low-res objects in the Containers."""
        return np.array([self.means[k] for k in self.lr_features]).astype(
            np.float32
        )

    @property
    def hr_means(self):
        """Means specific the high-res objects in the Containers."""
        return np.array([self.means[k] for k in self.hr_features]).astype(
            np.float32
        )

    @property
    def lr_stds(self):
        """Stdevs specific the low-res objects in the Containers."""
        return np.array([self.stds[k] for k in self.lr_features]).astype(
            np.float32
        )

    @property
    def hr_stds(self):
        """Stdevs specific the high-res objects in the Containers."""
        return np.array([self.stds[k] for k in self.hr_features]).astype(
            np.float32
        )

    @staticmethod
    def _normalize(array, means, stds):
        """Normalize an array with given means and stds."""
        return (array - means) / stds

    def normalize(self, lr, hr) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize a low-res / high-res pair with the stored means and
        stdevs."""
        return (
            self._normalize(lr, self.lr_means, self.lr_stds),
            self._normalize(hr, self.hr_means, self.hr_stds),
        )
