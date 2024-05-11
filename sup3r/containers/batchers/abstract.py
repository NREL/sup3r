"""Abstract Batcher class used to generate batches for training."""

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

import tensorflow as tf
from rex import safe_json_load

from sup3r.containers.samplers.base import Sampler, SamplerCollection

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


class AbstractBatchBuilder(SamplerCollection, ABC):
    """Collection with additional methods for collecting sampler data into
    batches and preparing batches for training."""

    def __init__(
        self,
        containers: List[Sampler],
        s_enhance,
        t_enhance,
        batch_size,
        max_workers,
    ):
        super().__init__(containers, s_enhance, t_enhance)
        self._sample_counter = 0
        self._batch_counter = 0
        self._data = None
        self._batches = None
        self.batch_size = batch_size
        self.max_workers = max_workers

    @property
    def batches(self):
        """Return iterable of batches prefetched from the data generator."""
        if self._batches is None:
            self._batches = self.prefetch()
        return self._batches

    def generator(self):
        """Generator over batches, which are composed of data samples."""
        while True:
            idx = self._sample_counter
            self._sample_counter += 1
            yield self[idx]

    @abstractmethod
    def get_output_signature(
        self,
    ) -> Union[Tuple[tf.TensorSpec, tf.TensorSpec], tf.TensorSpec]:
        """Get output signature used to define tensorflow dataset."""

    @property
    def data(self):
        """Tensorflow dataset."""
        if self._data is None:
            self._data = tf.data.Dataset.from_generator(
                self.generator, output_signature=self.get_output_signature()
            )
        return self._data

    def _parallel_map(self):
        """Perform call to map function to enable parallel sampling."""
        if self.all_container_pairs:
            data = self.data.map(
                lambda x, y: (x, y), num_parallel_calls=self.max_workers
            )
        else:
            data = self.data.map(
                lambda x: x, num_parallel_calls=self.max_workers
            )
        return data

    def prefetch(self):
        """Prefetch set of batches from dataset generator."""
        logger.info(
            f'Prefetching batches with batch_size = {self.batch_size}.'
        )
        data = self._parallel_map()
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
        batches = data.batch(self.batch_size)
        return batches.as_numpy_iterator()


class AbstractBatchQueue(AbstractBatchBuilder, ABC):
    """Abstract BatchQueue class. This class gets batches from a dataset
    generator and maintains a queue of normalized batches in a dedicated thread
    so the training routine can proceed as soon as batches as available."""

    BATCH_CLASS = Batch

    def __init__(
        self,
        containers: List[Sampler],
        s_enhance,
        t_enhance,
        batch_size,
        n_batches,
        queue_cap,
        max_workers,
    ):
        """
        Parameters
        ----------
        containers : List[Sampler]
            List of Sampler instances
        s_enhance : int
            Integer factor by which the spatial axes is to be enhanced.
        t_enhance : int
            Integer factor by which the temporal axes is to be enhanced.
        batch_size : int
            Number of observations / samples in a batch
        n_batches : int
            Number of batches in an epoch, this sets the iteration limit for
            this object.
        queue_cap : int
            Maximum number of batches the batch queue can store.
        max_workers : int
            Number of workers / threads to use for getting samples used to
            build batches.
        """
        super().__init__(
            containers, s_enhance, t_enhance, batch_size, max_workers
        )
        self._batch_counter = 0
        self._training = False
        self.n_batches = n_batches
        self.queue_cap = queue_cap
        self.queue_thread = threading.Thread(target=self.enqueue_batches)
        self.queue = self.get_queue()

    def _get_queue_shape(self) -> List[tuple]:
        """Get shape for queue. For SamplerPair containers shape is a list of
        length = 2. Otherwise its a list of length = 1.  In both cases the list
        elements are of shape (batch_size,
        *sample_shape, len(features))"""
        if self.all_container_pairs:
            shape = [
                (self.batch_size, *self.lr_shape),
                (self.batch_size, *self.hr_shape),
            ]
        else:
            shape = [(self.batch_size, *self.sample_shape, len(self.features))]
        return shape

    def get_queue(self):
        """Initialize FIFO queue for storing batches.

        Returns
        -------
        tensorflow.queue.FIFOQueue
            First in first out queue with `size = self.queue_cap`
        """
        shapes = self._get_queue_shape()
        dtypes = [tf.float32] * len(shapes)
        queue = tf.queue.FIFOQueue(
            self.queue_cap, dtypes=dtypes, shapes=self._get_queue_shape()
        )
        return queue

    @abstractmethod
    def batch_next(self, samples):
        """Returns wrapped collection of samples / observations. Performs
        coarsening on high-res data if Collection objects are Samplers and not
        SamplerPairs

        Returns
        -------
        Batch
            Simple Batch object with `low_res` and `high_res` attributes
        """

    def start(self) -> None:
        """Start thread to keep sample queue full for batches."""
        logger.info(f'Running {self.__class__.__name__}.queue_thread.start()')
        self._is_training = True
        self.queue_thread.start()

    def join(self) -> None:
        """Join thread to exit gracefully."""
        logger.info(f'Running {self.__class__.__name__}.queue_thread.join()')
        self.queue_thread.join()

    def stop(self) -> None:
        """Stop loading batches."""
        self._is_training = False
        self.join()

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self._batch_counter = 0
        return self

    def enqueue_batches(self) -> None:
        """Callback function for queue thread. While training the queue is
        checked for empty spots and filled. In the training thread, batches are
        removed from the queue."""
        while self._is_training:
            queue_size = self.queue.size().numpy()
            if queue_size < self.queue_cap:
                logger.info(f'{queue_size} batches in queue.')
                self.queue.enqueue(next(self.batches))

    def get_next(self) -> Batch:
        """Get next batch. This removes sets of samples from the queue and
        wraps them in the simple Batch class.

        Returns
        -------
        batch : Batch
            Batch object with batch.low_res and batch.high_res attributes
        """
        samples = self.queue.dequeue()
        batch = self.batch_next(samples)
        return batch

    def __next__(self) -> Batch:
        """
        Returns
        -------
        batch : Batch
            Batch object with batch.low_res and batch.high_res attributes
        """
        if self._batch_counter < self.n_batches:
            logger.info(
                f'Getting next batch: {self._batch_counter + 1} / '
                f'{self.n_batches}'
            )
            start = time.time()
            batch = self.get_next()
            logger.info(f'Built batch in {time.time() - start}.')
            self._batch_counter += 1
        else:
            raise StopIteration

        return batch

    @abstractmethod
    def get_output_signature(self):
        """Get tensorflow dataset output signature. If we are sampling from
        container pairs then this is a tuple for low / high res batches.
        Otherwise we are just getting high res batches and coarsening to get
        the corresponding low res batches."""


class AbstractNormedBatchQueue(AbstractBatchQueue):
    """Abstract NormedBatchQueue class. This extends the BatchQueue class to
    require implementation of `normalize` and `means`, `stds` constructor
    args."""

    def __init__(
        self,
        containers: List[Sampler],
        s_enhance,
        t_enhance,
        batch_size,
        n_batches,
        queue_cap,
        means: Union[Dict, str],
        stds: Union[Dict, str],
        max_workers=None,
    ):
        """
        Parameters
        ----------
        containers : List[Sampler]
            List of Sampler instances
        s_enhance : int
            Integer factor by which the spatial axes is to be enhanced.
        t_enhance : int
            Integer factor by which the temporal axes is to be enhanced.
        batch_size : int
            Number of observations / samples in a batch
        n_batches : int
            Number of batches in an epoch, this sets the iteration limit for
            this object.
        queue_cap : int
            Maximum number of batches the batch queue can store.
        means : Union[Dict, str]
            Either a .json path containing a dictionary or a dictionary of
            means which will be used to normalize batches as they are built.
        stds : Union[Dict, str]
            Either a .json path containing a dictionary or a dictionary of
            standard deviations which will be used to normalize batches as they
            are built.
        max_workers : int
            Number of workers / threads to use for getting samples used to
            build batches.
        """
        super().__init__(
            containers,
            s_enhance,
            t_enhance,
            batch_size,
            n_batches,
            queue_cap,
            max_workers,
        )
        self.means = (
            means if isinstance(means, dict) else safe_json_load(means)
        )
        self.stds = stds if isinstance(stds, dict) else safe_json_load(stds)
        self.container_index = self.get_container_index()
        self.container_weights = self.get_container_weights()
        self.max_workers = max_workers or self.batch_size

    @staticmethod
    def _normalize(array, means, stds):
        """Normalize an array with given means and stds."""
        return (array - means) / stds

    @abstractmethod
    def normalize(self, samples):
        """Normalize batch before sending out for training."""

    def get_next(self, **kwargs):
        """Get next batch of samples."""
        samples = self.queue.dequeue()
        samples = self.normalize(samples)
        batch = self.batch_next(samples, **kwargs)
        return batch
