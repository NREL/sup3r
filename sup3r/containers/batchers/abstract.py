"""Abstract Batcher class used to generate batches for training."""

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Tuple, Union

import tensorflow as tf

from sup3r.containers.samplers.base import CollectionSampler

logger = logging.getLogger(__name__)


class AbstractBatchBuilder(CollectionSampler, ABC):
    """Collection with additional methods for collecting sampler data into
    batches and preparing batches for training."""

    def __init__(self, containers, batch_size):
        super().__init__(containers)
        self._sample_counter = 0
        self._batch_counter = 0
        self._data = None
        self._batches = None
        self.batch_size = batch_size

    @property
    @abstractmethod
    def batches(self):
        """Return iterable of batches using `prefetch()`"""

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

    @abstractmethod
    def prefetch(self):
        """Prefetch set of batches from dataset generator."""


class AbstractBatchQueue(AbstractBatchBuilder, ABC):
    """Abstract BatchQueue class. This class gets batches from a BatchBuilder
    instance and maintains a queue of normalized batches in a dedicated thread
    so the training routine can proceed as soon as batches as available."""

    def __init__(self, containers, batch_size, n_batches, queue_cap):
        super().__init__(containers, batch_size)
        self._batch_counter = 0
        self._training = False
        self.n_batches = n_batches
        self.queue_cap = queue_cap
        self.queue = self.get_queue()
        self.queue_thread = threading.Thread(target=self.enqueue_batches)

    @abstractmethod
    def get_queue(self):
        """Initialize FIFO queue for storing batches."""

    @abstractmethod
    def batch_next(self, samples):
        """Returns wrapped collection of samples / observations."""

    def start(self):
        """Start thread to keep sample queue full for batches."""
        logger.info(
            f'Running {self.__class__.__name__}.queue_thread.start()')
        self._is_training = True
        self.queue_thread.start()

    def join(self):
        """Join thread to exit gracefully."""
        logger.info(
            f'Running {self.__class__.__name__}.queue_thread.join()')
        self.queue_thread.join()

    def stop(self):
        """Stop loading batches."""
        self._is_training = False
        self.join()

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self._batch_counter = 0
        return self

    @abstractmethod
    def enqueue_batches(self):
        """Callback function for queue thread."""

    def get_next(self, **kwargs):
        """Get next batch of samples."""
        samples = self.queue.dequeue()
        batch = self.batch_next(samples, **kwargs)
        return batch

    def __next__(self):
        """
        Returns
        -------
        batch : Batch
            Batch object with batch.low_res and batch.high_res attributes
        """
        if self._batch_counter < self.n_batches:
            logger.info(f'Getting next batch: {self._batch_counter + 1} / '
                        f'{self.n_batches}')
            start = time.time()
            batch = self.get_next()
            logger.info(f'Built batch in {time.time() - start}.')
            self._batch_counter += 1
        else:
            raise StopIteration

        return batch


class AbstractNormedBatchQueue(AbstractBatchQueue):
    """Abstract NormedBatchQueue class. This extends the BatchQueue class to
    require implementations of `get_means(), `get_stdevs()`, and
    `normalize()`."""

    def __init__(self, containers, batch_size, n_batches, queue_cap):
        super().__init__(containers, batch_size, n_batches, queue_cap)

    @abstractmethod
    def normalize(self, samples):
        """Normalize batch before sending out for training."""

    @abstractmethod
    def get_means(self):
        """Get means for the features in the containers."""

    @abstractmethod
    def get_stds(self):
        """Get standard deviations for the features in the containers."""

    def get_next(self, **kwargs):
        """Get next batch of samples."""
        samples = self.queue.dequeue()
        samples = self.normalize(samples)
        batch = self.batch_next(samples, **kwargs)
        return batch
