"""Batch handling classes for queued batch loads"""
import logging
import threading
from abc import ABC, abstractmethod

import numpy as np

from sup3r.preprocessing.mixin import HandlerStats
from sup3r.utilities.utilities import get_handler_weights

logger = logging.getLogger(__name__)


class AbstractBatchBuilder(ABC):
    """Abstract batch builder class. Need to implement data and gen methods"""

    def __init__(self, data_containers, batch_size):
        """
        Parameters
        ----------
        data_containers : list[DataContainer]
            List of DataContainer instances each with a `.size` property and a
            `.get_next` method to return the next (low_res, high_res) sample.
        batch_size : int
            Number of samples/observations to use for each batch. e.g. Batches
            will be (batch_size, spatial_1, spatial_2, temporal, features)
        """
        self.data_containers = data_containers
        self.batch_size = batch_size
        self.max_workers = None
        self.buffer_size = None
        self._data = None
        self._batches = None
        self._handler_weights = None
        self._lr_shape = None
        self._hr_shape = None
        self._sample_counter = 0

    def __iter__(self):
        self._sample_counter = 0
        return self

    @property
    def handler_weights(self):
        """Get weights used to sample from different data handlers based on
        relative sizes"""
        if self._handler_weights is None:
            self._handler_weights = get_handler_weights(self.data_containers)
        return self._handler_weights

    def get_handler_index(self):
        """Get random handler index based on handler weights"""
        indices = np.arange(0, len(self.data_containers))
        return np.random.choice(indices, p=self.handler_weights)

    def get_rand_handler(self):
        """Get random handler based on handler weights"""
        if self._sample_counter % self.batch_size == 0:
            self.handler_index = self.get_handler_index()
        return self.data_containers[self.handler_index]

    def __getitem__(self, index):
        """Get single observation / sample. Batches are built from
        self.batch_size samples."""
        handler = self.get_rand_handler()
        return handler.get_next()

    def __next__(self):
        return next(self.batches)

    @property
    @abstractmethod
    def lr_shape(self):
        """Shape of low resolution sample in a low-res / high-res pair.  (e.g.
        (spatial_1, spatial_2, temporal, features)) """

    @property
    @abstractmethod
    def hr_shape(self):
        """Shape of high resolution sample in a low-res / high-res pair.  (e.g.
        (spatial_1, spatial_2, temporal, features)) """

    @property
    @abstractmethod
    def data(self):
        """Return tensorflow dataset generator."""

    @abstractmethod
    def gen(self):
        """Generator method to enable Dataset.from_generator() call."""

    @property
    @abstractmethod
    def batches(self):
        """Prefetch set of batches from dataset generator."""


class AbstractBatchHandler(HandlerStats, ABC):
    """Abstract batch handler class. Need to implement queue, get_next,
    normalize, and specify BATCH_CLASS and VAL_CLASS."""

    BATCH_CLASS = None
    VAL_CLASS = None

    def __init__(self, data_containers, batch_size, n_batches, means_file,
                 stdevs_file, queue_cap):
        self.data_containers = data_containers
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.queue_cap = queue_cap
        self.means_file = means_file
        self.stdevs_file = stdevs_file
        self.val_data = []
        self._batch_pool = None
        self._batch_counter = 0
        self._queue = None
        self._is_training = False
        self._enqueue_thread = None
        HandlerStats.__init__(self, data_containers, means_file=means_file,
                              stdevs_file=stdevs_file)

    @property
    @abstractmethod
    def batch_pool(self):
        """Iterable set of batches. Can be implemented with BatchBuilder."""

    @property
    @abstractmethod
    def queue(self):
        """Queue to use for storing batches."""

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
            if queue_size < self.queue_cap:
                logger.info(f'{queue_size} batches in queue.')
                self.queue.enqueue(next(self.batch_pool))

    @abstractmethod
    def normalize(self, lr, hr):
        """Normalize a low-res / high-res pair with the stored means and
        stdevs."""

    @abstractmethod
    def get_next(self):
        """Get the next batch of observations."""

    def __next__(self):
        """
        Returns
        -------
        batch : Batch
            Batch object with batch.low_res and batch.high_res attributes
        """

        if self._batch_counter < self.n_batches:
            batch = self.get_next()
            self._batch_counter += 1
        else:
            raise StopIteration

        return batch
