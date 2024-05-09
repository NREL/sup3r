"""Batch handling classes for queued batch loads"""
import logging
import threading
from abc import ABC, abstractmethod

import numpy as np

from sup3r.utilities.utilities import get_handler_weights

logger = logging.getLogger(__name__)


class AbstractBatchBuilder(ABC):
    """Abstract batch builder class. Need to implement data and gen methods"""

    def __init__(self, data_handlers):
        self.data_handlers = data_handlers
        self.batch_size = None
        self.batches = None
        self._handler_weights = None
        self._sample_counter = 0

    def __iter__(self):
        self._sample_counter = 0
        return self

    @property
    def handler_weights(self):
        """Get weights used to sample from different data handlers based on
        relative sizes"""
        if self._handler_weights is None:
            self._handler_weights = get_handler_weights(self.data_handlers)
        return self._handler_weights

    def get_handler_index(self):
        """Get random handler index based on handler weights"""
        indices = np.arange(0, len(self.data_handlers))
        return np.random.choice(indices, p=self.handler_weights)

    def get_rand_handler(self):
        """Get random handler based on handler weights"""
        if self._sample_counter % self.batch_size == 0:
            self.handler_index = self.get_handler_index()
        return self.data_handlers[self.handler_index]

    def __next__(self):
        return next(self.batches)

    def __getitem__(self, index):
        """Get single observation / sample. Batches are built from
        self.batch_size samples."""
        handler = self.get_rand_handler()
        return handler.get_next()

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

    def prefetch(self):
        """Prefetch set of batches from dataset generator."""
        data = self.data.map(lambda x,y : (x,y),
                             num_parallel_calls=self.max_workers)
        data = data.prefetch(buffer_size=self.buffer_size)
        data = data.batch(self.batch_size)
        return data.as_numpy_iterator()


class AbstractBatchHandler(ABC):
    """Abstract batch handler class. Need to implement queue, get_next,
    normalize, and specify BATCH_CLASS and VAL_CLASS."""

    BATCH_CLASS = None
    VAL_CLASS = None

    def __init__(self, data_handlers, means_file, stdevs_file,
                 batch_size=32, n_batches=100, max_workers=None):
        self.data_handlers = data_handlers
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.queue_capacity = n_batches
        self.val_data = []
        self.batch_pool = None
        self._batch_counter = 0
        self._queue = None
        self._is_training = False
        self._enqueue_thread = None

        HandlerStats.__init__(self, data_handlers, means_file=means_file,
                              stdevs_file=stdevs_file)

        logger.info(f'Initialized {self.__class__.__name__} with '
                    f'{len(self.data_handlers)} data_handlers, '
                    f'means_file = {means_file}, stdevs_file = {stdevs_file}, '
                    f'batch_size = {batch_size}, n_batches = {n_batches}, '
                    f'max_workers = {max_workers}.')

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
            if queue_size < self.queue_capacity:
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
