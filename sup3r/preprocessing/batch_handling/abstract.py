"""Batch handling classes for queued batch loads"""
import logging
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from sup3r.preprocessing.utilities import get_handler_weights

logger = logging.getLogger(__name__)


class AbstractBatchBuilder(ABC):
    """Abstract class for batch builders. Just need to specify the lr_shape and
    hr_shape properties used to define the batch generator output signature for
    `tf.data.Dataset.from_generator(..., output_signature=...)"""

    def __init__(self, data_handlers, batch_size, buffer_size=None,
                 max_workers=None):
        """
        Parameters
        ----------
        data_handlers : list[DataHandler]
            List of DataHandler instances each with a `.size` property and a
            `.get_next` method to return the next (low_res, high_res) sample.
        batch_size : int
            Number of samples/observations to use for each batch. e.g. Batches
            will be (batch_size, spatial_1, spatial_2, temporal, features)
        buffer_size : int
            Number of samples to prefetch
        """
        self.data_handlers = data_handlers
        self.batch_size = batch_size
        self.buffer_size = buffer_size or 10 * batch_size
        self.max_workers = max_workers or self.batch_size
        self.handler_weights = get_handler_weights(data_handlers)
        self.handler_index = self.get_handler_index()
        self._sample_counter = 0
        self.batches = self.prefetch()

    def __iter__(self):
        self._sample_counter = 0
        return self

    def get_handler_index(self):
        """Get random handler index based on handler weights"""
        indices = np.arange(0, len(self.data_handlers))
        return np.random.choice(indices, p=self.handler_weights)

    def get_rand_handler(self):
        """Get random handler based on handler weights"""
        if self._sample_counter % self.batch_size == 0:
            self.handler_index = self.get_handler_index()
        return self.data_handlers[self.handler_index]

    @property
    @abstractmethod
    def lr_shape(self):
        """Shape of low-res batch array (n_obs, spatial_1, spatial_2, temporal,
        features). Used to define output_signature for
        tf.data.Dataset.from_generator()"""

    @property
    @abstractmethod
    def hr_shape(self):
        """Shape of high-res batch array (n_obs, spatial_1, spatial_2,
        temporal, features). Used to define output_signature for
        tf.data.Dataset.from_generator()"""

    @property
    def data(self):
        """Return tensorflow dataset generator."""
        data = tf.data.Dataset.from_generator(
            self.gen,
            output_signature=(tf.TensorSpec(self.lr_shape, tf.float32,
                                            name='low_resolution'),
                              tf.TensorSpec(self.hr_shape, tf.float32,
                                            name='high_resolution')))
        return data

    def __next__(self):
        return next(self.batches)

    def __getitem__(self, index):
        """Get single sample. Batches are built from self.batch_size
        samples."""
        handler = self.get_rand_handler()
        return handler.get_next()

    def gen(self):
        """Generator method to enable Dataset.from_generator() call."""
        while True:
            idx = self._sample_counter
            self._sample_counter += 1
            yield self[idx]

    def prefetch(self):
        """Prefetch set of batches for an epoch."""
        data = self.data.map(lambda x,y : (x,y),
                             num_parallel_calls=self.max_workers)
        data = data.prefetch(buffer_size=self.buffer_size)
        data = data.batch(self.batch_size)
        return data.as_numpy_iterator()
