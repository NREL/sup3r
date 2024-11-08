"""Abstract batch queue class used for multi-threaded batching / training.

TODO:
    (1) Figure out apparent "blocking" issue with threaded enqueue batches.
        max_workers=1 is the fastest?
    (2) Setup distributed data handling so this can work with data distributed
    over multiple nodes.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import tensorflow as tf

from sup3r.preprocessing.collections.base import Collection
from sup3r.utilities.utilities import RANDOM_GENERATOR, Timer

if TYPE_CHECKING:
    from sup3r.preprocessing.samplers import DualSampler, Sampler

logger = logging.getLogger(__name__)


class AbstractBatchQueue(Collection, ABC):
    """Abstract BatchQueue class. This class gets batches from a dataset
    generator and maintains a queue of batches in a dedicated thread so the
    training routine can proceed as soon as batches are available."""

    Batch = namedtuple('Batch', ['low_res', 'high_res'])

    def __init__(
        self,
        samplers: Union[List['Sampler'], List['DualSampler']],
        batch_size: int = 16,
        n_batches: int = 64,
        s_enhance: int = 1,
        t_enhance: int = 1,
        queue_cap: Optional[int] = None,
        transform_kwargs: Optional[dict] = None,
        max_workers: int = 1,
        thread_name: str = 'training',
        mode: str = 'lazy',
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
        queue_cap : int
            Maximum number of batches the batch queue can store.
        transform_kwargs : Union[Dict, None]
            Dictionary of kwargs to be passed to `self.transform`. This method
            performs smoothing / coarsening.
        max_workers : int
            Number of workers / threads to use for getting batches to fill
            queue
        thread_name : str
            Name of the queue thread. Default is 'training'. Used to set name
            to 'validation' for :class:`BatchHandler`, which has a training and
            validation queue.
        mode : str
            Loading mode. Default is 'lazy', which only loads data into memory
            as batches are queued. 'eager' will load all data into memory right
            away.
        """
        msg = (
            f'{self.__class__.__name__} requires a list of samplers. '
            f'Received type {type(samplers)}'
        )
        assert isinstance(samplers, list), msg
        super().__init__(containers=samplers)
        self._batch_count = 0
        self._queue_thread = None
        self._training_flag = threading.Event()
        self._thread_name = thread_name
        self.mode = mode
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.queue_cap = n_batches if queue_cap is None else queue_cap
        self.max_workers = max_workers
        self.container_index = self.get_container_index()
        self.queue = self.get_queue()
        self.transform_kwargs = transform_kwargs or {
            'smoothing_ignore': [],
            'smoothing': None,
        }
        self.timer = Timer()
        self.preflight()

    @property
    @abstractmethod
    def queue_shape(self):
        """Shape of objects stored in the queue. e.g. for single dataset queues
        this is (batch_size, *sample_shape, len(features)). For dual dataset
        queues this is [(batch_size, *lr_shape), (batch_size, *hr_shape)]"""

    def get_queue(self):
        """Return FIFO queue for storing batches."""
        return tf.queue.FIFOQueue(
            self.queue_cap,
            dtypes=[tf.float32] * len(self.queue_shape),
            shapes=self.queue_shape,
        )

    def preflight(self):
        """Run checks before kicking off the queue."""
        self.check_features()
        self.check_enhancement_factors()
        _ = self.check_shared_attr('sample_shape')

        sampler_bs = self.check_shared_attr('batch_size')
        msg = (
            f'Samplers have a different batch_size: {sampler_bs} than the '
            f'BatchQueue: {self.batch_size}'
        )
        assert sampler_bs == self.batch_size, msg

        if self.mode == 'eager':
            logger.info('Received mode = "eager".')
            _ = [c.compute() for c in self.containers]

    @property
    def queue_thread(self):
        """Get new queue thread."""
        if self._queue_thread is None or self._queue_thread._is_stopped:
            self._queue_thread = threading.Thread(
                target=self.enqueue_batches,
                name=self._thread_name,
            )
        return self._queue_thread

    def check_features(self):
        """Make sure all samplers have the same sets of features."""
        features = [list(c.features) for c in self.containers]
        msg = 'Received samplers with different sets of features.'
        assert all(feats == features[0] for feats in features), msg

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

    @abstractmethod
    def transform(self, samples, **kwargs):
        """Apply transform on batch samples. This can include smoothing /
        coarsening depending on the type of queue. e.g. coarsening could be
        included for a single dataset queue where low res samples are coarsened
        high res samples. For a dual dataset queue this will just include
        smoothing."""

    def post_proc(self, samples) -> Batch:
        """Performs some post proc on dequeued samples before sending out for
        training. Post processing can include coarsening on high-res data (if
        :class:`Collection` consists of :class:`Sampler` objects and not
        :class:`DualSampler` objects), smoothing, etc

        Returns
        -------
        Batch : namedtuple
             namedtuple with `low_res` and `high_res` attributes
        """
        lr, hr = self.transform(samples, **self.transform_kwargs)
        return self.Batch(low_res=lr, high_res=hr)

    def start(self) -> None:
        """Start thread to keep sample queue full for batches."""
        self._training_flag.set()
        if (
            not self.queue_thread.is_alive()
            and self.mode == 'lazy'
            and self.queue_cap > 0
        ):
            logger.info(f'Starting {self._thread_name} queue.')
            self.queue_thread.start()

    def stop(self) -> None:
        """Stop loading batches."""
        self._training_flag.clear()
        if self.queue_thread.is_alive():
            logger.info(f'Stopping {self._thread_name} queue.')
            self.queue_thread.join()

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self._batch_count = 0
        self.start()
        return self

    def get_batch(self) -> Batch:
        """Get batch from queue or directly from a ``Sampler`` through
        ``sample_batch``."""
        if (
            self.mode == 'eager'
            or self.queue_cap == 0
            or self.queue.size().numpy() == 0
        ):
            return self.sample_batch()
        return self.queue.dequeue()

    @property
    def running(self):
        """Boolean to check whether to keep enqueueing batches."""
        return (
            self._training_flag.is_set()
            and self.queue_thread.is_alive()
            and not self.queue.is_closed()
        )

    def enqueue_batches(self) -> None:
        """Callback function for queue thread. While training, the queue is
        checked for empty spots and filled. In the training thread, batches are
        removed from the queue."""
        log_time = time.time()
        while self.running:
            needed = self.queue_cap - self.queue.size().numpy()
            if needed == 1 or self.max_workers == 1:
                self.enqueue_batch()
            elif needed > 0:
                with ThreadPoolExecutor(self.max_workers) as exe:
                    _ = [exe.submit(self.enqueue_batch) for _ in range(needed)]
                logger.debug(
                    'Added %s enqueue futures to %s queue.',
                    needed,
                    self._thread_name,
                )
            if time.time() > log_time + 10:
                logger.debug(self.log_queue_info())
                log_time = time.time()

    def __next__(self) -> Batch:
        """Dequeue batch samples, squeeze if for a spatial only model, perform
        some post-proc like smoothing, coarsening, etc, and then send out for
        training as a namedtuple of low_res / high_res arrays.

        Returns
        -------
        batch : Batch
            Batch object with batch.low_res and batch.high_res attributes
        """
        if self._batch_count < self.n_batches:
            self.timer.start()
            samples = self.get_batch()
            if self.sample_shape[2] == 1:
                if isinstance(samples, (list, tuple)):
                    samples = tuple(s[..., 0, :] for s in samples)
                else:
                    samples = samples[..., 0, :]
            batch = self.post_proc(samples)
            self.timer.stop()
            self._batch_count += 1
            logger.debug(
                'Batch step %s finished in %s.',
                self._batch_count,
                self.timer.elapsed_str,
            )
        else:
            raise StopIteration
        return batch

    def get_container_index(self):
        """Get random container index based on weights"""
        indices = np.arange(0, len(self.containers))
        return RANDOM_GENERATOR.choice(indices, p=self.container_weights)

    def get_random_container(self):
        """Get random container based on container weights"""
        self.container_index = self.get_container_index()
        return self.containers[self.container_index]

    def sample_batch(self):
        """Get random sampler from collection and return a batch of samples
        from that sampler.

        Notes
        -----
        These samples are wrapped in an ``np.asarray`` call, so they have been
        loaded into memory.
        """
        return next(self.get_random_container())

    def log_queue_info(self):
        """Log info about queue size."""
        return '{} queue length: {} / {}.'.format(
            self._thread_name.title(),
            self.queue.size().numpy(),
            self.queue_cap,
        )

    def enqueue_batch(self):
        """Build batch and send to queue."""
        if self.running and self.queue.size().numpy() < self.queue_cap:
            self.queue.enqueue(self.sample_batch())

    @property
    def lr_shape(self):
        """Shape of low resolution sample in a low-res / high-res pair.  (e.g.
        (spatial_1, spatial_2, temporal, features))"""
        return (*self.lr_sample_shape, len(self.lr_features))

    @property
    def hr_shape(self):
        """Shape of high resolution sample in a low-res / high-res pair.  (e.g.
        (spatial_1, spatial_2, temporal, features))"""
        return (*self.hr_sample_shape, len(self.hr_features))
