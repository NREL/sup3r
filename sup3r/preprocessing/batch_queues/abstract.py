"""Abstract batch queue class used for multi-threaded batching / training.

TODO:
    (1) Figure out apparent "blocking" issue with threaded enqueue batches.
        max_workers=1 is the fastest?
    (2) Setup distributed data handling so this can work with data distributed
        over multiple nodes.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import tensorflow as tf

from sup3r.preprocessing.base import DsetTuple
from sup3r.preprocessing.collections.base import Collection
from sup3r.utilities.utilities import RANDOM_GENERATOR, Timer

if TYPE_CHECKING:
    from sup3r.preprocessing.samplers import DualSampler, Sampler

logger = logging.getLogger(__name__)


class AbstractBatchQueue(Collection, ABC):
    """Abstract BatchQueue class. This class gets batches from a dataset
    generator and maintains a queue of batches in a dedicated thread so the
    training routine can proceed as soon as batches are available."""

    BATCH_MEMBERS = ('low_res', 'high_res')

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
        verbose: bool = False,
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
        verbose : bool
            Whether to log timing information for batch steps.
        """
        msg = (
            f'{self.__class__.__name__} requires a list of samplers. '
            f'Received type {type(samplers)}'
        )
        assert isinstance(samplers, list), msg
        super().__init__(containers=samplers)
        self._batch_count = 0
        self._thread_name = thread_name
        self.running = False
        self.queue_cap = queue_cap or tf.data.AUTOTUNE
        self.mode = mode
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.max_workers = max_workers
        self.container_index = self.get_container_index()
        self.queue = None
        self.lr_sample_shape = (
            self.hr_sample_shape[0] // s_enhance,
            self.hr_sample_shape[1] // s_enhance,
            self.hr_sample_shape[2] // t_enhance,
        )
        self.transform_kwargs = transform_kwargs or {
            'smoothing_ignore': [],
            'smoothing': None,
        }
        self.verbose = verbose
        self.timer = Timer()
        self.preflight()

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

    def start(self) -> None:
        """Start thread to keep sample queue full for batches."""
        self.running = True
        self.queue = self.prefetch()
        logger.info(f'Starting {self._thread_name} thread.')

    def stop(self) -> None:
        """Stop loading batches."""
        self.running = False
        logger.info(f'Stopping {self._thread_name} thread.')

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self._batch_count = 0
        if not self.running:
            self.start()
        self.queue = self.prefetch()
        return self

    def prefetch(self):
        """Prefetch batches"""
        lr_shape, hr_shape = self.shapes
        output_signature = (
            tf.TensorSpec(lr_shape, tf.float32),
            tf.TensorSpec(hr_shape, tf.float32),
        )

        def worker_ds(_):
            return tf.data.Dataset.from_generator(
                self.gen, output_signature=output_signature
            )

        return (
            tf.data.Dataset.range(self.max_workers)
            .interleave(
                worker_ds,
                cycle_length=self.max_workers,
                num_parallel_calls=self.max_workers,
                deterministic=False,
            )
            .prefetch(self.queue_cap)
        )

    def get_batch(self):
        """Get samples from queue and perform any extra processing needed."""
        samples = next(iter(self.queue))
        return DsetTuple(**dict(zip(self.BATCH_MEMBERS, samples)))

    def __next__(self) -> DsetTuple:
        """Dequeue batch samples, squeeze if for a spatial only model, perform
        some post-proc like smoothing, coarsening, etc, and then send out for
        training as a namedtuple-like object of low_res / high_res arrays.

        Returns
        -------
        batch : Batch
            Batch object with batch.low_res and batch.high_res attributes
        """
        if self._batch_count < self.n_batches:
            self.timer.start()
            batch = self.get_batch()
            self.timer.stop()
            self._batch_count += 1
            if self.verbose:
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

    def _get_samples(self):
        """Get random sampler and return batch of samples from that sampler."""
        return next(self.get_random_container())

    def gen(self):
        """Get batch of samples, transform as needed, and yield low_res,
        high_res batch pair. This is the generator for
        ``tf.data.Dataset.from_generator``

        Notes
        -----
        These samples are wrapped in an ``np.asarray`` call, so they have been
        loaded into memory.
        """
        while self.running:
            out = self._get_samples()
            if not isinstance(out, tuple):
                samples = tf.convert_to_tensor(out, dtype=tf.float32)
            else:
                samples = tuple(
                    tf.convert_to_tensor(o, dtype=tf.float32) for o in out
                )
            if self.sample_shape[2] == 1:
                if isinstance(samples, (list, tuple)):
                    samples = tuple(s[..., 0, :] for s in samples)
                else:
                    samples = samples[..., 0, :]
            lr, hr = self.transform(samples, **self.transform_kwargs)
            yield lr, hr

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

    @property
    def shapes(self):
        """Shapes of batches returned by ``__next__``"""
        lr_shape, hr_shape = self.lr_shape, self.hr_shape
        if self.sample_shape[2] == 1:
            lr_shape = lr_shape[:2] + (lr_shape[-1],)
            hr_shape = hr_shape[:2] + (hr_shape[-1],)
        return (self.batch_size, *lr_shape), (self.batch_size, *hr_shape)
