"""Abstract batch queue class used for multi-threaded batching / training.

TODO: Setup distributed data handling so this can work with data distributed
over multiple nodes.
"""

import logging
import threading
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Dict, List, Optional, Tuple, Union
from warnings import warn

import numpy as np
import tensorflow as tf
from rex import safe_json_load

from sup3r.preprocessing.collections.base import Collection
from sup3r.preprocessing.samplers import DualSampler, Sampler
from sup3r.typing import T_Array
from sup3r.utilities.utilities import RANDOM_GENERATOR, Timer

logger = logging.getLogger(__name__)


class AbstractBatchQueue(Collection, ABC):
    """Abstract BatchQueue class. This class gets batches from a dataset
    generator and maintains a queue of batches in a dedicated thread so the
    training routine can proceed as soon as batches are available."""

    Batch = namedtuple('Batch', ['low_res', 'high_res'])

    def __init__(
        self,
        samplers: Union[List[Sampler], List[DualSampler]],
        batch_size: int = 16,
        n_batches: int = 64,
        s_enhance: int = 1,
        t_enhance: int = 1,
        means: Optional[Union[Dict, str]] = None,
        stds: Optional[Union[Dict, str]] = None,
        queue_cap: Optional[int] = None,
        transform_kwargs: Optional[dict] = None,
        max_workers: Optional[int] = None,
        default_device: Optional[str] = None,
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
        transform_kwargs : Union[Dict, None]
            Dictionary of kwargs to be passed to `self.transform`. This method
            performs smoothing / coarsening.
        max_workers : int
            Number of workers / threads to use for getting samples used to
            build batches.
        default_device : str
            Default device to use for batch queue (e.g. /cpu:0, /gpu:0). If
            None this will use the first GPU if GPUs are available otherwise
            the CPU.
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
        self._batch_counter = 0
        self._queue_thread = None
        self._default_device = default_device
        self._training_flag = threading.Event()
        self._thread_name = thread_name
        self.mode = mode
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.queue_cap = queue_cap or n_batches
        self.max_workers = max_workers or batch_size
        stats = self.get_stats(means=means, stds=stds)
        self.means, self.lr_means, self.hr_means = stats[:3]
        self.stds, self.lr_stds, self.hr_stds = stats[3:]
        self.container_index = self.get_container_index()
        self.queue = self.get_queue()
        self.batches = self.prep_batches()
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

    @property
    @abstractmethod
    def output_signature(self):
        """Signature of tensors returned by the queue. e.g. single
        TensorSpec(shape, dtype, name) for single dataset queues or tuples of
        TensorSpec for dual queues."""

    def get_queue(self):
        """Return FIFO queue for storing batches."""
        return tf.queue.FIFOQueue(
            self.queue_cap,
            dtypes=[tf.float32] * len(self.queue_shape),
            shapes=self.queue_shape,
        )

    def preflight(self):
        """Get data generator and run checks before kicking off the queue."""
        gpu_list = tf.config.list_physical_devices('GPU')
        self._default_device = self._default_device or (
            '/cpu:0' if len(gpu_list) == 0 else '/gpu:0'
        )
        msg = (
            'Queue cap needs to be at least 1 when batching in "lazy" mode, '
            f'but received queue_cap = {self.queue_cap}.'
        )
        assert self.mode == 'eager' or (
            self.queue_cap > 0 and self.mode == 'lazy'
        ), msg
        self.timer(self.check_features, log=True)()
        self.timer(self.check_enhancement_factors, log=True)()
        _ = self.check_shared_attr('sample_shape')
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
        features = [list(c.data.data_vars) for c in self.containers]
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

    def prep_batches(self):
        """Return iterable of batches prefetched from the data generator.

        TODO: Understand this better. Should prefetch be called more than just
        for initialization? Every epoch?
        """
        logger.debug(
            f'Prefetching {self._thread_name} batches with batch_size = '
            f'{self.batch_size}.'
        )
        with tf.device(self._default_device):
            data = tf.data.Dataset.from_generator(
                self.generator, output_signature=self.output_signature
            )
            data = self._parallel_map(data)
            data = data.prefetch(tf.data.AUTOTUNE)
            batches = data.batch(
                self.batch_size,
                drop_remainder=True,
                deterministic=False,
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        return batches.as_numpy_iterator()

    def generator(self):
        """Generator over samples. The samples are retrieved with the
        :meth:`get_samples` method through randomly selecting a sampler from
        the collection and then returning a sample from that sampler. Batches
        are constructed from a set (`batch_size`) of these samples.

        Returns
        -------
        samples : T_Array | Tuple[T_Array, T_Array]
            (lats, lons, times, n_features)
            Either an array or a 2-tuple of such arrays (in the case of queues
            with :class:`DualSampler` samplers.) These arrays are queued in a
            background thread and then dequeued during training.
        """
        while self._training_flag.is_set():
            yield self.get_samples()

    @abstractmethod
    def _parallel_map(self, data: tf.data.Dataset):
        """Perform call to map function to enable parallel sampling."""

    @abstractmethod
    def transform(self, samples, **kwargs):
        """Apply transform on batch samples. This can include smoothing /
        coarsening depending on the type of queue. e.g. coarsening could be
        included for a single dataset queue where low res samples are coarsened
        high res samples. For a dual dataset queue this will just include
        smoothing."""

    def _post_proc(self, samples) -> Batch:
        """Performs some post proc on dequeued samples before sending out for
        training. Post processing can include normalization, coarsening on
        high-res data (if :class:`Collection` consists of :class:`Sampler`
        objects and not :class:`DualSampler` objects), smoothing, etc

        Returns
        -------
        Batch : namedtuple
             namedtuple with `low_res` and `high_res` attributes
        """
        lr, hr = self.transform(samples, **self.transform_kwargs)
        lr, hr = self.normalize(lr, hr)
        return self.Batch(low_res=lr, high_res=hr)

    def start(self) -> None:
        """Start thread to keep sample queue full for batches."""
        self._training_flag.set()
        if not self.queue_thread.is_alive() and self.mode == 'lazy':
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
        self._batch_counter = 0
        self.timer(self.start)()
        return self

    def _enqueue_batch(self) -> None:
        batch = next(self.batches, None)
        if batch is not None:
            self.timer(self.queue.enqueue, log=True)(batch)
            msg = (
                f'{self._thread_name.title()} queue length: '
                f'{self.queue.size().numpy()} / {self.queue_cap}'
            )
            logger.debug(msg)

    def _get_batch(self) -> Batch:
        if self.mode == 'eager':
            return next(self.batches)
        return self.timer(self.queue.dequeue, log=True)()

    def enqueue_batches(self) -> None:
        """Callback function for queue thread. While training, the queue is
        checked for empty spots and filled. In the training thread, batches are
        removed from the queue."""
        try:
            while self._training_flag.is_set():
                if self.queue.size().numpy() < self.queue_cap:
                    self._enqueue_batch()
        except KeyboardInterrupt:
            logger.info(f'Stopping {self._thread_name.title()} queue.')
            self.stop()

    def __next__(self) -> Batch:
        """Dequeue batch samples, squeeze if for a spatial only model, perform
        some post-proc like normalization, smoothing, coarsening, etc, and then
        send out for training as a namedtuple of low_res / high_res arrays.

        Returns
        -------
        batch : Batch
            Batch object with batch.low_res and batch.high_res attributes
        """
        if self._batch_counter < self.n_batches:
            samples = self.timer(self._get_batch, log=True)()
            if self.sample_shape[2] == 1:
                if isinstance(samples, (list, tuple)):
                    samples = tuple(s[..., 0, :] for s in samples)
                else:
                    samples = samples[..., 0, :]
            batch = self.timer(self._post_proc, log=True)(samples)
            self._batch_counter += 1
        else:
            raise StopIteration
        return batch

    @staticmethod
    def _get_stats(means, stds, features):
        msg = (f'Some of the features: {features} not found in the provided '
               f'means: {means}')
        assert all(f in means for f in features), msg
        msg = (f'Some of the features: {features} not found in the provided '
               f'stds: {stds}')
        assert all(f in stds for f in features), msg
        f_means = np.array([means[k] for k in features]).astype(np.float32)
        f_stds = np.array([stds[k] for k in features]).astype(np.float32)
        return f_means, f_stds

    def get_stats(self, means, stds):
        """Get means / stds from given files / dicts and group these into
        low-res / high-res stats."""
        means = means if isinstance(means, dict) else safe_json_load(means)
        stds = stds if isinstance(stds, dict) else safe_json_load(stds)
        msg = (
            f'Received means = {means} with self.features = '
            f'{self.features}. Make sure the means are valid, since they '
            'clearly come from a different training run.'
        )

        if len(means) != len(self.features):
            logger.warning(msg)
            warn(msg)
        msg = (
            f'Received stds = {stds} with self.features = '
            f'{self.features}. Make sure the stds are valid, since they '
            'clearly come from a different training run.'
        )
        if len(stds) != len(self.features):
            logger.warning(msg)
            warn(msg)

        lr_means, lr_stds = self._get_stats(means, stds, self.lr_features)
        hr_means, hr_stds = self._get_stats(means, stds, self.hr_features)
        return means, lr_means, hr_means, stds, lr_stds, hr_stds

    @staticmethod
    def _normalize(array, means, stds):
        """Normalize an array with given means and stds."""
        return (array - means) / stds

    def normalize(self, lr, hr) -> Tuple[T_Array, T_Array]:
        """Normalize a low-res / high-res pair with the stored means and
        stdevs."""
        return (
            self._normalize(lr, self.lr_means, self.lr_stds),
            self._normalize(hr, self.hr_means, self.hr_stds),
        )

    def get_container_index(self):
        """Get random container index based on weights"""
        indices = np.arange(0, len(self.containers))
        return RANDOM_GENERATOR.choice(indices, p=self.container_weights)

    def get_random_container(self):
        """Get random container based on container weights

        TODO: This will select a random container for every sample, instead of
        every batch. Should we override this in the BatchHandler and use
        the batch_counter to do every batch?
        """
        self.container_index = self.get_container_index()
        return self.containers[self.container_index]

    def get_samples(self):
        """Get random sampler from collection and return a sample from that
        sampler."""
        return next(self.get_random_container())

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
