"""Abstract batch queue class used for multi-threaded batching / training.

TODO: Setup distributed data handling so this can work with data in memory but
distributed over multiple nodes.
"""

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from rex import safe_json_load

from sup3r.preprocessing.collections.samplers import SamplerCollection
from sup3r.preprocessing.samplers import DualSampler, Sampler
from sup3r.typing import T_Array
from sup3r.utilities.utilities import Timer

logger = logging.getLogger(__name__)


@dataclass
class Batch:
    """Basic single batch object, containing low_res and high_res data

    Parameters
    ----------
    low_res : T_Array
        4D | 5D array
        (batch_size, spatial_1, spatial_2, features)
        (batch_size, spatial_1, spatial_2, temporal, features)
    high_res : T_Array
        4D | 5D array
        (batch_size, spatial_1, spatial_2, features)
        (batch_size, spatial_1, spatial_2, temporal, features)
    """

    low_res: T_Array
    high_res: T_Array

    def __post_init__(self):
        self.shape = (self.low_res.shape, self.high_res.shape)

    def __len__(self):
        """Get the number of samples in this batch."""
        return len(self.low_res)


class AbstractBatchQueue(SamplerCollection, ABC):
    """Abstract BatchQueue class. This class gets batches from a dataset
    generator and maintains a queue of batches in a dedicated thread so the
    training routine can proceed as soon as batches are available."""

    BATCH_CLASS = Batch

    def __init__(
        self,
        samplers: Union[List[Sampler], List[DualSampler]],
        batch_size: Optional[int] = 16,
        n_batches: Optional[int] = 64,
        s_enhance: Optional[int] = 1,
        t_enhance: Optional[int] = 1,
        means: Optional[Union[Dict, str]] = None,
        stds: Optional[Union[Dict, str]] = None,
        queue_cap: Optional[int] = None,
        transform_kwargs: Optional[dict] = None,
        max_workers: Optional[int] = None,
        default_device: Optional[str] = None,
        thread_name: Optional[str] = 'training',
        mode: Optional[str] = 'lazy',
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
        super().__init__(
            samplers=samplers, s_enhance=s_enhance, t_enhance=t_enhance
        )
        self._batch_counter = 0
        self._queue = None
        self._queue_thread = None
        self._default_device = default_device
        self._running_queue = threading.Event()
        self.batches = None
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.queue_cap = queue_cap or n_batches
        self.max_workers = max_workers or batch_size
        stats = self.get_stats(means=means, stds=stds)
        self.means, self.lr_means, self.hr_means = stats[:3]
        self.stds, self.lr_stds, self.hr_stds = stats[3:]
        self.transform_kwargs = transform_kwargs or {
            'smoothing_ignore': [],
            'smoothing': None,
        }
        self.timer = Timer()
        self.preflight(mode=mode, thread_name=thread_name)

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

    def preflight(self, mode='lazy', thread_name='training'):
        """Get data generator and run checks before kicking off the queue."""
        gpu_list = tf.config.list_physical_devices('GPU')
        self._default_device = self._default_device or (
            '/cpu:0' if len(gpu_list) == 0 else '/gpu:0'
        )
        self.init_queue(thread_name=thread_name)
        self.batches = self.prep_batches()
        self.check_stats()
        self.check_features()
        self.check_enhancement_factors()
        if mode == 'eager':
            self.compute()

    def init_queue(self, thread_name='training'):
        """Define FIFO queue for storing batches and the thread to use for
        adding / removing from the queue during training."""
        dtypes = [tf.float32] * len(self.queue_shape)
        self._queue = tf.queue.FIFOQueue(
            self.queue_cap, dtypes=dtypes, shapes=self.queue_shape
        )
        self._queue_thread = threading.Thread(
            target=self.enqueue_batches,
            name=thread_name,
        )

    def check_features(self):
        """Make sure all samplers have the same sets of features."""
        features = [list(c.data.data_vars) for c in self.containers]
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

    def prep_batches(self):
        """Return iterable of batches prefetched from the data generator.

        TODO: Understand this better. Should prefetch be called more than just
        for initialization? Every epoch?
        """
        logger.debug(
            f'Prefetching {self._queue_thread.name} batches with batch_size = '
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
        """Generator over samples. The samples are retreived with the
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
        while self._running_queue.is_set():
            samples = self.get_samples()
            if not self.loaded:
                samples = (
                    tuple(sample.compute() for sample in samples)
                    if isinstance(samples, tuple)
                    else samples.compute()
                )
            yield samples

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

    def post_dequeue(self, samples) -> Batch:
        """Performs some post proc on dequeued samples before sending out for
        training. Post processing can include normalization, coarsening on
        high-res data (if :class:`Collection` consists of :class:`Sampler`
        objects and not :class:`DualSampler` objects), smoothing, etc

        Returns
        -------
        Batch
            Simple Batch object with `low_res` and `high_res` attributes
        """
        lr, hr = self.transform(samples, **self.transform_kwargs)
        lr, hr = self.normalize(lr, hr)
        return self.BATCH_CLASS(low_res=lr, high_res=hr)

    def start(self) -> None:
        """Start thread to keep sample queue full for batches."""
        if not self._queue_thread.is_alive():
            logger.info(f'Starting {self._queue_thread.name} queue.')
            self._running_queue.set()
            self._queue_thread.start()

    def stop(self) -> None:
        """Stop loading batches."""
        if self._queue_thread.is_alive():
            logger.info(f'Stopping {self._queue_thread.name} queue.')
            self._running_queue.clear()
            self._queue_thread.join()

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self._batch_counter = 0
        self.start()
        return self

    def enqueue_batches(self) -> None:
        """Callback function for queue thread. While training the queue is
        checked for empty spots and filled. In the training thread, batches are
        removed from the queue."""
        try:
            while self._running_queue.is_set():
                if self._queue.size().numpy() < self.queue_cap:
                    batch = next(self.batches, None)
                    if batch is not None:
                        self.timer(self._queue.enqueue, log=True)(batch)
        except KeyboardInterrupt:
            logger.info(
                f'Attempting to stop {self._queue.thread.name} batch queue.'
            )
            self.stop()

    def __next__(self) -> Batch:
        """Dequeue batch samples, squeeze if for a spatial only model, perform
        some post-proc like normalization, smoothing, coarsening, etc, and then
        send out for training as a :class:`Batch` object.

        Returns
        -------
        batch : Batch
            Batch object with batch.low_res and batch.high_res attributes
        """
        if self._batch_counter < self.n_batches:
            queue_size = self._queue.size().numpy()
            msg = (
                f'{queue_size} {"batch" if queue_size == 1 else "batches"}'
                f' in {self._queue_thread.name} queue.'
            )
            logger.debug(msg)
            samples = self.timer(self._queue.dequeue, log=True)()
            if self.sample_shape[2] == 1:
                if isinstance(samples, (list, tuple)):
                    samples = tuple([s[..., 0, :] for s in samples])
                else:
                    samples = samples[..., 0, :]
            batch = self.timer(self.post_dequeue, log=True)(samples)
            self._batch_counter += 1
        else:
            raise StopIteration
        return batch

    def get_stats(self, means, stds):
        """Get means / stds from given files / dicts and group these into
        low-res / high-res stats."""
        means = means if isinstance(means, dict) else safe_json_load(means)
        stds = stds if isinstance(stds, dict) else safe_json_load(stds)
        msg = f'Received means = {means} with self.features = {self.features}.'
        assert len(means) == len(self.features), msg
        msg = f'Received stds = {stds} with self.features = {self.features}.'
        assert len(stds) == len(self.features), msg

        lr_means = np.array([means[k] for k in self.lr_features]).astype(
            np.float32
        )
        hr_means = np.array([means[k] for k in self.hr_features]).astype(
            np.float32
        )
        lr_stds = np.array([stds[k] for k in self.lr_features]).astype(
            np.float32
        )
        hr_stds = np.array([stds[k] for k in self.hr_features]).astype(
            np.float32
        )
        return means, lr_means, hr_means, stds, lr_stds, hr_stds

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
