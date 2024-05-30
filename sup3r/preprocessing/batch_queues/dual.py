"""Base objects which generate, build, and operate on batches. Also can
interface with models."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import tensorflow as tf

from sup3r.preprocessing.batch_queues.abstract import AbstractBatchQueue
from sup3r.preprocessing.samplers import DualSampler

logger = logging.getLogger(__name__)


option_no_order = tf.data.Options()
option_no_order.experimental_deterministic = False

option_no_order.experimental_optimization.noop_elimination = True
option_no_order.experimental_optimization.apply_default_optimizations = True


class DualBatchQueue(AbstractBatchQueue):
    """Base BatchQueue for DualSampler containers."""

    def __init__(
        self,
        samplers: List[DualSampler],
        batch_size,
        n_batches,
        s_enhance,
        t_enhance,
        means: Union[Dict, str],
        stds: Union[Dict, str],
        queue_cap=None,
        max_workers=None,
        default_device: Optional[str] = None,
        thread_name: Optional[str] = "training"
    ):
        super().__init__(
            samplers=samplers,
            batch_size=batch_size,
            n_batches=n_batches,
            s_enhance=s_enhance,
            t_enhance=t_enhance,
            means=means,
            stds=stds,
            queue_cap=queue_cap,
            max_workers=max_workers,
            default_device=default_device,
            thread_name=thread_name
        )
        self.check_enhancement_factors()
        self.queue_shape = [
            (self.batch_size, *self.lr_shape),
            (self.batch_size, *self.hr_shape),
        ]

    def check_enhancement_factors(self):
        """Make sure each DualSampler has the same enhancment factors and they
        match those provided to the BatchQueue."""

        s_factors = [c.s_enhance for c in self.containers]
        msg = (
            f'Received s_enhance = {self.s_enhance} but not all '
            f'DualSamplers in the collection have the same value.'
        )
        assert all(self.s_enhance == s for s in s_factors), msg
        t_factors = [c.t_enhance for c in self.containers]
        msg = (
            f'Recived t_enhance = {self.t_enhance} but not all '
            f'DualSamplers in the collection have the same value.'
        )
        assert all(self.t_enhance == t for t in t_factors), msg

    def get_output_signature(self) -> Tuple[tf.TensorSpec, tf.TensorSpec]:
        """Get tensorflow dataset output signature. If we are sampling from
        container pairs then this is a tuple for low / high res batches.
        Otherwise we are just getting high res batches and coarsening to get
        the corresponding low res batches."""
        return (
            tf.TensorSpec(self.lr_shape, tf.float32, name='low_res'),
            tf.TensorSpec(self.hr_shape, tf.float32, name='high_res'),
        )

    def batch_next(self, samples):
        """Returns wrapped collection of samples / observations."""
        lr, hr = samples
        lr, hr = self.normalize(lr, hr)
        return self.BATCH_CLASS(low_res=lr, high_res=hr)

    def _parallel_map(self):
        """Perform call to map function for dual containers to enable parallel
        sampling."""
        return self.data.map(
            lambda x, y: (x, y), num_parallel_calls=self.max_workers
        )

    def _get_queue_shape(self) -> List[tuple]:
        """Get shape for DualSampler queue."""
        return [
            (self.batch_size, *self.lr_shape),
            (self.batch_size, *self.hr_shape),
        ]
