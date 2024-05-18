"""Base objects which generate, build, and operate on batches. Also can
interface with models."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import tensorflow as tf

from sup3r.containers.batchers.base import BatchQueue
from sup3r.containers.samplers import SamplerPair

logger = logging.getLogger(__name__)


option_no_order = tf.data.Options()
option_no_order.experimental_deterministic = False

option_no_order.experimental_optimization.noop_elimination = True
option_no_order.experimental_optimization.apply_default_optimizations = True


class PairBatchQueue(BatchQueue):
    """Base BatchQueue for SamplerPair containers."""

    def __init__(
        self,
        train_containers: List[SamplerPair],
        batch_size,
        n_batches,
        s_enhance,
        t_enhance,
        means: Union[Dict, str],
        stds: Union[Dict, str],
        val_containers: Optional[List[SamplerPair]] = None,
        queue_cap=None,
        max_workers=None,
        default_device: Optional[str] = None,
    ):
        super().__init__(
            train_containers=train_containers,
            batch_size=batch_size,
            n_batches=n_batches,
            s_enhance=s_enhance,
            t_enhance=t_enhance,
            means=means,
            stds=stds,
            val_containers=val_containers,
            queue_cap=queue_cap,
            max_workers=max_workers,
            default_device=default_device
        )
        self.check_enhancement_factors()

    def check_enhancement_factors(self):
        """Make sure each SamplerPair has the same enhancment factors and they
        match those provided to the BatchQueue."""

        s_factors = [c.s_enhance for c in self.containers]
        msg = (
            f'Received s_enhance = {self.s_enhance} but not all '
            f'SamplerPairs in the collection have the same value.'
        )
        assert all(self.s_enhance == s for s in s_factors), msg
        t_factors = [c.t_enhance for c in self.containers]
        msg = (
            f'Recived t_enhance = {self.t_enhance} but not all '
            f'SamplerPairs in the collection have the same value.'
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
