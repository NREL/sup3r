"""Base objects which generate, build, and operate on batches. Also can
interface with models."""

import logging

import tensorflow as tf
from scipy.ndimage import gaussian_filter

from sup3r.preprocessing.batch_queues.abstract import AbstractBatchQueue

logger = logging.getLogger(__name__)


option_no_order = tf.data.Options()
option_no_order.experimental_deterministic = False

option_no_order.experimental_optimization.noop_elimination = True
option_no_order.experimental_optimization.apply_default_optimizations = True


class DualBatchQueue(AbstractBatchQueue):
    """Base BatchQueue for DualSampler containers."""

    def __init__(self, *args, **kwargs):
        """
        See Also
        --------
        :class:`AbstractBatchQueue` for argument descriptions.
        """
        super().__init__(*args, **kwargs)
        self.check_enhancement_factors()

    @property
    def queue_shape(self):
        """Shape of objects stored in the queue."""
        return [
            (self.batch_size, *self.lr_shape),
            (self.batch_size, *self.hr_shape),
        ]

    @property
    def output_signature(self):
        """Signature of tensors returned by the queue."""
        return (
            tf.TensorSpec(self.lr_shape, tf.float32, name='low_res'),
            tf.TensorSpec(self.hr_shape, tf.float32, name='high_res'),
        )

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

    def _parallel_map(self):
        """Perform call to map function for dual containers to enable parallel
        sampling."""
        return self.data_gen.map(
            lambda x, y: (x, y), num_parallel_calls=self.max_workers
        )

    def transform(self, samples, smoothing=None, smoothing_ignore=None):
        """Perform smoothing if requested.

        Note
        ----
        This does not include temporal or spatial coarsening like
        :class:`SingleBatchQueue`
        """
        low_res, high_res = samples

        if smoothing is not None:
            feat_iter = [
                j
                for j in range(low_res.shape[-1])
                if self.features[j] not in smoothing_ignore
            ]
            for i in range(low_res.shape[0]):
                for j in feat_iter:
                    low_res[i, ..., j] = gaussian_filter(
                        low_res[i, ..., j], smoothing, mode='nearest'
                    )
        return low_res, high_res
