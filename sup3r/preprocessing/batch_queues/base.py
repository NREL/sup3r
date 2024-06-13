"""Base objects which generate, build, and operate on batches. Also can
interface with models."""

import logging

import tensorflow as tf

from sup3r.preprocessing.batch_queues.abstract import (
    AbstractBatchQueue,
)
from sup3r.preprocessing.batch_queues.utilities import smooth_data
from sup3r.utilities.utilities import (
    spatial_coarsening,
    temporal_coarsening,
)

logger = logging.getLogger(__name__)


option_no_order = tf.data.Options()
option_no_order.experimental_deterministic = False

option_no_order.experimental_optimization.noop_elimination = True
option_no_order.experimental_optimization.apply_default_optimizations = True


class SingleBatchQueue(AbstractBatchQueue):
    """Base BatchQueue class for single dataset containers"""

    @property
    def queue_shape(self):
        """Shape of objects stored in the queue."""
        return [(self.batch_size, *self.hr_shape)]

    @property
    def output_signature(self):
        """Signature of tensors returned by the queue."""
        return tf.TensorSpec(self.hr_shape, tf.float32, name='high_res')

    def transform(
        self,
        samples,
        smoothing=None,
        smoothing_ignore=None,
        temporal_coarsening_method='subsample',
    ):
        """Coarsen high res data to get corresponding low res batch.

        Parameters
        ----------
        samples : T_Array
            High resolution batch of samples.
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        smoothing : float | None
            Standard deviation to use for gaussian filtering of the coarse
            data. This can be tuned by matching the kinetic energy of a low
            resolution simulation with the kinetic energy of a coarsened and
            smoothed high resolution simulation. If None no smoothing is
            performed.
        smoothing_ignore : list | None
            List of features to ignore for the smoothing filter. None will
            smooth all features if smoothing kwarg is not None
        temporal_coarsening_method : str
            Method to use for temporal coarsening. Can be subsample, average,
            min, max, or total

        Returns
        -------
        low_res : T_Array
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        high_res : T_Array
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        """
        low_res = spatial_coarsening(samples, self.s_enhance)
        low_res = (
            low_res
            if self.t_enhance == 1
            else temporal_coarsening(
                low_res, self.t_enhance, temporal_coarsening_method
            )
        )
        smoothing_ignore = (
            smoothing_ignore if smoothing_ignore is not None else []
        )
        low_res = smooth_data(
            low_res, self.features, smoothing_ignore, smoothing
        )
        high_res = samples.numpy()[..., self.hr_features_ind]
        return low_res, high_res

    def _parallel_map(self):
        """Perform call to map function for single dataset containers to enable
        parallel sampling."""
        return self.data_gen.map(
            lambda x: x, num_parallel_calls=self.max_workers
        )
