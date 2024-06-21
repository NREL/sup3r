"""Data centric batch handler for dynamic batching strategy based on
performance during training"""

import logging

import numpy as np

from .base import SingleBatchQueue

logger = logging.getLogger(__name__)


class BatchQueueDC(SingleBatchQueue):
    """Sample from data based on spatial and temporal weights. These weights
    can be derived from validation training losses and updated during training
    or set a priori to construct a validation queue"""

    def __init__(self, *args, n_space_bins=1, n_time_bins=1, **kwargs):
        self.n_space_bins = n_space_bins
        self.n_time_bins = n_time_bins
        self._spatial_weights = np.ones(n_space_bins) / n_space_bins
        self._temporal_weights = np.ones(n_time_bins) / n_time_bins
        super().__init__(*args, **kwargs)

    def get_samples(self):
        """Update weights and get sample from sampled container."""
        sampler = self.get_random_container()
        sampler.update_weights(self.spatial_weights, self.temporal_weights)
        return next(sampler)

    @property
    def spatial_weights(self):
        """Get weights used to sample spatial bins."""
        return self._spatial_weights

    @property
    def temporal_weights(self):
        """Get weights used to sample temporal bins."""
        return self._temporal_weights

    def update_spatial_weights(self, value):
        """Set weights used to sample spatial bins. This is called by
        :class:`Sup3rGanDC` after an epoch to update weights based on model
        performance across validation samples."""
        self._spatial_weights = value

    def update_temporal_weights(self, value):
        """Set weights used to sample temporal bins. This is called by
        :class:`Sup3rGanDC` after an epoch to update weights based on model
        performance across validation samples."""
        self._temporal_weights = value


class ValBatchQueueDC(BatchQueueDC):
    """Queue to construct a single batch for each spatiotemporal validation
    bin. e.g. If we have 4 time bins and 1 space bin this will get `batch_size`
    samples for 4 batches, with `batch_size` samples from each bin. The model
    performance across these batches will determine the weights for how the
    training batch queue is sampled."""

    def __init__(self, *args, n_space_bins=1, n_time_bins=1, **kwargs):
        super().__init__(
            *args, n_space_bins=n_space_bins, n_time_bins=n_time_bins, **kwargs
        )
        self.n_batches = n_space_bins * n_time_bins

    @property
    def spatial_weights(self):
        """Sample entirely from this spatial bin determined by the batch
        number."""
        self._spatial_weights = np.eye(
            1,
            self.n_space_bins,
            self._batch_counter % self.n_space_bins,
            dtype=np.float32,
        )[0]
        return self._spatial_weights

    @property
    def temporal_weights(self):
        """Sample entirely from this temporal bin determined by the batch
        number."""
        self._temporal_weights = np.eye(
            1,
            self.n_time_bins,
            self._batch_counter % self.n_time_bins,
            dtype=np.float32,
        )[0]
        return self._temporal_weights
