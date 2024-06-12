"""Data centric batch handler for dynamic batching strategy based on
performance during training"""

import logging

import numpy as np

from sup3r.preprocessing.batch_queues.base import SingleBatchQueue

logger = logging.getLogger(__name__)


class BatchQueueDC(SingleBatchQueue):
    """Sample from data based on spatial and temporal weights. These weights
    can be derived from validation training losses and updated during training
    or set a priori to construct a validation queue"""

    def __init__(self, *args, n_space_bins=1, n_time_bins=1, **kwargs):
        self.spatial_weights = np.ones(n_space_bins) / n_space_bins
        self.temporal_weights = np.ones(n_time_bins) / n_time_bins
        super().__init__(*args, **kwargs)

    def __getitem__(self, keys):
        """Update weights and get sample from sampled container."""
        sampler = self.get_random_container()
        sampler.update_weights(self.spatial_weights, self.temporal_weights)
        return next(sampler)


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
        self.n_space_bins = n_space_bins
        self.n_time_bins = n_time_bins
        self.n_batches = n_space_bins * n_time_bins

    @property
    def spatial_weights(self):
        """Sample entirely from this spatial bin determined by the batch
        number."""
        weights = np.zeros(self.n_space_bins)
        weights[self._batch_counter % self.n_space_bins] = 1

    @property
    def temporal_weights(self):
        """Sample entirely from this temporal bin determined by the batch
        number."""
        weights = np.zeros(self.n_time_bins)
        weights[self._batch_counter % self.n_time_bins] = 1
