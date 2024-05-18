"""Sampler objects. These take in data objects / containers and can them sample
from them. These samples can be used to build batches."""

import logging

from sup3r.containers.samplers.abstract import (
    AbstractSampler,
)
from sup3r.utilities.utilities import uniform_box_sampler, uniform_time_sampler

logger = logging.getLogger(__name__)


class Sampler(AbstractSampler):
    """Base sampler class."""

    def get_sample_index(self):
        """Randomly gets spatial sample and time sample

        Parameters
        ----------
        data_shape : tuple
            Size of available region for sampling
            (spatial_1, spatial_2, temporal)
        sample_shape : tuple
            Size of observation to sample
            (spatial_1, spatial_2, temporal)

        Returns
        -------
        sample_index : tuple
            Tuple of sampled spatial grid, time slice, and features indices.
            Used to get single observation like self.data[sample_index]
        """
        spatial_slice = uniform_box_sampler(self.shape, self.sample_shape[:2])
        time_slice = uniform_time_sampler(self.shape, self.sample_shape[2])
        return (*spatial_slice, time_slice, slice(None))
