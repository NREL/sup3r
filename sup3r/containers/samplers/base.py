import logging
from typing import List, Tuple

import numpy as np

from sup3r.containers.base import Container, ContainerPair
from sup3r.containers.samplers.abstract import (
    AbstractCollectionSampler,
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
        temporal_slice = uniform_time_sampler(self.shape, self.sample_shape[2])
        return (*spatial_slice, temporal_slice, slice(None))


class SamplerPair(ContainerPair, AbstractSampler):
    """Pair of sampler objects, one for low resolution and one for high
    resolution."""

    def __init__(self, lr_container: Sampler, hr_container: Sampler):
        self.lr_container = lr_container
        self.hr_container = hr_container
        self.s_enhance, self.t_enhance = self.get_enhancement_factors()

    def get_enhancement_factors(self):
        """Compute spatial / temporal enhancement factors based on relative
        shapes of the low / high res containers."""
        lr_shape, hr_shape = self.sample_shape
        s_enhance = hr_shape[0] / lr_shape[0]
        t_enhance = hr_shape[2] / lr_shape[2]
        return s_enhance, t_enhance

    @property
    def sample_shape(self) -> Tuple[tuple, tuple]:
        """Shape of the data sample to select when `get_next()` is called."""
        return (self.lr_container.sample_shape, self.hr_container.sample_shape)

    def get_sample_index(self) -> Tuple[tuple, tuple]:
        """Get paired sample index, consisting of index for the low res sample
        and the index for the high res sample with the same spatiotemporal
        extent."""
        lr_index = self.lr_container.get_sample_index()
        hr_index = [slice(s.start * self.s_enhance, s.stop * self.s_enhance)
                    for s in lr_index[:2]]
        hr_index += [slice(s.start * self.t_enhance, s.stop * self.t_enhance)
                     for s in lr_index[2:-1]]
        hr_index += [slice(None)]
        hr_index = tuple(hr_index)
        return (lr_index, hr_index)

    @property
    def size(self):
        """Return size used to compute container weights."""
        return np.prod(self.shape)


class CollectionSampler(AbstractCollectionSampler):
    """Base collection sampler class."""

    def __init__(self, containers: List[Container]):
        super().__init__(containers)
        self.all_container_pairs = self.check_all_container_pairs()

    def check_all_container_pairs(self):
        """Check if all containers are pairs of low and high res or single
        containers"""
        return all(isinstance(container, ContainerPair)
                   for container in self.containers)

    def get_container_weights(self):
        """Get weights used to sample from different containers based on
        relative sizes"""
        sizes = [c.size for c in self.containers]
        weights = sizes / np.sum(sizes)
        return weights.astype(np.float32)

    def get_container_index(self):
        """Get random container index based on weights"""
        indices = np.arange(0, len(self.containers))
        return np.random.choice(indices, p=self.container_weights)

    def get_random_container(self):
        """Get random container based on container weights"""
        if self._sample_counter % self.batch_size == 0:
            self.container_index = self.get_container_index()
        return self.containers[self.container_index]
