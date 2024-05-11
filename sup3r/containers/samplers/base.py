"""Sampler objects. These take in data objects / containers and can them sample
from them. These samples can be used to build batches."""

import logging
from typing import List, Tuple

import numpy as np

from sup3r.containers.base import ContainerPair
from sup3r.containers.samplers.abstract import (
    AbstractSampler,
    AbstractSamplerCollection,
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

    def __init__(self, lr_container: Sampler, hr_container: Sampler,
                 s_enhance, t_enhance):
        super().__init__(lr_container, hr_container)
        self.lr_container = lr_container
        self.hr_container = hr_container
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        self.check_for_consistent_shapes()

    def check_for_consistent_shapes(self):
        """Make sure container shapes are compatible with enhancement
        factors."""
        enhanced_shape = (self.lr_container.shape[0] * self.s_enhance,
                          self.lr_container.shape[1] * self.s_enhance,
                          self.lr_container.shape[2] * self.t_enhance)
        msg = (f'hr_container.shape {self.hr_container.shape} and enhanced '
               f'lr_container.shape {enhanced_shape} are not compatible with '
               'the given enhancement factors')
        assert self.hr_container.shape == enhanced_shape, msg
        s_enhance = self.hr_sample_shape[0] // self.lr_sample_shape[0]
        t_enhance = self.hr_sample_shape[2] // self.lr_sample_shape[2]
        msg = (f'Received s_enhance = {self.s_enhance} but based on sample '
               f'shapes it should be {s_enhance}.')
        assert self.s_enhance == s_enhance, msg
        msg = (f'Received t_enhance = {self.t_enhance} but based on sample '
               f'shapes it should be {t_enhance}.')
        assert self.t_enhance == t_enhance, msg

    @property
    def sample_shape(self) -> Tuple[tuple, tuple]:
        """Shape of the data sample to select when `get_next()` is called."""
        return (self.lr_sample_shape, self.hr_sample_shape)

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
    def lr_only_features(self):
        """Features to use for training only and not output"""
        tof = [fn for fn in self.lr_container.features
               if fn not in self.hr_out_features
               and fn not in self.hr_exo_features]
        return tof

    @property
    def lr_features(self):
        """Get a list of low-resolution features. All low-resolution features
        are used for training."""
        return self.lr_container.features

    @property
    def hr_features(self):
        """Get a list of high-resolution features. This is hr_exo_features plus
        hr_out_features."""
        return self.hr_container.features

    @property
    def hr_exo_features(self):
        """Get a list of high-resolution features that are only used for
        training e.g., mid-network high-res topo injection. These must come at
        the end of the high-res feature set."""
        return self.hr_container.hr_exo_features

    @property
    def hr_out_features(self):
        """Get a list of high-resolution features that are intended to be
        output by the GAN. Does not include high-resolution exogenous features
        """
        return self.hr_container.hr_out_features

    @property
    def size(self):
        """Return size used to compute container weights."""
        return np.prod(self.shape)

    @property
    def lr_sample_shape(self):
        """Get lr sample shape"""
        return self.lr_container.sample_shape

    @property
    def hr_sample_shape(self):
        """Get hr sample shape"""
        return self.hr_container.sample_shape


class SamplerCollection(AbstractSamplerCollection):
    """Base collection sampler class."""

    def __init__(self, containers: List[Sampler], s_enhance, t_enhance):
        super().__init__(containers, s_enhance, t_enhance)
        self.check_collection_consistency()
        self.all_container_pairs = self.check_all_container_pairs()

    def check_collection_consistency(self):
        """Make sure all samplers in the collection have the same sample
        shape."""
        sample_shapes = [c.sample_shape for c in self.containers]
        msg = ('All samplers must have the same sample_shape. Received '
               'inconsistent collection.')
        assert all(s == sample_shapes[0] for s in sample_shapes), msg

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
