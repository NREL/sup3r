"""Collection objects consisting of lists of :class:`Sampler` instances"""

import logging
from typing import List, Union

import numpy as np

from sup3r.preprocessing.collections.base import Collection
from sup3r.preprocessing.samplers.base import Sampler
from sup3r.preprocessing.samplers.dual import DualSampler

logger = logging.getLogger(__name__)

np.random.seed(42)


class SamplerCollection(Collection):
    """Collection of :class:`Sampler` objects with methods for sampling across
    the collection."""

    def __init__(
        self,
        samplers: Union[List[Sampler], List[DualSampler]],
        s_enhance,
        t_enhance,
    ):
        super().__init__(containers=samplers)
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        self.container_index = self.get_container_index()
        _ = self.check_shared_attr('sample_shape')

    def get_container_index(self):
        """Get random container index based on weights"""
        indices = np.arange(0, len(self.containers))
        return np.random.choice(indices, p=self.container_weights)

    def get_random_container(self):
        """Get random container based on container weights

        TODO: This will select a random container for every sample, instead of
        every batch. Should we override this in the BatchHandler and use
        the batch_counter to do every batch?
        """
        self.container_index = self.get_container_index()
        return self.containers[self.container_index]

    def get_samples(self):
        """Get random sampler from collection and return a sample from that
        sampler."""
        return next(self.get_random_container())

    @property
    def lr_shape(self):
        """Shape of low resolution sample in a low-res / high-res pair.  (e.g.
        (spatial_1, spatial_2, temporal, features))"""
        return (*self.lr_sample_shape, len(self.lr_features))

    @property
    def hr_shape(self):
        """Shape of high resolution sample in a low-res / high-res pair.  (e.g.
        (spatial_1, spatial_2, temporal, features))"""
        return (*self.hr_sample_shape, len(self.hr_features))
