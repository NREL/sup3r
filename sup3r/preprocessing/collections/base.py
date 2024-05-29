"""Base collection classes. These are objects that contain sets / lists of
containers like batch handlers. Of course these also contain data so they're
containers too!"""

from typing import List, Union

import numpy as np

from sup3r.preprocessing.base import Container, DualContainer
from sup3r.preprocessing.samplers.base import Sampler
from sup3r.preprocessing.samplers.dual import DualSampler


class Collection(Container):
    """Object consisting of a set of containers."""

    def __init__(
        self,
        containers: Union[
            List[Container],
            List[DualContainer],
            List[Sampler],
            List[DualSampler],
        ],
    ):
        self._containers = containers
        self.data = tuple([c.data for c in self._containers])
        self.all_container_pairs = self.check_all_container_pairs()
        self.features = self.containers[0].features

    @property
    def containers(
        self,
    ) -> Union[
        List[Container], List[DualContainer], List[Sampler], List[DualSampler]
    ]:
        """Returns a list of containers."""
        return self._containers

    @containers.setter
    def containers(self, containers: List[Container]):
        self._containers = containers

    @property
    def container_weights(self):
        """Get weights used to sample from different containers based on
        relative sizes"""
        sizes = [c.size for c in self.containers]
        weights = sizes / np.sum(sizes)
        return weights.astype(np.float32)

    def check_all_container_pairs(self):
        """Check if all containers are pairs of low and high res or single
        containers"""
        return all(
            isinstance(container, (DualContainer, DualSampler))
            for container in self.containers
        )
