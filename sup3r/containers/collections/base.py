"""Base collection classes. These are objects that contain sets / lists of
containers like batch handlers. Of course these also contain data so they're
containers too!"""

from typing import List, Union

import numpy as np

from sup3r.containers.base import Container, ContainerPair
from sup3r.containers.samplers.base import Sampler
from sup3r.containers.samplers.pair import SamplerPair


class Collection(Container):
    """Object consisting of a set of containers."""

    def __init__(
        self,
        containers: Union[
            List[Container],
            List[ContainerPair],
            List[Sampler],
            List[SamplerPair],
        ],
    ):
        self._containers = containers
        self.data = [c.data for c in self._containers]
        self.all_container_pairs = self.check_all_container_pairs()
        self.features = self.containers[0].features
        self.shape = self.containers[0].shape

    @property
    def containers(
        self,
    ) -> Union[
        List[Container], List[ContainerPair], List[Sampler], List[SamplerPair]
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
            isinstance(container, (ContainerPair, SamplerPair))
            for container in self.containers
        )
