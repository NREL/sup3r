from abc import ABC, abstractmethod
from typing import Tuple

from sup3r.containers.base import Collection, Container


class AbstractSampler(Container, ABC):
    """Sampler class for iterating through contained things."""

    def __init__(self):
        self._counter = 0
        self._size = None

    @abstractmethod
    def get_sample_index(self):
        """Get index used to select sample from contained data. e.g.
        self[index]."""

    def get_next(self):
        """Get "next" thing in the container. e.g. data observation or batch of
        observations"""
        return self[self.get_sample_index()]

    @property
    @abstractmethod
    def sample_shape(self) -> Tuple:
        """Shape of the data sample to select when `get_next()` is called."""

    def __next__(self):
        """Iterable next method"""
        return self.get_next()

    def __iter__(self):
        self._counter = 0
        return self

    def __len__(self):
        return self._size


class AbstractCollectionSampler(Collection, ABC):
    """Collection subclass with additional methods for sampling containers
    from the collection."""

    def __init__(self, containers):
        super().__init__(containers)
        self.container_weights = None
        self.s_enhance, self.t_enhance = self.get_enhancement_factors()

    @abstractmethod
    def get_container_weights(self):
        """List of normalized container sizes used to weight them when randomly
        sampling."""

    @abstractmethod
    def get_container_index(self) -> int:
        """Get random container index based on weights."""

    @abstractmethod
    def get_random_container(self) -> Container:
        """Get random container based on weights."""

    def __getitem__(self, index):
        """Get data sample from sampled container."""
        container = self.get_random_container()
        return container.get_next()

    @property
    def sample_shape(self):
        """Get shape of sample to select when sampling container collection."""
        return self.containers[0].sample_shape

    def get_enhancement_factors(self):
        """Get enhancement factors from container properties."""
        return self.containers[0].get_enhancement_factors()
