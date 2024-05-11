from abc import ABC, abstractmethod
from typing import List

from sup3r.containers.base import Container


class AbstractCollection(ABC):
    """Object consisting of a set of containers."""

    def __init__(self, containers: List[Container]):
        super().__init__()
        self._containers = containers

    @property
    def containers(self) -> List[Container]:
        """Returns a list of containers."""
        return self._containers

    @containers.setter
    def containers(self, containers: List[Container]):
        self._containers = containers

    @property
    @abstractmethod
    def data(self):
        """Data available in the collection of containers."""

    @property
    @abstractmethod
    def features(self):
        """Get set of features available in the container collection."""

    @property
    @abstractmethod
    def shape(self):
        """Get full available shape to sample from when selecting sample_size
        samples."""
