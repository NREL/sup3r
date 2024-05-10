from abc import ABC, abstractmethod
from typing import List

import numpy as np


class DataObject(ABC):
    """Lowest level object. This is the thing contained by Container
    classes."""

    @property
    @abstractmethod
    def data(self):
        """Raw data."""

    @data.setter
    def data(self, data):
        """Set raw data."""
        self._data = data

    @property
    def shape(self):
        """Shape of raw data"""
        return self.data.shape

    @abstractmethod
    def __getitem__(self, key):
        """Method for accessing self.data."""


class AbstractContainer(DataObject, ABC):
    """Low level object with access to data, knowledge of the data shape, and
    what variables / features are contained."""

    def __init__(self):
        self._data = None

    @property
    @abstractmethod
    def data(self) -> DataObject:
        """Data in the container."""

    @data.setter
    def data(self, data):
        """Define contained data."""
        self._data = data

    @property
    def size(self):
        """'Size' of container."""
        return np.prod(self.shape)

    @property
    @abstractmethod
    def features(self):
        """Set of features in the container."""


class AbstractCollection(ABC):
    """Object consisting of a set of containers."""

    def __init__(self, containers):
        super().__init__()
        self._containers = containers

    @property
    def containers(self) -> List[AbstractContainer]:
        """Returns a list of containers."""
        return self._containers

    @containers.setter
    def containers(self, containers):
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
