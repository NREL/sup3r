from abc import ABC, abstractmethod

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
