"""Abstract container classes. These are the fundamental objects that all
classes which interact with data (e.g. handlers, wranglers, samplers, batchers)
are based on."""
from abc import ABC, abstractmethod


class DataObject(ABC):
    """Lowest level object. This is the thing contained by Container
    classes."""

    def __init__(self):
        self._data = None
        self._features = None
        self._shape = None

    @property
    def data(self):
        """Raw data."""
        if self._data is None:
            msg = (f'This {self.__class__.__name__} contains no data.')
            raise ValueError(msg)
        return self._data

    @data.setter
    def data(self, data):
        """Set raw data."""
        self._data = data

    @property
    def shape(self):
        """Shape of raw data"""
        return self._shape

    @shape.setter
    def shape(self, shape):
        """Shape of raw data"""
        self._shape = shape

    @property
    def features(self):
        """Set of features in the data object."""
        return self._features

    @features.setter
    def features(self, features):
        """Set the features in the data object."""
        self._features = features

    @abstractmethod
    def __getitem__(self, key):
        """Method for accessing self.data."""


class AbstractContainer(DataObject, ABC):
    """Very basic thing _containing_ a data object."""

    def __init__(self, obj: DataObject):
        super().__init__()
        self.obj = obj
