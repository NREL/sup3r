"""Abstract Loader class merely for loading data from file paths. This data
can be loaded lazily or eagerly."""

from abc import ABC, abstractmethod

from sup3r.containers.base import Container
from sup3r.utilities.utilities import expand_paths


class AbstractLoader(Container, ABC):
    """Container subclass with methods for loading data to set data
    atttribute."""

    def __init__(self, file_paths):
        self.file_paths = expand_paths(file_paths)
        self._data = None

    @property
    def data(self):
        """Load data if not already."""
        if self._data is None:
            self._data = self.load()
        return self._data

    @abstractmethod
    def load(self):
        """Get data using provided file_paths."""
