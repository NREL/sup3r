"""Abstract Loader class merely for loading data from file paths. This data
can be loaded lazily or eagerly."""

from abc import ABC, abstractmethod

from sup3r.containers.abstract import AbstractContainer
from sup3r.utilities.utilities import expand_paths


class AbstractLoader(AbstractContainer, ABC):
    """Container subclass with methods for loading data to set data
    atttribute."""

    def __init__(self, file_paths, features=()):
        """
        Parameters
        ----------
        file_paths : str | pathlib.Path | list
            Location(s) of files to load
        features : list
            list of all features extracted or to extract.
        """
        self.file_paths = expand_paths(file_paths)
        self._features = features
        self._data = None
        self._shape = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, trace):
        pass

    @property
    def data(self):
        """Load data if not already."""
        if self._data is None:
            self._data = self.load()
        return self._data

    @abstractmethod
    def load(self):
        """Get data using provided file_paths."""
