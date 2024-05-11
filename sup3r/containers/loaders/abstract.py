"""Abstract Loader class merely for loading data from file paths. This data
can be loaded lazily or eagerly."""

from abc import ABC, abstractmethod

from sup3r.containers.base import Container
from sup3r.utilities.utilities import expand_paths


class AbstractLoader(Container, ABC):
    """Container subclass with methods for loading data to set data
    atttribute."""

    def __init__(self, file_paths, features=(), lr_only_features=(),
                 hr_exo_features=()):
        """
        Parameters
        ----------
        file_paths : str | pathlib.Path | list
            Location(s) of files to load
        features : list
            list of all features extracted or to extract.
        lr_only_features : list | tuple
            List of feature names or patt*erns that should only be included in
            the low-res training set and not the high-res observations.
        hr_exo_features : list | tuple
            List of feature names or patt*erns that should be included in the
            high-resolution observation but not expected to be output from the
            generative model. An example is high-res topography that is to be
            injected mid-network.
        """
        super().__init__(features, lr_only_features, hr_exo_features)
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
