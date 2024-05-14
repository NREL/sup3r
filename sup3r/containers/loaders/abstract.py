"""Abstract Loader class merely for loading data from file paths. This data
can be loaded lazily or eagerly."""

from abc import ABC, abstractmethod

from sup3r.containers.abstract import AbstractContainer
from sup3r.utilities.utilities import expand_paths


class AbstractLoader(AbstractContainer, ABC):
    """Container subclass with methods for loading data to set data
    atttribute."""

    def __init__(self,
                 file_paths):
        """
        Parameters
        ----------
        file_paths : str | pathlib.Path | list
            Globbable path str(s) or pathlib.Path for file locations.
        """
        super().__init__()
        self.file_paths = file_paths
        self.data = self.load()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, trace):
        self.data.close()

    @property
    def file_paths(self):
        """Get file paths for input data"""
        return self._file_paths

    @file_paths.setter
    def file_paths(self, file_paths):
        """Set file paths attr and do initial glob / sort

        Parameters
        ----------
        file_paths : str | list
            A list of files to extract raster data from. Each file must have
            the same number of timesteps. Can also pass a string or list of
            strings with a unix-style file path which will be passed through
            glob.glob
        """
        self._file_paths = expand_paths(file_paths)
        msg = ('No valid files provided to DataHandler. '
               f'Received file_paths={file_paths}. Aborting.')
        assert file_paths is not None and len(self._file_paths) > 0, msg

    @abstractmethod
    def load(self):
        """Get data using provided file_paths."""
