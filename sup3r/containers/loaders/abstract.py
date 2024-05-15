"""Abstract Loader class merely for loading data from file paths. This data
can be loaded lazily or eagerly."""

from abc import ABC, abstractmethod

import dask.array

from sup3r.containers.abstract import AbstractContainer
from sup3r.utilities.utilities import expand_paths


class AbstractLoader(AbstractContainer, ABC):
    """Container subclass with methods for loading data to set data
    atttribute."""

    def __init__(self,
                 file_paths,
                 features):
        """
        Parameters
        ----------
        file_paths : str | pathlib.Path | list
            Location(s) of files to load
        features : list
            list of all features wanted from the file_paths.
        """
        super().__init__()
        self.file_paths = file_paths
        self.features = features

    @abstractmethod
    def res(self):
        """Lowest level file_path handler. e.g. h5py.File(), xr.open_dataset(),
        rex.Resource(), etc."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, trace):
        self.res.close()

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
    def load(self) -> dask.array:
        """Get data using provided file_paths."""
