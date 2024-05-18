"""Abstract Loader class merely for loading data from file paths. This data
can be loaded lazily or eagerly."""

from abc import ABC, abstractmethod

import numpy as np

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
        self._res = None
        self._data = None
        self.file_paths = file_paths
        self.features = features

    @property
    def data(self):
        """'Load' data when access is requested."""
        if self._data is None:
            self._data = self.load().astype(np.float32)
        return self._data

    @property
    def res(self):
        """Lowest level file_path handler. e.g. h5py.File(), xr.open_dataset(),
        rex.Resource(), etc."""
        if self._res is None:
            self._res = self._get_res()
        return self._res

    @abstractmethod
    def _get_res(self):
        """Get lowest level file interface."""

    @abstractmethod
    def get(self, feature):
        """Method for retrieving features for `.res`. This can depend on the
        specific methods / attributes of `.res`"""

    @abstractmethod
    def scale_factor(self, feature):
        """Return scale factor for the given feature if the data is stored in
        scaled format."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, trace):
        self.close()

    def close(self):
        """Close `self.res`."""
        self.res.close()

    def __getitem__(self, keys):
        """Get item from data."""
        return self.data[keys]

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
