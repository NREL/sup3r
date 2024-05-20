"""Abstract Loader class merely for loading data from file paths. This data
can be loaded lazily or eagerly."""

from abc import ABC, abstractmethod

import numpy as np

from sup3r.containers.abstract import AbstractContainer
from sup3r.utilities.utilities import expand_paths


class Loader(AbstractContainer, ABC):
    """Base loader. "Loads" files so that a `.data` attribute provides access
    to the data in the files as a dask array with shape (lats, lons, time,
    features). This object provides a `__getitem__` method that can be used by
    :class:`Sampler` objects to build batches or by :class:`Extracter` objects
    to derive / extract specific features / regions / time_periods."""

    def __init__(
        self,
        file_paths,
        features='all',
        res_kwargs=None,
        chunks='auto',
        mode='lazy',
    ):
        """
        Parameters
        ----------
        file_paths : str | pathlib.Path | list
            Location(s) of files to load
        features : list | str | None
            list of all features wanted from the file_paths. If 'all' then all
            available features will be loaded. If None then only the base
            file_path interface will be exposed for downstream extraction of
            meta data like lat_lon / time_index
        res_kwargs : dict
            kwargs for `.res` object
        chunks : tuple
            Tuple of chunk sizes to use for call to dask.array.from_array().
            Note: The ordering here corresponds to the default ordering given
            by `.res`.
        mode : str
            Options are ('lazy', 'eager') for how to load data.
        """
        super().__init__()
        self._res = None
        self._data = None
        self._res_kwargs = res_kwargs or {}
        self.file_paths = file_paths
        self.features = self.parse_requested_features(features)
        self.mode = mode
        self.chunks = chunks

    def parse_requested_features(self, features):
        """Parse the feature input and return corresponding feature list."""
        features = [] if features is None else features
        if features == 'all':
            features = self.get_loadable_features()
        return features

    def get_loadable_features(self):
        """Get loadable features excluding coordinate / time fields."""
        return [
            f
            for f in self.res
            if not f.startswith(('lat', 'lon', 'time', 'meta'))
        ]

    @property
    def data(self):
        """'Load' data when access is requested."""
        if self._data is None and any(self.features):
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
        msg = (
            f'No valid files provided to {self.__class__.__name__}. '
            f'Received file_paths={file_paths}. Aborting.'
        )
        assert file_paths is not None and len(self._file_paths) > 0, msg

    def load(self):
        """Dask array with features in last dimension. Either lazily loaded
        (mode = 'lazy') or loaded into memory right away (mode = 'eager').

        Returns
        -------
        dask.array.core.Array
            (spatial, time, features) or (spatial_1, spatial_2, time, features)
        """
        data = self._get_features(self.features)

        if self.mode == 'eager':
            data = data.compute()

        return data

    @abstractmethod
    def _get_features(self, features):
        """Get specific features from base resource."""
