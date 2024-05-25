"""Abstract Loader class merely for loading data from file paths. This data
can be loaded lazily or eagerly."""

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from sup3r.containers.base import Container
from sup3r.utilities.utilities import expand_paths


class Loader(Container, ABC):
    """Base loader. "Loads" files so that a `.data` attribute provides access
    to the data in the files as a dask array with shape (lats, lons, time,
    features). This object provides a `__getitem__` method that can be used by
    :class:`Sampler` objects to build batches or by :class:`Extracter` objects
    to derive / extract specific features / regions / time_periods."""

    BASE_LOADER = None

    FEATURE_NAMES: ClassVar = {
        'elevation': 'topography',
        'orog': 'topography',
    }

    DIM_NAMES: ClassVar = {
        'lat': 'south_north',
        'lon': 'west_east',
        'latitude': 'south_north',
        'longitude': 'west_east',
        'plev': 'level'
    }

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
        features : str | list
            List of features to include in the loaded data. If 'all'
            this includes all features available in the file_paths. To select
            specific features provide a list.
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
        self.res_kwargs = res_kwargs or {}
        self.file_paths = file_paths
        self.mode = mode
        self.chunks = chunks
        self.res = self.BASE_LOADER(self.file_paths, **self.res_kwargs)
        self.data = self._standardize(self.load(), self.FEATURE_NAMES).astype(
            np.float32
        )
        features = (
            list(self.data.features)
            if features == 'all'
            else features
        )
        self.data = self.data.slice_dset(features=features)

    def _standardize(self, data, standard_names):
        """Standardize fields in the dataset using the `standard_names`
        dictionary."""
        rename_map = {feat: feat.lower() for feat in data.data_vars}
        data = data.rename(rename_map)
        data = data.rename(
            {k: v for k, v in standard_names.items() if k in data}
        )
        return data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, trace):
        self.close()

    def close(self):
        """Close `self.res`."""
        self.res.close()
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
        msg = (
            f'No valid files provided to {self.__class__.__name__}. '
            f'Received file_paths={file_paths}. Aborting.'
        )
        assert file_paths is not None and len(self._file_paths) > 0, msg

    @abstractmethod
    def load(self):
        """xarray.DataArray features in last dimension. Either lazily loaded
        (mode = 'lazy') or loaded into memory right away (mode = 'eager').

        Returns
        -------
        dask.array.core.Array
            (spatial, time, features) or (spatial_1, spatial_2, time, features)
        """
