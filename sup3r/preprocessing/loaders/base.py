"""Abstract Loader class merely for loading data from file paths. This data
can be loaded lazily or eagerly."""

from abc import ABC, abstractmethod
from typing import Callable, ClassVar

import numpy as np
import xarray as xr

from sup3r.preprocessing.base import Container
from sup3r.preprocessing.utilities import Dimension, expand_paths


class Loader(Container, ABC):
    """Base loader. "Loads" files so that a `.data` attribute provides access
    to the data in the files as a dask array with shape (lats, lons, time,
    features). This object provides a `__getitem__` method that can be used by
    :class:`Sampler` objects to build batches or by :class:`Extracter` objects
    to derive / extract specific features / regions / time_periods."""

    BASE_LOADER: Callable = xr.open_dataset

    FEATURE_NAMES: ClassVar = {
        'elevation': 'topography',
        'orog': 'topography',
        'hgt': 'topography',
    }

    DIM_NAMES: ClassVar = {
        'lat': Dimension.SOUTH_NORTH,
        'lon': Dimension.WEST_EAST,
        'xlat': Dimension.SOUTH_NORTH,
        'xlong': Dimension.WEST_EAST,
        'latitude': Dimension.SOUTH_NORTH,
        'longitude': Dimension.WEST_EAST,
        'plev': Dimension.PRESSURE_LEVEL,
        'xtime': Dimension.TIME,
    }

    def __init__(
        self,
        file_paths,
        features='all',
        res_kwargs=None,
        chunks='auto',
    ):
        """
        Parameters
        ----------
        file_paths : str | pathlib.Path | list
            Location(s) of files to load
        features : list | str
            Features to return in loaded dataset. If 'all' then all available
            features will be returned.
        res_kwargs : dict
            kwargs for `.res` object
        chunks : tuple
            Tuple of chunk sizes to use for call to dask.array.from_array().
            Note: The ordering here corresponds to the default ordering given
            by `.res`.
        """
        super().__init__()
        self._res = None
        self._data = None
        self.res_kwargs = res_kwargs or {}
        self.file_paths = file_paths
        self.chunks = chunks
        self.res = self.BASE_LOADER(self.file_paths, **self.res_kwargs)
        self.data = self.rename(self.load(), self.FEATURE_NAMES).astype(
            np.float32
        )
        self.data[Dimension.LONGITUDE] = (
            self.data[Dimension.LONGITUDE] + 180.0
        ) % 360.0 - 180.0
        self.data = self.data[features] if features != 'all' else self.data
        self.add_attrs()

    def add_attrs(self):
        """Add meta data to dataset."""
        attrs = {'source_files': self.file_paths}
        self.data.attrs.update(attrs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, trace):
        self.res.close()

    def rename(self, data, standard_names):
        """Standardize fields in the dataset using the `standard_names`
        dictionary."""
        rename_map = {
            feat: feat.lower()
            for feat in [
                *list(data.data_vars),
                *list(data.coords),
                *list(data.dims),
            ]
        }
        data = data.rename(
            {k: v for k, v in rename_map.items() if v != Dimension.TIME}
        )
        data = data.swap_dims(
            {
                k: v
                for k, v in rename_map.items()
                if v == Dimension.TIME and k in data
            }
        )
        data = data.rename(
            {k: v for k, v in standard_names.items() if k in data}
        )
        return data

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
        """xarray.DataArray features in last dimension.

        Returns
        -------
        dask.array.core.Array
            (spatial, time, features) or (spatial_1, spatial_2, time, features)
        """
