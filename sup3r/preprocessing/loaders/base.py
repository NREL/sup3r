"""Abstract Loader class merely for loading data from file paths. This data is
always loaded lazily."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime as dt
from typing import Callable

import numpy as np
import xarray as xr

from sup3r.preprocessing.base import Container
from sup3r.preprocessing.names import (
    FEATURE_NAMES,
)
from sup3r.preprocessing.utilities import expand_paths

from .utilities import standardize_names, standardize_values

logger = logging.getLogger(__name__)


class BaseLoader(Container, ABC):
    """Base loader. "Loads" files so that a `.data` attribute provides access
    to the data in the files as a dask array with shape (lats, lons, time,
    features). This object provides a `__getitem__` method that can be used by
    :class:`Sampler` objects to build batches or by :class:`Rasterizer` objects
    to derive / extract specific features / regions / time_periods."""

    BASE_LOADER: Callable = xr.open_mfdataset

    def __init__(
        self,
        file_paths,
        features='all',
        res_kwargs=None,
        chunks='auto',
        BaseLoader=None
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
        chunks : dict | str
            Dictionary of chunk sizes to use for call to
            `dask.array.from_array()` or xr.Dataset().chunk(). Will be
            converted to a tuple when used in `from_array().`
        BaseLoader : Callable
            Optional base loader method update. This is a function which takes
            `file_paths` and `**kwargs` and returns an initialized base loader
            with those arguments. The default for h5 is a method which returns
            MultiFileWindX(file_paths, **kwargs) and for nc the default is
            xarray.open_mfdataset(file_paths, **kwargs)
        """
        super().__init__()
        self._data = None
        self.res_kwargs = res_kwargs or {}
        self.file_paths = file_paths
        self.chunks = chunks
        BASE_LOADER = BaseLoader or self.BASE_LOADER
        self.res = BASE_LOADER(self.file_paths, **self.res_kwargs)
        self.data = self.load().astype(np.float32)
        self.data = standardize_names(self.data, FEATURE_NAMES)
        self.data = standardize_values(self.data)
        self.data = self.data[features] if features != 'all' else self.data
        self.add_attrs()

    def add_attrs(self):
        """Add meta data to dataset."""
        attrs = {
            'source_files': str(self.file_paths),
            'date_modified': dt.utcnow().isoformat(),
        }
        if hasattr(self.res, 'global_attrs'):
            attrs['global_attrs'] = self.res.global_attrs

        if hasattr(self.res, 'h5'):
            attrs.update(
                {
                    f: dict(self.res.h5[f.split('/')[0]].attrs)
                    for f in self.res.datasets
                }
            )
        elif hasattr(self.res, 'attrs'):
            attrs.update(self.res.attrs)
        self.data.attrs.update(attrs)

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
