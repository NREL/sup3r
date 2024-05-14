"""Base loading classes. These are containers which also load data from
file_paths and include some sampling ability to interface with batcher
classes."""

import logging

import numpy as np
import xarray as xr
from rex import MultiFileWindX

from sup3r.containers.loaders.abstract import AbstractLoader

logger = logging.getLogger(__name__)


class LoaderNC(AbstractLoader):
    """Base NETCDF loader. "Loads" netcdf files so that a `.data` attribute
    provides access to the data in the files. This object provides a
    `__getitem__` method that can be used by Sampler objects to build batches
    or by Wrangler objects to derive / extract specific features / regions /
    time_periods."""

    def __init__(
        self, file_paths, features, res_kwargs=None, mode='lazy'
    ):
        """
        Parameters
        ----------
        file_paths : str | pathlib.Path | list
            Location(s) of files to load
        features : list
            list of all features wanted from the file_paths.
        res_kwargs : dict
            kwargs for xr.open_mfdataset()
        mode : str
            Options are ('lazy', 'eager') for how to load data.
        """
        super().__init__(file_paths)
        self.features = features
        self._res_kwargs = res_kwargs or {}
        self._mode = mode

    @property
    def shape(self):
        """Return shape of extent available for sampling."""
        if self._shape is None:
            self._shape = (*self.data["latitude"].shape,
                           len(self.data["time"]))
        return self._shape

    def load(self):
        """Xarray dataset either lazily loaded (mode = 'lazy') or loaded into
        memory right away (mode = 'eager').

        Returns
        -------
        xr.Dataset()
            xarray dataset with the requested features
        """
        data = xr.open_mfdataset(self.file_paths, **self._res_kwargs)
        msg = (f'Loading {self.file_paths} with kwargs = '
               f'{self._res_kwargs} and mode = {self._mode}')
        logger.info(msg)

        if self._mode == 'eager':
            data = data.compute()

        return data[self.features]

    def __getitem__(self, key):
        """Get observation/sample. Should return a single sample from the
        underlying data with shape (spatial_1, spatial_2, temporal,
        features)."""

        out = self.data.isel(
            south_north=key[0],
            west_east=key[1],
            time=key[2],
        )

        if self._mode == 'lazy':
            out = out.compute()

        out = out.to_dataarray().values
        return np.transpose(out, axes=(2, 3, 1, 0))


class LoaderH5(AbstractLoader):
    """Base H5 loader. "Loads" h5 files so that a `.data` attribute
    provides access to the data in the files. This object provides a
    `__getitem__` method that can be used by Sampler objects to build batches
    or by Wrangler objects to derive / extract specific features / regions /
    time_periods."""

    def __init__(
        self, file_paths, features, res_kwargs=None, mode='lazy'
):
        """
        Parameters
        ----------
        file_paths : str | pathlib.Path | list
            Location(s) of files to load
        features : list
            list of all features wanted from the file_paths.
        res_kwargs : dict
            kwargs for MultiFileWindX
        mode : str
            Options are ('lazy', 'eager') for how to load data.
        """
        super().__init__(file_paths)
        self.features = features
        self._res_kwargs = res_kwargs or {}
        self._mode = mode

    @property
    def shape(self):
        """Return shape of extent available for sampling."""
        if self._shape is None:
            self._shape = (*self.data["latitude"].shape,
                           len(self.data["time"]))
        return self._shape

    def load(self):
        """Xarray dataset either lazily loaded (mode = 'lazy') or loaded into
        memory right away (mode = 'eager').

        Returns
        -------
        xr.Dataset()
            xarray dataset with the requested features
        """
        data = MultiFileWindX(self.file_paths, **self._res_kwargs)
        msg = (f'Loading {self.file_paths} with kwargs = '
               f'{self._res_kwargs} and mode = {self._mode}')
        logger.info(msg)

        if self._mode == 'eager':
            data = data[:]

        return data

    def __getitem__(self, key):
        """Get observation/sample. Should return a single sample from the
        underlying data with shape (spatial_1, spatial_2, temporal,
        features)."""
        return self.data[key]
