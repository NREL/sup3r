"""Base loading classes. These are containers which also load data from
file_paths and include some sampling ability to interface with batcher
classes."""

import logging

import numpy as np
import xarray as xr

from sup3r.containers.loaders.abstract import AbstractLoader

logger = logging.getLogger(__name__)


class LoaderNC(AbstractLoader):
    """Base loader. Loads precomputed netcdf files (usually from
    a DataHandler.to_netcdf() call after populating DataHandler.data).
    Provides `__getitem__` method for use by Sampler objects."""

    def __init__(
        self, file_paths, features, res_kwargs=None, mode='lazy'
    ):
        """
        Parameters
        ----------
        file_paths : str | pathlib.Path | list
            Location(s) of files to load
        features : list
            list of all features extracted or to extract.
        res_kwargs : dict
            kwargs for xr.open_mfdataset()
        mode : str
            Options are ('lazy', 'eager') for how to load data.
        """
        super().__init__(file_paths, features)
        self._res_kwargs = res_kwargs or {}
        self._mode = mode

        logger.info(f'Initialized {self.__class__.__name__} with '
                    f'files = {self.file_paths}, features = {self.features}, '
                    f'res_kwargs = {self._res_kwargs}, mode = {self._mode}.')

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
