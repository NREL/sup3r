"""Base loading classes. These are containers which also load data from
file_paths and include some sampling ability to interface with batcher
classes."""

import logging

import dask
import xarray as xr

from sup3r.containers.loaders import Loader

logger = logging.getLogger(__name__)


class LoaderNC(Loader):
    """Base NETCDF loader. "Loads" netcdf files so that a `.data` attribute
    provides access to the data in the files. This object provides a
    `__getitem__` method that can be used by Sampler objects to build batches
    or by Wrangler objects to derive / extract specific features / regions /
    time_periods."""

    def load(self) -> dask.array:
        """Dask array with features in last dimension. Either lazily loaded
        (mode = 'lazy') or loaded into memory right away (mode = 'eager').

        Returns
        -------
        dask.array.core.Array
            (spatial, time, features) or (spatial_1, spatial_2, time, features)
        """
        data = self.res[self.features].to_dataarray().data
        data = dask.array.moveaxis(data, 0, -1)
        data = dask.array.moveaxis(data, 0, -2)

        if self.mode == 'eager':
            data = data.compute()

        return data

    def _get_res(self):
        """Lowest level interface to data."""
        return xr.open_mfdataset(self.file_paths, **self._res_kwargs)
