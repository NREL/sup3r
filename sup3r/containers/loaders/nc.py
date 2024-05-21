"""Base loading classes. These are containers which also load data from
file_paths and include some sampling ability to interface with batcher
classes."""

import logging

import dask.array as da
import numpy as np
import xarray as xr

from sup3r.containers.loaders import Loader

logger = logging.getLogger(__name__)


class LoaderNC(Loader):
    """Base NETCDF loader. "Loads" netcdf files so that a `.data` attribute
    provides access to the data in the files. This object provides a
    `__getitem__` method that can be used by Sampler objects to build batches
    or by Wrangler objects to derive / extract specific features / regions /
    time_periods."""

    def BASE_LOADER(self, file_paths, **kwargs):
        """Lowest level interface to data."""
        if isinstance(self.chunks, tuple):
            kwargs['chunks'] = dict(
                zip(['time', 'latitude', 'longitude', 'level'], self.chunks)
            )
        return xr.open_mfdataset(file_paths, **kwargs)

    def load(self):
        """Load netcdf xarray.Dataset()."""
        lats = self.res['latitude'].data
        lons = self.res['longitude'].data
        if len(lats.shape) == 1:
            lons, lats = da.meshgrid(lons, lats)
        rename_dict = {'latitude': 'south_north', 'longitude': 'west_east'}
        for k, v in rename_dict.items():
            if k in self.res.dims:
                self.res = self.res.rename({k: v})
        self.res = self.res.assign_coords(
            {'latitude': (('south_north', 'west_east'), lats)}
        )
        self.res = self.res.assign_coords(
            {'longitude': (('south_north', 'west_east'), lons)}
        )
        return self.res.astype(np.float32)
