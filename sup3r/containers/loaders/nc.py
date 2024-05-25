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
        return xr.open_mfdataset(file_paths, **kwargs)

    def enforce_descending_lats(self, dset):
        """Make sure latitudes are in descneding order so that min lat / lon is
        at lat_lon[-1, 0]."""
        invert_lats = dset['latitude'][-1, 0] > dset['latitude'][0, 0]
        if invert_lats:
            for var in ['latitude', 'longitude', *list(dset.data_vars)]:
                if 'south_north' in dset[var].dims:
                    dset[var] = (
                        dset[var].dims,
                        dset[var].sel(south_north=slice(None, None, -1)).data,
                    )
        return dset

    def load(self):
        """Load netcdf xarray.Dataset()."""
        res = self._standardize(self.res, self.DIM_NAMES)
        lats = res['south_north'].data
        lons = res['west_east'].data
        times = res.indexes['time']

        if hasattr(times, 'to_datetimeindex'):
            times = times.to_datetimeindex()

        if len(lats.shape) == 1:
            lons, lats = da.meshgrid(lons, lats)

        out = res.assign_coords(
            {
                'latitude': (('south_north', 'west_east'), lats),
                'longitude': (('south_north', 'west_east'), lons),
                'time': times,
            }
        )
        out = out.drop_vars(('south_north', 'west_east'))
        if isinstance(self.chunks, tuple):
            chunks = dict(
                zip(['south_north', 'west_east', 'time', 'level'], self.chunks)
            )
            out = out.chunk(chunks)
        return self.enforce_descending_lats(out).astype(np.float32)
