"""Base loading classes. These are containers which also load data from
file_paths and include some sampling ability to interface with batcher
classes."""

import logging

import dask.array as da
import numpy as np
import xarray as xr

from sup3r.preprocessing.utilities import Dimension, ordered_dims

from .base import Loader

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
        """Make sure latitudes are in descending order so that min lat / lon is
        at lat_lon[-1, 0]."""
        invert_lats = (
            dset[Dimension.LATITUDE][-1, 0] > dset[Dimension.LATITUDE][0, 0]
        )
        if invert_lats:
            for var in [
                Dimension.LATITUDE,
                Dimension.LONGITUDE,
                *list(dset.data_vars),
            ]:
                if Dimension.SOUTH_NORTH in dset[var].dims:
                    dset[var] = (
                        dset[var].dims,
                        dset[var].sel(south_north=slice(None, None, -1)).data,
                    )
        return dset

    def unstagger_variables(self, dset):
        """Unstagger variables with staggered dimensions. Usually used in WRF
        output."""
        raise NotImplementedError

    def enforce_descending_levels(self, dset):
        """Make sure levels are in descending order so that max pressure is at
        level[0]."""
        invert_levels = (
            dset[Dimension.PRESSURE_LEVEL][-1]
            > dset[Dimension.PRESSURE_LEVEL][0]
            if Dimension.PRESSURE_LEVEL in dset
            else False
        )
        if invert_levels:
            for var in list(dset.data_vars):
                if Dimension.PRESSURE_LEVEL in dset[var].dims:
                    dset[var] = (
                        dset[var].dims,
                        dset[var].sel(level=slice(None, None, -1)).data,
                    )
        return dset

    def load(self):
        """Load netcdf xarray.Dataset()."""
        res = self.rename(self.res, self.DIM_NAMES)
        lats = res[Dimension.SOUTH_NORTH].data.squeeze()
        lons = res[Dimension.WEST_EAST].data.squeeze()

        time_independent = (
            Dimension.TIME not in res.coords and Dimension.TIME not in res.dims
        )

        if not time_independent:
            times = (
                res.indexes[Dimension.TIME]
                if Dimension.TIME in res.indexes
                else res[Dimension.TIME]
            )

            if hasattr(times, 'to_datetimeindex'):
                times = times.to_datetimeindex()

            res = res.assign_coords({Dimension.TIME: times})

        if len(lats.shape) == 1:
            lons, lats = da.meshgrid(lons, lats)

        coords = {
            Dimension.LATITUDE: (
                (Dimension.SOUTH_NORTH, Dimension.WEST_EAST),
                lats.astype(np.float32),
            ),
            Dimension.LONGITUDE: (
                (Dimension.SOUTH_NORTH, Dimension.WEST_EAST),
                lons.astype(np.float32),
            ),
        }
        out = res.assign_coords(coords)
        out = out.drop_vars((Dimension.SOUTH_NORTH, Dimension.WEST_EAST))

        if isinstance(self.chunks, tuple):
            chunks = dict(zip(ordered_dims(out.dims), self.chunks))
            out = out.chunk(chunks)
        out = self.enforce_descending_lats(out)
        return self.enforce_descending_levels(out).astype(np.float32)
