"""Base loading classes. These are containers which also load data from
file_paths and include some sampling ability to interface with batcher
classes."""

import logging

import dask.array as da
import xarray as xr

from sup3r.containers.loaders import Loader

logger = logging.getLogger(__name__)


class LoaderNC(Loader):
    """Base NETCDF loader. "Loads" netcdf files so that a `.data` attribute
    provides access to the data in the files. This object provides a
    `__getitem__` method that can be used by Sampler objects to build batches
    or by Wrangler objects to derive / extract specific features / regions /
    time_periods."""

    def _get_res(self):
        """Lowest level interface to data."""
        return xr.open_mfdataset(self.file_paths, **self._res_kwargs)

    def _get_features(self, features):
        """We perform an axis shift here from (time, ...) to (..., time)
        ordering. The final stack puts features in the last channel."""
        if isinstance(features, (list, tuple)):
            data = [self._get_features(f) for f in features]
        elif isinstance(features, str) and features in self.res:
            data = da.moveaxis(self.res[features].data, 0, -1)
        elif isinstance(features, str) and features.lower() in self.res:
            data = self._get_features(features.lower())
        else:
            msg = f'{features} not found in {self.file_paths}.'
            logger.error(msg)
            raise KeyError(msg)
        if isinstance(data, list):
            data = da.stack(data, axis=-1)
        return data
