"""Basic container objects can perform transformations / extractions on the
contained data."""

import logging
import os
from abc import ABC

import dask.array as da
import h5py
import numpy as np
import xarray as xr

from sup3r.containers.loaders import Loader
from sup3r.containers.wranglers.abstract import AbstractWrangler

np.random.seed(42)

logger = logging.getLogger(__name__)


class Wrangler(AbstractWrangler, ABC):
    """Base Wrangler object."""

    def __init__(
        self,
        container: Loader,
        features,
        target,
        shape,
        time_slice=slice(None),
        transform_function=None,
        cache_kwargs=None
    ):
        """
        Parameters
        ----------
        container : Loader
            Loader type container with `.data` attribute exposing data to
            wrangle.
        features : list
            List of feature names to extract from file_paths.
        target : tuple
            (lat, lon) lower left corner of raster. Either need target+shape or
            raster_file.
        shape : tuple
            (rows, cols) grid size. Either need target+shape or raster_file.
        time_slice : slice
            Slice specifying extent and step of temporal extraction. e.g.
            slice(start, stop, step). If equal to slice(None, None, 1)
            the full time dimension is selected.
        transform_function : function
            Optional operation on loader.data. For example, if you want to
            derive U/V and you used the Loader to expose windspeed/direction,
            provide a function that operates on windspeed/direction and returns
            U/V. The final `.data` attribute will be the output of this
            function.
        cache_kwargs : dict
            Dictionary with kwargs for caching wrangled data. This should at
            minimum include a 'cache_pattern' key, value. This pattern must
            have a {feature} format key and either a h5 or nc file extension,
            based on desired output type.

            Can also include a 'chunks' key, value with a dictionary of tuples
            for each feature. e.g. {'cache_pattern': ..., 'chunks':
            {'windspeed_100m': (20, 100, 100)}} where the chunks ordering is
            (time, lats, lons)

            Note: This is only for saving cached data. If you want to reload
            the cached files load them with a Loader object.
        """
        super().__init__(
            container=container,
            features=features,
            target=target,
            shape=shape,
            time_slice=time_slice,
            transform_function=transform_function,
            cache_kwargs=cache_kwargs
        )

    def cache_data(self):
        """Cache data to file with file type based on user provided
        cache_pattern."""
        cache_pattern = self.cache_kwargs['cache_pattern']
        chunks = self.cache_kwargs.get('chunks', None)
        msg = 'cache_pattern must have {feature} format key.'
        assert '{feature}' in cache_pattern, msg
        _, ext = os.path.splitext(cache_pattern)
        coords = {
            'latitude': (('south_north', 'west_east'), self.lat_lon[..., 0]),
            'longitude': (('south_north', 'west_east'), self.lat_lon[..., 1]),
            'time': self.time_index.values,
        }
        for fidx, feature in enumerate(self.features):
            out_file = cache_pattern.format(feature=feature)
            if not os.path.exists(out_file):
                logger.info(f'Writing {feature} to {out_file}.')
                data = self.data[..., fidx]
                if ext == '.h5':
                    self._write_h5(
                        out_file,
                        feature,
                        np.transpose(data, axes=(2, 0, 1)),
                        coords,
                        chunks,
                    )
                elif ext == '.nc':
                    self._write_netcdf(
                        out_file,
                        feature,
                        np.transpose(data, axes=(2, 0, 1)),
                        coords,
                    )
                else:
                    msg = (
                        'cache_pattern must have either h5 or nc '
                        f'extension. Recived {ext}.'
                    )
                    logger.error(msg)
                    raise ValueError(msg)

    def _write_h5(self, out_file, feature, data, coords, chunks=None):
        """Cache data to h5 file using user provided chunks value."""
        chunks = chunks or {}
        with h5py.File(out_file, 'w') as f:
            _, lats = coords['latitude']
            _, lons = coords['longitude']
            times = coords['time'].astype(int)
            data_dict = dict(
                zip(
                    ['time_index', 'latitude', 'longitude', feature],
                    [
                        da.from_array(times),
                        da.from_array(lats),
                        da.from_array(lons),
                        data,
                    ],
                )
            )
            for dset, vals in data_dict.items():
                d = f.require_dataset(
                    f'/{dset}',
                    dtype=vals.dtype,
                    shape=vals.shape,
                    chunks=chunks.get(dset, None),
                )
                da.store(vals, d)
                logger.info(f'Added {dset} to {out_file}.')

    def _write_netcdf(self, out_file, feature, data, coords):
        """Cache data to a netcdf file."""
        data_vars = {feature: (('time', 'south_north', 'west_east'), data)}
        out = xr.Dataset(data_vars=data_vars, coords=coords)
        out.to_netcdf(out_file)
