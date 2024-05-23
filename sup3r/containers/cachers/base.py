"""Basic objects that can cache extracted / derived data."""

import logging
import os
from typing import Dict, Union

import dask.array as da
import h5py
import numpy as np
import xarray as xr

from sup3r.containers.abstract import AbstractContainer
from sup3r.containers.base import Container
from sup3r.containers.derivers import Deriver
from sup3r.containers.extracters import Extracter

np.random.seed(42)

logger = logging.getLogger(__name__)


class Cacher(AbstractContainer):
    """Base extracter object."""

    def __init__(
        self,
        container: Union[Container, Extracter, Deriver],
        cache_kwargs: Dict,
    ):
        """
        Parameters
        ----------
        container : Union[Extracter, Deriver]
            Extracter or Deriver type container containing data to cache
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
        super().__init__()
        self.container = container
        self.out_files = self.cache_data(cache_kwargs)

    def cache_data(self, kwargs):
        """Cache data to file with file type based on user provided
        cache_pattern.

        Parameters
        ----------
        cache_kwargs : dict
            Can include 'cache_pattern' and 'chunks'. 'chunks' is a dictionary
            of tuples (time, lats, lons) for each feature specifying the chunks
            for h5 writes. 'cache_pattern' must have a {feature} format key.
        """
        cache_pattern = kwargs['cache_pattern']
        chunks = kwargs.get('chunks', None)
        msg = 'cache_pattern must have {feature} format key.'
        assert '{feature}' in cache_pattern, msg
        _, ext = os.path.splitext(cache_pattern)
        write_features = [
            f for f in self.features if len(self.container[f].shape) == 3
        ]
        out_files = [cache_pattern.format(feature=f) for f in write_features]
        for feature, out_file in zip(write_features, out_files):
            if not os.path.exists(out_file):
                logger.info(f'Writing {feature} to {out_file}.')
                if ext == '.h5':
                    self._write_h5(
                        out_file,
                        feature,
                        np.transpose(self.container[feature], axes=(2, 0, 1)),
                        self.container.data.coords,
                        chunks,
                    )
                elif ext == '.nc':
                    self._write_netcdf(
                        out_file,
                        feature,
                        np.transpose(self.container[feature], axes=(2, 0, 1)),
                        self.container.data.coords,
                    )
                else:
                    msg = (
                        'cache_pattern must have either h5 or nc '
                        f'extension. Recived {ext}.'
                    )
                    logger.error(msg)
                    raise ValueError(msg)
        logger.info(f'Finished writing {out_files}.')
        return out_files

    def _write_h5(self, out_file, feature, data, coords, chunks=None):
        """Cache data to h5 file using user provided chunks value."""
        chunks = chunks or {}
        with h5py.File(out_file, 'w') as f:
            lats = coords['latitude'].data
            lons = coords['longitude'].data
            times = coords['time'].astype(int)
            data_dict = dict(
                zip(
                    ['time_index', 'latitude', 'longitude', feature],
                    [da.from_array(times), lats, lons, data],
                )
            )
            for dset, vals in data_dict.items():
                if dset in ('latitude', 'longitude'):
                    dset = f'meta/{dset}'
                d = f.require_dataset(
                    f'/{dset}',
                    dtype=vals.dtype,
                    shape=vals.shape,
                    chunks=chunks.get(dset, None),
                )
                da.store(vals, d)
                logger.debug(f'Added {dset} to {out_file}.')

    def _write_netcdf(self, out_file, feature, data, coords):
        """Cache data to a netcdf file."""
        data_vars = {feature: (('time', 'south_north', 'west_east'), data)}
        out = xr.Dataset(data_vars=data_vars, coords=coords)
        out.to_netcdf(out_file)
