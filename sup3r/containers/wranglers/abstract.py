"""Basic container objects can perform transformations / extractions on the
contained data."""

import logging
import os
from abc import ABC, abstractmethod

import dask.array as da
import h5py
import numpy as np
import xarray as xr

from sup3r.containers.abstract import AbstractContainer
from sup3r.containers.loaders.base import Loader

np.random.seed(42)

logger = logging.getLogger(__name__)


class AbstractWrangler(AbstractContainer, ABC):
    """Loader subclass with additional methods for wrangling data. e.g.
    Extracting specific spatiotemporal extents and features and deriving new
    features."""

    def __init__(
        self,
        container: Loader,
        features,
        target=(),
        shape=(),
        time_slice=slice(None),
        transform_function=None,
        cache_kwargs=None,
    ):
        """
        Parameters
        ----------
        loader : Container
            Loader type container. Initialized on file_paths pointing to data
            that will now be wrangled.
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
            or slice(None) the full time dimension is selected.
        transform_function : function
            Optional operation on loader.data. For example, if you want to
            derive U/V and you used the Loader to expose windspeed/direction,
            provide a function that operates on windspeed/direction and returns
            U/V. The final `.data` attribute will be the output of this
            function.

            Note: This function needs to include a `self` argument. This
            enables access to the members of the Wrangler instance. For
            example::

                def transform_ws_wd(self, data):

                    from sup3r.utilities.utilities import transform_rotate_wind
                    ws, wd = data[..., 0], data[..., 1]
                    u, v = transform_rotate_wind(ws, wd, self.lat_lon)
                    data[..., 0], data[..., 1] = u, v

                    return data

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
        self.time_slice = time_slice
        self.features = features
        self.transform_function = transform_function
        self.cache_kwargs = cache_kwargs
        self._grid_shape = shape
        self._target = target
        self._data = None
        self._lat_lon = None
        self._time_index = None
        self._raster_index = None

    @property
    def target(self):
        """Return the true value based on the closest lat lon instead of the
        user provided value self._target, which is used to find the closest lat
        lon."""
        return self.lat_lon[-1, 0]

    @property
    def grid_shape(self):
        """Return the grid_shape based on the raster_index, since
        self._grid_shape does not need to be provided as an input if the
        raster_file is."""
        return self.lat_lon.shape[:-1]

    @property
    def raster_index(self):
        """Get array of indices used to select the spatial region of
        interest."""
        if self._raster_index is None:
            self._raster_index = self.get_raster_index()
        return self._raster_index

    @property
    def time_index(self):
        """Get the time index for the time period of interest."""
        if self._time_index is None:
            self._time_index = self.get_time_index()
        return self._time_index

    @property
    def lat_lon(self):
        """Get 2D grid of coordinates with `target` as the lower left
        coordinate. (lats, lons, 2)"""
        if self._lat_lon is None:
            self._lat_lon = self.get_lat_lon()
        return self._lat_lon

    @property
    def data(self):
        """Get extracted feature data."""
        if self._data is None:
            data = self.extract_features()
            if self.transform_function is not None:
                data = self.transform_function(self, data)
            self._data = data
        return self._data

    @abstractmethod
    def extract_features(self):
        """'Extract' requested features to dask.array (lats, lons, time,
        features)"""

    @abstractmethod
    def get_raster_index(self):
        """Get array of indices used to select the spatial region of
        interest."""

    @abstractmethod
    def get_time_index(self):
        """Get the time index for the time period of interest."""

    @abstractmethod
    def get_lat_lon(self):
        """Get 2D grid of coordinates with `target` as the lower left
        coordinate. (lats, lons, 2)"""

    def __getitem__(self, key):
        return self.data[key]

    @property
    def shape(self):
        """Define spatiotemporal shape of extracted extent."""
        return (*self.grid_shape, len(self.time_index))

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
