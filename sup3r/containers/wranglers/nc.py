"""Basic container objects can perform transformations / extractions on the
contained data."""

import logging
from abc import ABC

import numpy as np

from sup3r.containers.loaders import Loader
from sup3r.containers.wranglers.base import Wrangler

np.random.seed(42)

logger = logging.getLogger(__name__)


class WranglerNC(Wrangler, ABC):
    """Wrangler subclass for h5 files specifically."""

    def __init__(
        self,
        container: Loader,
        features,
        target=None,
        shape=None,
        time_slice=slice(None),
        transform_function=None,
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
        raster_file : str | None
            File for raster_index array for the corresponding target and shape.
            If specified the raster_index will be loaded from the file if it
            exists or written to the file if it does not yet exist. If None and
            raster_index is not provided raster_index will be calculated
            directly. Either need target+shape, raster_file, or raster_index
            input.
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
        """
        super().__init__(
            container=container,
            features=features,
            target=target,
            shape=shape,
            time_slice=time_slice,
            transform_function=transform_function,
        )
        self.check_target_and_shape()

    def check_target_and_shape(self):
        """NETCDF files tend to use a regular grid so if either target or shape
        is not given we can easily find the values that give the maximum
        extent."""
        full_lat_lon = self._get_full_lat_lon()
        if self._target is None:
            lat = (
                full_lat_lon[-1, 0, 0]
                if self._has_descending_lats()
                else full_lat_lon[0, 0, 0]
            )
            lon = (
                full_lat_lon[-1, 0, 1]
                if self._has_descending_lats()
                else full_lat_lon[0, 0, 1]
            )
            self._target = (lat, lon)
        if self._grid_shape is None:
            self._grid_shape = full_lat_lon.shape[:-1]

    def _get_full_lat_lon(self):
        lats = self.container.res['latitude'].data
        lons = self.container.res['longitude'].data
        if len(lats.shape) == 1:
            lons, lats = np.meshgrid(lons, lats)
        return np.dstack([lats, lons])

    def _has_descending_lats(self):
        lats = self._get_full_lat_lon()[:, 0, 0]
        return lats[0] > lats[-1]

    def get_raster_index(self):
        """Get set of slices or indices selecting the requested region from
        the contained data."""
        row, col = self.get_closest_row_col(
            self._get_full_lat_lon(), self._target
        )
        if self._has_descending_lats():
            lat_slice = slice(row, row - self._grid_shape[0], -1)
        else:
            lat_slice = slice(row, row + self._grid_shape[0])
        lon_slice = slice(col, col + self._grid_shape[1])
        return (lat_slice, lon_slice)

    @staticmethod
    def get_closest_row_col(lat_lon, target):
        """Get closest indices to target lat lon

        Parameters
        ----------
        lat_lon : ndarray
            Array of lat/lon
            (spatial_1, spatial_2, 2)
            Last dimension in order of (lat, lon)
        target : tuple
            (lat, lon) for target coordinate

        Returns
        -------
        row : int
            row index for closest lat/lon to target lat/lon
        col : int
            col index for closest lat/lon to target lat/lon
        """
        dist = np.hypot(
            lat_lon[..., 0] - target[0], lat_lon[..., 1] - target[1]
        )
        row, col = np.where(dist == np.min(dist))
        return row[0], col[0]

    def get_time_index(self):
        """Get the time index corresponding to the requested time_slice"""
        return self.container.res['time'].values[self.time_slice]

    def get_lat_lon(self):
        """Get the 2D array of coordinates corresponding to the requested
        target and shape."""
        return self._get_full_lat_lon()[*self.raster_index]

    def extract_features(self):
        """Extract the requested features for the requested target + grid_shape
        + time_slice."""
        return self.container[*self.raster_index, self.time_slice]
