"""Basic container object that can perform extractions on the contained NETCDF
data."""

import logging
from abc import ABC

import numpy as np

from sup3r.containers.extracters.base import Extracter
from sup3r.containers.loaders import Loader

np.random.seed(42)

logger = logging.getLogger(__name__)


class ExtracterNC(Extracter, ABC):
    """Extracter subclass for h5 files specifically."""

    def __init__(
        self,
        container: Loader,
        target=None,
        shape=None,
        time_slice=slice(None),
    ):
        """
        Parameters
        ----------
        container : Loader
            Loader type container with `.data` attribute exposing data to
            extract.
        target : tuple
            (lat, lon) lower left corner of raster. Either need target+shape or
            raster_file.
        shape : tuple
            (rows, cols) grid size. Either need target+shape or raster_file.
        time_slice : slice
            Slice specifying extent and step of temporal extraction. e.g.
            slice(start, stop, step). If equal to slice(None, None, 1)
            the full time dimension is selected.
        """
        super().__init__(
            container=container,
            target=target,
            shape=shape,
            time_slice=time_slice,
        )

    def check_target_and_shape(self, full_lat_lon):
        """NETCDF files tend to use a regular grid so if either target or shape
        is not given we can easily find the values that give the maximum
        extent."""
        if not self._target:
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
        if not self._grid_shape:
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
        full_lat_lon = self._get_full_lat_lon()
        self.check_target_and_shape(full_lat_lon)
        row, col = self.get_closest_row_col(full_lat_lon, self._target)
        if self._has_descending_lats():
            lat_slice = slice(row - self._grid_shape[0] + 1, row + 1)
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
        lat_lon = self._get_full_lat_lon()[*self.raster_index]
        if self._has_descending_lats():
            lat_lon = lat_lon[::-1]
        return lat_lon

    def extract_features(self):
        """Extract the requested features for the requested target + grid_shape
        + time_slice."""
        out = self.container[*self.raster_index, self.time_slice]
        if self._has_descending_lats():
            out = out[::-1]
        return out
