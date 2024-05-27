"""Basic container object that can perform extractions on the contained NETCDF
data."""

import logging
from abc import ABC
from warnings import warn

import dask.array as da
import numpy as np

from sup3r.containers.extracters.base import Extracter
from sup3r.containers.loaders import Loader

logger = logging.getLogger(__name__)


class BaseExtracterNC(Extracter, ABC):
    """Extracter subclass for h5 files specifically."""

    def __init__(
        self,
        loader: Loader,
        features='all',
        target=None,
        shape=None,
        time_slice=slice(None),
    ):
        """
        Parameters
        ----------
        loader : Loader
            Loader type container with `.data` attribute exposing data to
            extract.
        features : str | None | list
            List of features in include in the final extracted data. If 'all'
            this includes all features available in the loader. If None this
            results in a dataset with just lat / lon / time. To select specific
            features provide a list.
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
            loader=loader,
            features=features,
            target=target,
            shape=shape,
            time_slice=time_slice,
        )

    def extract_data(self):
        """Get rasterized data."""
        return self.loader.isel(
            south_north=self.raster_index[0],
            west_east=self.raster_index[1],
            time=self.time_slice,
        )

    def check_target_and_shape(self, full_lat_lon):
        """NETCDF files tend to use a regular grid so if either target or shape
        is not given we can easily find the values that give the maximum
        extent."""
        if not self._target:
            self._target = full_lat_lon[-1, 0, :]
        if not self._grid_shape:
            self._grid_shape = full_lat_lon.shape[:-1]

    def get_raster_index(self):
        """Get set of slices or indices selecting the requested region from
        the contained data."""
        self.check_target_and_shape(self.full_lat_lon)
        row, col = self.get_closest_row_col(self.full_lat_lon, self._target)
        lat_slice = slice(row - self._grid_shape[0] + 1, row + 1)
        lon_slice = slice(col, col + self._grid_shape[1])
        return self._check_raster_index(lat_slice, lon_slice)

    def _check_raster_index(self, lat_slice, lon_slice):
        """Check if raster index has bounds which exceed available region and
        crop if so."""
        lat_start, lat_end = lat_slice.start, lat_slice.stop
        lon_start, lon_end = lon_slice.start, lon_slice.stop
        lat_start = max(lat_start, 0)
        lat_end = min(lat_end, self.full_lat_lon.shape[0])
        lon_start = max(lon_start, 0)
        lon_end = min(lon_end, self.full_lat_lon.shape[1])
        new_lat_slice = slice(lat_start, lat_end)
        new_lon_slice = slice(lon_start, lon_end)
        msg = (
            f'Computed lat_slice = {lat_slice} exceeds available region. '
            f'Using {new_lat_slice}'
        )
        if lat_slice != new_lat_slice:
            logger.warning(msg)
            warn(msg)
        msg = (
            f'Computed lon_slice = {lon_slice} exceeds available region. '
            f'Using {new_lon_slice}'
        )
        if lon_slice != new_lon_slice:
            logger.warning(msg)
            warn(msg)
        return new_lat_slice, new_lon_slice

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
        return da.unravel_index(da.argmin(dist, axis=None), dist.shape)

    def get_lat_lon(self):
        """Get the 2D array of coordinates corresponding to the requested
        target and shape."""
        return self.full_lat_lon[*self.raster_index]
