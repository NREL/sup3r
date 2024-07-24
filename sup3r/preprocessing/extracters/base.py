"""Basic objects that can perform spatial / temporal extractions of requested
features on 3D loaded data."""

import logging
from warnings import warn

import dask.array as da
import numpy as np

from sup3r.preprocessing.base import Container
from sup3r.preprocessing.utilities import _compute_if_dask, _parse_time_slice

logger = logging.getLogger(__name__)


class BaseExtracter(Container):
    """Container subclass with additional methods for extracting a
    spatiotemporal extent from contained data.

    Note
    ----
    This `Extracter` base class is for 3D rasterized data. This usually
    comes from NETCDF files but can also be cached H5 files saved from
    previously rasterized data. For 3D, whether H5 or NETCDF, the full domain
    will be extracted automatically if no target / shape are provided."""

    def __init__(
        self,
        loader,
        features='all',
        target=None,
        shape=None,
        time_slice=slice(None),
        threshold=0.5
    ):
        """
        Parameters
        ----------
        loader : Loader
            Loader type container with `.data` attribute exposing data to
            extract.
        features : list | str
            Features to return in loaded dataset. If 'all' then all available
            features will be returned.
        target : tuple
            (lat, lon) lower left corner of raster. Either need target+shape or
            raster_file.
        shape : tuple
            (rows, cols) grid size. Either need target+shape or raster_file.
        time_slice : slice
            Slice specifying extent and step of temporal extraction. e.g.
            slice(start, stop, step). If equal to slice(None, None, 1) the full
            time dimension is selected.
        threshold : float
            Nearest neighbor euclidean distance threshold. If the coordinates
            are more than this value away from the target lat/lon, an error is
            raised.
        """
        super().__init__(data=loader.data)
        self.loader = loader
        self.threshold = threshold
        self.time_slice = time_slice
        self.grid_shape = shape
        self.target = target
        self.full_lat_lon = self.data.lat_lon
        self.raster_index = self.get_raster_index()
        self.time_index = (
            loader.time_index[self.time_slice]
            if 'time' in loader.data.indexes
            else None
        )
        self._lat_lon = None
        self.data = self.extract_data()
        self.data = self.data[features]

    @property
    def time_slice(self):
        """Return time slice for extracted time period."""
        return self._time_slice

    @time_slice.setter
    def time_slice(self, value):
        """Set and sanitize the time slice."""
        self._time_slice = _parse_time_slice(value)

    @property
    def target(self):
        """Return the true value based on the closest lat lon instead of the
        user provided value self._target, which is used to find the closest lat
        lon."""
        return _compute_if_dask(self.lat_lon[-1, 0])

    @target.setter
    def target(self, value):
        """Set the private target attribute. Ultimately target is determined by
        lat_lon but _target is set to bottom left corner of the full domain if
        None and then used to get the raster_index, which is then used to get
        the lat_lon"""
        self._target = value

    @property
    def grid_shape(self):
        """Return the grid_shape based on the raster_index, since
        self._grid_shape does not need to be provided as an input if the
        raster_file is."""
        return self.lat_lon.shape[:-1]

    @grid_shape.setter
    def grid_shape(self, value):
        """Set the private grid_shape attribute. Ultimately grid_shape is
        determined by lat_lon but _grid_shape is set to the full domain if None
        and then used to get the raster_index, which is then used to get the
        lat_lon"""
        self._grid_shape = value

    @property
    def lat_lon(self):
        """Get 2D grid of coordinates with `target` as the lower left
        coordinate. (lats, lons, 2)"""
        if self._lat_lon is None:
            self._lat_lon = self.get_lat_lon()
        return self._lat_lon

    def extract_data(self):
        """Get rasterized data."""
        return self.loader.isel(
            south_north=self.raster_index[0],
            west_east=self.raster_index[1],
            time=self.time_slice)

    def check_target_and_shape(self, full_lat_lon):
        """The data is assumed to use a regular grid so if either target or
        shape is not given we can easily find the values that give the maximum
        extent."""
        if self._target is None:
            self._target = full_lat_lon[-1, 0, :]
        if self._grid_shape is None:
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
            f'Computed lat_slice = {_compute_if_dask(lat_slice)} exceeds '
            f'available region. Using {_compute_if_dask(new_lat_slice)}.'
        )
        if lat_slice != new_lat_slice:
            logger.warning(msg)
            warn(msg)
        msg = (
            f'Computed lon_slice = {_compute_if_dask(lon_slice)} exceeds '
            f'available region. Using {_compute_if_dask(new_lon_slice)}.'
        )
        if lon_slice != new_lon_slice:
            logger.warning(msg)
            warn(msg)
        return new_lat_slice, new_lon_slice

    def get_closest_row_col(self, lat_lon, target):
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
        row, col = da.unravel_index(da.argmin(dist, axis=None), dist.shape)
        if dist.min() > self.threshold:
            msg = ('The distance between the closest coordinate: '
                   f'{_compute_if_dask(lat_lon[row, col])} in the grid from '
                   f'{self.loader.file_paths} and the requested target '
                   f'{target} exceeds the given threshold: {self.threshold}).')
            logger.error(msg)
            raise RuntimeError(msg)
        return row, col

    def get_lat_lon(self):
        """Get the 2D array of coordinates corresponding to the requested
        target and shape."""
        return self.full_lat_lon[self.raster_index[0], self.raster_index[1]]
