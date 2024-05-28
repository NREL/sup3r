"""Basic objects that can perform spatial / temporal extractions of requested
features on loaded data."""

import logging
from abc import ABC, abstractmethod

from sup3r.containers.base import Container
from sup3r.containers.loaders.base import Loader

logger = logging.getLogger(__name__)


class Extracter(Container, ABC):
    """Container subclass with additional methods for extracting a
    spatiotemporal extent from contained data."""

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
        super().__init__()
        self.loader = loader
        self.time_slice = time_slice
        self._grid_shape = shape
        self._target = target
        self._lat_lon = None
        self._time_index = None
        self._raster_index = None
        self._full_lat_lon = None
        self.data = self.extract_data().slice_dset(features=features)

    @property
    def time_slice(self):
        """Return time slice for extracted time period."""
        return self._time_slice

    @time_slice.setter
    def time_slice(self, value):
        """Set and sanitize the time slice."""
        self._time_slice = value if value is not None else slice(None)

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
    def full_lat_lon(self):
        """Get full lat/lon grid from loader."""
        if self._full_lat_lon is None:
            self._full_lat_lon = self.loader.lat_lon
        return self._full_lat_lon

    @property
    def time_index(self):
        """Get the time index for the time period of interest."""
        if self._time_index is None:
            self._time_index = self.loader.time_index[self.time_slice]
        return self._time_index

    @property
    def lat_lon(self):
        """Get 2D grid of coordinates with `target` as the lower left
        coordinate. (lats, lons, 2)"""
        if self._lat_lon is None:
            self._lat_lon = self.get_lat_lon()
        return self._lat_lon

    @abstractmethod
    def get_raster_index(self):
        """Get array of indices used to select the spatial region of
        interest."""

    def get_time_index(self):
        """Get the time index corresponding to the requested time_slice"""
        return self.loader['time'][self.time_slice]

    @abstractmethod
    def get_lat_lon(self):
        """Get 2D grid of coordinates with `target` as the lower left
        coordinate. (lats, lons, 2)"""

    @abstractmethod
    def extract_data(self):
        """Get extracted data by slicing loader.data with calculated
        raster_index and time_slice.

        Returns
        -------
        xr.Dataset
            xr.Dataset() object with extracted features. When `self.data` is
            set with this, `self._data` will be wrapped with
            :class:`DataWrapper` class so that `self.data` will return a
            :class:`DataWrapper` object.
        """
