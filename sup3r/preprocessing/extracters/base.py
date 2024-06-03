"""Basic objects that can perform spatial / temporal extractions of requested
features on loaded data."""

import logging
from abc import ABC, abstractmethod

from sup3r.preprocessing.abstract import DatasetWrapper
from sup3r.preprocessing.base import Container
from sup3r.preprocessing.loaders.base import Loader

logger = logging.getLogger(__name__)


class Extracter(Container, ABC):
    """Container subclass with additional methods for extracting a
    spatiotemporal extent from contained data."""

    def __init__(
        self,
        loader: Loader,
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
        self.grid_shape = shape
        self.target = target
        self.full_lat_lon = self.loader.lat_lon
        self.raster_index = self.get_raster_index()
        self.time_index = (
            loader.time_index[self.time_slice]
            if not loader.time_independent
            else None
        )
        self._lat_lon = None
        self.data = self.extract_data()

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

    @abstractmethod
    def get_raster_index(self):
        """Get array of indices used to select the spatial region of
        interest."""

    @abstractmethod
    def get_lat_lon(self):
        """Get 2D grid of coordinates with `target` as the lower left
        coordinate. (lats, lons, 2)"""

    @abstractmethod
    def extract_data(self) -> DatasetWrapper:
        """Get extracted data by slicing loader.data with calculated
        raster_index and time_slice.

        Returns
        -------
        DatasetWrapper
            Wrapped xr.Dataset() object with extracted features.
        """
