"""Basic container objects can perform transformations / extractions on the
contained data."""

import logging
from abc import ABC, abstractmethod

import numpy as np

from sup3r.containers.base import Container
from sup3r.containers.loaders.base import Loader

np.random.seed(42)

logger = logging.getLogger(__name__)


class Extracter(Container, ABC):
    """Container subclass with additional methods for extracting a
    spatiotemporal extent from contained data."""

    def __init__(
        self,
        container: Loader,
        target,
        shape,
        time_slice=slice(None)
    ):
        """
        Parameters
        ----------
        container : Loader
            Loader type container with `.data` attribute exposing data to
            extract.
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
        """
        super().__init__(container)
        self.time_slice = time_slice
        self._grid_shape = shape
        self._target = target
        self._lat_lon = None
        self._time_index = None
        self._raster_index = None
        self.shape = (*self.grid_shape, len(self.time_index))
        self.data = self.extract_features().astype(np.float32)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, trace):
        self.close()

    def close(self):
        """Close Loader."""
        self.container.close()

    @property
    def target(self):
        """Return the true value based on the closest lat lon instead of the
        user provided value self._target, which is used to find the closest lat
        lon."""
        return self.lat_lon[0, 0]

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
