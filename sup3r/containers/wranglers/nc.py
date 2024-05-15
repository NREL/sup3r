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
        target,
        shape,
        time_slice=slice(None),
        transform_function=None
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
            transform_function=transform_function
        )

    def get_raster_index(self):
        """Get set of slices or indices selecting the requested region from
        the contained data."""
        full_lat_lon = self.container.res[['latitude', 'longitude']]
        row, col = self.get_closest_row_col(full_lat_lon, self.target)
        lat_slice = slice(row, row + self.grid_shape[0])
        lon_slice = slice(col, col + self.grid_shape[1])
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
        dist = np.hypot(lat_lon[..., 0] - target[0],
                        lat_lon[..., 1] - target[1])
        row, col = np.where(dist == np.min(dist))
        row = row[0]
        col = col[0]
        return row, col

    def get_time_index(self):
        """Get the time index corresponding to the requested time_slice"""
        return self.container.res.time_index[self.time_slice]

    def get_lat_lon(self):
        """Get the 2D array of coordinates corresponding to the requested
        target and shape."""
        return self.container.res[['latitude', 'longitude']][
            self.raster_index
        ].reshape((*self.grid_shape, 2))
