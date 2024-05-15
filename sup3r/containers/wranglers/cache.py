"""Basic container objects can perform transformations / extractions on the
contained data."""

import logging
import os
from abc import ABC

import numpy as np

from sup3r.containers.loaders import Loader
from sup3r.containers.wranglers.base import Wrangler

np.random.seed(42)

logger = logging.getLogger(__name__)


class WranglerH5(Wrangler, ABC):
    """Wrangler subclass for h5 files specifically."""

    def __init__(
        self,
        container: Loader,
        features,
        target=(),
        shape=(),
        raster_file=None,
        time_slice=slice(None),
        max_delta=20,
        transform_function=None,
    ):
        """
        Parameters
        ----------
        container : Loader
            Loader type container with `.data` attribute exposing data to
            wrangle.
        features : list
            List of feature names to extract from data exposed through Loader.
            These are not necessarily the same as the features used to
            initialize the Loader.
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
        max_delta : int
            Optional maximum limit on the raster shape that is retrieved at
            once. If shape is (20, 20) and max_delta=10, the full raster will
            be retrieved in four chunks of (10, 10). This helps adapt to
            non-regular grids that curve over large distances.
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
        self.raster_file = raster_file
        self.max_delta = max_delta
        if self.raster_file is not None:
            self.save_raster_index()

    def save_raster_index(self):
        """Save raster index to cache file."""
        np.savetxt(self.raster_file, self.raster_index)
        logger.info(f'Saved raster_index to {self.raster_file}')

    def get_raster_index(self):
        """Get set of slices or indices selecting the requested region from
        the contained data."""
        if self.raster_file is None or not os.path.exists(self.raster_file):
            logger.info(f'Calculating raster_index for target={self.target}, '
                        f'shape={self.shape}.')
            raster_index = self.container.res.get_raster_index(
                self.target, self.grid_shape, max_delta=self.max_delta
            )
        else:
            raster_index = np.loadtxt(self.raster_file)
            logger.info(f'Loaded raster_index from {self.raster_file}')

        return raster_index

    def get_time_index(self):
        """Get the time index corresponding to the requested time_slice"""
        return self.container.res.time_index[self.time_slice]

    def get_lat_lon(self):
        """Get the 2D array of coordinates corresponding to the requested
        target and shape."""
        return (
            self.container.res.meta[['latitude', 'longitude']]
            .iloc[self.raster_index.flatten()]
            .values.reshape((*self.grid_shape, 2))
        )

    def extract_features(self):
        """Extract the requested features for the requested target + grid_shape
        + time_slice."""
        out = self.container.data[self.raster_index.flatten(), self.time_slice]
        return out.reshape((*self.shape, len(self.features)))
