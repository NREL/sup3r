"""Basic container object that can perform extractions on the contained H5
data."""

import logging
import os
from abc import ABC

import numpy as np

from sup3r.containers.extracters.base import Extracter
from sup3r.containers.loaders import Loader

np.random.seed(42)

logger = logging.getLogger(__name__)


class ExtracterH5(Extracter, ABC):
    """Extracter subclass for h5 files specifically."""

    def __init__(
        self,
        container: Loader,
        target=(),
        shape=(),
        time_slice=slice(None),
        raster_file=None,
        max_delta=20,
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
        raster_file : str | None
            File for raster_index array for the corresponding target and shape.
            If specified the raster_index will be loaded from the file if it
            exists or written to the file if it does not yet exist. If None and
            raster_index is not provided raster_index will be calculated
            directly. Either need target+shape, raster_file, or raster_index
            input.
        max_delta : int
            Optional maximum limit on the raster shape that is retrieved at
            once. If shape is (20, 20) and max_delta=10, the full raster will
            be retrieved in four chunks of (10, 10). This helps adapt to
            non-regular grids that curve over large distances.
        """
        super().__init__(
            container=container,
            target=target,
            shape=shape,
            time_slice=time_slice
        )
        self.raster_file = raster_file
        self.max_delta = max_delta
        if self.raster_file is not None and not os.path.exists(
            self.raster_file
        ):
            self.save_raster_index()

    def save_raster_index(self):
        """Save raster index to cache file."""
        np.savetxt(self.raster_file, self.raster_index)
        logger.info(f'Saved raster_index to {self.raster_file}')

    def get_raster_index(self):
        """Get set of slices or indices selecting the requested region from
        the contained data."""
        if self.raster_file is None or not os.path.exists(self.raster_file):
            logger.info(
                f'Calculating raster_index for target={self._target}, '
                f'shape={self._grid_shape}.'
            )
            raster_index = self.container.res.get_raster_index(
                self._target, self._grid_shape, max_delta=self.max_delta
            )
        else:
            raster_index = np.loadtxt(self.raster_file)
            logger.info(f'Loaded raster_index from {self.raster_file}')

        return raster_index

    def get_time_index(self):
        """Get the time index corresponding to the requested time_slice"""
        if 'time_index' in self.container.res:
            raw_time_index = self.container.res['time_index']
        elif hasattr(self.container.res, 'time_index'):
            raw_time_index = self.container.res.time_index
        else:
            msg = (f'Could not get time_index from {self.container.res}')
            logger.error(msg)
            raise RuntimeError(msg)
        return raw_time_index[self.time_slice]

    def get_lat_lon(self):
        """Get the 2D array of coordinates corresponding to the requested
        target and shape."""
        return (
            self.container.res.meta[['latitude', 'longitude']]
            .iloc[self.raster_index.flatten()]
            .values.reshape((*self.raster_index.shape, 2))
        )

    def extract_features(self):
        """Extract the requested features for the requested target + grid_shape
        + time_slice."""
        out = self.container[self.raster_index.flatten(), self.time_slice]
        return out.reshape((*self.shape, len(self.features)))
