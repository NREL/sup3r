"""Basic container object that can perform extractions on the contained H5
data."""

import logging
import os
from abc import ABC

import numpy as np
import xarray as xr

from sup3r.preprocessing.common import Dimension
from sup3r.preprocessing.extracters.base import Extracter
from sup3r.preprocessing.loaders import LoaderH5

logger = logging.getLogger(__name__)


class BaseExtracterH5(Extracter, ABC):
    """Extracter subclass for h5 files specifically."""

    def __init__(
        self,
        loader: LoaderH5,
        target=None,
        shape=None,
        time_slice=slice(None),
        raster_file=None,
        max_delta=20,
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
        self.raster_file = raster_file
        self.max_delta = max_delta
        super().__init__(
            loader=loader,
            target=target,
            shape=shape,
            time_slice=time_slice,
        )
        if self.raster_file is not None and not os.path.exists(
            self.raster_file
        ):
            self.save_raster_index()

    def extract_data(self):
        """Get rasterized data.

        TODO: Generalize this to handle non-flattened H5 data. Would need to
        encapsulate the flatten call somewhere.
        """
        dims = (Dimension.SOUTH_NORTH, Dimension.WEST_EAST)
        coords = {
            Dimension.LATITUDE: (dims, self.lat_lon[..., 0]),
            Dimension.LONGITUDE: (dims, self.lat_lon[..., 1]),
            Dimension.TIME: self.time_index,
        }
        data_vars = {}
        for f in self.loader.features:
            dat = self.loader[f].data[self.raster_index.flatten()]
            if Dimension.TIME in self.loader[f].dims:
                dat = dat[..., self.time_slice].reshape(
                    (*self.grid_shape, len(self.time_index))
                )
                data_vars[f] = ((*dims, Dimension.TIME), dat)
            else:
                dat = dat.reshape(self.grid_shape)
                data_vars[f] = (dims, dat)
        attrs = {'source_files': self.loader.file_paths}
        return xr.Dataset(coords=coords, data_vars=data_vars, attrs=attrs)

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
            raster_index = self.loader.res.get_raster_index(
                self._target, self._grid_shape, max_delta=self.max_delta
            )
        else:
            raster_index = np.loadtxt(self.raster_file)
            logger.info(f'Loaded raster_index from {self.raster_file}')

        return raster_index

    def get_lat_lon(self):
        """Get the 2D array of coordinates corresponding to the requested
        target and shape."""
        lat_lon = self.full_lat_lon[self.raster_index.flatten()].reshape(
            (*self.raster_index.shape, -1)
        )
        return lat_lon
