"""Basic container object that can perform extractions on the contained H5
data."""

import logging
import os

import numpy as np
import xarray as xr

from sup3r.preprocessing.loaders import LoaderH5
from sup3r.preprocessing.names import Dimension

from .base import BaseExtracter

logger = logging.getLogger(__name__)


class ExtendedExtracter(BaseExtracter):
    """Extended `Extracter` class which also handles the flattened data format
    used for some H5 files (e.g. Wind Toolkit or NSRDB data)

    Arguments added to parent class:

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

    See Also
    --------
    :class:`Extracter` for description of other arguments.
    """

    def __init__(
        self,
        loader: LoaderH5,
        features='all',
        target=None,
        shape=None,
        time_slice=slice(None),
        raster_file=None,
        max_delta=20,
        threshold=None
    ):
        self.raster_file = raster_file
        self.max_delta = max_delta
        super().__init__(
            loader=loader,
            features=features,
            target=target,
            shape=shape,
            time_slice=time_slice,
            threshold=threshold
        )
        if self.raster_file is not None and not os.path.exists(
            self.raster_file
        ):
            self.save_raster_index()

    def extract_data(self):
        """Get rasterized data."""
        if not self.loader.flattened:
            return super().extract_data()
        return self._extract_flat_data()

    def _extract_flat_data(self):
        """Extract data from flattened source data, usually coming from WTK
        or NSRDB data."""
        dims = (Dimension.SOUTH_NORTH, Dimension.WEST_EAST)
        coords = {
            Dimension.LATITUDE: (dims, self.lat_lon[..., 0]),
            Dimension.LONGITUDE: (dims, self.lat_lon[..., 1]),
            Dimension.TIME: self.time_index,
        }
        data_vars = {}
        feats = list(self.loader.data_vars)
        data = self.loader[feats].isel(
            **{Dimension.FLATTENED_SPATIAL: self.raster_index.flatten()}
        )
        for f in self.loader.data_vars:
            if Dimension.TIME in self.loader[f].dims:
                dat = data[f].isel({Dimension.TIME: self.time_slice})
                dat = dat.data.reshape(
                    (*self.grid_shape, len(self.time_index))
                )
                data_vars[f] = ((*dims, Dimension.TIME), dat)
            else:
                dat = data[f].data.reshape(self.grid_shape)
                data_vars[f] = (dims, dat)
        return xr.Dataset(
            coords=coords, data_vars=data_vars, attrs=self.loader.attrs
        )

    def save_raster_index(self):
        """Save raster index to cache file."""
        os.makedirs(os.path.dirname(self.raster_file), exist_ok=True)
        np.savetxt(self.raster_file, self.raster_index)
        logger.info(f'Saved raster_index to {self.raster_file}')

    def get_raster_index(self):
        """Get set of slices or indices selecting the requested region from
        the contained data."""

        if not self.loader.flattened:
            return super().get_raster_index()
        return self._get_flat_data_raster_index()

    def _get_flat_data_raster_index(self):
        """Get raster index for the flattened source data, which usually comes
        from WTK or NSRDB data."""

        if self.raster_file is None or not os.path.exists(self.raster_file):
            logger.info(
                f'Calculating raster_index for target={self._target}, '
                f'shape={self._grid_shape}.'
            )
            raster_index = self.loader.res.get_raster_index(
                self._target, self._grid_shape, max_delta=self.max_delta
            )
        else:
            raster_index = np.loadtxt(self.raster_file).astype(np.int32)
            logger.info(f'Loaded raster_index from {self.raster_file}')

        return raster_index

    def get_lat_lon(self):
        """Get the 2D array of coordinates corresponding to the requested
        target and shape."""

        if not self.loader.flattened:
            return super().get_lat_lon()
        return self._get_flat_data_lat_lon()

    def _get_flat_data_lat_lon(self):
        """Get lat lon for flattened source data."""
        lat_lon = self.full_lat_lon[self.raster_index.flatten()].reshape(
            (*self.raster_index.shape, -1)
        )
        return lat_lon
