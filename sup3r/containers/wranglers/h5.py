"""Basic container objects can perform transformations / extractions on the
contained data."""

import logging
from abc import ABC

import numpy as np

from sup3r.containers.wranglers.abstract import AbstractWrangler

np.random.seed(42)

logger = logging.getLogger(__name__)


class WranglerH5(AbstractWrangler, ABC):
    """Wrangler subclass for h5 files specifically."""

    def __init__(self,
                 file_paths,
                 features,
                 target,
                 shape,
                 raster_file=None,
                 temporal_slice=slice(None, None, 1),
                 res_kwargs=None,
                 ):
        """
        Parameters
        ----------
        file_paths : str | pathlib.Path | list
            Globbable path str(s) or pathlib.Path for file locations.
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
        temporal_slice : slice
            Slice specifying extent and step of temporal extraction. e.g.
            slice(start, stop, time_pruning). If equal to slice(None, None, 1)
            the full time dimension is selected.
        res_kwargs : dict | None
            Dictionary of kwargs to pass to xarray.open_mfdataset.
        """
        super().__init__(file_paths, features=features)
        self.res_kwargs = res_kwargs or {}
        self.raster_file = raster_file
        self.temporal_slice = temporal_slice
        self.target = target
        self.grid_shape = shape
        self.time_index = self.get_time_index()
        self.lat_lon = self.get_lat_lon()
        self.raster_index = self.get_raster_index()
        self.data = self.load()
