"""Basic container objects can perform transformations / extractions on the
contained data."""

import logging
from abc import ABC

import numpy as np

from sup3r.containers.extracters.abstract import AbstractExtracter
from sup3r.containers.loaders import Loader

np.random.seed(42)

logger = logging.getLogger(__name__)


class Extracter(AbstractExtracter, ABC):
    """Base extracter object."""

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
        super().__init__(
            container=container,
            target=target,
            shape=shape,
            time_slice=time_slice
        )
