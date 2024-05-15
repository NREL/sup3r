"""Basic container objects can perform transformations / extractions on the
contained data."""

import logging
from abc import ABC

import numpy as np

from sup3r.containers.loaders import Loader
from sup3r.containers.wranglers.abstract import AbstractWrangler

np.random.seed(42)

logger = logging.getLogger(__name__)


class Wrangler(AbstractWrangler, ABC):
    """Base Wrangler object."""

    def __init__(
        self,
        container: Loader,
        features,
        target,
        shape,
        time_slice=slice(None),
        transform_function=None,
        cache_kwargs=None
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
        cache_kwargs : dict
            Dictionary with kwargs for caching wrangled data. This should at
            minimum include a 'cache_pattern' key, value. This pattern must
            have a {feature} format key and either a h5 or nc file extension,
            based on desired output type.

            Can also include a 'chunks' key, value with a dictionary of tuples
            for each feature. e.g. {'cache_pattern': ..., 'chunks':
            {'windspeed_100m': (20, 100, 100)}} where the chunks ordering is
            (time, lats, lons)

            Note: This is only for saving cached data. If you want to reload
            the cached files load them with a Loader object.
        """
        super().__init__(
            container=container,
            features=features,
            target=target,
            shape=shape,
            time_slice=time_slice,
            transform_function=transform_function,
            cache_kwargs=cache_kwargs
        )
