"""Basic container objects can perform transformations / extractions on the
contained data."""

import logging

import numpy as np

from sup3r.containers.cachers import Cacher
from sup3r.containers.derivers import DeriverH5, DeriverNC
from sup3r.containers.extracters import ExtracterH5, ExtracterNC
from sup3r.containers.loaders import Loader

np.random.seed(42)

logger = logging.getLogger(__name__)


class WranglerH5(DeriverH5):
    """Wrangler subclass for H5 files specifically."""

    def __init__(
        self,
        container: Loader,
        features,
        target=(),
        shape=(),
        time_slice=slice(None),
        transform=None,
        cache_kwargs=None,
        raster_file=None,
        max_delta=20,
    ):
        """
        Parameters
        ----------
        container : Loader
            Loader type container with `.data` attribute exposing data to
            wrangle.
        extract_features : list
            List of feature names to derive from data exposed through Loader
            for the spatiotemporal extent specified by target + shape.
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
        transform : function
            Optional operation on extracter data. For example, if you want to
            derive U/V and you used the :class:`Extracter` to expose
            windspeed/direction, provide a function that operates on
            windspeed/direction and returns U/V. The final `.data` attribute
            will be the output of this function.

            Note: This function needs to include a `self` argument. This
            enables access to the members of the :class:`Deriver` instance. For
            example::

                def transform_ws_wd(self, data: Container):

                    from sup3r.utilities.utilities import transform_rotate_wind
                    ws, wd = data['windspeed'], data['winddirection']
                    u, v = transform_rotate_wind(ws, wd, self.lat_lon)
                    self['U'], self['V'] = u, v
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
        extracter = ExtracterH5(
            container=container,
            target=target,
            shape=shape,
            time_slice=time_slice,
            raster_file=raster_file,
            max_delta=max_delta,
        )
        super().__init__(extracter, features=features, transform=transform)

        if cache_kwargs is not None:
            Cacher(self, cache_kwargs)


class WranglerNC(DeriverNC):
    """Wrangler subclass for NETCDF files specifically."""

    def __init__(
        self,
        container: Loader,
        features,
        target=(),
        shape=(),
        time_slice=slice(None),
        transform=None,
        cache_kwargs=None,
    ):
        """
        Parameters
        ----------
        container : Loader
            Loader type container with `.data` attribute exposing data to
            wrangle.
        extract_features : list
            List of feature names to derive from data exposed through Loader
            for the spatiotemporal extent specified by target + shape.
        target : tuple
            (lat, lon) lower left corner of raster. Either need target+shape or
            raster_file.
        shape : tuple
            (rows, cols) grid size. Either need target+shape or raster_file.
        time_slice : slice
            Slice specifying extent and step of temporal extraction. e.g.
            slice(start, stop, step). If equal to slice(None, None, 1)
            the full time dimension is selected.
        transform : function
            Optional operation on extracter data. For example, if you want to
            derive U/V and you used the :class:`Extracter` to expose
            windspeed/direction, provide a function that operates on
            windspeed/direction and returns U/V. The final `.data` attribute
            will be the output of this function.

            Note: This function needs to include a `self` argument. This
            enables access to the members of the :class:`Deriver` instance. For
            example::

                def transform_ws_wd(self, data: Container):

                    from sup3r.utilities.utilities import transform_rotate_wind
                    ws, wd = data['windspeed'], data['winddirection']
                    u, v = transform_rotate_wind(ws, wd, self.lat_lon)
                    self['U'], self['V'] = u, v
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
        extracter = ExtracterNC(
            container=container,
            target=target,
            shape=shape,
            time_slice=time_slice,
        )
        super().__init__(extracter, features=features, transform=transform)

        if cache_kwargs is not None:
            Cacher(self, cache_kwargs)
