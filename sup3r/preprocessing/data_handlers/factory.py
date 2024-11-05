"""DataHandler objects, which are built through composition of
:class:`~sup3r.preprocessing.rasterizers.Rasterizer`,
:class:`~sup3r.preprocessing.loaders.Loader`,
:class:`~sup3r.preprocessing.derivers.Deriver`, and
:class:`~sup3r.preprocessing.cachers.Cacher` classes.

TODO: If ``.data`` is a ``Sup3rDataset`` with more than one member only the
high resolution data is cached. We could extend caching and loading to write /
load both with groups
"""

import logging
from typing import Callable, Dict, Optional, Union

from rex import MultiFileNSRDBX

from sup3r.preprocessing.base import (
    Sup3rDataset,
)
from sup3r.preprocessing.cachers import Cacher
from sup3r.preprocessing.cachers.utilities import _check_for_cache
from sup3r.preprocessing.derivers import Deriver
from sup3r.preprocessing.derivers.methods import (
    RegistryH5SolarCC,
    RegistryH5WindCC,
)
from sup3r.preprocessing.loaders import Loader
from sup3r.preprocessing.rasterizers import Rasterizer
from sup3r.preprocessing.utilities import (
    expand_paths,
    log_args,
    parse_to_list,
)

logger = logging.getLogger(__name__)


class DataHandler(Deriver):
    """Base DataHandler. Composes
    :class:`~sup3r.preprocessing.rasterizers.Rasterizer`,
    :class:`~sup3r.preprocessing.loaders.Loader`,
    :class:`~sup3r.preprocessing.derivers.Deriver`, and
    :class:`~sup3r.preprocessing.cachers.Cacher` classes."""

    @log_args
    def __init__(
        self,
        file_paths,
        features='all',
        res_kwargs: Optional[dict] = None,
        chunks: Union[str, Dict[str, int]] = 'auto',
        target: Optional[tuple] = None,
        shape: Optional[tuple] = None,
        time_slice: Union[slice, tuple, list, None] = slice(None),
        threshold: Optional[float] = None,
        time_roll: int = 0,
        time_shift: Optional[int] = None,
        hr_spatial_coarsen: int = 1,
        nan_method_kwargs: Optional[dict] = None,
        BaseLoader: Optional[Callable] = None,
        FeatureRegistry: Optional[dict] = None,
        interp_kwargs: Optional[dict] = None,
        cache_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        file_paths : str | list | pathlib.Path
            file_paths input to LoaderClass
        features : list | str
            Features to load and / or derive. If 'all' then all available raw
            features will be loaded. Specify explicit feature names for
            derivations.
        res_kwargs : dict
            Additional keyword arguments passed through to the ``BaseLoader``.
            BaseLoader is usually xr.open_mfdataset for NETCDF files and
            MultiFileResourceX for H5 files.
        chunks : dict | str
            Dictionary of chunk sizes to pass through to
            ``dask.array.from_array()`` or ``xr.Dataset().chunk()``. Will be
            converted to a tuple when used in ``from_array()``. These are the
            methods for H5 and NETCDF data, respectively. This argument can
            be "auto" or None in addition to a dictionary. None will not do
            any chunking and load data into memory as ``np.array``
        target : tuple
            (lat, lon) lower left corner of raster. Either need target+shape
            or raster_file.
        shape : tuple
            (rows, cols) grid size. Either need target+shape or raster_file.
        time_slice : slice
            Slice specifying extent and step of temporal extraction. e.g.
            slice(start, stop, step). If equal to slice(None, None, 1) the
            full time dimension is selected.
        threshold : float
            Nearest neighbor euclidean distance threshold. If the coordinates
            are more than this value away from the target lat/lon, an error is
            raised.
        time_roll : int
            Number of steps to roll along the time axis. `Passed to
            xr.Dataset.roll()`
        time_shift : int | None
            Number of minutes to shift time axis. This can be used, for
            example, to shift the time index for daily data so that the time
            stamp for a given day starts at the zeroth minute instead of at
            noon, as is the case for most GCM data.
        hr_spatial_coarsen : int
            Spatial coarsening factor. Passed to `xr.Dataset.coarsen()`
        nan_method_kwargs : str | dict | None
            Keyword arguments for nan handling. If 'mask', time steps with nans
            will be dropped. Otherwise this should be a dict of kwargs which
            will be passed to
            :py:meth:`sup3r.preprocessing.accessor.Sup3rX.interpolate_na`.
        BaseLoader : Callable
            Base level file loader wrapped by
            :class:`~sup3r.preprocessing.loaders.Loader`. This is usually
            xr.open_mfdataset for NETCDF files and MultiFileResourceX for H5
            files.
        FeatureRegistry : dict
            Dictionary of
            :class:`~sup3r.preprocessing.derivers.methods.DerivedFeature`
            objects used for derivations
        interp_kwargs : dict | None
            Dictionary of kwargs for level interpolation. Can include "method"
            and "run_level_check" keys. Method specifies how to perform height
            interpolation. e.g. Deriving u_20m from u_10m and u_100m. Options
            are "linear" and "log". See
            :py:meth:`sup3r.preprocessing.derivers.Deriver.do_level_interpolation`
        cache_kwargs : dict | None
            Dictionary with kwargs for caching wrangled data. This should at
            minimum include a `cache_pattern` key, value. This pattern must
            have a {feature} format key and either a h5 or nc file extension,
            based on desired output type. See class:`Cacher` for description
            of more arguments.
        kwargs : dict
            Dictionary of additional keyword args for
            :class:`~sup3r.preprocessing.rasterizers.Rasterizer`, used
            specifically for rasterizing flattened data
        """  # pylint: disable=line-too-long

        features = parse_to_list(features=features)
        cached_files, cached_features, _, missing_features = _check_for_cache(
            features=features, cache_kwargs=cache_kwargs
        )

        just_coords = not features
        raster_feats = 'all' if any(missing_features) else []
        self.rasterizer = self.loader = self.cache = None

        if any(cached_features):
            self.cache = Loader(
                file_paths=cached_files,
                res_kwargs=res_kwargs,
                chunks=chunks,
                BaseLoader=BaseLoader,
            )
            self.rasterizer = self.loader = self.cache

        if any(missing_features) or just_coords:
            self.rasterizer = Rasterizer(
                file_paths=file_paths,
                res_kwargs=res_kwargs,
                features=raster_feats,
                chunks=chunks,
                target=target,
                shape=shape,
                time_slice=time_slice,
                threshold=threshold,
                BaseLoader=BaseLoader,
                **kwargs,
            )
            self.loader = self.rasterizer.loader

        self.time_slice = self.rasterizer.time_slice
        self.lat_lon = self.rasterizer.lat_lon
        self._rasterizer_hook()
        self.data = self.rasterizer

        if missing_features:
            super().__init__(
                data=self.data,
                features=features,
                time_roll=time_roll,
                time_shift=time_shift,
                hr_spatial_coarsen=hr_spatial_coarsen,
                nan_method_kwargs=nan_method_kwargs,
                FeatureRegistry=FeatureRegistry,
                interp_kwargs=interp_kwargs,
            )

        if self.cache is not None:
            self.data[cached_features] = self.cache.data[cached_features]
            self.rasterizer.file_paths = (
                expand_paths(file_paths) + cached_files
            )

        if cache_kwargs is not None and 'cache_pattern' in cache_kwargs:
            _ = Cacher(data=self.data, cache_kwargs=cache_kwargs)
        self._deriver_hook()

    def _rasterizer_hook(self):
        """Hook in after rasterizer initialization. Implement this to
        extend class functionality with operations after default rasterizer
        initialization. e.g. If special methods are required to add more
        data to the rasterized data or to perform some pre-processing
        before derivations.

        Examples
        --------
         - adding a special method to extract / regrid clearsky_ghi from an
         nsrdb source file prior to derivation of clearsky_ratio.
         - apply bias correction to rasterized data before deriving new
         features
        """

    def _deriver_hook(self):
        """Hook in after deriver initialization. Implement this to extend
        class functionality with operations after default deriver
        initialization. e.g. If special methods are required to derive
        additional features which might depend on non-standard inputs (e.g.
        other source files than those used by the loader)."""


class DailyDataHandler(DataHandler):
    """General data handler class with daily data as an additional attribute.
    xr.Dataset coarsen method employed to compute averages / mins / maxes over
    daily windows. Special treatment of clearsky_ratio, which requires
    derivation from total clearsky_ghi and total ghi.

    TODO:
    (1) Not a fan of manually adding cs_ghi / ghi and then removing. Maybe
    this could be handled through a derivation instead

    (2) We assume daily and hourly data here but we could generalize this to
    go from daily -> any time step. This would then enable the CC models to do
    arbitrary temporal enhancement.
    """

    def __init__(self, file_paths, features, **kwargs):
        """Add features required for daily cs ratio derivation if not
        requested."""

        features = parse_to_list(features=features)
        self.requested_features = features.copy()
        if 'clearsky_ratio' in features:
            needed = [
                f
                for f in self.FEATURE_REGISTRY['clearsky_ratio'].inputs
                if f not in features
            ]
            features.extend(needed)
        super().__init__(file_paths=file_paths, features=features, **kwargs)

    _signature_objs = (__init__, DataHandler)

    def _deriver_hook(self):
        """Hook to run daily coarsening calculations after derivations of
        hourly variables. Replaces data with daily averages / maxes / mins
        / sums"""
        msg = (
            'Data needs to be hourly with at least 24 hours, but data '
            'shape is {}.'.format(self.data.shape)
        )

        day_steps = int(24 * 3600 / self.time_step)
        assert len(self.time_index) % day_steps == 0, msg
        assert len(self.time_index) > day_steps, msg

        n_data_days = int(len(self.time_index) / day_steps)

        logger.info(
            'Calculating daily average datasets for {} training '
            'data days.'.format(n_data_days)
        )
        daily_data = self.data.coarsen(time=day_steps).mean()
        feats = [f for f in self.features if 'clearsky_ratio' not in f]
        feats = (
            feats
            if 'clearsky_ratio' not in self.features
            else [*feats, 'total_clearsky_ghi', 'total_ghi']
        )
        for fname in feats:
            if '_max_' in fname:
                daily_data[fname] = (
                    self.data[fname].coarsen(time=day_steps).max()
                )
            if '_min_' in fname:
                daily_data[fname] = (
                    self.data[fname].coarsen(time=day_steps).min()
                )
            if 'total_' in fname:
                daily_data[fname] = (
                    self.data[fname.split('total_')[-1]]
                    .coarsen(time=day_steps)
                    .sum()
                )

        if 'clearsky_ratio' in self.features:
            daily_data['clearsky_ratio'] = (
                daily_data['total_ghi'] / daily_data['total_clearsky_ghi']
            )

        logger.info(
            'Finished calculating daily average datasets for {} '
            'training data days.'.format(n_data_days)
        )
        hourly_data = self.data[self.requested_features]
        daily_data = daily_data[self.requested_features]
        hourly_data.attrs.update({'name': 'hourly'})
        daily_data.attrs.update({'name': 'daily'})
        self.data = Sup3rDataset(daily=daily_data, hourly=hourly_data)


def DataHandlerFactory(cls, BaseLoader=None, FeatureRegistry=None, name=None):
    """Build composite objects that load from file_paths, rasterize a specified
    region, derive new features, and cache derived data.

    Parameters
    ----------
    BaseLoader : Callable
        Optional base loader update. The default for H5 is MultiFileWindX and
        for NETCDF the default is xarray
    FeatureRegistry : Dict[str, DerivedFeature]
        Dictionary of compute methods for features. This is used to look up how
        to derive features that are not contained in the raw loaded data.
    name : str
        Optional class name, used to resolve `repr(Class)` and distinguish
        partially initialized DataHandlers with different FeatureRegistrys
    """

    class FactoryDataHandler(cls):
        """FactoryDataHandler object. Is a partially initialized instance with
        `BaseLoader`, `FeatureRegistry`, and `name` set."""

        FEATURE_REGISTRY = FeatureRegistry or None
        BASE_LOADER = BaseLoader or None
        __name__ = name or 'FactoryDataHandler'

        def __init__(self, file_paths, features='all', **kwargs):
            """
            Parameters
            ----------
            file_paths : str | list | pathlib.Path
                file_paths input to LoaderClass
            features : list | str
                Features to load and / or derive. If 'all' then all available
                raw features will be loaded. Specify explicit feature names for
                derivations.
            kwargs : dict
                kwargs for parent class, except for FeatureRegistry and
                BaseLoader
            """
            super().__init__(
                file_paths,
                features=features,
                BaseLoader=self.BASE_LOADER,
                FeatureRegistry=self.FEATURE_REGISTRY,
                **kwargs,
            )

        _signature_objs = (cls,)
        _skip_params = ('FeatureRegistry', 'BaseLoader')

    return FactoryDataHandler


DataHandlerH5SolarCC = DataHandlerFactory(
    DailyDataHandler,
    BaseLoader=MultiFileNSRDBX,
    FeatureRegistry=RegistryH5SolarCC,
    name='DataHandlerH5SolarCC',
)


DataHandlerH5WindCC = DataHandlerFactory(
    DailyDataHandler,
    BaseLoader=MultiFileNSRDBX,
    FeatureRegistry=RegistryH5WindCC,
    name='DataHandlerH5WindCC',
)
