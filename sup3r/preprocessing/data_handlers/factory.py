"""Basic objects can perform transformations / extractions on the contained
data."""

import logging
from typing import ClassVar

from rex import MultiFileNSRDBX

from sup3r.preprocessing.base import (
    FactoryMeta,
    Sup3rDataset,
    TypeGeneralClass,
)
from sup3r.preprocessing.cachers import Cacher
from sup3r.preprocessing.derivers import Deriver
from sup3r.preprocessing.derivers.methods import (
    RegistryH5,
    RegistryH5SolarCC,
    RegistryH5WindCC,
    RegistryNC,
)
from sup3r.preprocessing.extracters import DirectExtracter
from sup3r.preprocessing.utilities import get_class_kwargs, parse_to_list

logger = logging.getLogger(__name__)


def DataHandlerFactory(
    BaseLoader=None, FeatureRegistry=None, name='TypeSpecificDataHandler'
):
    """Build composite objects that load from file_paths, extract specified
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
        Optional name for class built from factory. This will display in
        logging.

    """

    class TypeSpecificDataHandler(Deriver, metaclass=FactoryMeta):
        """Handler class returned by factory. Composes `Extracter`, `Loader`
        and `Deriver` classes."""

        __name__ = name
        _legos = (Deriver, DirectExtracter)

        if BaseLoader is not None:
            BASE_LOADER = BaseLoader

        FEATURE_REGISTRY = (
            FeatureRegistry
            if FeatureRegistry is not None
            else Deriver.FEATURE_REGISTRY
        )

        def __init__(self, file_paths, features='all', **kwargs):
            """
            Parameters
            ----------
            file_paths : str | list | pathlib.Path
                file_paths input to DirectExtracterClass
            features : list
                Features to derive from loaded data.
            **kwargs : dict
                Dictionary of keyword args for DirectExtracter, Deriver, and
                Cacher
            """
            [cacher_kwargs, deriver_kwargs, extracter_kwargs] = (
                get_class_kwargs([Cacher, Deriver, DirectExtracter], kwargs)
            )
            features = parse_to_list(features=features)
            self.extracter = DirectExtracter(
                file_paths=file_paths, **extracter_kwargs
            )
            self._extracter_hook()
            super().__init__(
                data=self.extracter.data, features=features, **deriver_kwargs
            )
            self._deriver_hook()
            cache_kwargs = cacher_kwargs.get('cache_kwargs', {})
            if cache_kwargs is not None and 'cache_pattern' in cache_kwargs:
                _ = Cacher(data=self.data, **cacher_kwargs)

        def _extracter_hook(self):
            """Hook in after extracter initialization. Implement this to extend
            class functionality with operations after default extracter
            initialization. e.g. If special methods are required to add more
            data to the extracted data or to perform some pre-processing before
            derivations.

            Examples
            --------
             - adding a special method to extract / regrid clearsky_ghi from an
             nsrdb source file prior to derivation of clearsky_ratio.
             - apply bias correction to extracted data before deriving new
             features
            """

        def _deriver_hook(self):
            """Hook in after deriver initialization. Implement this to extend
            class functionality with operations after default deriver
            initialization. e.g. If special methods are required to derive
            additional features which might depend on non-standard inputs (e.g.
            other source files than those used by the loader)."""

        def __getattr__(self, attr):
            """Look for attribute in extracter and then loader if not found in
            self.

            TODO: Not a fan of the hardcoded list here. Find better way.
            """
            if attr in [
                'lat_lon',
                'target',
                'grid_shape',
                'time_slice',
                'time_index',
            ]:
                return getattr(self.extracter, attr)
            try:
                return Deriver.__getattr__(self, attr)
            except Exception as e:
                msg = f'{self.__class__.__name__} has no attribute "{attr}"'
                raise AttributeError(msg) from e

        def __repr__(self):
            return f"<class '{self.__module__}.__name__'>"

    return TypeSpecificDataHandler


def DailyDataHandlerFactory(
    BaseLoader=None, FeatureRegistry=None, name='DailyDataHandler'
):
    """Handler factory for data handlers with additional daily_data.

    TODO: Not a fan of manually adding cs_ghi / ghi and then removing. Maybe
    this could be handled through a derivation instead
    """

    class DailyDataHandler(
        DataHandlerFactory(
            BaseLoader=BaseLoader, FeatureRegistry=FeatureRegistry
        )
    ):
        """General data handler class with daily data as an additional
        attribute. xr.Dataset coarsen method employed to compute averages /
        mins / maxes over daily windows. Special treatment of clearsky_ratio,
        which requires derivation from total clearsky_ghi and total ghi.

        TODO: We assume daily and hourly data here but we could generalize this
        to go from daily -> any time step. This would then enable the CC models
        to do arbitrary temporal enhancement.
        """

        __name__ = name

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
            super().__init__(
                file_paths=file_paths, features=features, **kwargs
            )

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

    return DailyDataHandler


DataHandlerH5 = DataHandlerFactory(
    FeatureRegistry=RegistryH5, name='DataHandlerH5'
)
DataHandlerNC = DataHandlerFactory(
    FeatureRegistry=RegistryNC, name='DataHandlerNC'
)


class DataHandler(TypeGeneralClass):
    """`DataHandler` class which parses input file type and returns
    appropriate `TypeSpecificDataHandler`."""

    _legos = (DataHandlerNC, DataHandlerH5)
    TypeSpecificClass: ClassVar = dict(zip(['nc', 'h5'], _legos))


def _base_loader(file_paths, **kwargs):
    return MultiFileNSRDBX(file_paths, **kwargs)


DataHandlerH5SolarCC = DailyDataHandlerFactory(
    BaseLoader=_base_loader,
    FeatureRegistry=RegistryH5SolarCC,
    name='DataHandlerH5SolarCC',
)


DataHandlerH5WindCC = DailyDataHandlerFactory(
    BaseLoader=_base_loader,
    FeatureRegistry=RegistryH5WindCC,
    name='DataHandlerH5WindCC',
)
