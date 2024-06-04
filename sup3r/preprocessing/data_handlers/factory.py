"""Basic objects can perform transformations / extractions on the contained
data."""

import logging

import numpy as np
from rex import MultiFileNSRDBX
from scipy.stats import mode

from sup3r.preprocessing.cachers import Cacher
from sup3r.preprocessing.common import FactoryMeta, lowered
from sup3r.preprocessing.derivers import Deriver
from sup3r.preprocessing.derivers.methods import (
    RegistryH5,
    RegistryH5SolarCC,
    RegistryH5WindCC,
    RegistryNC,
)
from sup3r.preprocessing.extracters import (
    BaseExtracterH5,
    BaseExtracterNC,
)
from sup3r.preprocessing.loaders import LoaderH5, LoaderNC
from sup3r.utilities.utilities import get_class_kwargs

logger = logging.getLogger(__name__)


def DataHandlerFactory(
    ExtracterClass,
    LoaderClass,
    BaseLoader=None,
    FeatureRegistry=None,
    name='Handler',
):
    """Build composite objects that load from file_paths, extract specified
    region, derive new features, and cache derived data.

    Parameters
    ----------
    ExtracterClass : class
        :class:`Extracter` class to use in this object composition.
    LoaderClass : class
        :class:`Loader` class to use in this object composition.
    BaseLoader : class
        Optional base loader update. The default for h5 is MultiFileWindX and
        for nc the default is xarray
    name : str
        Optional name for class built from factory. This will display in
        logging.

    """

    class Handler(Deriver, metaclass=FactoryMeta):
        __name__ = name

        if BaseLoader is not None:
            BASE_LOADER = BaseLoader

        def __init__(self, file_paths, features, **kwargs):
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
            cache_kwargs = kwargs.pop('cache_kwargs', None)
            loader_kwargs = get_class_kwargs(LoaderClass, kwargs)
            deriver_kwargs = get_class_kwargs(Deriver, kwargs)
            extracter_kwargs = get_class_kwargs(ExtracterClass, kwargs)
            features = lowered(features)
            self.loader = LoaderClass(file_paths, **loader_kwargs)
            self._loader_hook()
            self.extracter = ExtracterClass(
                self.loader,
                **extracter_kwargs,
            )
            self._extracter_hook()
            super().__init__(
                self.extracter.data,
                features=features,
                **deriver_kwargs,
                FeatureRegistry=FeatureRegistry,
            )
            self._deriver_hook()
            if cache_kwargs is not None:
                _ = Cacher(self, cache_kwargs)

        def _loader_hook(self):
            """Hook in after loader initialization. Implement this to extend
            class functionality with operations after default loader
            initialization. e.g. Extra preprocessing like renaming variables,
            ensuring correct dimension ordering with non-standard dimensions,
            etc."""
            pass

        def _extracter_hook(self):
            """Hook in after extracter initialization. Implement this to extend
            class functionality with operations after default extracter
            initialization. e.g. If special methods are required to add more
            data to the extracted data - Prime example is adding a special
            method to extract / regrid clearsky_ghi from an nsrdb source file
            prior to derivation of clearsky_ratio."""
            pass

        def _deriver_hook(self):
            """Hook in after deriver initialization. Implement this to extend
            class functionality with operations after default deriver
            initialization. e.g. If special methods are required to derive
            additional features which might depend on non-standard inputs (e.g.
            other source files than those used by the loader)."""
            pass

        def __getattr__(self, attr):
            """Look for attribute in extracter and then loader if not found in
            self."""
            if attr in ['lat_lon', 'grid_shape', 'time_slice', 'time_index']:
                return getattr(self.extracter, attr)
            try:
                return Deriver.__getattr__(self, attr)
            except Exception as e:
                msg = f'{self.__class__.__name__} has no attribute "{attr}"'
                raise AttributeError(msg) from e

    return Handler


def DailyDataHandlerFactory(
    ExtracterClass,
    LoaderClass,
    BaseLoader=None,
    FeatureRegistry=None,
    name='Handler',
):
    """Handler factory for data handlers with additional daily_data."""

    BaseHandler = DataHandlerFactory(
        ExtracterClass,
        LoaderClass=LoaderClass,
        BaseLoader=BaseLoader,
        FeatureRegistry=FeatureRegistry,
    )

    class DailyHandler(BaseHandler):
        """General data handler class with daily data as an additional
        attribute. xr.Dataset coarsen method employed to compute averages /
        mins / maxes over daily windows. Special treatment of clearsky_ratio,
        which requires derivation from total clearsky_ghi and total ghi"""

        __name__ = name

        def __init__(self, file_paths, features, **kwargs):
            """Add features required for daily cs ratio derivation if not
            requested."""

            self.requested_features = features.copy()
            if 'clearsky_ratio' in features:
                needed = [
                    f for f in ['clearsky_ghi', 'ghi'] if f not in features
                ]
                features.extend(needed)
            super().__init__(file_paths, features, **kwargs)

        def _deriver_hook(self):
            """Hook to run daily coarsening calculations after derivations of
            hourly variables. Replaces data with daily averages / maxes / mins
            / sums"""
            msg = (
                'Data needs to be hourly with at least 24 hours, but data '
                'shape is {}.'.format(self.data.shape)
            )

            day_steps = int(
                24 // float(mode(self.time_index.diff().seconds / 3600).mode)
            )
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
            self.data = self.data[self.requested_features]
            self.daily_data = daily_data[self.requested_features]
            self.daily_data_slices = [
                slice(x[0], x[-1] + 1)
                for x in np.array_split(
                    np.arange(len(self.time_index)), n_data_days
                )
            ]
            self.data.attrs = {'name': 'hourly'}
            self.daily_data.attrs = {'name': 'daily'}

    return DailyHandler


DataHandlerH5 = DataHandlerFactory(
    BaseExtracterH5, LoaderH5, FeatureRegistry=RegistryH5, name='DataHandlerH5'
)
DataHandlerNC = DataHandlerFactory(
    BaseExtracterNC, LoaderNC, FeatureRegistry=RegistryNC, name='DataHandlerNC'
)


def _base_loader(file_paths, **kwargs):
    return MultiFileNSRDBX(file_paths, **kwargs)


DataHandlerH5SolarCC = DailyDataHandlerFactory(
    BaseExtracterH5,
    LoaderH5,
    BaseLoader=_base_loader,
    FeatureRegistry=RegistryH5SolarCC,
    name='DataHandlerH5SolarCC',
)


DataHandlerH5WindCC = DailyDataHandlerFactory(
    BaseExtracterH5,
    LoaderH5,
    BaseLoader=_base_loader,
    FeatureRegistry=RegistryH5WindCC,
    name='DataHandlerH5WindCC',
)
