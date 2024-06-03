"""Basic objects can perform transformations / extractions on the contained
data."""

import logging

import pandas as pd
from rex import MultiFileNSRDBX

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
            if attr in ['lat_lon', 'grid_shape', 'time_slice']:
                return getattr(self.extracter, attr)
            return super().__getattr__(attr)

    return Handler


def DailyDataHandlerFactory(
    ExtracterClass,
    LoaderClass,
    BaseLoader=None,
    FeatureRegistry=None,
    name='Handler',
):
    """Handler factory for daily data handlers."""

    BaseHandler = DataHandlerFactory(
        ExtracterClass,
        LoaderClass=LoaderClass,
        BaseLoader=BaseLoader,
        FeatureRegistry=FeatureRegistry
    )

    class DailyHandler(BaseHandler):
        """General data handler class for daily data. DatasetWrapper coarsen
        method inherited from xr.Dataset employed to compute averages / mins /
        maxes over daily windows."""

        __name__ = name

        def _extracter_hook(self):
            """Hook to run daily coarsening calculations after extraction and
            replaces data with daily averages / maxes / mins to then be used in
            derivations."""

            msg = (
                'Data needs to be hourly with at least 24 hours, but data '
                'shape is {}.'.format(self.extracter.data.shape)
            )
            assert self.extracter.data.shape[2] % 24 == 0, msg
            assert self.extracter.data.shape[2] > 24, msg

            n_data_days = int(self.extracter.data.shape[2] / 24)

            logger.info(
                'Calculating daily average datasets for {} training '
                'data days.'.format(n_data_days)
            )
            daily_data = self.extracter.data.coarsen(time=24).mean()
            for fname in self.extracter.features:
                if '_max_' in fname:
                    daily_data[fname] = (
                        self.extracter.data[fname].coarsen(time=24).max()
                    )
                if '_min_' in fname:
                    daily_data[fname] = (
                        self.extracter.data[fname].coarsen(time=24).min()
                    )

            logger.info(
                'Finished calculating daily average datasets for {} '
                'training data days.'.format(n_data_days)
            )
            self.extracter.data = daily_data
            self.extracter.time_index = pd.to_datetime(
                daily_data.indexes['time']
            )

    return DailyHandler


def CompositeDailyHandlerFactory(
    ExtracterClass,
    LoaderClass,
    BaseLoader=None,
    FeatureRegistry=None,
    name='Handler',
):
    """Builds a data handler with `.data` and `.daily_data` attributes coming
    from a standard data handler and a :class:`DailyDataHandler`,
    respectively."""

    BaseHandler = DataHandlerFactory(
        ExtracterClass=ExtracterClass,
        LoaderClass=LoaderClass,
        BaseLoader=BaseLoader,
        FeatureRegistry=FeatureRegistry)

    DailyHandler = DailyDataHandlerFactory(
        ExtracterClass=ExtracterClass,
        LoaderClass=LoaderClass,
        BaseLoader=BaseLoader,
        FeatureRegistry=FeatureRegistry,
    )

    class CompositeDailyHandler(BaseHandler):
        """Handler composed of a daily handler and standard handler, which
        provide `.daily_data` and `.data` respectively."""

        __name__ = name

        def __init__(self, file_paths, features, **kwargs):
            """
            Parameters
            ----------
            file_paths : str | list | pathlib.Path
                file_paths input to Loader
            features : list
                Features to derive from loaded data.
            **kwargs : dict
                Dictionary of keyword args for Loader, Extracter, Deriver, and
                Cacher
            """
            super().__init__(file_paths, features, **kwargs)

            self.daily_data = DailyHandler(
                file_paths, features, **kwargs
            ).data

    return CompositeDailyHandler


DataHandlerH5 = DataHandlerFactory(
    BaseExtracterH5, LoaderH5, FeatureRegistry=RegistryH5, name='DataHandlerH5'
)
DataHandlerNC = DataHandlerFactory(
    BaseExtracterNC, LoaderNC, FeatureRegistry=RegistryNC, name='DataHandlerNC'
)


def _base_loader(file_paths, **kwargs):
    return MultiFileNSRDBX(file_paths, **kwargs)


DataHandlerH5SolarCC = CompositeDailyHandlerFactory(
    BaseExtracterH5,
    LoaderH5,
    BaseLoader=_base_loader,
    FeatureRegistry=RegistryH5SolarCC,
    name='DataHandlerH5SolarCC',
)


DataHandlerH5WindCC = CompositeDailyHandlerFactory(
    BaseExtracterH5,
    LoaderH5,
    BaseLoader=_base_loader,
    FeatureRegistry=RegistryH5WindCC,
    name='DataHandlerH5WindCC',
)
