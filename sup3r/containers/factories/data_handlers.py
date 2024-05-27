"""Basic objects can perform transformations / extractions on the contained
data."""

import logging

from sup3r.containers.cachers import Cacher
from sup3r.containers.derivers import Deriver
from sup3r.containers.derivers.methods import (
    RegistryH5,
    RegistryNC,
)
from sup3r.containers.extracters import (
    BaseExtracterH5,
    BaseExtracterNC,
)
from sup3r.containers.factories.common import FactoryMeta
from sup3r.containers.loaders import LoaderH5, LoaderNC
from sup3r.utilities.utilities import get_class_kwargs

logger = logging.getLogger(__name__)


def ExtracterFactory(
    ExtracterClass, LoaderClass, BaseLoader=None, name='DirectExtracter'
):
    """Build composite :class:`Extracter` objects that also load from
    file_paths. Inputs are required to be provided as keyword args so that they
    can be split appropriately across different classes.

    Parameters
    ----------
    ExtracterClass : class
        :class:`Extracter` class to use in this object composition.
    LoaderClass : class
        :class:`Loader` class to use in this object composition.
    BaseLoader : function
        Optional base loader method update. This is a function which takes
        `file_paths` and `**kwargs` and returns an initialized base loader with
        those arguments. The default for h5 is a method which returns
        MultiFileWindX(file_paths, **kwargs) and for nc the default is
        xarray.open_mfdataset(file_paths, **kwargs)
    name : str
        Optional name for class built from factory. This will display in
        logging.
    """

    class DirectExtracter(ExtracterClass, metaclass=FactoryMeta):
        __name__ = name

        if BaseLoader is not None:
            BASE_LOADER = BaseLoader

        def __init__(self, file_paths, **kwargs):
            """
            Parameters
            ----------
            file_paths : str | list | pathlib.Path
                file_paths input to LoaderClass
            **kwargs : dict
                Dictionary of keyword args for Extracter and Loader
            """
            loader_kwargs = get_class_kwargs(LoaderClass, kwargs)
            extracter_kwargs = get_class_kwargs(ExtracterClass, kwargs)
            self.loader = LoaderClass(file_paths, **loader_kwargs)
            super().__init__(loader=self.loader, **extracter_kwargs)

    return DirectExtracter


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

        def __init__(
            self, file_paths, features, load_features='all', **kwargs
        ):
            """
            Parameters
            ----------
            file_paths : str | list | pathlib.Path
                file_paths input to DirectExtracterClass
            features : list
                Features to derive from loaded data.
            load_features : list
                Features to load for use in derivations.
            **kwargs : dict
                Dictionary of keyword args for DirectExtracter, Deriver, and
                Cacher
            """
            cache_kwargs = kwargs.pop('cache_kwargs', None)
            loader_kwargs = get_class_kwargs(LoaderClass, kwargs)
            deriver_kwargs = get_class_kwargs(Deriver, kwargs)
            extracter_kwargs = get_class_kwargs(ExtracterClass, kwargs)
            self.loader = LoaderClass(
                file_paths, features=load_features, **loader_kwargs
            )
            self._loader_hook()
            self.extracter = ExtracterClass(
                self.loader, features=load_features, **extracter_kwargs
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


ExtracterH5 = ExtracterFactory(BaseExtracterH5, LoaderH5, name='ExtracterH5')
ExtracterNC = ExtracterFactory(BaseExtracterNC, LoaderNC, name='ExtracterNC')
DataHandlerH5 = DataHandlerFactory(
    BaseExtracterH5, LoaderH5, FeatureRegistry=RegistryH5, name='DataHandlerH5'
)
DataHandlerNC = DataHandlerFactory(
    BaseExtracterNC, LoaderNC, FeatureRegistry=RegistryNC, name='DataHandlerNC'
)
