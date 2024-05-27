"""Basic objects can perform transformations / extractions on the contained
data."""

import logging

import numpy as np

from sup3r.containers.cachers import Cacher
from sup3r.containers.derivers import Deriver
from sup3r.containers.derivers.methods import (
    RegistryH5,
    RegistryNC,
    RegistryNCforCC,
    RegistryNCforCCwithPowerLaw,
)
from sup3r.containers.extracters import (
    ExtracterH5,
    ExtracterNC,
    ExtracterNCforCC,
)
from sup3r.containers.factories.common import FactoryMeta
from sup3r.containers.loaders import LoaderH5, LoaderNC
from sup3r.utilities.utilities import get_class_kwargs

np.random.seed(42)

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
                Dictionary of keyword args for Extracter
            """
            loader = LoaderClass(file_paths)
            super().__init__(loader=loader, **kwargs)

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
    DirectExtracterClass = ExtracterFactory(
        ExtracterClass, LoaderClass, BaseLoader=BaseLoader
    )

    class Handler(Deriver, metaclass=FactoryMeta):
        __name__ = name

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
            deriver_kwargs = get_class_kwargs(Deriver, kwargs)
            extracter_kwargs = get_class_kwargs(DirectExtracterClass, kwargs)
            extracter = DirectExtracterClass(
                file_paths, features=load_features, **extracter_kwargs
            )
            super().__init__(
                extracter.data,
                features=features,
                **deriver_kwargs,
                FeatureRegistry=FeatureRegistry,
            )
            if cache_kwargs is not None:
                _ = Cacher(self, cache_kwargs)

    return Handler


DirectExtracterH5 = ExtracterFactory(
    ExtracterH5, LoaderH5, name='DirectExtracterH5'
)
DirectExtracterNC = ExtracterFactory(
    ExtracterNC, LoaderNC, name='DirectExtracterNC'
)
DataHandlerH5 = DataHandlerFactory(
    ExtracterH5, LoaderH5, FeatureRegistry=RegistryH5, name='DataHandlerH5'
)
DataHandlerNC = DataHandlerFactory(
    ExtracterNC, LoaderNC, FeatureRegistry=RegistryNC, name='DataHandlerNC'
)

DataHandlerNCforCC = DataHandlerFactory(
    ExtracterNCforCC,
    LoaderNC,
    FeatureRegistry=RegistryNCforCC,
    name='DataHandlerNCforCC',
)

DataHandlerNCforCCwithPowerLaw = DataHandlerFactory(
    ExtracterNCforCC,
    LoaderNC,
    FeatureRegistry=RegistryNCforCCwithPowerLaw,
    name='DataHandlerNCforCCwithPowerLaw',
)
