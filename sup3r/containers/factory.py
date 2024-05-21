"""Basic container objects can perform transformations / extractions on the
contained data."""

import logging

import numpy as np

from sup3r.containers.cachers import Cacher
from sup3r.containers.derivers import ExtendedDeriver
from sup3r.containers.derivers.methods import RegistryH5, RegistryNC
from sup3r.containers.extracters import ExtracterH5, ExtracterNC
from sup3r.containers.loaders import LoaderH5, LoaderNC
from sup3r.utilities.utilities import _get_class_kwargs

np.random.seed(42)

logger = logging.getLogger(__name__)


def extracter_factory(ExtracterClass, LoaderClass, BaseLoader=None):
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
    """

    class DirectExtracter(ExtracterClass):
        if BaseLoader is not None:
            BASE_LOADER = BaseLoader

        def __init__(self, file_paths, **kwargs):
            """
            Parameters
            ----------
            file_paths : str | list | pathlib.Path
                file_paths input to LoaderClass
            features : list | None
                List of features to load
            **kwargs : dict
                Dictionary of keyword args for Extracter
            """
            loader = LoaderClass(file_paths)
            super().__init__(loader=loader, **kwargs)

    return DirectExtracter


def handler_factory(
    ExtracterClass,
    LoaderClass,
    BaseLoader=None,
    FeatureRegistry=None,
):
    """Build composite objects that load from file_paths, extract specified
    region, derive new features, and cache derived data.

    Parameters
    ----------
    DeriverClass : class
        :class:`Deriver` class to use in this object composition.
    ExtracterClass : class
        :class:`Extracter` class to use in this object composition.
    LoaderClass : class
        :class:`Loader` class to use in this object composition.
    BaseLoader : class
        Optional base loader update. The default for h5 is MultiFileWindX and
        for nc the default is xarray
    """
    DirectExtracterClass = extracter_factory(
        ExtracterClass, LoaderClass, BaseLoader=BaseLoader
    )

    class Handler(ExtendedDeriver):
        def __init__(self, file_paths, **kwargs):
            """
            Parameters
            ----------
            file_paths : str | list | pathlib.Path
                file_paths input to DirectExtracterClass
            **kwargs : dict
                Dictionary of keyword args for DirectExtracter, Deriver, and
                Cacher
            """
            cache_kwargs = kwargs.pop('cache_kwargs', None)
            extracter_kwargs = _get_class_kwargs(DirectExtracterClass, kwargs)
            deriver_kwargs = _get_class_kwargs(ExtendedDeriver, kwargs)
            extracter = DirectExtracterClass(file_paths, **extracter_kwargs)
            super().__init__(
                extracter, **deriver_kwargs, FeatureRegistry=FeatureRegistry
            )
            if cache_kwargs is not None:
                _ = Cacher(self, cache_kwargs)

    return Handler


DirectExtracterH5 = extracter_factory(ExtracterH5, LoaderH5)
DirectExtracterNC = extracter_factory(ExtracterNC, LoaderNC)
DataHandlerH5 = handler_factory(
    ExtracterH5, LoaderH5, FeatureRegistry=RegistryH5
)
DataHandlerNC = handler_factory(
    ExtracterNC, LoaderNC, FeatureRegistry=RegistryNC
)
