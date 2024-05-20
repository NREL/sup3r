"""Basic container objects can perform transformations / extractions on the
contained data."""

import logging

import numpy as np

from sup3r.containers.cachers import Cacher
from sup3r.containers.derivers import DeriverH5, DeriverNC
from sup3r.containers.extracters import ExtracterH5, ExtracterNC
from sup3r.containers.loaders import LoaderH5, LoaderNC
from sup3r.utilities.utilities import _get_class_kwargs

np.random.seed(42)

logger = logging.getLogger(__name__)


def extracter_factory(ExtracterClass, LoaderClass):
    """Build composite :class:`Extracter` objects that also load from
    file_paths. Inputs are required to be provided as keyword args so that they
    can be split appropriately across different classes.

    Parameters
    ----------
    ExtracterClass : class
        :class:`Extracter` class to use in this object composition.
    LoaderClass : class
        :class:`Loader` class to use in this object composition.
    """

    class DirectExtracter(ExtracterClass):

        def __init__(self, file_paths, features=None, **kwargs):
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
            loader = LoaderClass(file_paths, features)
            super().__init__(container=loader, **kwargs)

    return DirectExtracter


def handler_factory(DeriverClass, DirectExtracterClass, FeatureRegistry=None):
    """Build composite objects that load from file_paths, extract specified
    region, derive new features, and cache derived data.

    Parameters
    ----------
    DirectExtracterClass : class
        Object composed of a :class:`Loader` and :class:`Extracter` class.
        Created with the :func:`extracter_factory` method
    DeriverClass : class
        :class:`Deriver` class to use in this object composition.
    FeatureRegistry : Dict
        Optional FeatureRegistry dictionary to use for derivation method
        lookups. When the :class:`Deriver` is asked to derive a feature that
        is not found in the :class:`Extracter` data it will look for a method
        to derive the feature in the registry.
    """

    class Handler(DeriverClass):

        if FeatureRegistry is not None:
            FEATURE_REGISTRY = FeatureRegistry

        def __init__(self, file_paths, load_features='all', **kwargs):
            """
            Parameters
            ----------
            file_paths : str | list | pathlib.Path
                file_paths input to DirectExtracterClass
            load_features : list
                List of features to load and use in region extraction and
                derivations
            **kwargs : dict
                Dictionary of keyword args for DirectExtracter, Deriver, and
                Cacher
            """
            cache_kwargs = kwargs.pop('cache_kwargs', None)
            extracter_kwargs = _get_class_kwargs(DirectExtracterClass, kwargs)
            extracter_kwargs['features'] = load_features
            deriver_kwargs = _get_class_kwargs(DeriverClass, kwargs)

            extracter = DirectExtracterClass(file_paths, **extracter_kwargs)
            super().__init__(extracter, **deriver_kwargs)
            for attr in ['time_index', 'lat_lon']:
                setattr(self, attr, getattr(extracter, attr))

            if cache_kwargs is not None:
                _ = Cacher(self, cache_kwargs)

    return Handler


DirectExtracterH5 = extracter_factory(ExtracterH5, LoaderH5)
DirectExtracterNC = extracter_factory(ExtracterNC, LoaderNC)
DataHandlerH5 = handler_factory(DeriverH5, DirectExtracterH5)
DataHandlerNC = handler_factory(DeriverNC, DirectExtracterNC)
