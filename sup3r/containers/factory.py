"""Basic container objects can perform transformations / extractions on the
contained data."""

import logging
from inspect import signature

import numpy as np

from sup3r.containers.cachers import Cacher
from sup3r.containers.derivers import DeriverH5, DeriverNC
from sup3r.containers.extracters import ExtracterH5, ExtracterNC
from sup3r.containers.loaders import LoaderH5, LoaderNC

np.random.seed(42)

logger = logging.getLogger(__name__)


def _merge(dicts):
    out = {}
    for d in dicts:
        out.update(d)
    return out


def _get_possible_class_args(Class):
    class_args = list(signature(Class.__init__).parameters.keys())
    if Class.__bases__ == (object,):
        return class_args
    for base in Class.__bases__:
        class_args += _get_possible_class_args(base)
    return class_args


def _get_class_kwargs(Class, kwargs):
    class_args = _get_possible_class_args(Class)
    return {k: v for k, v in kwargs.items() if k in class_args}


def extracter_factory(ExtracterClass, LoaderClass):
    """Build composite :class:`Extracter` objects that also load from
    file_paths. Inputs are required to be provided as keyword args so that they
    can be split appropriately across different classes."""

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


def deriver_factory(DirectExtracterClass, DeriverClass):
    """Build composite :class:`Deriver` objects that also load from
    file_paths and extract specified region. Inputs are required to be provided
    as keyword args so that they can be split appropriately across different
    classes."""

    class DirectDeriver(DirectExtracterClass):
        def __init__(self, features, load_features='all', **kwargs):
            """
            Parameters
            ----------
            features : list
                List of features to derive from loaded features
            load_features : list
                List of features to load and use in region extraction and
                derivations
            **kwargs : dict
                Dictionary of keyword args for DirectExtracter, Deriver, and
                Cacher
            """
            extracter_kwargs = _get_class_kwargs(DirectExtracterClass, kwargs)
            deriver_kwargs = _get_class_kwargs(DeriverClass, kwargs)

            super().__init__(features=load_features, **extracter_kwargs)
            _ = DeriverClass(self, features=features, **deriver_kwargs)

    return DirectDeriver


def wrangler_factory(DirectDeriverClass):
    """Inputs are required to be provided as keyword args so that they can be
    split appropriately across different classes."""

    class Wrangler(DirectDeriverClass):
        def __init__(self, features, load_features='all', **kwargs):
            """
            Parameters
            ----------
            features : list
                List of features to derive from loaded features
            load_features : list
                List of features to load and use in region extraction and
                derivations
            **kwargs : dict
                Dictionary of keyword args for DirectExtracter, Deriver, and
                Cacher
            """
            cache_kwargs = kwargs.pop('cache_kwargs', None)
            super().__init__(
                features=features,
                load_features=load_features,
                **kwargs,
            )
            _ = Cacher(self, cache_kwargs)

    return Wrangler


DirectExtracterH5 = extracter_factory(ExtracterH5, LoaderH5)
DirectExtracterNC = extracter_factory(ExtracterNC, LoaderNC)
DirectDeriverH5 = deriver_factory(DirectExtracterH5, DeriverH5)
DirectDeriverNC = deriver_factory(DirectExtracterNC, DeriverNC)
WranglerH5 = wrangler_factory(DirectDeriverH5)
WranglerNC = wrangler_factory(DirectDeriverNC)
