"""Basic container objects can perform transformations / extractions on the
contained data."""

import logging

import numpy as np

from sup3r.containers.derivers.base import Deriver
from sup3r.containers.derivers.factory import RegistryH5

np.random.seed(42)

logger = logging.getLogger(__name__)


class DeriverH5(Deriver):
    """Container subclass with additional methods for transforming / deriving
    data exposed through an :class:`Extracter` object. Specifically for H5 data
    """

    FEATURE_REGISTRY = RegistryH5
