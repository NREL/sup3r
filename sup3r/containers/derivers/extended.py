"""Basic container objects can perform transformations / extractions on the
contained data."""

import logging

import numpy as np

from sup3r.containers.derivers.base import Deriver
from sup3r.containers.derivers.methods import (
    RegistryH5,
    RegistryNC,
)
from sup3r.containers.extracters.base import Extracter
from sup3r.utilities.utilities import spatial_coarsening

np.random.seed(42)

logger = logging.getLogger(__name__)


class ExtendedDeriver(Deriver):
    """Extends base :class:`Deriver` class with time_roll and
    hr_spatial_coarsen args."""

    def __init__(
        self,
        container: Extracter,
        features,
        time_roll=0,
        hr_spatial_coarsen=1,
        FeatureRegistry=None,
    ):
        super().__init__(container, features, FeatureRegistry=FeatureRegistry)

        if time_roll != 0:
            logger.debug('Applying time roll to data array')
            self.data.roll(time=time_roll)

        if hr_spatial_coarsen > 1:
            logger.debug(
                f'Applying hr_spatial_coarsen = {hr_spatial_coarsen} '
                'to data array'
            )
            for f in ['latitude', 'longitude', *self.data.data_vars]:
                self.data[f] = (
                    self.data[f].dims,
                    spatial_coarsening(
                        self.data[f],
                        s_enhance=hr_spatial_coarsen,
                        obs_axis=False,
                    ),
                )


class DeriverNC(ExtendedDeriver):
    """Container subclass with additional methods for transforming / deriving
    data exposed through an :class:`Extracter` object. Specifically for NETCDF
    data"""

    FEATURE_REGISTRY = RegistryNC


class DeriverH5(ExtendedDeriver):
    """Container subclass with additional methods for transforming / deriving
    data exposed through an :class:`Extracter` object. Specifically for H5 data
    """

    FEATURE_REGISTRY = RegistryH5
