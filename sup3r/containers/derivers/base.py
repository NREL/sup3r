"""Basic container objects can perform transformations / extractions on the
contained data."""

import logging
import re
from inspect import signature

import numpy as np
import xarray as xr

from sup3r.containers.abstract import AbstractContainer
from sup3r.containers.derivers.methods import (
    RegistryBase,
)
from sup3r.containers.extracters.base import Extracter
from sup3r.utilities.utilities import Feature, spatial_coarsening

np.random.seed(42)

logger = logging.getLogger(__name__)


class Deriver(AbstractContainer):
    """Container subclass with additional methods for transforming / deriving
    data exposed through an :class:`Extracter` object."""

    FEATURE_REGISTRY = RegistryBase

    def __init__(self, container: Extracter, features, FeatureRegistry=None):
        """
        Parameters
        ----------
        container : Container
            Extracter type container exposing `.data` for a specified
            spatiotemporal extent
        features : list
            List of feature names to derive from the :class:`Extracter` data.
            The :class:`Extracter` object contains the features available to
            use in the derivation. e.g. extracter.features = ['windspeed',
            'winddirection'] with self.features = ['U', 'V']
        FeatureRegistry : Dict
            Optional FeatureRegistry dictionary to use for derivation method
            lookups. When the :class:`Deriver` is asked to derive a feature
            that is not found in the :class:`Extracter` data it will look for a
            method to derive the feature in the registry.
        """
        if FeatureRegistry is not None:
            self.FEATURE_REGISTRY = FeatureRegistry

        super().__init__()
        self.container = container
        self.data = container.data
        self.features = features
        self.update_data()

    def update_data(self):
        """Update contained data with results of derivations. If the features
        in self.features are not found in data the calls to `__getitem__`
        will run derivations for features found in the feature registry."""
        for f in self.features:
            self.data[f] = (('south_north', 'west_east', 'time'), self[f])
        self.data = self.data[self.features]

    def _check_for_compute(self, feature):
        """Get compute method from the registry if available. Will check for
        pattern feature match in feature registry. e.g. if U_100m matches a
        feature registry entry of U_(.*)m"""
        for pattern in self.FEATURE_REGISTRY:
            if re.match(pattern.lower(), feature.lower()):
                method = self.FEATURE_REGISTRY[pattern]
                if isinstance(method, str):
                    return self._check_for_compute(method)
                compute = method.compute
                params = signature(compute).parameters
                kwargs = {
                    k: getattr(Feature(feature), k)
                    for k in params
                    if hasattr(Feature(feature), k)
                }
                return compute(self.container, **kwargs)
        return None

    def __getitem__(self, keys):
        if keys not in self:
            compute_check = self._check_for_compute(keys)
            if compute_check is not None and isinstance(compute_check, str):
                return self[compute_check]
            if compute_check is not None:
                return compute_check
            msg = (
                f'Could not find {keys} in contained data or in the '
                'FeatureRegistry.'
            )
            logger.error(msg)
            raise KeyError(msg)
        return super().__getitem__(keys)


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
            self.data = np.roll(self.data, time_roll, axis=2)

        if hr_spatial_coarsen > 1:
            logger.debug('Applying hr spatial coarsening to data array')
            coords = {
                coord: spatial_coarsening(
                    self.data[coord],
                    s_enhance=hr_spatial_coarsen,
                    obs_axis=False,
                )
                for coord in ['latitude', 'longitude']
            }
            coords['time'] = self.data['time']
            data_vars = {
                f: (
                    ('latitude', 'longitude', 'time'),
                    spatial_coarsening(
                        self.data[f],
                        s_enhance=hr_spatial_coarsen,
                        obs_axis=False,
                    ),
                )
                for f in self.features
            }
            self.data = xr.Dataset(coords=coords, data_vars=data_vars)
