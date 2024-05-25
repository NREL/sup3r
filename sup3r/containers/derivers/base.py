"""Basic objects that can perform derivations of new features from loaded /
extracted features."""

import logging
import re
from inspect import signature

import numpy as np
import xarray as xr

from sup3r.containers.abstract import Data
from sup3r.containers.base import Container
from sup3r.containers.derivers.methods import (
    RegistryBase,
)
from sup3r.utilities.utilities import Feature, spatial_coarsening

np.random.seed(42)

logger = logging.getLogger(__name__)


class BaseDeriver(Container):
    """Container subclass with additional methods for transforming / deriving
    data exposed through an :class:`Extracter` object."""

    FEATURE_REGISTRY = RegistryBase

    def __init__(self, data: Data, features, FeatureRegistry=None):
        """
        Parameters
        ----------
        data : Data
            wrapped xr.Dataset() (:class:`Data`) with data to use for
            derivations. Usually comes from the `.data` attribute of a
            :class:`Extracter` object.
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

        super().__init__(data=data)
        for f in features:
            self.data[f.lower()] = self.derive(f.lower())
        self.data = self.data.slice_dset(features=features)

    def _check_for_compute(self, feature):
        """Get compute method from the registry if available. Will check for
        pattern feature match in feature registry. e.g. if U_100m matches a
        feature registry entry of U_(.*)m

        Notes
        -----
        Features are all saved as lower case names and __contains__ checks will
        use feature.lower()
        """
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
                return compute(self.data, **kwargs)
        return None

    def derive(self, feature):
        """Routine to derive requested features. Employs a little recursion to
        locate differently named features with a name map in the feture
        registry. i.e. if  `FEATURE_REGISTRY` containers a key, value pair like
        "windspeed": "wind_speed" then requesting "windspeed" will ultimately
        return a compute method (or fetch from raw data) for "wind_speed"""
        if feature not in self.data.variables:
            compute_check = self._check_for_compute(feature)
            if compute_check is not None and isinstance(compute_check, str):
                logger.debug(f'Found alternative name {compute_check} for '
                             f'feature {feature}. Continuing with search for '
                             'compute method.')
                return self.compute[compute_check]
            if compute_check is not None:
                logger.debug(f'Found compute method for {feature}. Proceeding '
                             'with derivation.')
                return compute_check
            msg = (
                f'Could not find {feature} in contained data or in the '
                'available compute methods.'
            )
            logger.error(msg)
            raise RuntimeError(msg)
        return self.data[feature]


class Deriver(BaseDeriver):
    """Extends base :class:`BaseDeriver` class with time_roll and
    hr_spatial_coarsen args."""

    def __init__(
        self,
        data: Data,
        features,
        time_roll=0,
        hr_spatial_coarsen=1,
        FeatureRegistry=None,
    ):
        super().__init__(data, features, FeatureRegistry=FeatureRegistry)

        if time_roll != 0:
            logger.debug('Applying time roll to data array')
            self.data = self.data.roll(time=time_roll)

        if hr_spatial_coarsen > 1:
            logger.debug('Applying hr spatial coarsening to data array')
            coords = self.data.coords
            coords = {
                coord: (
                    self.dims[:2],
                    spatial_coarsening(
                        self.data[coord],
                        s_enhance=hr_spatial_coarsen,
                        obs_axis=False,
                    ),
                )
                for coord in ['latitude', 'longitude']
            }
            data_vars = {}
            for feat in self.features:
                dat = self.data[feat]
                data_vars[feat] = (
                    (self.dims[:len(dat.shape)]),
                    spatial_coarsening(
                        dat,
                        s_enhance=hr_spatial_coarsen,
                        obs_axis=False,
                    ),
                )
            self.data = Data(xr.Dataset(coords=coords, data_vars=data_vars))
