"""Basic container objects can perform transformations / extractions on the
contained data."""

import logging
import re
from inspect import signature

import dask.array as da
import numpy as np

from sup3r.containers.base import Container
from sup3r.containers.derivers.methods import (
    RegistryBase,
    RegistryH5,
    RegistryNC,
)
from sup3r.containers.extracters.base import Extracter
from sup3r.utilities.utilities import Feature, parse_keys

np.random.seed(42)

logger = logging.getLogger(__name__)


class Deriver(Container):
    """Container subclass with additional methods for transforming / deriving
    data exposed through an :class:`Extracter` object."""

    FEATURE_REGISTRY = RegistryBase

    def __init__(self, container: Extracter, features, transform=None):
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
        transform : function
            Optional operation on extracter data. This should not be used for
            deriving new features from extracted features. That should be
            handled by compute method lookups in the FEATURE_REGISTRY. This is
            for transformations like rotations, inversions, spatial / temporal
            coarsening, etc.

            For example::

                def coarsening_transform(extracter: Container):
                    from sup3r.utilities.utilities import spatial_coarsening
                    data = spatial_coarsening(extracter.data, s_enhance=2,
                                              obs_axis=False)
                    extracter._lat_lon = spatial_coarsening(extracter.lat_lon,
                                                            s_enhance=2,
                                                            obs_axis=False)
                    return data
        """
        super().__init__(container)
        self._data = None
        self.features = features
        self.transform = transform
        self.update_data()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, trace):
        self.close()

    def close(self):
        """Close Extracter."""
        self.container.close()

    def update_data(self):
        """Update contained data with results of transformation and
        derivations. If the features in self.features are not found in data
        after the transform then the calls to `__getitem__` will run
        derivations for features found in the feature registry."""
        if self.transform is not None:
            self.container.data = self.transform(self.container)
        self.data = da.stack([self[feat] for feat in self.features], axis=-1)

    def _check_for_compute(self, feature):
        """Get compute method from the registry if available. Will check for
        pattern feature match in feature registry. e.g. if U_100m matches a
        feature registry entry of U_(.*)m"""
        for pattern in self.FEATURE_REGISTRY:
            if re.match(pattern.lower(), feature.lower()):
                compute = self.FEATURE_REGISTRY[pattern].compute
                kwargs = {}
                params = signature(compute).parameters
                if 'height' in params:
                    kwargs.update({'height': Feature.get_height(feature)})
                if 'pressure' in params:
                    kwargs.update({'pressure': Feature.get_pressure(feature)})
                return compute(self.container, **kwargs)
        return None

    def _check_self(self, key, key_slice):
        """Check if the requested key is available in derived data or a self
        attribute."""
        if self.data is not None and key in self:
            return self.data[*key_slice, self.index(key)]
        if hasattr(self, key):
            return getattr(self, key)
        return None

    def _check_container(self, key, key_slice):
        """Check if the requested key is available in the container data (if it
        has not been derived yet) or a container attribute."""
        if self.container.data is not None and key in self.container:
            return self.container.data[*key_slice, self.index(key)]
        if hasattr(self.container, key):
            return getattr(self.container, key)
        return None

    def __getitem__(self, keys):
        key, key_slice = parse_keys(keys)
        if isinstance(key, str):
            self_check = self._check_self(key, key_slice)
            if self_check is not None:
                return self_check
            container_check = self._check_container(key, key_slice)
            if container_check is not None:
                return container_check
            compute_check = self._check_for_compute(key)
            if compute_check is not None:
                return compute_check
            raise ValueError(f'Could not get item for "{keys}"')
        return self.data[key, key_slice]


class DeriverNC(Deriver):
    """Container subclass with additional methods for transforming / deriving
    data exposed through an :class:`Extracter` object. Specifically for NETCDF
    data"""

    FEATURE_REGISTRY = RegistryNC


class DeriverH5(Deriver):
    """Container subclass with additional methods for transforming / deriving
    data exposed through an :class:`Extracter` object. Specifically for H5 data
    """

    FEATURE_REGISTRY = RegistryH5
