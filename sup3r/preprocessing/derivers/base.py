"""Basic objects that can perform derivations of new features from loaded /
extracted features."""

import logging
import re
from inspect import signature
from typing import Type, Union

import dask.array as da
import numpy as np

from sup3r.preprocessing.base import Container
from sup3r.preprocessing.utilities import Dimension, parse_to_list
from sup3r.typing import T_Array, T_Dataset
from sup3r.utilities.interpolation import Interpolator

from .methods import DerivedFeature, RegistryBase
from .utilities import parse_feature

logger = logging.getLogger(__name__)


class BaseDeriver(Container):
    """Container subclass with additional methods for transforming / deriving
    data exposed through an :class:`Extracter` object."""

    FEATURE_REGISTRY = RegistryBase

    def __init__(self, data: T_Dataset, features, FeatureRegistry=None):
        """
        Parameters
        ----------
        data : T_Dataset
            Data to use for derivations. Usually comes from the `.data`
            attribute of a :class:`Extracter` object.
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
        features = parse_to_list(data=data, features=features)
        new_features = [f for f in features if f not in self.data]
        for f in new_features:
            self.data[f] = self.derive(f)
        self.data = (
            self.data[[Dimension.LATITUDE, Dimension.LONGITUDE]]
            if not features
            else self.data
            if features == 'all'
            else self.data[features]
        )

    def _check_registry(self, feature) -> Union[Type[DerivedFeature], None]:
        """Check if feature or matching pattern is in the feature registry
        keys. Return the corresponding value if found."""
        if feature.lower() in self.FEATURE_REGISTRY:
            return self.FEATURE_REGISTRY[feature.lower()]
        for pattern in self.FEATURE_REGISTRY:
            if re.match(pattern.lower(), feature.lower()):
                return self.FEATURE_REGISTRY[pattern]
        return None

    def check_registry(self, feature) -> Union[T_Array, str, None]:
        """Get compute method from the registry if available. Will check for
        pattern feature match in feature registry. e.g. if U_100m matches a
        feature registry entry of U_(.*)m
        """
        method = self._check_registry(feature)
        if isinstance(method, str):
            return method
        if method is not None and hasattr(method, 'inputs'):
            fstruct = parse_feature(feature)
            inputs = [fstruct.map_wildcard(i) for i in method.inputs]
            if all(f in self.data for f in inputs):
                logger.debug(
                    f'Found compute method ({method}) for {feature}. '
                    'Proceeding with derivation.'
                )
                return self._run_compute(feature, method)
        return None

    def _run_compute(self, feature, method):
        """If we have all the inputs we can run the compute method."""
        compute = method.compute
        params = signature(compute).parameters
        fstruct = parse_feature(feature)
        kwargs = {
            k: getattr(fstruct, k) for k in params if hasattr(fstruct, k)
        }
        return compute(self.data, **kwargs)

    def map_new_name(self, feature, pattern):
        """If the search for a derivation method first finds an alternative
        name for the feature we want to derive, by matching a wildcard pattern,
        we need to replace the wildcard with the specific height or pressure we
        want and continue the search for a derivation method with this new
        name."""
        fstruct = parse_feature(feature)
        pstruct = parse_feature(pattern)
        if fstruct.height is not None:
            new_feature = pstruct.basename + f'_{fstruct.height}m'
        elif fstruct.pressure is not None:
            new_feature = pstruct.basename + f'_{fstruct.pressure}pa'
        else:
            new_feature = pattern
        logger.debug(
            f'Found alternative name {new_feature} for '
            f'feature {feature}. Continuing with search for '
            f'compute method for {new_feature}.'
        )
        return new_feature

    def derive(self, feature) -> T_Array:
        """Routine to derive requested features. Employs a little recursion to
        locate differently named features with a name map in the feature
        registry. i.e. if  `FEATURE_REGISTRY` contains a key, value pair like
        "windspeed": "wind_speed" then requesting "windspeed" will ultimately
        return a compute method (or fetch from raw data) for "wind_speed

        Note
        ----
        Features are all saved as lower case names and __contains__ checks will
        use feature.lower()
        """

        fstruct = parse_feature(feature)
        if feature not in self.data:
            compute_check = self.check_registry(feature)
            if compute_check is not None and isinstance(compute_check, str):
                new_feature = self.map_new_name(feature, compute_check)
                return self.derive(new_feature)

            if compute_check is not None:
                return compute_check

            if fstruct.basename in self.data.data_vars:
                logger.debug(f'Attempting level interpolation for {feature}.')
                return self.do_level_interpolation(feature)

            msg = (
                f'Could not find {feature} in contained data or in the '
                'available compute methods.'
            )
            logger.error(msg)
            raise RuntimeError(msg)
        return self.data[feature, ...].astype(np.float32)

    def add_single_level_data(self, feature, lev_array, var_array):
        """When doing level interpolation we should include the single level
        data available. e.g. If we have U_100m already and want to
        interpolation U_40m from multi-level data U we should add U_100m at
        height 100m before doing interpolation since 100 could be a closer
        level to 40m than those available in U."""
        fstruct = parse_feature(feature)
        pattern = fstruct.basename + '_(.*)'
        var_list = []
        lev_list = []
        for f in list(self.data.data_vars):
            if re.match(pattern.lower(), f):
                var_list.append(self.data[f])
                pstruct = parse_feature(f)
                lev = (
                    pstruct.height
                    if pstruct.height is not None
                    else pstruct.pressure
                )
                lev_list.append(lev)

        if len(var_list) > 0:
            var_array = da.concatenate(
                [var_array, da.stack(var_list, axis=-1)], axis=-1
            )
            lev_array = da.concatenate(
                [
                    lev_array,
                    da.broadcast_to(
                        da.from_array(lev_list),
                        (*var_array.shape[:-1], len(lev_list)),
                    ),
                ],
                axis=-1,
            )
        return lev_array, var_array

    def do_level_interpolation(self, feature) -> T_Array:
        """Interpolate over height or pressure to derive the given feature."""
        fstruct = parse_feature(feature)
        var_array: T_Array = self.data[fstruct.basename, ...]
        if fstruct.height is not None:
            level = [fstruct.height]
            msg = (
                f'To interpolate {fstruct.basename} to {feature} the loaded '
                'data needs to include "zg" and "topography".'
            )
            assert (
                'zg' in self.data.data_vars
                and 'topography' in self.data.data_vars
            ), msg
            lev_array = (
                self.data['zg', ...]
                - da.broadcast_to(
                    self.data['topography', ...].T,
                    self.data['zg', ...].T.shape,
                ).T
            )
        else:
            level = [fstruct.pressure]
            msg = (
                f'To interpolate {fstruct.basename} to {feature} the loaded '
                'data needs to include "level" (a.k.a pressure at multiple '
                'levels).'
            )
            assert Dimension.PRESSURE_LEVEL in self.data, msg
            lev_array = da.broadcast_to(
                self.data[Dimension.PRESSURE_LEVEL, ...], var_array.shape
            )

        lev_array, var_array = self.add_single_level_data(
            feature, lev_array, var_array
        )
        interp_method = 'linear'
        if fstruct.basename in ('u', 'v') and fstruct.height < 100:
            interp_method = 'log'
        out = Interpolator.interp_to_level(
            lev_array=lev_array,
            var_array=var_array,
            level=level,
            interp_method=interp_method,
        )
        return out


class Deriver(BaseDeriver):
    """Extends base :class:`BaseDeriver` class with time_roll and
    hr_spatial_coarsen args."""

    def __init__(
        self,
        data: T_Dataset,
        features,
        time_roll=0,
        hr_spatial_coarsen=1,
        nan_method_kwargs=None,
        FeatureRegistry=None,
    ):
        """
        Parameters
        ----------
        data : T_Dataset
            Data used for derivations
        features: list
            List of features to derive
        time_roll: int
            Number of steps to shift the time axis. `Passed to
            xr.Dataset.roll()`
        hr_spatial_coarsen: int
            Spatial coarsening factor. Passed to `xr.Dataset.coarsen()`
        nan_method_kwargs: str | dict | None
            Keyword arguments for nan handling. If 'mask', time steps with nans
            will be dropped. Otherwise this should be a dict of kwargs which
            will be passed to :meth:`Sup3rX.interpolate_na`.
        FeatureRegistry : dict
            Dictionary of :class:`DerivedFeature` objects used for derivations
        """

        super().__init__(
            data=data, features=features, FeatureRegistry=FeatureRegistry
        )

        if time_roll != 0:
            logger.debug(f'Applying time_roll={time_roll} to data array')
            self.data = self.data.roll(**{Dimension.TIME: time_roll})

        if hr_spatial_coarsen > 1:
            logger.debug(
                f'Applying hr_spatial_coarsen={hr_spatial_coarsen} '
                'to data array'
            )
            self.data = self.data.coarsen(
                {
                    Dimension.SOUTH_NORTH: hr_spatial_coarsen,
                    Dimension.WEST_EAST: hr_spatial_coarsen,
                }
            ).mean()

        if nan_method_kwargs is not None:
            if nan_method_kwargs['method'] == 'mask':
                dim = nan_method_kwargs.get('dim', Dimension.TIME)
                axes = [i for i in range(4) if i != self.data.dims.index(dim)]
                mask = np.isnan(self.data.as_array()).any(axes)
                self.data = self.data.drop_isel(**{dim: mask})

            elif np.isnan(self.data.as_array()).any():
                logger.info(
                    f'Filling nan values with nan_method_kwargs='
                    f'{nan_method_kwargs}'
                )
                self.data = self.data.interpolate_na(**nan_method_kwargs)
