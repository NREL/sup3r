"""Basic objects that can perform derivations of new features from loaded /
extracted features."""

import logging
import re
from inspect import signature
from typing import Union

import dask.array as da

from sup3r.preprocessing.abstract import Data, XArrayWrapper
from sup3r.preprocessing.base import Container
from sup3r.preprocessing.common import Dimension
from sup3r.preprocessing.derivers.methods import (
    RegistryBase,
)
from sup3r.typing import T_Array
from sup3r.utilities.interpolation import Interpolator

logger = logging.getLogger(__name__)


def parse_feature(feature):
    """Parse feature name to get the "basename" (i.e. U for U_100m), the height
    (100 for U_100m), and pressure if available (1000 for U_1000pa)."""

    class FeatureStruct:
        def __init__(self):
            height = re.findall(r'_\d+m', feature)
            pressure = re.findall(r'_\d+pa', feature)
            self.basename = (
                feature.replace(height[0], '')
                if height
                else feature.replace(pressure[0], '')
                if pressure
                else feature.split('_(.*)')[0]
                if '_(.*)' in feature
                else feature
            )
            self.height = int(height[0][1:-1]) if height else None
            self.pressure = int(pressure[0][1:-2]) if pressure else None

        def map_wildcard(self, pattern):
            """Return given pattern with wildcard replaced with height if
            available, pressure if available, or just return the basename."""
            if '(.*)' not in pattern:
                return pattern
            return (
                f"{pattern.split('_(.*)')[0]}_{self.height}m"
                if self.height
                else f"{pattern.split('_(.*)')[0]}_{self.pressure}pa"
                if self.pressure
                else f"{pattern.split('_(.*)')[0]}"
            )

    return FeatureStruct()


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

        super().__init__(data=data, features=features)
        for f in self.features:
            self.data[f] = self.derive(f)
        self.data = self.data[self.features]

    def _check_for_compute(self, feature) -> Union[T_Array, str]:
        """Get compute method from the registry if available. Will check for
        pattern feature match in feature registry. e.g. if U_100m matches a
        feature registry entry of U_(.*)m
        """
        for pattern in self.FEATURE_REGISTRY:
            if re.match(pattern.lower(), feature.lower()):
                method = self.FEATURE_REGISTRY[pattern]
                if isinstance(method, str):
                    return method
                if hasattr(method, 'inputs'):
                    fstruct = parse_feature(feature)
                    inputs = [fstruct.map_wildcard(i) for i in method.inputs]
                    if inputs in self.data:
                        logger.debug(
                            f'Found compute method for {feature}. Proceeding '
                            'with derivation.'
                        )
                        return self._run_compute(feature, method).data
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
        locate differently named features with a name map in the feture
        registry. i.e. if  `FEATURE_REGISTRY` containers a key, value pair like
        "windspeed": "wind_speed" then requesting "windspeed" will ultimately
        return a compute method (or fetch from raw data) for "wind_speed

        Notes
        -----
        Features are all saved as lower case names and __contains__ checks will
        use feature.lower()
        """

        fstruct = parse_feature(feature)
        if feature not in self.data.data_vars:
            compute_check = self._check_for_compute(feature)
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
        return self.data[feature].data

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
        for f in self.data.features:
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
        var_array = self.data[fstruct.basename].data
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
                self.data['zg'].data
                - da.broadcast_to(
                    self.data['topography'].data.T, self.data['zg'].T.shape
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
                self.data[Dimension.PRESSURE_LEVEL].data, var_array.shape
            )

        lev_array, var_array = self.add_single_level_data(
            feature, lev_array, var_array
        )
        out = Interpolator.interp_to_level(
            lev_array=lev_array, var_array=var_array, level=level
        )
        return out


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
            logger.debug(f'Applying time_roll={time_roll} to data array')
            self.data = self.data.roll(time=time_roll)

        if hr_spatial_coarsen > 1:
            logger.debug(
                f'Applying hr_spatial_coarsen={hr_spatial_coarsen} '
                'to data array'
            )
            out = self.data.coarsen(
                {
                    Dimension.SOUTH_NORTH: hr_spatial_coarsen,
                    Dimension.WEST_EAST: hr_spatial_coarsen,
                }
            ).mean()

            self.data = XArrayWrapper(
                coords=out.coords, data_vars=out.data_vars
            )
