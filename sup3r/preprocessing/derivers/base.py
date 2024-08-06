"""Basic objects that can perform derivations of new features from loaded /
rasterized features."""

import logging
import re
from inspect import signature
from typing import Type, Union

import dask.array as da
import numpy as np
import xarray as xr

from sup3r.preprocessing.accessor import Sup3rX
from sup3r.preprocessing.base import Container, Sup3rDataset
from sup3r.preprocessing.names import Dimension
from sup3r.preprocessing.utilities import (
    _rechunk_if_dask,
    parse_to_list,
)
from sup3r.typing import T_Array
from sup3r.utilities.interpolation import Interpolator

from .methods import DerivedFeature, RegistryBase
from .utilities import parse_feature

logger = logging.getLogger(__name__)


class BaseDeriver(Container):
    """Container subclass with additional methods for transforming / deriving
    data exposed through an :class:`Rasterizer` object."""

    FEATURE_REGISTRY = RegistryBase

    def __init__(
        self,
        data: Union[Sup3rX, Sup3rDataset],
        features,
        FeatureRegistry=None,
        interp_method='linear',
    ):
        """
        Parameters
        ----------
        data : Union[Sup3rX, Sup3rDataset]
            Data to use for derivations. Usually comes from the `.data`
            attribute of a :class:`Rasterizer` object.
        features : list
            List of feature names to derive from the :class:`Rasterizer` data.
            The :class:`Rasterizer` object contains the features available to
            use in the derivation. e.g. rasterizer.features = ['windspeed',
            'winddirection'] with self.features = ['U', 'V']
        FeatureRegistry : Dict
            Optional FeatureRegistry dictionary to use for derivation method
            lookups. When the :class:`Deriver` is asked to derive a feature
            that is not found in the :class:`Rasterizer` data it will look for
            a method to derive the feature in the registry.
        interp_method : str
            Interpolation method to use for height interpolation. e.g. Deriving
            u_20m from u_10m and u_100m. Options are "linear" and "log"
        """
        if FeatureRegistry is not None:
            self.FEATURE_REGISTRY = FeatureRegistry

        super().__init__(data=data)
        self.interp_method = interp_method
        features = parse_to_list(data=data, features=features)
        new_features = [f for f in features if f not in self.data]
        for f in new_features:
            self.data[f] = self.derive(f)
        self.data = (
            self.data[list(self.data.coords)]
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

    def _get_inputs(self, feature):
        """Get method inputs and map any wildcards to height or pressure
        (depending on the name of "feature")"""
        method = self._check_registry(feature)
        fstruct = parse_feature(feature)
        return [fstruct.map_wildcard(i) for i in getattr(method, 'inputs', [])]

    def get_inputs(self, feature):
        """Get inputs for the given feature and inputs for those inputs."""
        inputs = self._get_inputs(feature)
        more_inputs = []
        for inp in inputs:
            more_inputs.extend(self._get_inputs(inp))
        return inputs + more_inputs

    def no_overlap(self, feature):
        """Check if any of the nested inputs for 'feature' contain 'feature'"""
        return feature not in self.get_inputs(feature)

    def check_registry(self, feature) -> Union[T_Array, str, None]:
        """Get compute method from the registry if available. Will check for
        pattern feature match in feature registry. e.g. if u_100m matches a
        feature registry entry of u_(.*)m
        """
        method = self._check_registry(feature)
        if isinstance(method, str):
            return method
        if hasattr(method, 'inputs'):
            fstruct = parse_feature(feature)
            inputs = [fstruct.map_wildcard(i) for i in method.inputs]
            missing = [f for f in inputs if f not in self.data]
            can_derive = all(self.no_overlap(m) for m in missing)
            logger.debug('Found compute method (%s) for %s.', method, feature)
            if any(missing) and can_derive:
                logger.debug(
                    'Missing required features %s. '
                    'Trying to derive these first.',
                    missing,
                )
                for f in missing:
                    self.data[f] = self.derive(f)
                return self._run_compute(feature, method)
            if not missing:
                logger.debug(
                    'All required features %s found. Proceeding.', inputs
                )
                return self._run_compute(feature, method)
            if not can_derive:
                logger.debug(
                    'Some of the method inputs reference %s itself. '
                    'We will try height interpolation instead.',
                    feature,
                )
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
        if '*' not in pattern:
            new_feature = pattern
        elif fstruct.height is not None:
            new_feature = pstruct.basename + f'_{fstruct.height}m'
        elif fstruct.pressure is not None:
            new_feature = pstruct.basename + f'_{fstruct.pressure}pa'
        else:
            msg = (
                f'Found matching pattern "{pattern}" for feature '
                f'"{feature}" but could not construct a valid new feature '
                'name'
            )
            logger.error(msg)
            raise RuntimeError(msg)
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

            if fstruct.basename in self.data.features:
                logger.debug(f'Attempting level interpolation for {feature}.')
                return self.do_level_interpolation(
                    feature, interp_method=self.interp_method
                )

            msg = (
                f'Could not find "{feature}" in contained data or in the '
                'available compute methods.'
            )
            logger.error(msg)
            raise RuntimeError(msg)
        return self.data[feature]

    def add_single_level_data(self, feature, lev_array, var_array):
        """When doing level interpolation we should include the single level
        data available. e.g. If we have u_100m already and want to interpolate
        u_40m from multi-level data U we should add u_100m at height 100m
        before doing interpolation, since 100 could be a closer level to 40m
        than those available in U."""
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
                lev_list.append(np.float32(lev))

        if len(var_list) > 0:
            var_array = np.concatenate(
                [var_array, da.stack(var_list, axis=-1)], axis=-1
            )
            sl_shape = (*var_array.shape[:-1], len(lev_list))
            single_levs = da.broadcast_to(da.from_array(lev_list), sl_shape)
            lev_array = np.concatenate([lev_array, single_levs], axis=-1)
        return lev_array, var_array

    def do_level_interpolation(
        self, feature, interp_method='linear'
    ) -> xr.DataArray:
        """Interpolate over height or pressure to derive the given feature."""
        fstruct = parse_feature(feature)
        var_array = self.data[fstruct.basename, ...]
        if fstruct.height is not None:
            level = [fstruct.height]
            msg = (
                f'To interpolate {fstruct.basename} to {feature} the loaded '
                'data needs to include "zg" and "topography" or have a '
                f'"{Dimension.HEIGHT}" dimension.'
            )
            can_calc_height = (
                'zg' in self.data.features
                and 'topography' in self.data.features
            )
            have_height = Dimension.HEIGHT in self.data.dims
            assert can_calc_height or have_height, msg

            if can_calc_height:
                lev_array = self.data[['zg', 'topography']].as_array()
                lev_array = lev_array[..., 0] - lev_array[..., 1]
            else:
                lev_array = da.broadcast_to(
                    self.data[Dimension.HEIGHT, ...].astype(np.float32),
                    var_array.shape,
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
                self.data[Dimension.PRESSURE_LEVEL, ...].astype(np.float32),
                var_array.shape,
            )

        lev_array, var_array = self.add_single_level_data(
            feature, lev_array, var_array
        )
        out = Interpolator.interp_to_level(
            lev_array=lev_array,
            var_array=var_array,
            level=np.float32(level),
            interp_method=interp_method,
        )
        return xr.DataArray(
            data=_rechunk_if_dask(out),
            dims=Dimension.dims_3d(),
            attrs=self.data[fstruct.basename].attrs,
        )


class Deriver(BaseDeriver):
    """Extends base :class:`BaseDeriver` class with time_roll and
    hr_spatial_coarsen args."""

    def __init__(
        self,
        data: Union[Sup3rX, Sup3rDataset],
        features,
        time_roll=0,
        hr_spatial_coarsen=1,
        nan_method_kwargs=None,
        FeatureRegistry=None,
        interp_method='linear',
    ):
        """
        Parameters
        ----------
        data : Union[Sup3rX, Sup3rDataset]
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
        interp_method : str
            Interpolation method to use for height interpolation. e.g. Deriving
            u_20m from u_10m and u_100m. Options are "linear" and "log"
        """

        super().__init__(
            data=data,
            features=features,
            FeatureRegistry=FeatureRegistry,
            interp_method=interp_method,
        )

        if time_roll != 0:
            logger.debug(f'Applying time_roll={time_roll} to data array')
            self.data = self.data.roll(**{Dimension.TIME: time_roll})

        if hr_spatial_coarsen > 1:
            logger.debug(
                f'Applying hr_spatial_coarsen={hr_spatial_coarsen} to data.'
            )
            self.data = self.data.coarsen(
                {
                    Dimension.SOUTH_NORTH: hr_spatial_coarsen,
                    Dimension.WEST_EAST: hr_spatial_coarsen,
                },
                boundary='trim',
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
