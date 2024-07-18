"""Interpolator class with methods for pressure and height interpolation"""

import logging
from warnings import warn

import dask.array as da
import numpy as np

from sup3r.preprocessing.utilities import (
    _compute_chunks_if_dask,
    _compute_if_dask,
)
from sup3r.typing import T_Array
from sup3r.utilities.utilities import RANDOM_GENERATOR

logger = logging.getLogger(__name__)


class Interpolator:
    """Class for handling pressure and height interpolation"""

    @classmethod
    def get_level_masks(cls, lev_array, level):
        """Get the masks used to select closest surrounding levels in the
        lev_array to requested interpolation level.

        Parameters
        ----------
        var_array : T_Array
            Array of variable data, for example u-wind in a 4D array of shape
            (lat, lon, time, level)
        lev_array : T_Array
            Height or pressure values for the corresponding entries in
            var_array, in the same shape as var_array. If this is height and
            the requested levels are hub heights above surface, lev_array
            should be the geopotential height corresponding to every var_array
            index relative to the surface elevation (subtract the elevation at
            the surface from the geopotential height)
        level : float
            level to interpolate to (e.g. final desired hub height
            above surface elevation)

        Returns
        -------
        mask1 : T_Array
            Array of bools selecting the entries with the closest levels to the
            one requested.
            (lat, lon, time, level)
        mask2 : T_Array
            Array of bools selecting the entries with the second closest levels
            to the one requested.
            (lat, lon, time, level)
        """
        over_mask = lev_array > level
        under_levs = (
            da.ma.masked_array(lev_array, over_mask)
            if ~over_mask.sum() >= lev_array[..., 0].size
            else lev_array
        )
        diff1 = da.abs(under_levs - level)
        lev_indices = da.broadcast_to(
            da.arange(lev_array.shape[-1]), lev_array.shape
        )
        mask1 = lev_indices == da.argmin(diff1, axis=-1, keepdims=True)

        over_levs = (
            da.ma.masked_array(lev_array, ~over_mask)
            if over_mask.sum() >= lev_array[..., 0].size
            else da.ma.masked_array(lev_array, mask1)
        )
        diff2 = da.abs(over_levs - level)
        mask2 = lev_indices == da.argmin(diff2, axis=-1, keepdims=True)
        return mask1, mask2

    @classmethod
    def _lin_interp(cls, lev_samps, var_samps, level):
        """Linearly interpolate between levels."""
        diff = lev_samps[1] - lev_samps[0]
        alpha = (level - lev_samps[0]) / diff
        alpha = da.where(diff == 0, 0, alpha)
        return var_samps[0] * (1 - alpha) + var_samps[1] * alpha

    @classmethod
    def _log_interp(cls, lev_samps, var_samps, level):
        """Interpolate between levels with log profile.

        Note
        ----
        Here we fit the function a * log(h - h0 + 1) + v0 to the two given
        levels and variable values. So a is calculated with `v1 = a * log(h1 -
        h0 + 1) + v0` where v1, v0 are var_samps[0], var_samps[1] and h1, h0
        are lev_samps[1], lev_samps[0]
        """
        mask = lev_samps[0] < lev_samps[1]
        h0 = da.where(mask, lev_samps[0], lev_samps[1])
        h1 = da.where(mask, lev_samps[1], lev_samps[0])
        v0 = da.where(mask, var_samps[0], var_samps[1])
        v1 = da.where(mask, var_samps[1], var_samps[0])
        coeff = da.where(h1 == h0, 0, (v1 - v0) / np.log(h1 - h0 + 1))
        coeff = da.where(level < h0, -coeff, coeff)
        return coeff * np.log(da.abs(level - h0) + 1) + v0

    @classmethod
    def interp_to_level(
        cls,
        lev_array: T_Array,
        var_array: T_Array,
        level,
        interp_method='linear',
    ):
        """Interpolate var_array to the given level.

        Parameters
        ----------
        var_array : xr.DataArray
            Array of variable data, for example u-wind in a 4D array of shape
            (lat, lon, time, level)
        lev_array : xr.DataArray
            Height or pressure values for the corresponding entries in
            var_array, in the same shape as var_array. If this is height and
            the requested levels are hub heights above surface, lev_array
            should be the geopotential height corresponding to every var_array
            index relative to the surface elevation (subtract the elevation at
            the surface from the geopotential height)
        level : float
            level or levels to interpolate to (e.g. final desired hub height
            above surface elevation)

        Returns
        -------
        out : T_Array
            Interpolated var_array
            (lat, lon, time)
        """
        cls._check_lev_array(lev_array, levels=[level])
        levs = da.ma.masked_array(lev_array, da.isnan(lev_array))
        mask1, mask2 = cls.get_level_masks(levs, level)
        lev1 = _compute_chunks_if_dask(lev_array[mask1])
        lev1 = lev1.reshape(mask1.shape[:-1])
        lev2 = _compute_chunks_if_dask(lev_array[mask2])
        lev2 = lev2.reshape(mask2.shape[:-1])
        var1 = _compute_chunks_if_dask(var_array[mask1])
        var1 = var1.reshape(mask1.shape[:-1])
        var2 = _compute_chunks_if_dask(var_array[mask2])
        var2 = var2.reshape(mask2.shape[:-1])

        if interp_method == 'log':
            out = cls._log_interp(
                lev_samps=[lev1, lev2], var_samps=[var1, var2], level=level
            )
        else:
            out = cls._lin_interp(
                lev_samps=[lev1, lev2], var_samps=[var1, var2], level=level
            )

        return out

    @classmethod
    def _check_lev_array(cls, lev_array, levels):
        """Check if the requested levels are consistent with the given
        lev_array and if there are any nans in the lev_array."""

        if np.isnan(lev_array).all():
            msg = 'All pressure level height data is NaN!'
            logger.error(msg)
            raise RuntimeError(msg)

        nans = np.isnan(lev_array)
        logger.debug('Level array shape: {}'.format(lev_array.shape))

        lowest_height = np.min(lev_array, axis=-1)
        highest_height = np.max(lev_array, axis=-1)
        bad_min = min(levels) < lowest_height
        bad_max = max(levels) > highest_height

        if nans.any():
            nans = _compute_if_dask(nans)
            msg = (
                'Approximately {:.2f}% of the vertical level '
                'array is NaN. Data will be interpolated or extrapolated '
                'past these NaN values.'.format(100 * nans.sum() / nans.size)
            )
            logger.warning(msg)
            warn(msg)

        # This and the next if statement can return warnings in the case of
        # pressure inversions, in which case the "lowest" or "highest" pressure
        # does not correspond to the lowest or highest height. Interpolation
        # can be performed without issue in this case.
        if bad_min.any():
            bad_min = _compute_if_dask(bad_min)
            lev_array = _compute_if_dask(lev_array)
            msg = (
                'Approximately {:.2f}% of the lowest vertical levels '
                '(maximum value of {:.3f}, minimum value of {:.3f}) '
                'were greater than the minimum requested level: {}'.format(
                    100 * bad_min.sum() / bad_min.size,
                    lev_array[..., 0].max(),
                    lev_array[..., 0].min(),
                    min(levels),
                )
            )
            logger.warning(msg)
            warn(msg)

        if bad_max.any():
            bad_max = _compute_if_dask(bad_max)
            lev_array = _compute_if_dask(lev_array)
            msg = (
                'Approximately {:.2f}% of the highest vertical levels '
                '(minimum value of {:.3f}, maximum value of {:.3f}) '
                'were lower than the maximum requested level: {}'.format(
                    100 * bad_max.sum() / bad_max.size,
                    lev_array[..., -1].min(),
                    lev_array[..., -1].max(),
                    max(levels),
                )
            )
            logger.warning(msg)
            warn(msg)

    @classmethod
    def prep_level_interp(cls, var_array, lev_array, levels):
        """Prepare var_array interpolation. Check level ranges and add noise to
        mask locations.

        Parameters
        ----------
        var_array : T_Array
            Array of variable data, for example u-wind in a 4D array of shape
            (time, vertical, lat, lon)
        lev_array : T_Array
            Array of height or pressure values corresponding to the wrf source
            data in the same shape as var_array. If this is height and the
            requested levels are hub heights above surface, lev_array should be
            the geopotential height corresponding to every var_array index
            relative to the surface elevation (subtract the elevation at the
            surface from the geopotential height)
        levels : float | list
            level or levels to interpolate to (e.g. final desired hub heights
            above surface elevation)

        Returns
        -------
        lev_array : T_Array
            Array of levels with noise added to mask locations.
        levels : list
            List of levels to interpolate to.
        """

        msg = (
            'Input arrays must be the same shape.'
            f'\nvar_array: {var_array.shape}'
            f'\nh_array: {lev_array.shape}'
        )
        assert var_array.shape == lev_array.shape, msg

        levels = (
            [levels]
            if isinstance(levels, (int, float, np.float32))
            else levels
        )

        cls._check_lev_array(lev_array, levels)

        # if multiple vertical levels have identical heights at the desired
        # interpolation level, interpolation to that value will fail because
        # linear slope will be NaN. This is most common if you have multiple
        # pressure levels at zero height at the surface in the case that the
        # data didnt provide underground data.
        for level in levels:
            mask = lev_array == level
            random = RANDOM_GENERATOR.uniform(-1e-5, 0, size=mask.sum())
            lev_array = da.ma.masked_array(lev_array, mask)
            lev_array = da.ma.filled(lev_array, random)

        return lev_array, levels
