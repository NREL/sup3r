"""Interpolator class with methods for pressure and height interpolation"""
import numpy as np
import logging
from warnings import warn
from scipy.interpolate import interp1d

from sup3r.utilities.utilities import forward_average


logger = logging.getLogger(__name__)


class Interpolator:
    """Class for handling pressure and height interpolation"""

    @classmethod
    def calc_height(cls, data, raster_index, time_slice=slice(None)):
        """Calculate height from the ground

        Parameters
        ----------
        data : xarray
            netcdf data object
        raster_index : list
            List of slices for raster index of spatial domain
        time_slice : slice
            slice of time to extract

        Returns
        -------
        height_arr : ndarray
            (temporal, vertical_level, spatial_1, spatial_2)
            4D array of heights above ground. In meters.
        """
        if all(field in data for field in ('PHB', 'PH', 'HGT')):
            # Base-state Geopotential(m^2/s^2)
            if any('stag' in d for d in data['PHB'].dims):
                gp = cls.unstagger_var(data, 'PHB', raster_index, time_slice)
            else:
                gp = cls.extract_var(data, 'PHB', raster_index, time_slice)

            # Perturbation Geopotential (m^2/s^2)
            if any('stag' in d for d in data['PH'].dims):
                gp += cls.unstagger_var(data, 'PH', raster_index, time_slice)
            else:
                gp += cls.extract_var(data, 'PH', raster_index, time_slice)

            # Terrain Height (m)
            hgt = data['HGT'][(time_slice,) + tuple(raster_index)]
            if gp.shape != hgt.shape:
                hgt = np.repeat(np.expand_dims(hgt, axis=1), gp.shape[-3],
                                axis=1)
            hgt = gp / 9.81 - hgt
            del gp

        elif all(field in data for field in ('zg', 'orog')):
            if len(data['orog'].dims) == 3:
                hgt = data['orog'][(0,) + tuple(raster_index)]
            else:
                hgt = data['orog'][tuple(raster_index)]
            gp = data['zg'][(time_slice, slice(None),) + tuple(raster_index)]
            hgt = np.repeat(np.expand_dims(hgt, axis=0), gp.shape[1], axis=0)
            hgt = np.repeat(np.expand_dims(hgt, axis=0), gp.shape[0], axis=0)
            hgt = gp - hgt
            del gp

        else:
            msg = ('Need either PHB/PH/HGT or zg/orog in data to perform '
                   'height interpolation')
            raise ValueError(msg)
        return np.array(hgt)

    @classmethod
    def extract_var(cls, data, var, raster_index, time_slice=slice(None)):
        """Extract WRF variable values. This is meant to extract values from
        fields without staggered dimensions

        Parameters
        ----------
        data : xarray
            netcdf data object
        var : str
            Name of variable to be extracted
        raster_index : list
            List of slices for raster index of spatial domain
        time_slice : slice
            slice of time to extract

        Returns
        -------
        ndarray
            Extracted array of variable values.
        """

        idx = [time_slice, slice(None), raster_index[0], raster_index[1]]

        assert not any('stag' in d for d in data[var].dims)

        return np.array(data[var][tuple(idx)], dtype=np.float32)

    @classmethod
    def unstagger_var(cls, data, var, raster_index, time_slice=slice(None)):
        """Unstagger WRF variable values. Some variables use a staggered grid
        with values associated with grid cell edges. We want to center these
        values.

        Parameters
        ----------
        data : xarray
            netcdf data object
        var : str
            Name of variable to be unstaggered
        raster_index : list
            List of slices for raster index of spatial domain
        time_slice : slice
            slice of time to extract

        Returns
        -------
        ndarray
            Unstaggered array of variable values.
        """

        idx = [time_slice, slice(None), raster_index[0], raster_index[1]]
        assert any('stag' in d for d in data[var].dims)

        if 'stag' in data[var].dims[2]:
            idx[2] = slice(idx[2].start, idx[2].stop + 1)
        if 'stag' in data[var].dims[3]:
            idx[3] = slice(idx[3].start, idx[3].stop + 1)

        array_in = np.array(data[var][tuple(idx)], dtype=np.float32)

        for i, d in enumerate(data[var].dims):
            if 'stag' in d:
                array_in = np.apply_along_axis(forward_average, i, array_in)

        return array_in

    @classmethod
    def calc_pressure(cls, data, var, raster_index, time_slice=slice(None)):
        """Calculate pressure array

        Parameters
        ----------
        data : xarray
            netcdf data object
        var : str
            Feature to extract from data
        raster_index : list
            List of slices for raster index of spatial domain
        time_slice : slice
            slice of time to extract

        Returns
        -------
        height_arr : ndarray
            (temporal, vertical_level, spatial_1, spatial_2)
            4D array of pressure levels in pascals
        """
        idx = (time_slice, slice(None),) + tuple(raster_index)
        p_array = np.zeros(data[var][idx].shape, dtype=np.float32)
        for i in range(p_array.shape[1]):
            p_array[:, i, ...] = data.plev[i]

        return p_array

    @classmethod
    def interp_to_level(cls, var_array, lev_array, levels):
        """Interpolate var_array to given level(s) based on h_array.
        Interpolation is linear and done for every 'z' column of [var, h] data.

        Parameters
        ----------
        var_array : ndarray
            Array of variable data, for example u-wind in a 4D array of shape
            (time, vertical, lat, lon)
        lev_array : ndarray
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
        out_array : ndarray
            Array of interpolated values.
        """

        msg = ('Input arrays must be the same shape.'
               f'\nvar_array: {var_array.shape}'
               f'\nh_array: {lev_array.shape}')
        assert var_array.shape == lev_array.shape, msg

        levels = ([levels] if isinstance(levels, (int, float, np.float32))
                  else levels)

        if np.isnan(lev_array).all():
            msg = 'All pressure level height data is NaN!'
            logger.error(msg)
            raise RuntimeError(msg)

        nans = np.isnan(lev_array)
        logger.debug('lev_array.shape: {}'.format(lev_array.shape))
        bad_min = min(levels) < lev_array[:, 0, :, :]
        bad_max = max(levels) > lev_array[:, -1, :, :]

        if nans.any():
            msg = ('Approximately {:.2f}% of the vertical level '
                   'array is NaN. Data will be interpolated or extrapolated '
                   'past these NaN values.'
                   .format(100 * nans.sum() / nans.size))
            logger.warning(msg)
            warn(msg)

        if bad_min.any():
            msg = ('Approximately {:.2f}% of the lowest vertical levels '
                   '(maximum value of {:.3f}, minimum value of {:.3f}) '
                   'were greater than the minimum requested level: {}'
                   .format(100 * bad_min.sum() / bad_min.size,
                           lev_array[:, 0, :, :].max(),
                           lev_array[:, 0, :, :].min(), min(levels)))
            logger.warning(msg)
            warn(msg)

        if bad_max.any():
            msg = ('Approximately {:.2f}% of the highest vertical levels '
                   '(minimum value of {:.3f}, maximum value of {:.3f}) '
                   'were lower than the maximum requested level: {}'
                   .format(100 * bad_max.sum() / bad_max.size,
                           lev_array[:, -1, :, :].min(),
                           lev_array[:, -1, :, :].max(),
                           max(levels)))
            logger.warning(msg)
            warn(msg)

        array_shape = var_array.shape

        # Flatten h_array and var_array along lat, long axis
        shape = (len(levels), array_shape[-4], np.product(array_shape[-2:]))
        out_array = np.zeros(shape, dtype=np.float32).T

        # if multiple vertical levels have identical heights at the desired
        # interpolation level, interpolation to that value will fail because
        # linear slope will be NaN. This is most common if you have multiple
        # pressure levels at zero height at the surface in the case that the
        # data didnt provide underground data.
        for level in levels:
            mask = (lev_array == level)
            lev_array[mask] += np.random.uniform(-1e-5, 0, size=mask.sum())

        # iterate through time indices
        for idt in range(array_shape[0]):
            shape = (array_shape[-3], np.product(array_shape[-2:]))
            h_tmp = lev_array[idt].reshape(shape).T
            var_tmp = var_array[idt].reshape(shape).T
            not_nan = ~np.isnan(h_tmp) & ~np.isnan(var_tmp)

            # Interp each vertical column of height and var to requested levels
            zip_iter = zip(h_tmp, var_tmp, not_nan)
            out_array[:, idt, :] = np.array(
                [interp1d(h[mask], var[mask], fill_value='extrapolate')(levels)
                 for h, var, mask in zip_iter], dtype=np.float32)

        # Reshape out_array
        if isinstance(levels, (float, np.float32, int)):
            shape = (1, array_shape[-4], array_shape[-2], array_shape[-1])
            out_array = out_array.T.reshape(shape)
        else:
            shape = (len(levels), array_shape[-4], array_shape[-2],
                     array_shape[-1])
            out_array = out_array.T.reshape(shape)

        return out_array

    @classmethod
    def interp_var_to_height(cls, data, var, raster_index, heights,
                             time_slice=slice(None)):
        """Interpolate var_array to given level(s) based on h_array.
        Interpolation is linear and done for every 'z' column of [var, h] data.

        Parameters
        ----------
        data : xarray
            netcdf data object
        var : str
            Name of variable to be interpolated
        raster_index : list
            List of slices for raster index of spatial domain
        heights : float | list
            level or levels to interpolate to (e.g. final desired hub heights)
        time_slice : slice
            slice of time to extract

        Returns
        -------
        out_array : ndarray
            Array of interpolated values.
        """
        if len(data[var].dims) == 5:
            raster_index = [0] + raster_index
        logger.debug(f'Interpolating {var} to heights (meters): {heights}')
        hgt = cls.calc_height(data, raster_index, time_slice)
        logger.info(f'Computed height array with min/max: {np.nanmin(hgt)} / '
                    f'{np.nanmax(hgt)}')
        if data[var].dims == ('plev',):
            arr = np.array(data[var])
            arr = np.expand_dims(arr, axis=(0, 2, 3))
            arr = np.repeat(arr, hgt.shape[0], axis=0)
            arr = np.repeat(arr, hgt.shape[2], axis=2)
            arr = np.repeat(arr, hgt.shape[3], axis=3)
        elif all('stag' not in d for d in data[var].dims):
            arr = cls.extract_var(data, var, raster_index, time_slice)
        else:
            arr = cls.unstagger_var(data, var, raster_index, time_slice)
        return cls.interp_to_level(arr, hgt, heights)[0]

    @classmethod
    def interp_var_to_pressure(cls, data, var, raster_index, pressures,
                               time_slice=slice(None)):
        """Interpolate var_array to given level(s) based on h_array.
        Interpolation is linear and done for every 'z' column of [var, h] data.

        Parameters
        ----------
        data : xarray
            netcdf data object
        var : str
            Name of variable to be interpolated
        raster_index : list
            List of slices for raster index of spatial domain
        pressures : float | list
            level or levels to interpolate to (e.g. final desired hub heights)
        time_slice : slice
            slice of time to extract

        Returns
        -------
        out_array : ndarray
            Array of interpolated values.
        """
        logger.debug(f'Interpolating {var} to pressures (Pa): {pressures}')
        if len(data[var].dims) == 5:
            raster_index = [0] + raster_index

        if all('stag' not in d for d in data[var].dims):
            arr = cls.extract_var(data, var, raster_index, time_slice)
        else:
            arr = cls.unstagger_var(data, var, raster_index, time_slice)

        p_levels = cls.calc_pressure(data, var, raster_index, time_slice)

        return cls.interp_to_level(arr[:, ::-1], p_levels[:, ::-1],
                                   pressures)[0]
