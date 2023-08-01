"""Rescale ERA5 wind components according to log profile"""

import logging
import os
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from glob import glob
from warnings import warn

import numpy as np
import xarray as xr
from netCDF4 import Dataset
from rex import init_logger
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from sup3r.utilities.interpolation import Interpolator

init_logger(__name__, log_level='DEBUG')
init_logger('sup3r', log_level='DEBUG')


logger = logging.getLogger(__name__)


class LogLinInterpolator:
    """Open ERA5 file, log interpolate wind components between 0 -
    max_log_height, linearly interpolate components above max_log_height
    meters, and save to file"""

    def __init__(self, infile, outfile, output_heights=None,
                 input_heights=None, max_log_height=100,
                 only_fixed_levels=True):
        """Initialize log interpolator.

        Parameters
        ----------
        infile : str
            Path to ERA5 data to use for windspeed log interpolation. Assumed
            to contain zg, orog, and at least u/v at 10m and 100m.
        outfile : str
            Path to save output after log interpolation.
        output_heights : None | list
            Heights to interpolate to. If None this defaults to [40, 80].
        input_heights : None | list
            Explicit heights to use in interpolation. e.g. If this is [10, 100]
            then u/v at 10 and 100m will be included explicitly in the input
            array used for interpolation. interpolate to. If None this defaults
            to [10, 100].
        max_log_height : int
            Maximum height to use for log interpolation. Above this linear
            interpolation will be used.
        only_fixed_levels : bool
            Use only fixed levels for log interpolation. Fixed levels are those
            that were not computed from pressure levels but instead added along
            with wind components at explicit heights (e.g u_10m, v_10m, u_100m,
            v_100m)
        """
        self.infile = infile
        self.outfile = outfile
        self.new_heights = output_heights or [40, 80]
        self.input_heights = input_heights or [10, 100]
        self.max_log_height = max_log_height
        self.u = None
        self.v = None
        self.u_new = None
        self.v_new = None
        self.heights = None
        self.fixed_level_mask = None

        msg = (f'{self.infile} does not exist. Skipping.')
        assert os.path.exists(self.infile), msg

        if only_fixed_levels:
            self.fixed_level_mask = [True] * len(self.input_heights)

        msg = (f'Initializing LogInterpolator with infile={infile}, '
               f'outfile={outfile}, new_heights={self.new_heights}')
        logger.info(msg)

    def load(self):
        """Load ERA5 data and create wind component arrays"""
        with xr.open_dataset(self.infile) as res:
            gp = res['zg'].values
            sfc_hgt = np.repeat(res['orog'].values[:, np.newaxis, ...],
                                gp.shape[1], axis=1)
            self.heights = gp - sfc_hgt

            if self.fixed_level_mask is not None:
                self.fixed_level_mask += [False] * self.heights.shape[1]

            u_arr = []
            v_arr = []
            height_arr = []
            shape = (self.heights.shape[0], 1, *self.heights.shape[2:])
            for height in self.input_heights:
                u_arr.append(res[f'u_{height}m'].values[:, np.newaxis, ...])
                v_arr.append(res[f'v_{height}m'].values[:, np.newaxis, ...])
                height_arr.append(np.full(shape, height))
            u_arr.append(res['u'].values)
            v_arr.append(res['v'].values)
            height_arr.append(self.heights)

            self.heights = np.concatenate(height_arr, axis=1)
            self.u = np.concatenate(u_arr, axis=1)
            self.v = np.concatenate(v_arr, axis=1)

    def interpolate_wind(self, max_workers=None):
        """Interpolate u/v wind components below 100m using log profile"""
        self.u_new = self.interp_ws_to_height(self.u, self.heights,
                                              self.new_heights,
                                              self.fixed_level_mask,
                                              self.max_log_height,
                                              max_workers)
        self.v_new = self.interp_ws_to_height(self.v, self.heights,
                                              self.new_heights,
                                              self.fixed_level_mask,
                                              self.max_log_height,
                                              max_workers)

    def save_output(self):
        """Save interpolated wind components to outfile"""
        dirname = os.path.dirname(self.outfile)
        os.makedirs(dirname, exist_ok=True)
        os.system(f'cp {self.infile} {self.outfile}')
        ds = Dataset(self.outfile, 'a')
        for i, height in enumerate(self.new_heights):
            variable = ds.variables['u_10m']
            name = f'u_{height}m'
            if name not in ds.variables:
                _ = ds.createVariable(name,
                                      np.float32,
                                      dimensions=variable.dimensions)
                ds.variables[name][:] = self.u_new[i, ...]
                ds.variables[name].units = 'm s**-1'
                ds.variables[name].long_name = f'{height} meter U Component'
            variable = ds.variables['v_10m']
            name = f'v_{height}m'
            if name not in ds.variables:
                ds.createVariable(name,
                                  np.float32,
                                  dimensions=variable.dimensions)
                ds.variables[name][:] = self.v_new[i, ...]
                ds.variables[name].units = 'm s**-1'
                ds.variables[name].long_name = f'{height} meter V Component'
        ds.close()
        logger.info(f'Saved interpolated output to {self.outfile}.')

    @classmethod
    def run(cls, infile, outfile, output_heights=None, input_heights=None,
            only_fixed_levels=True, max_log_height=100, overwrite=False,
            max_workers=None):
        """Run interpolation and save output

        Parameters
        ----------
        infile : str
            Path to ERA5 data to use for windspeed log interpolation. Assumed
            to contain zg, orog, and at least u/v at 10m and 100m.
        outfile : str
            Path to save output after log interpolation.
        output_heights : None | list
            Heights to interpolate to. If None this defaults to [40, 80].
        input_heights : None | list
            Explicit heights to use in interpolation. e.g. If this is [10, 100]
            then u/v at 10 and 100m will be included explicitly in the input
            array used for interpolation. interpolate to. If None this defaults
            to [10, 100].
        only_fixed_levels : bool
            Use only fixed levels for log interpolation. Fixed levels are those
            that were not computed from pressure levels but instead added along
            with wind components at explicit heights (e.g u_10m, v_10m, u_100m,
            v_100m)
        max_log_height : int
            Maximum height to use for log interpolation. Above this linear
            interpolation will be used.
        overwrite : bool
            Whether to overwrite existing outfile.
        max_workers : None | int
            Number of workers to use for interpolating over timesteps.
        """
        if os.path.exists(outfile) and not overwrite:
            logger.info(f'{outfile} exists and overwrite=False. Skipping.')
        else:
            log_interp = cls(infile, outfile,
                             output_heights=output_heights,
                             input_heights=input_heights,
                             only_fixed_levels=only_fixed_levels,
                             max_log_height=max_log_height)
            log_interp.load()
            log_interp.interpolate_wind(max_workers=max_workers)
            log_interp.save_output()

    @classmethod
    def run_multiple(cls, infiles, out_dir, heights=None,
                     overwrite=False, max_workers=None):
        """Run log interpolation on multiple files

        Parameters
        ----------
        infiles : str | list
            List of ERA5 data files or a globbable string to use for windspeed
            log interpolation. Assumed to contain u/v at 10m, 100m, and at
            least one height between.
        out_dir : str
            Directory to save output after log interpolation.
        heights : None | list
            Heights to interpolate to. If None this defaults to [40, 80].
        overwrite : bool
            Whether to overwrite exisitng outfiles.
        max_workers : None | bool
            Number of workers to use for thread pool when running multiple log
            interpolation routines.
        """
        futures = []
        if isinstance(infiles, str):
            infiles = glob(infiles)
        if max_workers == 1:
            for _, file in enumerate(infiles):
                outfile = os.path.basename(file).replace('.nc',
                                                         '_log_interp.nc')
                outfile = os.path.join(out_dir, outfile)
                cls.run(file, outfile, heights, overwrite,
                        max_workers=max_workers)

        else:
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                for i, file in enumerate(infiles):
                    outfile = os.path.basename(file).replace('.nc',
                                                             '_log_interp.nc')
                    outfile = os.path.join(out_dir, outfile)
                    futures.append(exe.submit(cls.run, file, outfile, heights,
                                              overwrite, max_workers))
                    logger.info(
                        f'{i + 1} of {len(infiles)} futures submitted.')
            for i, future in enumerate(as_completed(futures)):
                future.result()
                logger.info(f'{i + 1} of {len(futures)} futures complete.')

    @classmethod
    def pbl_interp_to_height(cls, lev_array, var_array, levels,
                             fixed_level_mask=None,
                             max_log_height=100):
        """Fit ws log law to data below max_log_height.

        Parameters
        ----------
        lev_array : ndarray
            1D Array of height values corresponding to the wrf source
            data in the same shape as var_array.
        var_array : ndarray
            1D Array of variable data, for example u-wind in a 1D array of
            shape
        levels : float | list
            level or levels to interpolate to (e.g. final desired hub heights
            above surface elevation)
        fixed_level_mask : ndarray | None
            Optional mask to use only fixed levels. Fixed levels are those that
            were not computed from pressure levels but instead added along with
            wind components at explicit heights (e.g u_10m, v_10m, u_100m,
            v_100m)
        max_log_height : int
            Max height for using log interpolation.

        Returns
        -------
        values : ndarray
            Array of interpolated windspeed values below max_log_height.
        """
        def ws_log_profile(z, a, b):
            return a * np.log(z) + b

        lev_array_samp = lev_array.copy()
        var_array_samp = var_array.copy()
        if fixed_level_mask is not None:
            lev_array_samp = lev_array_samp[fixed_level_mask]
            var_array_samp = var_array_samp[fixed_level_mask]

        levels = np.array(levels)
        lev_mask = (0 < levels) & (levels <= max_log_height)
        var_mask = (0 < lev_array_samp) & (lev_array_samp <= max_log_height)

        try:
            popt, _ = curve_fit(ws_log_profile, lev_array_samp[var_mask],
                                var_array_samp[var_mask])
            log_ws = ws_log_profile(levels[lev_mask], *popt)
        except Exception as e:
            msg = ('Log interp failed with (h, ws) = '
                   f'({lev_array_samp[var_mask]}, '
                   f'{var_array_samp[var_mask]}). {e} '
                   'Using linear interpolation.')
            logger.warning(msg)
            warn(msg)
            log_ws = interp1d(lev_array[var_mask], var_array[var_mask],
                              fill_value='extrapolate')(levels[lev_mask])
        return log_ws

    @classmethod
    def _interp_ws_to_height(cls, lev_array, var_array, levels,
                             fixed_level_mask=None,
                             max_log_height=100):
        """Fit ws log law to data below max_log_height and linearly
        interpolate data above.

        Parameters
        ----------
        lev_array : ndarray
            1D Array of height values corresponding to the wrf source
            data in the same shape as var_array.
        var_array : ndarray
            1D Array of variable data, for example u-wind in a 1D array of
            shape
        levels : float | list
            level or levels to interpolate to (e.g. final desired hub heights
            above surface elevation)
        fixed_level_mask : ndarray | None
            Optional mask to use only fixed levels. Fixed levels are those that
            were not computed from pressure levels but instead added along with
            wind components at explicit heights (e.g u_10m, v_10m, u_100m,
            v_100m)
        max_log_height : int
            Max height for using log interpolation.

        Returns
        -------
        values : ndarray
            Array of interpolated windspeed values at the requested heights.
        """
        levels = np.array(levels)

        log_ws = None
        lin_ws = None

        hgt_check = (any(levels < max_log_height)
                     and any(lev_array < max_log_height))
        if hgt_check:
            log_ws = cls.pbl_interp_to_height(
                lev_array, var_array, levels,
                fixed_level_mask=fixed_level_mask,
                max_log_height=max_log_height)

        if any(levels > max_log_height):
            lev_mask = levels >= max_log_height
            var_mask = lev_array >= max_log_height

            if len(lev_array[var_mask]) > 1:
                lin_ws = interp1d(lev_array[var_mask], var_array[var_mask],
                                  fill_value='extrapolate')(levels[lev_mask])
            else:
                msg = ('Requested interpolation levels are outside the '
                       f'available range: lev_array={lev_array}, '
                       f'levels={levels}. Using linear extrapolation.')
                lin_ws = interp1d(lev_array, var_array,
                                  fill_value='extrapolate')(levels[lev_mask])
                logger.warning(msg)
                warn(msg)

        if log_ws is not None and lin_ws is not None:
            out = np.concatenate([log_ws, lin_ws])

        if log_ws is not None and lin_ws is None:
            out = log_ws

        if lin_ws is not None and log_ws is None:
            out = lin_ws

        if log_ws is None and lin_ws is None:
            msg = (f'No interpolation was performed for lev_array={lev_array} '
                   f'and levels={levels}')
            raise RuntimeError(msg)

        return out

    @classmethod
    def _get_timestep_interp_input(cls, lev_array, var_array, idt):
        """Get interpolation input for given timestep

        Parameters
        ----------
        lev_array : ndarray
            1D Array of height values corresponding to the wrf source
            data in the same shape as var_array.
        var_array : ndarray
            1D Array of variable data, for example u-wind in a 1D array of
            shape
        idt : int
            Time index to interpolate

        Returns
        -------
        h_t : ndarray
            1D array of height values for the requested time
        v_t : ndarray
            1D array of variable data for the requested time
        mask : ndarray
            1D array of bool values masking nans and heights < 0

        """
        array_shape = var_array.shape
        shape = (array_shape[-3], np.product(array_shape[-2:]))
        h_t = lev_array[idt].reshape(shape).T
        var_t = var_array[idt].reshape(shape).T
        mask = ~np.isnan(h_t) & ~np.isnan(var_t)

        return h_t, var_t, mask

    @classmethod
    def interp_single_ts(cls, hgt_t, var_t, mask, levels,
                         fixed_level_mask=None, max_log_height=100):
        """Perform interpolation for a single timestep specified by the index
        idt

        Parameters
        ----------
        hgt_t : ndarray
            1D Array of height values for a specific time.
        var_t : ndarray
            1D Array of variable data for a specific time.
        mask : ndarray
            1D Array of bool values to mask out nans and heights below 0.
        levels : float | list
            level or levels to interpolate to (e.g. final desired hub heights
            above surface elevation)
        fixed_level_mask : ndarray | None
            Optional mask to use only fixed levels. Fixed levels are those
            that were not computed from pressure levels but instead added along
            with wind components at explicit heights (e.g u_10m, v_10m, u_100m,
            v_100m)
        max_log_height : int
            Max height for using log interpolation.

        Returns
        -------
        out_array : ndarray
            Array of interpolated values.
        """

        # Interp each vertical column of height and var to requested levels
        zip_iter = zip(hgt_t, var_t, mask)
        return np.array(
            [cls._interp_ws_to_height(h[mask], var[mask], levels,
                                      fixed_level_mask=fixed_level_mask,
                                      max_log_height=max_log_height)
             for h, var, mask in zip_iter], dtype=np.float32)

    @classmethod
    def interp_ws_to_height(cls, var_array, lev_array, levels,
                            fixed_level_mask=None,
                            max_log_height=100, max_workers=None):
        """Interpolate windspeed array to given level(s) based on h_array.
        Interpolation is done using windspeed log profile and is done for every
        'z' column of [var, h] data.

        Parameters
        ----------
        var_array : ndarray
            Array of variable data, for example u-wind in a 4D array of shape
            (time, vertical, lat, lon)
        lev_array : ndarray
            Array of height values corresponding to the wrf source
            data in the same shape as var_array. lev_array should be
            the geopotential height corresponding to every var_array index
            relative to the surface elevation (subtract the elevation at the
            surface from the geopotential height)
        levels : float | list
            level or levels to interpolate to (e.g. final desired hub heights
            above surface elevation)
        fixed_level_mask : ndarray | None
            Optional mask to use only fixed levels. Fixed levels are those
            that were not computed from pressure levels but instead added along
            with wind components at explicit heights (e.g u_10m, v_10m, u_100m,
            v_100m)
        max_log_height : int
            Max height for using log interpolation.
        max_workers : None | int
            Number of workers to use for interpolating over timesteps.

        Returns
        -------
        out_array : ndarray
            Array of interpolated values.
        """
        lev_array, levels = Interpolator.prep_level_interp(var_array,
                                                           lev_array,
                                                           levels)

        array_shape = var_array.shape

        # Flatten h_array and var_array along lat, long axis
        shape = (len(levels), array_shape[-4], np.product(array_shape[-2:]))
        out_array = np.zeros(shape, dtype=np.float32).T

        # iterate through time indices
        futures = {}
        if max_workers == 1:
            for idt in range(array_shape[0]):
                h_t, v_t, mask = cls._get_timestep_interp_input(lev_array,
                                                                var_array,
                                                                idt)
                out_array[:, idt, :] = cls.interp_single_ts(
                    h_t, v_t, mask, levels=levels,
                    fixed_level_mask=fixed_level_mask,
                    max_log_height=max_log_height)
                logger.info(
                    f'{idt + 1} of {array_shape[0]} timesteps finished.')

        else:
            with ProcessPoolExecutor(max_workers=max_workers) as exe:
                for idt in range(array_shape[0]):
                    h_t, v_t, mask = cls._get_timestep_interp_input(lev_array,
                                                                    var_array,
                                                                    idt)
                    future = exe.submit(cls.interp_single_ts, h_t, v_t, mask,
                                        levels=levels,
                                        fixed_level_mask=fixed_level_mask,
                                        max_log_height=max_log_height)
                    futures[future] = idt
                    logger.info(
                        f'{idt + 1} of {array_shape[0]} futures submitted.')
            for i, future in enumerate(as_completed(futures)):
                out_array[:, futures[future], :] = future.result()
                logger.info(f'{i + 1} of {len(futures)} futures complete.')

        # Reshape out_array
        if isinstance(levels, (float, np.float32, int)):
            shape = (1, array_shape[-4], array_shape[-2], array_shape[-1])
            out_array = out_array.T.reshape(shape)
        else:
            shape = (len(levels), array_shape[-4], array_shape[-2],
                     array_shape[-1])
            out_array = out_array.T.reshape(shape)

        return out_array
