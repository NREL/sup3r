"""Rescale ERA5 wind components according to log profile"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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


class LogInterpolator:
    """Open ERA5 file, rescale wind components between 0 - 100 meters, and save
    to file"""

    def __init__(self, infile, outfile, heights=None):
        """Initialize log interpolator.

        Parameters
        ----------
        infile : str
            Path to ERA5 data to use for windspeed log interpolation. Assumed
            to contain zg, orog, and at least u/v at 10m and 100m.
        outfile : str
            Path to save output after log interpolation.
        heights : None | list
            Heights to interpolate to. If None this defaults to [40, 80].
        """
        self.infile = infile
        self.outfile = outfile
        self.new_heights = heights or [40, 80]
        self.DATA_COUNT = 0
        self.SUCESS_COUNT = 0
        self.u = None
        self.v = None
        self.u_new = None
        self.v_new = None
        self.heights = None
        msg = (f'Initializing LogInterpolator with infile={infile}, '
               f'outfile={outfile}, new_heights={self.new_heights}')
        logger.info(msg)

    def load(self):
        """Load ERA5 data and create wind component arrays"""
        with xr.open_dataset(self.infile) as res:
            self.heights = res['zg'].values - res['orog'].values
            u_10m = res['u_10m'].values
            v_10m = res['v_10m'].values
            u_100m = res['u_100m'].values
            v_100m = res['v_100m'].values
            u = res['u'].values
            v = res['v'].values
            shape = (self.heights.shape[0], 1, *self.heights.shape[2:])
            self.heights = np.concatenate([np.full(shape, 10),
                                           np.full(shape, 100),
                                           self.heights], axis=1)
            self.u = np.concatenate([u_10m[:, np.newaxis, ...],
                                     u_100m[:, np.newaxis, ...],
                                     u], axis=1)
            self.v = np.concatenate([v_10m[:, np.newaxis, ...],
                                     v_100m[:, np.newaxis, ...],
                                     v], axis=1)

    def interpolate_wind(self):
        """Interpolate windspeed using log profile and winddirection
        linearly"""
        self.SUCESS_COUNT = 0
        self.DATA_COUNT = 0
        self.u_new = self.interp_ws_to_height(self.u, self.heights,
                                              self.new_heights)
        msg = (f'{self.SUCESS_COUNT + 1} of {self.DATA_COUNT + 1} points '
               'used log interpolation for U.')
        logger.info(msg)
        self.SUCESS_COUNT = 0
        self.DATA_COUNT = 0
        self.v_new = self.interp_ws_to_height(self.v, self.heights,
                                              self.new_heights)
        msg = (f'{self.SUCESS_COUNT + 1} of {self.DATA_COUNT + 1} points '
               'used log interpolation for V.')
        logger.info(msg)

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
    def run(cls, infile, outfile, heights=None, overwrite=False):
        """Run interpolation and save output

        Parameters
        ----------
        infile : str
            Path to ERA5 data to use for windspeed log interpolation. Assumed
            to contain u/v at 10m, 100m, and at least one height between.
        outfile : str
            Path to save output after log interpolation.
        heights : None | list
            Heights to interpolate to. If None this defaults to [40, 80].
        overwrite : bool
            Whether to overwrite exisitng outfile.
        """
        if os.path.exists(outfile) and not overwrite:
            logger.info(f'{outfile} exists and overwrite=False. Skipping.')
        else:
            log_interp = cls(infile, outfile, heights)
            log_interp.load()
            log_interp.interpolate_wind()
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
                cls.run(file, outfile, heights, overwrite)

        else:
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                for i, file in enumerate(infiles):
                    outfile = os.path.basename(file).replace('.nc',
                                                             '_log_interp.nc')
                    outfile = os.path.join(out_dir, outfile)
                    futures.append(exe.submit(cls.run, file, outfile, heights,
                                              overwrite))
                    logger.info(
                        f'{i + 1} of {len(infiles)} futures submitted.')
            for i, future in enumerate(as_completed(futures)):
                future.result()
                logger.info(f'{i + 1} of {len(futures)} futures complete.')

    def ws_log_interp(self, lev_array, var_array, levels):
        """Fit ws log law to data and get requested level values from fit.

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

        Returns
        -------
        values : ndarray
            Array of interpolated windspeed values at the requested heights.
        """
        def ws_log_profile(z, a, b):
            return a * np.log(z) + b

        levels = np.array(levels)
        lev_mask = levels <= 100
        var_mask = lev_array <= 100

        self.DATA_COUNT += 1
        try:
            popt, _ = curve_fit(ws_log_profile, lev_array[var_mask],
                                var_array[var_mask])
            out = ws_log_profile(levels[lev_mask], *popt)
            if any(levels > 100) and any(lev_array > 100):
                ws = interp1d(lev_array[~var_mask], var_array[~var_mask],
                              fill_value='extrapolate')(levels[~lev_mask])
                out = np.concatenate([out, ws])
            self.SUCESS_COUNT += 1
        except Exception as e:
            msg = ('Log interp failed with (h, ws) = '
                   f'({lev_array[var_mask]}, {var_array[var_mask]}). {e} '
                   'Using linear interpolation.')
            logger.warning(msg)
            warn(msg)
            out = interp1d(lev_array, var_array,
                           fill_value='extrapolate')(levels)
        return out

    def interp_ws_to_height(self, var_array, lev_array, levels):
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
        for idt in range(array_shape[0]):
            shape = (array_shape[-3], np.product(array_shape[-2:]))
            h_tmp = lev_array[idt].reshape(shape).T
            var_tmp = var_array[idt].reshape(shape).T
            not_nan = ~np.isnan(h_tmp) & ~np.isnan(var_tmp)
            pos_hgt = h_tmp > 0.0
            mask = not_nan & pos_hgt

            # Interp each vertical column of height and var to requested levels
            zip_iter = zip(h_tmp, var_tmp, mask)
            out_array[:, idt, :] = np.array(
                [self.ws_log_interp(h[mask], var[mask], levels)
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
