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
from sup3r.utilities.utilities import invert_uv, transform_rotate_wind

init_logger(__name__, log_level='DEBUG')
init_logger('sup3r', log_level='DEBUG')


logger = logging.getLogger(__name__)


def ws_log_interp(lev_array, var_array, levels):
    """Fit ws log law to data and get requested level values from fit.

    Parameters
    ----------
    lev_array : ndarray
        1D Array of height values corresponding to the wrf source
        data in the same shape as var_array.
    var_array : ndarray
        1D Array of variable data, for example u-wind in a 1D array of shape
    levels : float | list
        level or levels to interpolate to (e.g. final desired hub heights
        above surface elevation)

    Returns
    -------
    values : ndarray
        Array of interpolated windspeed values at the requested heights.
    """
    numer = var_array[-1] * np.log(lev_array[0])
    numer -= (var_array[0] * np.log(lev_array[-1]))
    denom = var_array[-1] - var_array[0]
    z0 = np.exp(numer / denom)

    def ws_log_profile(z, d, uf, psi):
        return uf / 0.41 * (np.log((z - d) / z0) + psi)

    levels = np.array(levels)
    lev_mask = levels <= 100
    var_mask = lev_array <= 100
    try:
        popt, _ = curve_fit(ws_log_profile, lev_array[var_mask],
                            var_array[var_mask])
        out = ws_log_profile(levels[lev_mask], *popt)
        if any(levels > 100) and any(lev_array > 100):
            ws = interp1d(lev_array[~var_mask], var_array[~var_mask],
                          fill_value='extrapolate')(levels[~lev_mask])
            out = np.concatenate([out, ws])
    except Exception:
        msg = (f'Log interp failed with (h, ws) = ({lev_array}, {var_array}). '
               'Using linear interpolation.')
        logger.warning(msg)
        warn(msg)
        out = interp1d(lev_array, var_array, fill_value='extrapolate')(levels)
    return out



class LogScaler:
    """Open ERA5 file, rescale wind components between 0 - 100 meters, and save
    to file"""

    def __init__(self, infile, outfile, heights=None):
        """Initialize log scaler.

        Parameters
        ----------
        infile : str
            Path to ERA5 data to use for windspeed log interpolation. Assumed
            to contain u/v at 10m, 100m, and at least one height between.
        outfile : str
            Path to save output after log interpolation.
        heights : None | list
            Heights to interpolate to. If None this defaults to [40, 80].
        """
        self.infile = infile
        self.outfile = outfile
        self.new_heights = heights or [40, 80]

    def load(self):
        """Load ERA5 data and create wind component arrays"""
        with xr.open_dataset(self.infile) as res:
            self.heights = res['zg'].values - res['orog'].values
            self.u_10m = res['u_10m'].values
            self.v_10m = res['v_10m'].values
            self.u_100m = res['u_100m'].values
            self.v_100m = res['v_100m'].values
            self.u = res['u'].values
            self.v = res['v'].values
            shape = (self.heights.shape[0], 1, *self.heights.shape[2:])
            self.heights = np.concatenate([np.full(shape, 10),
                                           np.full(shape, 100),
                                           self.heights], axis=1)
            self.u = np.concatenate([self.u_10m[:, np.newaxis, ...],
                                     self.u_100m[:, np.newaxis, ...],
                                     self.u], axis=1)
            self.v = np.concatenate([self.v_10m[:, np.newaxis, ...],
                                     self.v_100m[:, np.newaxis, ...],
                                     self.v], axis=1)
            self.u = np.transpose(self.u, axes=(2, 3, 0, 1))
            self.v = np.transpose(self.v, axes=(2, 3, 0, 1))
            lons, lats = np.meshgrid(res['longitude'].values,
                                     res['latitude'].values)
            self.lat_lon = np.stack([lats, lons], axis=2)
            self.ws = []
            self.wd = []
            for i in range(self.heights.shape[1]):
                ws, wd = invert_uv(self.u[..., i],
                                   self.v[..., i], self.lat_lon)
                self.ws.append(ws)
                self.wd.append(wd)
            self.ws = np.stack(self.ws)
            self.wd = np.stack(self.wd)
            self.ws = np.transpose(self.ws, axes=(3, 0, 1, 2))
            self.wd = np.transpose(self.wd, axes=(3, 0, 1, 2))

    def interpolate_wind(self):
        """Interpolate windspeed using log profile and winddirection
        linearly"""
        self.ws = Interpolator.interp_ws_to_height(self.ws,
                                                   self.heights,
                                                   self.new_heights)
        self.wd = Interpolator.interp_to_level(self.wd, self.heights,
                                               self.new_heights)
        self.ws = np.transpose(self.ws, axes=(0, 2, 3, 1))
        self.wd = np.transpose(self.wd, axes=(0, 2, 3, 1))

    def convert_wind(self):
        """Convert ws and wd back to u and v components"""
        self.u = []
        self.v = []
        for i in range(len(self.new_heights)):
            u, v = transform_rotate_wind(self.ws[i], self.wd[i], self.lat_lon)
            self.u.append(u)
            self.v.append(v)
        self.u = np.stack(self.u)
        self.v = np.stack(self.v)
        self.u = np.transpose(self.u, axes=(0, 3, 1, 2))
        self.v = np.transpose(self.v, axes=(0, 3, 1, 2))

    def save_output(self):
        """Save interpolated wind components to outfile"""
        os.system(f'cp {self.infile} {self.outfile}')
        ds = Dataset(self.outfile, 'a')
        for i, height in enumerate(self.new_heights):
            variable = ds.variables['u_10m']
            name = f'u_{height}m'
            if name not in ds.variables:
                _ = ds.createVariable(name,
                                      np.float32,
                                      dimensions=variable.dimensions)
                ds.variables[name][:] = self.u[i, ...]
                ds.variables[name].units = 'm s**-1'
                ds.variables[name].long_name = f'{height} meter U Component'
            variable = ds.variables['v_10m']
            name = f'v_{height}m'
            if name not in ds.variables:
                ds.createVariable(name,
                                  np.float32,
                                  dimensions=variable.dimensions)
                ds.variables[name][:] = self.v[i, ...]
                ds.variables[name].units = 'm s**-1'
                ds.variables[name].long_name = f'{height} meter V Component'
        ds.close()

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
        log_scaler = cls(infile, outfile, heights)
        if not os.path.exists(outfile) or overwrite:
            log_scaler.load()
            log_scaler.interpolate_wind()
            log_scaler.convert_wind()
            log_scaler.save_output()

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
        os.makedirs(out_dir, exist_ok=True)
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            for i, file in enumerate(infiles):
                outfile = os.path.basename(file).replace('.nc',
                                                         '_log_interp.nc')
                outfile = os.path.join(out_dir, outfile)
                futures.append(exe.submit(cls.run, file, outfile, heights,
                                          overwrite))
                logger.info(f'{i + 1} of {len(infiles)} futures submitted.')
        for i, future in enumerate(as_completed(futures)):
            future.result()
            logger.info(f'{i + 1} of {len(futures)} futures complete.')
