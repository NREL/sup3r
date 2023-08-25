"""Rescale ERA5 wind components according to log profile"""

import logging
import os
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from glob import glob
from typing import ClassVar
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

    DEFAULT_OUTPUT_HEIGHTS: ClassVar[dict] = {
        'u': [40, 80, 120, 160, 200],
        'v': [40, 80, 120, 160, 200],
        'temperature': [10, 40, 80, 100, 120, 160, 200],
        'pressure': [0, 100, 200],
        'relative_humidity': [80, 100, 120],
    }

    def __init__(
        self,
        infile,
        outfile,
        output_heights=None,
        variables=None,
        max_log_height=100,
    ):
        """Initialize log interpolator.

        Parameters
        ----------
        infile : str
            Path to ERA5 data to use for windspeed log interpolation. Assumed
            to contain zg, orog, and at least u/v at 10m and 100m.
        outfile : str
            Path to save output after log interpolation.
        output_heights : None | dict
            Dictionary of heights to interpolate to for each variables.
            If None this defaults to DEFAULT_OUTPUT_HEIGHTS.
        variables : list
            List of variables to interpolate. If None this defaults to ['u',
            'v']
        max_log_height : int
            Maximum height to use for log interpolation. Above this linear
            interpolation will be used.
        """
        self.infile = infile
        self.outfile = outfile

        msg = ('output_heights must be a dictionary with variables as keys '
               f'and lists of heights as values. Received: {output_heights}.')
        assert output_heights is None or isinstance(output_heights, dict), msg

        self.new_heights = output_heights or self.DEFAULT_OUTPUT_HEIGHTS
        self.max_log_height = max_log_height
        self.variables = ['u', 'v'] if variables is None else variables
        self.data_dict = {}
        self.new_data = {}

        msg = f'{self.infile} does not exist. Skipping.'
        assert os.path.exists(self.infile), msg

        msg = (f'Initializing {self.__class__.__name__} with infile={infile}, '
               f'outfile={outfile}, new_heights={self.new_heights}, '
               f'variables={variables}.')
        logger.info(msg)

    def _load_single_var(self, variable):
        """Load ERA5 data for the given variable.

        Parameters
        ----------
        variable : str
            Name of variable to load. (e.g. u, v, temperature)

        Returns
        -------
        heights : ndarray
            Array of heights for the given variable. Includes heights from
            variables at single levels (e.g. u_10m).
        var_arr : ndarray
            Array of values for the given variable. Includes values from single
            level fields for the given variable. (e.g. u_10m)
        """
        logger.info(f'Loading {self.infile} for {variable}.')
        with xr.open_dataset(self.infile) as res:
            gp = res['zg'].values
            sfc_hgt = np.repeat(res['orog'].values[:, np.newaxis, ...],
                                gp.shape[1],
                                axis=1)
            heights = gp - sfc_hgt

            input_heights = []
            for var in res:
                if f'{variable}_' in var:
                    height = var.split(f'{variable}_')[-1].strip('m')
                    input_heights.append(height)

            var_arr = []
            height_arr = []
            shape = (heights.shape[0], 1, *heights.shape[2:])
            for height in input_heights:
                var_arr.append(res[f'{variable}_{height}m'].values[:,
                                                                   np.newaxis,
                                                                   ...])
                height_arr.append(np.full(shape, height, dtype=np.float32))

            if variable in res:
                var_arr.append(res[f'{variable}'].values)
                height_arr.append(heights)
            var_arr = np.concatenate(var_arr, axis=1)
            heights = np.concatenate(height_arr, axis=1)

            fixed_level_mask = np.full(heights.shape[1], True)
            if variable in ('u', 'v'):
                fixed_level_mask[:] = False
                for i, _ in enumerate(input_heights):
                    fixed_level_mask[i] = True

        return heights, var_arr, fixed_level_mask

    def load(self):
        """Load ERA5 data and create data arrays"""
        self.data_dict = {}
        for var in self.variables:
            self.data_dict[var] = {}
            out = self._load_single_var(var)
            self.data_dict[var]['heights'] = out[0]
            self.data_dict[var]['data'] = out[1]
            self.data_dict[var]['mask'] = out[2]

    def interpolate_vars(self, max_workers=None):
        """Interpolate u/v wind components below 100m using log profile.
        Interpolate non wind data linearly."""
        for var, arrs in self.data_dict.items():
            max_log_height = self.max_log_height
            if var not in ('u', 'v'):
                max_log_height = -np.inf
            logger.info(
                f'Interpolating {var} to heights = {self.new_heights[var]}.')

            self.new_data[var] = self.interp_var_to_height(
                var_array=arrs['data'],
                lev_array=arrs['heights'],
                levels=self.new_heights[var],
                fixed_level_mask=arrs['mask'],
                max_log_height=max_log_height,
                max_workers=max_workers,
            )

    def save_output(self):
        """Save interpolated data to outfile"""
        dirname = os.path.dirname(self.outfile)
        os.makedirs(dirname, exist_ok=True)
        os.system(f'cp {self.infile} {self.outfile}')
        ds = Dataset(self.outfile, 'a')
        logger.info(f'Creating {self.outfile}.')
        for var, data in self.new_data.items():
            for i, height in enumerate(self.new_heights[var]):
                name = f'{var}_{height}m'
                logger.info(f'Adding {name} to {self.outfile}.')
                if name not in ds.variables:
                    _ = ds.createVariable(
                        name,
                        np.float32,
                        dimensions=('time', 'latitude', 'longitude'),
                    )
                ds.variables[name][:] = data[i, ...]
                ds.variables[name].long_name = f'{height} meter {var}'

                units = None
                if 'u_' in var or 'v_' in var:
                    units = 'm s**-1'
                if 'pressure' in var:
                    units = 'Pa'
                if 'temperature' in var:
                    units = 'C'

                if units is not None:
                    ds.variables[name].units = units

        ds.close()
        logger.info(f'Saved interpolated output to {self.outfile}.')

    @classmethod
    def init_dims(cls, old_ds, new_ds, dims):
        """Initialize dimensions in new dataset from old dataset

        Parameters
        ----------
        old_ds : Dataset
            Dataset() object from old file
        new_ds : Dataset
            Dataset() object for new file
        dims : tuple
            Tuple of dimensions. e.g. ('time', 'latitude', 'longitude')

        Returns
        -------
        new_ds : Dataset
            Dataset() object for new file with dimensions initialized.
        """
        for var in dims:
            new_ds.createDimension(var, len(old_ds[var]))
            _ = new_ds.createVariable(var, old_ds[var].dtype, dimensions=var)
            new_ds[var][:] = old_ds[var][:]
            new_ds[var].units = old_ds[var].units
        return new_ds

    @classmethod
    def get_tmp_file(cls, file):
        """Get temp file for given file. Then only needed variables will be
        written to the given file."""
        tmp_file = file.replace('.nc', '_tmp.nc')
        return tmp_file

    @classmethod
    def run(
        cls,
        infile,
        outfile,
        output_heights=None,
        variables=None,
        max_log_height=100,
        overwrite=False,
        max_workers=None,
    ):
        """Run interpolation and save output

        Parameters
        ----------
        infile : str
            Path to ERA5 data to use for windspeed log interpolation. Assumed
            to contain zg, orog, and at least u/v at 10m and 100m.
        outfile : str
            Path to save output after log interpolation.
        output_heights : None | list
            Heights to interpolate to. If None this defaults to [10, 40, 80,
            100, 120, 160, 200].
        variables : list
            List of variables to interpolate. If None this defaults to u and v.
        max_log_height : int
            Maximum height to use for log interpolation. Above this linear
            interpolation will be used.
        max_workers : None | int
            Number of workers to use for interpolating over timesteps.
        overwrite : bool
            Whether to overwrite existing files.
        """
        log_interp = cls(
            infile,
            outfile,
            output_heights=output_heights,
            variables=variables,
            max_log_height=max_log_height,
        )
        if os.path.exists(outfile) and not overwrite:
            logger.info(
                f'{outfile} already exists and overwrite=False. Skipping.')
        else:
            log_interp.load()
            log_interp.interpolate_vars(max_workers=max_workers)
            log_interp.save_output()

    @classmethod
    def run_multiple(
        cls,
        infiles,
        out_dir,
        output_heights=None,
        max_log_height=100,
        overwrite=False,
        variables=None,
        max_workers=None,
    ):
        """Run interpolation and save output

        Parameters
        ----------
        infiles : str | list
            Glob-able path or to ERA5 data or list of files to use for
            windspeed log interpolation. Assumed to contain zg, orog, and at
            least u/v at 10m.
        out_dir : str
            Path to save output directory after log interpolation.
        output_heights : None | list
            Heights to interpolate to. If None this defaults to [40, 80].
        max_log_height : int
            Maximum height to use for log interpolation. Above this linear
            interpolation will be used.
        variables : list
            List of variables to interpolate. If None this defaults to u and v.
        overwrite : bool
            Whether to overwrite existing outfile.
        max_workers : None | int
            Number of workers to use for interpolating over timesteps.
        """
        futures = []
        if isinstance(infiles, str):
            infiles = glob(infiles)
        if max_workers == 1:
            for _, file in enumerate(infiles):
                outfile = os.path.basename(file).replace(
                    '.nc', '_all_interp.nc')
                outfile = os.path.join(out_dir, outfile)
                cls.run(
                    file,
                    outfile,
                    output_heights=output_heights,
                    max_log_height=max_log_height,
                    overwrite=overwrite,
                    variables=variables,
                )

        else:
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                for i, file in enumerate(infiles):
                    outfile = os.path.basename(file).replace(
                        '.nc', '_all_interp.nc')
                    outfile = os.path.join(out_dir, outfile)
                    futures.append(
                        exe.submit(cls.run,
                                   file,
                                   outfile,
                                   output_heights=output_heights,
                                   variables=variables,
                                   max_log_height=max_log_height,
                                   overwrite=overwrite))
                    logger.info(
                        f'{i + 1} of {len(infiles)} futures submitted.')
            for i, future in enumerate(as_completed(futures)):
                future.result()
                logger.info(f'{i + 1} of {len(futures)} futures complete.')

    @classmethod
    def pbl_interp_to_height(cls,
                             lev_array,
                             var_array,
                             levels,
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
        good : bool
            Check if log interpolation went without issue.
        """

        def ws_log_profile(z, a, b):
            return a * np.log(z) + b

        lev_array_samp = lev_array.copy()
        var_array_samp = var_array.copy()
        if fixed_level_mask is not None:
            lev_array_samp = lev_array_samp[fixed_level_mask]
            var_array_samp = var_array_samp[fixed_level_mask]

        good = True
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
            good = False
            logger.warning(msg)
            warn(msg)
            log_ws = interp1d(
                lev_array[var_mask],
                var_array[var_mask],
                fill_value='extrapolate',
            )(levels[lev_mask])
        return log_ws, good

    @classmethod
    def _interp_var_to_height(cls,
                              lev_array,
                              var_array,
                              levels,
                              fixed_level_mask=None,
                              max_log_height=100):
        """Fit ws log law to wind data below max_log_height and linearly
        interpolate data above. Linearly interpolate non wind data.

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
            Array of interpolated data values at the requested heights.
        good : bool
            Check if interpolation went without issue.
        """
        levels = np.array(levels)

        log_ws = None
        lin_ws = None
        good = True

        hgt_check = any(levels < max_log_height) and any(
            lev_array < max_log_height)
        if hgt_check:
            log_ws, good = cls.pbl_interp_to_height(
                lev_array,
                var_array,
                levels,
                fixed_level_mask=fixed_level_mask,
                max_log_height=max_log_height)

        if any(levels > max_log_height):
            lev_mask = levels >= max_log_height
            var_mask = lev_array >= max_log_height
            if len(lev_array[var_mask]) > 1:
                lin_ws = interp1d(lev_array[var_mask],
                                  var_array[var_mask],
                                  fill_value='extrapolate')(levels[lev_mask])
            elif len(lev_array) > 1:
                msg = ('Requested interpolation levels are outside the '
                       f'available range: lev_array={lev_array}, '
                       f'levels={levels}. Using linear extrapolation.')
                lin_ws = interp1d(lev_array,
                                  var_array,
                                  fill_value='extrapolate')(levels[lev_mask])
                good = False
                logger.warning(msg)
                warn(msg)
            else:
                msg = ('Data seems to be all NaNs. Something may have gone '
                       'wrong during download.')
                raise OSError(msg)

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

        return out, good

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
    def interp_single_ts(cls,
                         hgt_t,
                         var_t,
                         mask,
                         levels,
                         fixed_level_mask=None,
                         max_log_height=100):
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
        out_array = []
        checks = []
        for h, var, mask in zip_iter:
            val, check = cls._interp_var_to_height(
                h[mask],
                var[mask],
                levels,
                fixed_level_mask=fixed_level_mask[mask],
                max_log_height=max_log_height,
            )
            out_array.append(val)
            checks.append(check)
        return np.array(out_array), np.array(checks)

    @classmethod
    def interp_var_to_height(cls,
                             var_array,
                             lev_array,
                             levels,
                             fixed_level_mask=None,
                             max_log_height=100,
                             max_workers=None):
        """Interpolate data array to given level(s) based on h_array.
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
        lev_array, levels = Interpolator.prep_level_interp(
            var_array, lev_array, levels)

        array_shape = var_array.shape

        # Flatten h_array and var_array along lat, long axis
        shape = (len(levels), array_shape[-4], np.product(array_shape[-2:]))
        out_array = np.zeros(shape, dtype=np.float32).T
        total_checks = []

        # iterate through time indices
        futures = {}
        if max_workers == 1:
            for idt in range(array_shape[0]):
                h_t, v_t, mask = cls._get_timestep_interp_input(
                    lev_array, var_array, idt)
                out, checks = cls.interp_single_ts(
                    h_t,
                    v_t,
                    mask,
                    levels=levels,
                    fixed_level_mask=fixed_level_mask,
                    max_log_height=max_log_height,
                )
                out_array[:, idt, :] = out
                total_checks.append(checks)

                logger.info(
                    f'{idt + 1} of {array_shape[0]} timesteps finished.')

        else:
            with ProcessPoolExecutor(max_workers=max_workers) as exe:
                for idt in range(array_shape[0]):
                    h_t, v_t, mask = cls._get_timestep_interp_input(
                        lev_array, var_array, idt)
                    future = exe.submit(
                        cls.interp_single_ts,
                        h_t,
                        v_t,
                        mask,
                        levels=levels,
                        fixed_level_mask=fixed_level_mask,
                        max_log_height=max_log_height,
                    )
                    futures[future] = idt
                    logger.info(
                        f'{idt + 1} of {array_shape[0]} futures submitted.')
            for i, future in enumerate(as_completed(futures)):
                out, checks = future.result()
                out_array[:, futures[future], :] = out
                total_checks.append(checks)
                logger.info(f'{i + 1} of {len(futures)} futures complete.')

        total_checks = np.concatenate(total_checks)
        good_count = total_checks.sum()
        total_count = len(total_checks)
        logger.info('Percent of points interpolated without issue: '
                    f'{100 * good_count / total_count:.2f}')

        # Reshape out_array
        if isinstance(levels, (float, np.float32, int)):
            shape = (1, array_shape[-4], array_shape[-2], array_shape[-1])
            out_array = out_array.T.reshape(shape)
        else:
            shape = (len(levels), array_shape[-4], array_shape[-2],
                     array_shape[-1])
            out_array = out_array.T.reshape(shape)

        return out_array
