"""Download ERA5 file for the given year and month

NOTE: To use this you need to have cdsapi package installed and a ~/.cdsapirc
file with a url and api key.  Follow the instructions here:
https://cds.climate.copernicus.eu/api-how-to
"""

import logging
import os
from calendar import monthrange
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed,
                                )
from glob import glob
from typing import ClassVar
from warnings import warn

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset

from sup3r.utilities.interpolate_log_profile import LogLinInterpolator

logger = logging.getLogger(__name__)

try:
    import cdsapi

    CDS_API_CLIENT = cdsapi.Client()
except ImportError as e:
    msg = f'Could not import cdsapi package. {e}'
    logger.error(msg)


class EraDownloader:
    """Class to handle ERA5 downloading, variable renaming, file combination,
    and interpolation."""

    msg = ('To download ERA5 data you need to have a ~/.cdsapirc file '
           'with a valid url and api key. Follow the instructions here: '
           'https://cds.climate.copernicus.eu/api-how-to')
    req_file = os.path.join(os.path.expanduser('~'), '.cdsapirc')
    assert os.path.exists(req_file), msg

    VALID_VARIABLES: ClassVar[list] = [
        'u', 'v', 'pressure', 'temperature', 'relative_humidity',
        'specific_humidity', 'total_precipitation',
    ]

    KEEP_VARIABLES: ClassVar[list] = ['orog']
    KEEP_VARIABLES += [f'{v}_' for v in VALID_VARIABLES]

    DEFAULT_RENAMED_VARS: ClassVar[list] = [
        'zg', 'orog', 'u', 'v', 'u_10m', 'v_10m', 'u_100m', 'v_100m',
        'temperature', 'pressure',
    ]
    DEFAULT_DOWNLOAD_VARS: ClassVar[list] = [
        '10m_u_component_of_wind', '10m_v_component_of_wind',
        '100m_u_component_of_wind', '100m_v_component_of_wind',
        'u_component_of_wind', 'v_component_of_wind', '2m_temperature',
        'temperature', 'surface_pressure', 'relative_humidity',
        'total_precipitation',
    ]

    SFC_VARS: ClassVar[list] = [
        '10m_u_component_of_wind', '10m_v_component_of_wind',
        '100m_u_component_of_wind', '100m_v_component_of_wind',
        'surface_pressure', '2m_temperature', 'geopotential',
        'total_precipitation',
    ]
    LEVEL_VARS: ClassVar[list] = [
        'u_component_of_wind', 'v_component_of_wind', 'geopotential',
        'temperature', 'relative_humidity', 'specific_humidity',
    ]
    NAME_MAP: ClassVar[dict] = {
        'u10': 'u_10m',
        'v10': 'v_10m',
        'u100': 'u_100m',
        'v100': 'v_100m',
        't': 'temperature',
        't2m': 'temperature_2m',
        'u': 'u',
        'v': 'v',
        'sp': 'pressure_0m',
        'r': 'relative_humidity',
        'q': 'specific_humidity',
        'tp': 'total_precipitation',
    }

    def __init__(self,
                 year,
                 month,
                 area,
                 levels,
                 combined_out_pattern,
                 interp_out_pattern=None,
                 run_interp=True,
                 overwrite=False,
                 required_shape=None,
                 variables=None):
        """Initialize the class.

        Parameters
        ----------
        year : int
            Year of data to download.
        month : int
            Month of data to download.
        area : list
            Domain area of the data to download.
            [max_lat, min_lon, min_lat, max_lon]
        levels : list
            List of pressure levels to download.
        combined_out_pattern : str
            Pattern for combined monthly output file. Must include year and
            month format keys.  e.g. 'era5_{year}_{month}_combined.nc'
        interp_out_pattern : str | None
            Pattern for interpolated monthly output file. Must include year and
            month format keys.  e.g. 'era5_{year}_{month}_interp.nc'
        run_interp : bool
            Whether to run interpolation after downloading and combining files.
        overwrite : bool
            Whether to overwrite existing files.
        required_shape : tuple | None
            Required shape of data to download. Used to check downloaded data.
            Should be (n_levels, n_lats, n_lons). If None, no check is
            performed.
        variables : list | None
            Variables to download. If None this defaults to just gepotential
            and wind components.
        """
        self.year = year
        self.month = month
        self.area = area
        self.levels = levels
        self.run_interp = run_interp
        self.overwrite = overwrite
        self.combined_out_pattern = combined_out_pattern
        self.interp_out_pattern = interp_out_pattern
        self._interp_file = None
        self._combined_file = None
        self._variables = variables
        self.hours = [str(n).zfill(2) + ":00" for n in range(0, 24)]
        self.sfc_file_variables = ['geopotential']
        self.level_file_variables = ['geopotential']

        self.shape_check(required_shape, levels)
        self.check_good_vars(self.variables)
        self.prep_var_lists(self.variables)

        msg = ('Initialized EraDownloader with: '
               f'year={self.year}, month={self.month}, area={self.area}, '
               f'levels={self.levels}, variables={self.variables}')
        logger.info(msg)

    @property
    def variables(self):
        """Get list of requested variables"""
        if self._variables is None:
            self._variables = self.VALID_VARIABLES
        return self._variables

    @property
    def days(self):
        """Get list of days for the requested month"""
        return [
            str(n).zfill(2)
            for n in np.arange(1,
                               monthrange(self.year, self.month)[1] + 1)
        ]

    @property
    def interp_file(self):
        """Get name of file with interpolated variables"""
        if self._interp_file is None:
            if self.interp_out_pattern is not None and self.run_interp:
                self._interp_file = self.interp_out_pattern.format(
                    year=self.year, month=str(self.month).zfill(2))
                os.makedirs(os.path.dirname(self._interp_file), exist_ok=True)
        return self._interp_file

    @property
    def combined_file(self):
        """Get name of file from combined surface and level files"""
        if self._combined_file is None:
            self._combined_file = self.combined_out_pattern.format(
                year=self.year, month=str(self.month).zfill(2))
            os.makedirs(os.path.dirname(self._combined_file), exist_ok=True)
        return self._combined_file

    @property
    def surface_file(self):
        """Get name of file with variables from single level download"""
        basedir = os.path.dirname(self.combined_file)
        basename = f'sfc_{"_".join(self.variables)}_{self.year}_'
        basename += f'{str(self.month).zfill(2)}.nc'
        return os.path.join(basedir, basename)

    @property
    def level_file(self):
        """Get name of file with variables from pressure level download"""
        basedir = os.path.dirname(self.combined_file)
        basename = f'levels_{"_".join(self.variables)}_{self.year}_'
        basename += f'{str(self.month).zfill(2)}.nc'
        return os.path.join(basedir, basename)

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
        tmp_file = file.replace(".nc", "_tmp.nc")
        return tmp_file

    def shape_check(self, required_shape, levels):
        """Check given required shape"""
        if required_shape is None or len(required_shape) == 3:
            self.required_shape = required_shape
        elif len(required_shape) == 2 and len(levels) != required_shape[0]:
            self.required_shape = (len(levels), *required_shape)
        else:
            msg = f'Received weird required_shape: {required_shape}.'
            logger.error(msg)
            raise OSError(msg)

    def check_good_vars(self, variables):
        """Make sure requested variables are valid.

        Parameters
        ----------
        variables : list
            List of variables to download. Can be any of VALID_VARIABLES
        """
        good = all(var in self.VALID_VARIABLES for var in variables)
        if not good:
            msg = (f'Received variables {variables} not in valid variables '
                   f'list {self.VALID_VARIABLES}')
            logger.error(msg)
            raise OSError(msg)

    def _prep_var_lists(self, variables):
        """Add all downloadable variables for the generic requested variables.
        e.g. if variable = 'u' add all downloadable u variables to list."""
        d_vars = []
        vars = variables.copy()
        for i, v in enumerate(vars):
            if v in ('u', 'v'):
                vars[i] = f'{v}_'
        for var in vars:
            for d_var in self.SFC_VARS + self.LEVEL_VARS:
                if var in d_var:
                    d_vars.append(d_var)
        return d_vars

    def prep_var_lists(self, variables):
        """Create surface and level variable lists based on requested
        variables."""
        variables = self._prep_var_lists(variables)
        for var in variables:
            if var in self.SFC_VARS and var not in self.sfc_file_variables:
                self.sfc_file_variables.append(var)
            elif (var in self.LEVEL_VARS
                  and var not in self.level_file_variables):
                self.level_file_variables.append(var)
            else:
                msg = f'Requested {var} is not available for download.'
                logger.warning(msg)
                warn(msg)

    def download_process_combine(self):
        """Run the download routine."""
        sfc_check = len(self.sfc_file_variables) > 0
        level_check = (len(self.level_file_variables) > 0
                       and self.levels is not None)
        if self.level_file_variables:
            msg = (f'{self.level_file_variables} requested but no levels'
                   ' were provided.')
            if self.levels is None:
                logger.warning(msg)
                warn(msg)
        if sfc_check:
            self.download_surface_file()
        if level_check:
            self.download_levels_file()
        if sfc_check or level_check:
            self.process_and_combine()

    def download_levels_file(self):
        """Download file with requested pressure levels"""
        if not os.path.exists(self.level_file) or self.overwrite:
            msg = (f'Downloading {self.level_file_variables} to '
                   f'{self.level_file}.')
            logger.info(msg)
            CDS_API_CLIENT.retrieve(
                'reanalysis-era5-pressure-levels', {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': self.level_file_variables,
                    'pressure_level': self.levels,
                    'year': self.year,
                    'month': self.month,
                    'day': self.days,
                    'time': self.hours,
                    'area': self.area,
                }, self.level_file)
        else:
            logger.info(f'File already exists: {self.level_file}.')

    def download_surface_file(self):
        """Download surface file"""
        if not os.path.exists(self.surface_file) or self.overwrite:
            msg = (f'Downloading {self.sfc_file_variables} to '
                   f'{self.surface_file}.')
            logger.info(msg)
            CDS_API_CLIENT.retrieve(
                'reanalysis-era5-single-levels', {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': self.sfc_file_variables,
                    'year': self.year,
                    'month': self.month,
                    'day': self.days,
                    'time': self.hours,
                    'area': self.area,
                }, self.surface_file)
        else:
            logger.info(f'File already exists: {self.surface_file}.')

    def process_surface_file(self):
        """Rename variables and convert geopotential to geopotential height."""

        dims = ('time', 'latitude', 'longitude')
        tmp_file = self.get_tmp_file(self.surface_file)
        with Dataset(self.surface_file, "r") as old_ds:
            with Dataset(tmp_file, "w") as ds:
                ds = self.init_dims(old_ds, ds, dims)

                ds = self.convert_z('orog', 'Orography', old_ds, ds)

                ds = self.map_vars(old_ds, ds)
        os.system(f'mv {tmp_file} {self.surface_file}')
        logger.info(f'Finished processing {self.surface_file}. Moved '
                    f'{tmp_file} to {self.surface_file}.')

    def map_vars(self, old_ds, ds):
        """Map variables from old dataset to new dataset

        Parameters
        ----------
        old_ds : Dataset
            Dataset() object from old file
        ds : Dataset
            Dataset() object for new file

        Returns
        -------
        ds : Dataset
            Dataset() object for new file with new variables written.
        """
        for old_name, new_name in self.NAME_MAP.items():
            if old_name in old_ds.variables:
                _ = ds.createVariable(new_name,
                                      np.float32,
                                      dimensions=old_ds[old_name].dimensions,
                                      )
                vals = old_ds.variables[old_name][:]
                if 'temperature' in new_name:
                    vals -= 273.15
                ds.variables[new_name][:] = vals
        return ds

    def convert_z(self, standard_name, long_name, old_ds, ds):
        """Convert z to given height variable

        Parameters
        ----------
        standard_name : str
            New variable name. e.g. 'zg' or 'orog'
        long_name : str
            Long name for new variable. e.g. 'Geopotential Height' or
            'Orography'
        old_ds : Dataset
            Dataset() object from tmp file
        ds : Dataset
            Dataset() object for new file

        Returns
        -------
        ds : Dataset
            Dataset() object for new file with new height variable written.
        """

        _ = ds.createVariable(standard_name,
                              np.float32,
                              dimensions=old_ds['z'].dimensions)
        ds.variables[standard_name][:] = old_ds['z'][:] / 9.81
        ds.variables[standard_name].long_name = long_name
        ds.variables[standard_name].standard_name = 'zg'
        ds.variables[standard_name].units = 'm'
        return ds

    def process_level_file(self):
        """Convert geopotential to geopotential height."""

        dims = ('time', 'level', 'latitude', 'longitude')
        tmp_file = self.get_tmp_file(self.level_file)
        with Dataset(self.level_file, "r") as old_ds:
            with Dataset(tmp_file, "w") as ds:
                ds = self.init_dims(old_ds, ds, dims)

                ds = self.convert_z('zg', 'Geopotential Height', old_ds, ds)

                ds = self.map_vars(old_ds, ds)

                if 'pressure' in self.variables:
                    tmp = np.zeros(ds.variables['zg'].shape)
                    for i in range(tmp.shape[1]):
                        tmp[:, i, :, :] = ds.variables['level'][i] * 100
                    _ = ds.createVariable('pressure',
                                          np.float32,
                                          dimensions=dims)
                    ds.variables['pressure'][:] = tmp[...]
                    ds.variables['pressure'].long_name = 'Pressure'
                    ds.variables['pressure'].units = 'Pa'

        os.system(f'mv {tmp_file} {self.level_file}')
        logger.info(f'Finished processing {self.level_file}. Moved '
                    f'{tmp_file} to {self.level_file}.')

    def process_and_combine(self):
        """Process variables and combine."""

        if not os.path.exists(self.combined_file) or self.overwrite:
            files = []
            if os.path.exists(self.level_file):
                logger.info(f'Processing {self.level_file}.')
                self.process_level_file()
                files.append(self.level_file)
            if os.path.exists(self.surface_file):
                logger.info(f'Processing {self.surface_file}.')
                self.process_surface_file()
                files.append(self.surface_file)

            logger.info(f'Combining {files} and {self.surface_file} '
                        f'to {self.combined_file}.')
            with xr.open_mfdataset(files) as ds:
                ds.to_netcdf(self.combined_file)
            logger.info(f'Finished writing {self.combined_file}')

            if os.path.exists(self.level_file):
                os.remove(self.level_file)
            if os.path.exists(self.surface_file):
                os.remove(self.surface_file)

    def good_file(self, file, required_shape):
        """Check if file has the required shape and variables.

        Parameters
        ----------
        file : str
            Name of file to check for required variables and shape
        required_shape : tuple
            Required shape for data. Should be (n_levels, n_lats, n_lons).

        Returns
        -------
        bool
            Whether or not data has required shape and variables.
        """
        out = self.check_single_file(file,
                                     var_list=self.variables,
                                     check_nans=False,
                                     check_heights=False,
                                     required_shape=required_shape)
        good_vars, good_shape, _, _ = out
        check = good_vars and good_shape
        return check

    def check_existing_files(self):
        """If files exist already check them for good shape and required
        variables. Remove them if there was a problem so we can continue with
        routine from scratch."""
        if os.path.exists(self.combined_file):
            try:
                check = self.good_file(self.combined_file, self.required_shape)
                if not check:
                    msg = f'Bad file: {self.combined_file}'
                    logger.error(msg)
                    raise OSError(msg)
                else:
                    if os.path.exists(self.level_file):
                        os.remove(self.level_file)
                    if os.path.exists(self.surface_file):
                        os.remove(self.surface_file)
                    logger.info(f'{self.combined_file} already exists and '
                                f'overwrite={self.overwrite}. Skipping.')
            except Exception as e:
                logger.info(f'Something wrong with {self.combined_file}. {e}')
                if os.path.exists(self.combined_file):
                    os.remove(self.combined_file)
                check = self.interp_file is not None and os.path.exists(
                    self.interp_file)
                if check:
                    os.remove(self.interp_file)

    def run_interpolation(self, max_workers=None, **kwargs):
        """Run interpolation to get final final. Runs log interpolation up to
        max_log_height (usually 100m) and linear interpolation above this."""
        LogLinInterpolator.run(infile=self.combined_file,
                               outfile=self.interp_file,
                               max_workers=max_workers,
                               variables=self.variables,
                               overwrite=self.overwrite,
                               **kwargs)

    def get_monthly_file(self, interp_workers=None, **interp_kwargs):
        """Download level and surface files, process variables, and combine
        processed files. Includes checks for shape and variables and option to
        interpolate."""

        if os.path.exists(self.combined_file) and self.overwrite:
            os.remove(self.combined_file)

        self.check_existing_files()

        if not os.path.exists(self.combined_file):
            self.download_process_combine()

        if self.run_interp:
            self.run_interpolation(max_workers=interp_workers, **interp_kwargs)

        if self.interp_file is not None and os.path.exists(self.interp_file):
            if self.already_pruned(self.interp_file):
                logger.info(f'{self.interp_file} pruned already.')
            else:
                self.prune_output(self.interp_file)

    @classmethod
    def all_months_exist(cls, year, file_pattern):
        """Check if all months in the requested year exist.

        Parameters
        ----------
        year : int
            Year of data to download.
        file_pattern : str
            Pattern for monthly output file. Must include year and month format
            keys. e.g. 'era5_{year}_{month}_combined.nc'

        Returns
        -------
        bool
            True if all months in the requested year exist.
        """
        return all(
            os.path.exists(
                file_pattern.format(year=year, month=str(month).zfill(2)))
            for month in range(1, 13))

    @classmethod
    def already_pruned(cls, infile):
        """Check if file has been pruned already."""

        pruned = True
        with Dataset(infile, 'r') as ds:
            for var in ds.variables:
                if not any(name in var for name in cls.KEEP_VARIABLES):
                    logger.info(f'Pruning {var} in {infile}.')
                    pruned = False
        return pruned

    @classmethod
    def prune_output(cls, infile):
        """Prune output file to keep just single level variables"""

        logger.info(f'Pruning {infile}.')
        tmp_file = cls.get_tmp_file(infile)
        with Dataset(infile, 'r') as old_ds:
            with Dataset(tmp_file, 'w') as new_ds:
                new_ds = cls.init_dims(old_ds, new_ds,
                                       ('time', 'latitude', 'longitude'))
                for var in old_ds.variables:
                    if any(name in var for name in cls.KEEP_VARIABLES):
                        old_var = old_ds[var]
                        vals = old_var[:]
                        _ = new_ds.createVariable(
                            var, old_var.dtype, dimensions=old_var.dimensions)
                        new_ds[var][:] = vals
                        if hasattr(old_var, 'units'):
                            new_ds[var].units = old_var.units
                        if hasattr(old_var, 'standard_name'):
                            standard_name = old_var.standard_name
                            new_ds[var].standard_name = standard_name
                        if hasattr(old_var, 'long_name'):
                            new_ds[var].long_name = old_var.long_name
        os.system(f'mv {tmp_file} {infile}')
        logger.info(f'Finished pruning variables in {infile}. Moved '
                    f'{tmp_file} to {infile}.')

    @classmethod
    def run_month(cls,
                  year,
                  month,
                  area,
                  levels,
                  combined_out_pattern,
                  interp_out_pattern=None,
                  run_interp=True,
                  overwrite=False,
                  required_shape=None,
                  interp_workers=None,
                  variables=None,
                  **interp_kwargs):
        """Run routine for all months in the requested year.

        Parameters
        ----------
        year : int
            Year of data to download.
        month : int
            Month of data to download.
        area : list
            Domain area of the data to download.
            [max_lat, min_lon, min_lat, max_lon]
        levels : list
            List of pressure levels to download.
        combined_out_pattern : str
            Pattern for combined monthly output file. Must include year and
            month format keys.  e.g. 'era5_{year}_{month}_combined.nc'
        interp_out_pattern : str | None
            Pattern for interpolated monthly output file. Must include year and
            month format keys.  e.g. 'era5_{year}_{month}_interp.nc'
        run_interp : bool
            Whether to run interpolation after downloading and combining files.
        overwrite : bool
            Whether to overwrite existing files.
        required_shape : tuple | None
            Required shape of data to download. Used to check downloaded data.
            Should be (n_levels, n_lats, n_lons).  If None, no check is
            performed.
        interp_workers : int | None
            Max number of workers to use for interpolation.
        variables : list | None
            Variables to download. If None this defaults to just gepotential
            and wind components.
        **interp_kwargs : dict
            Keyword args for LogLinInterpolator.run()
        """
        downloader = cls(year=year,
                         month=month,
                         area=area,
                         levels=levels,
                         combined_out_pattern=combined_out_pattern,
                         interp_out_pattern=interp_out_pattern,
                         run_interp=run_interp,
                         overwrite=overwrite,
                         required_shape=required_shape,
                         variables=variables)
        downloader.get_monthly_file(interp_workers=interp_workers,
                                    **interp_kwargs)

    @classmethod
    def run_year(cls,
                 year,
                 area,
                 levels,
                 combined_out_pattern,
                 combined_yearly_file,
                 interp_out_pattern=None,
                 interp_yearly_file=None,
                 run_interp=True,
                 overwrite=False,
                 required_shape=None,
                 max_workers=None,
                 interp_workers=None,
                 variables=None,
                 **interp_kwargs):
        """Run routine for all months in the requested year.

        Parameters
        ----------
        year : int
            Year of data to download.
        area : list
            Domain area of the data to download.
            [max_lat, min_lon, min_lat, max_lon]
        levels : list
            List of pressure levels to download.
        combined_out_pattern : str
            Pattern for combined monthly output file. Must include year and
            month format keys.  e.g. 'era5_{year}_{month}_combined.nc'
        combined_yearly_file : str
            Name of yearly file made from monthly combined files.
        interp_out_pattern : str | None
            Pattern for interpolated monthly output file. Must include year and
            month format keys.  e.g. 'era5_{year}_{month}_interp.nc'
        interp_yearly_file : str
            Name of yearly file made from monthly interp files.
        run_interp : bool
            Whether to run interpolation after downloading and combining files.
        overwrite : bool
            Whether to overwrite existing files.
        required_shape : tuple | None
            Required shape of data to download. Used to check downloaded data.
            Should be (n_levels, n_lats, n_lons).  If None, no check is
            performed.
        max_workers : int
            Max number of workers to use for downloading and processing monthly
            files.
        interp_workers : int | None
            Max number of workers to use for interpolation.
        variables : list | None
            Variables to download. If None this defaults to just gepotential
            and wind components.
        **interp_kwargs : dict
            Keyword args for LogLinInterpolator.run()
        """
        if max_workers == 1:
            for month in range(1, 13):
                cls.run_month(year=year,
                              month=month,
                              area=area,
                              levels=levels,
                              combined_out_pattern=combined_out_pattern,
                              interp_out_pattern=interp_out_pattern,
                              run_interp=run_interp,
                              overwrite=overwrite,
                              required_shape=required_shape,
                              interp_workers=interp_workers,
                              variables=variables,
                              **interp_kwargs)
        else:
            futures = {}
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                for month in range(1, 13):
                    future = exe.submit(
                        cls.run_month,
                        year=year,
                        month=month,
                        area=area,
                        levels=levels,
                        combined_out_pattern=combined_out_pattern,
                        interp_out_pattern=interp_out_pattern,
                        run_interp=run_interp,
                        overwrite=overwrite,
                        required_shape=required_shape,
                        interp_workers=interp_workers,
                        variables=variables,
                        **interp_kwargs)
                    futures[future] = {'year': year, 'month': month}
                    logger.info(f'Submitted future for year {year} and month '
                                f'{month}.')
            for future in as_completed(futures):
                future.result()
                v = futures[future]
                logger.info(f'Finished future for year {v["year"]} and month '
                            f'{v["month"]}.')

        cls.make_yearly_file(year, combined_out_pattern, combined_yearly_file)

        if run_interp:
            cls.make_yearly_file(year, interp_out_pattern, interp_yearly_file)

    @classmethod
    def make_yearly_file(cls, year, file_pattern, yearly_file):
        """Combine monthly files into a single file.

        Parameters
        ----------
        year : int
            Year of monthly data to make into a yearly file.
        file_pattern : str
            File pattern for monthly files. Must have year and month format
            keys. e.g. './era_uv_{year}_{month}_combined.nc'
        yearly_file : str
            Name of yearly file made from monthly files.
        """
        msg = (f'Not all monthly files with file_patten {file_pattern} for '
               f'year {year} exist.')
        assert cls.all_months_exist(year, file_pattern), msg

        files = [
            file_pattern.format(year=year, month=str(month).zfill(2))
            for month in range(1, 13)
        ]

        if not os.path.exists(yearly_file):
            with xr.open_mfdataset(files) as res:
                logger.info(f'Combining {files}')
                os.makedirs(os.path.dirname(yearly_file), exist_ok=True)
                res.to_netcdf(yearly_file)
                logger.info(f'Saved {yearly_file}')
        else:
            logger.info(f'{yearly_file} already exists.')

    @classmethod
    def _check_single_file(cls,
                           res,
                           var_list=None,
                           check_nans=True,
                           check_heights=True,
                           max_interp_height=200,
                           required_shape=None,
                           max_workers=10):
        """Make sure given files include the given variables. Check for NaNs
        and required shape.

        Parameters
        ----------
        res : xr.open_dataset() object
            opened xarray data handler.
        var_list : list
            List of variables to check.
        check_nans : bool
            Whether to check data for NaNs.
        check_heights : bool
            Whether to check for heights above max interpolation height.
        max_interp_height : int
            Maximum height for interpolated output. Need raw heights above this
            to avoid extrapolation.
        required_shape : None | tuple
            Required shape for data. Should be (n_levels, n_lats, n_lons).
            If None the shape check will be skipped.
        max_workers : int | None
            Max number of workers to use in height check routine.

        Returns
        -------
        good_vars : bool
            Whether file includes all given variables
        good_shape : bool
            Whether shape matches required shape
        good_hgts : bool
            Whether there exists a height above the max interpolation height
            for each spatial location and timestep
        nan_pct : float
            Percent of data which consists of NaNs across all given variables.
        """
        good_vars = all(var in res for var in var_list)
        res_shape = (*res['level'].shape, *res['latitude'].shape,
                     *res['longitude'].shape,
                     )
        good_shape = ('NA' if required_shape is None else
                      (res_shape == required_shape))
        good_hgts = ('NA' if not check_heights else cls.check_heights(
            res,
            max_interp_height=max_interp_height,
            max_workers=max_workers,
        ))
        nan_pct = ('NA' if not check_nans else cls.get_nan_pct(
            res, var_list=var_list))

        if not good_vars:
            mask = [var not in res for var in var_list]
            missing_vars = np.array(var_list)[mask]
            logger.error(f'Missing variables: {missing_vars}.')
        if good_shape != 'NA' and not good_shape:
            logger.error(f'Bad shape: {res_shape} != {required_shape}.')

        return good_vars, good_shape, good_hgts, nan_pct

    @classmethod
    def check_heights(cls, res, max_interp_height=200, max_workers=10):
        """Make sure there are heights higher than max interpolation height

        Parameters
        ----------
        res : xr.open_dataset() object
            opened xarray data handler.
        max_interp_height : int
            Maximum height for interpolated output. Need raw heights above this
            to avoid extrapolation.
        max_workers : int | None
            Max number of workers to use for process pool height check.

        Returns
        -------
        bool
            Whether there is a height above max_interp_height for every spatial
            location and timestep
        """
        gp = res['zg'].values
        sfc_hgt = np.repeat(res['orog'].values[:, np.newaxis, ...],
                            gp.shape[1],
                            axis=1)
        heights = gp - sfc_hgt
        heights = heights.reshape(heights.shape[0], heights.shape[1], -1)
        checks = []
        logger.info(
            f'Checking heights with max_interp_height={max_interp_height}.')

        if max_workers == 1:
            for idt in range(heights.shape[0]):
                checks.append(
                    cls._check_heights_single_ts(
                        heights[idt], max_interp_height=max_interp_height))
                msg = f'Finished check for {idt + 1} of {heights.shape[0]}.'
                logger.debug(msg)
        else:
            futures = []
            with ProcessPoolExecutor(max_workers=max_workers) as exe:
                for idt in range(heights.shape[0]):
                    future = exe.submit(cls._check_heights_single_ts,
                                        heights[idt],
                                        max_interp_height=max_interp_height,
                                        )
                    futures.append(future)
                    msg = (f'Submitted height check for {idt + 1} of '
                           f'{heights.shape[0]}')
                    logger.info(msg)
            for i, future in enumerate(as_completed(futures)):
                checks.append(future.result())
                msg = (f'Finished height check for {i + 1} of '
                       f'{heights.shape[0]}')
                logger.info(msg)

        return all(checks)

    @classmethod
    def _check_heights_single_ts(cls, heights, max_interp_height=200):
        """Make sure there are heights higher than max interpolation height for
        a single timestep

        Parameters
        ----------
        heights : ndarray
            Array of heights for single timestep and all spatial locations
        max_interp_height : int
            Maximum height for interpolated output. Need raw heights above this
            to avoid extrapolation.

        Returns
        -------
        bool
            Whether there is a height above max_interp_height for every spatial
            location
        """
        checks = [any(h > max_interp_height) for h in heights.T]
        return all(checks)

    @classmethod
    def get_nan_pct(cls, res, var_list=None):
        """Get percentage of data which consists of NaNs, across the given
        variables

        Parameters
        ----------
        res : xr.open_dataset() object
            opened xarray data handler.
        var_list : list
            List of variables to check.
            If None: ['zg', 'orog', 'u', 'v', 'u_10m', 'v_10m',
                      'u_100m', 'v_100m']

        Returns
        -------
        nan_pct : float
            Percent of data which consists of NaNs across all given variables.
        """
        elem_count = 0
        nan_count = 0
        for var in var_list:
            logger.info(f'Checking NaNs for {var}.')
            nans = np.isnan(res[var].values)
            if nans.any():
                nan_count += nans.sum()
            elem_count += nans.size
        return 100 * nan_count / elem_count

    @classmethod
    def check_single_file(cls,
                          file,
                          var_list=None,
                          check_nans=True,
                          check_heights=True,
                          max_interp_height=200,
                          required_shape=None,
                          max_workers=10):
        """Make sure given files include the given variables. Check for NaNs
        and required shape.

        Parameters
        ----------
        file : str
            Name of file to check.
        var_list : list
            List of variables to check.
        check_nans : bool
            Whether to check data for NaNs.
        check_heights : bool
            Whether to check for heights above max interpolation height.
        max_interp_height : int
            Maximum height for interpolated output. Need raw heights above this
            to avoid extrapolation.
        required_shape : None | tuple
            Required shape for data. Should be (n_levels, n_lats, n_lons).
            If None the shape check will be skipped.
        max_workers : int | None
            Max number of workers to use for process pool height check.

        Returns
        -------
        good_vars : bool
            Whether file includes all given variables
        good_shape : bool
            Whether shape matches required shape
        good_hgts : bool
            Whether there is a height above max_interp_height for every spatial
            location at every timestep.
        nan_pct : float
            Percent of data which consists of NaNs across all given variables.
        """
        good = True
        nan_pct = None
        good_shape = None
        good_vars = None
        good_hgts = None
        var_list = (var_list if var_list is not None else cls.VALID_VARIABLES)
        try:
            res = xr.open_dataset(file)
        except Exception as e:
            msg = f'Unable to open {file}. {e}'
            logger.warning(msg)
            warn(msg)
            good = False

        if good:
            out = cls._check_single_file(res,
                                         var_list,
                                         check_nans=check_nans,
                                         check_heights=check_heights,
                                         max_interp_height=max_interp_height,
                                         required_shape=required_shape,
                                         max_workers=max_workers)
            good_vars, good_shape, good_hgts, nan_pct = out
        return good_vars, good_shape, good_hgts, nan_pct

    @classmethod
    def run_files_checks(cls,
                         file_pattern,
                         var_list=None,
                         required_shape=None,
                         check_nans=True,
                         check_heights=True,
                         max_interp_height=200,
                         max_workers=None,
                         height_check_workers=10):
        """Make sure given files include the given variables. Check for NaNs
        and required shape.

        Parameters
        ----------
        file_pattern : str | list
            glob-able file pattern for files to check.
        var_list : list | None
            List of variables to check. If None:
            ['zg', 'orog', 'u', 'v', 'u_10m', 'v_10m', 'u_100m', 'v_100m']
        required_shape : None | tuple
            Required shape for data. Should include (n_levels, n_lats, n_lons).
            If None the shape check will be skipped.
        check_nans : bool
            Whether to check data for NaNs.
        check_heights : bool
            Whether to check for heights above max interpolation height.
        max_interp_height : int
            Maximum height for interpolated output. Need raw heights above this
            to avoid extrapolation.
        max_workers : int | None
            Number of workers to use for thread pool file checks.
        height_check_workers : int | None
            Number of workers to use for process pool height check.

        Returns
        -------
        df : pd.DataFrame
            DataFrame describing file check results.  Has columns ['file',
            'good_vars', 'good_shape', 'good_hgts', 'nan_pct']

        """
        if isinstance(file_pattern, str):
            files = glob(file_pattern)
        else:
            files = file_pattern
        df = pd.DataFrame(columns=[
            'file', 'good_vars', 'good_shape', 'good_hgts', 'nan_pct'
        ])
        df['file'] = [os.path.basename(file) for file in files]
        if max_workers == 1:
            for i, file in enumerate(files):
                logger.info(f'Checking {file}.')
                out = cls.check_single_file(
                    file,
                    var_list=var_list,
                    check_nans=check_nans,
                    check_heights=check_heights,
                    max_interp_height=max_interp_height,
                    max_workers=height_check_workers,
                    required_shape=required_shape)
                df.loc[i, df.columns[1:]] = out
                logger.info(f'Finished checking {file}.')
        else:
            futures = {}
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                for i, file in enumerate(files):
                    future = exe.submit(cls.check_single_file,
                                        file=file,
                                        var_list=var_list,
                                        check_nans=check_nans,
                                        check_heights=check_heights,
                                        max_interp_height=max_interp_height,
                                        max_workers=height_check_workers,
                                        required_shape=required_shape)
                    msg = (f'Submitted file check future for {file}. Future '
                           f'{i + 1} of {len(files)}.')
                    logger.info(msg)
                    futures[future] = i
            for i, future in enumerate(as_completed(futures)):
                out = future.result()
                df.loc[futures[future], df.columns[1:]] = out
                msg = (f'Finished checking {df["file"].iloc[futures[future]]}.'
                       f' Future {i + 1} of {len(files)}.')
                logger.info(msg)
        return df
