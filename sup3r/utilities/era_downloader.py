"""Download ERA5 file for the given year and month

NOTE: To use this you need to have cdsapi package installed and a ~/.cdsapirc
file with a url and api key.  Follow the instructions here:
https://cds.climate.copernicus.eu/api-how-to
"""

import logging
import os
from calendar import monthrange
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from glob import glob
from typing import ClassVar
from warnings import warn

import numpy as np
import pandas as pd
import xarray as xr

from sup3r.utilities.interpolate_log_profile import LogLinInterpolator

logger = logging.getLogger(__name__)


class EraDownloader:
    """Class to handle ERA5 downloading, variable renaming, file combination,
    and interpolation.
    """

    # variables available on a single level (e.g. surface)
    SFC_VARS: ClassVar[list] = [
        '10m_u_component_of_wind', '10m_v_component_of_wind',
        '100m_u_component_of_wind', '100m_v_component_of_wind',
        'surface_pressure', '2m_temperature', 'geopotential',
        'total_precipitation', "convective_available_potential_energy",
        "2m_dewpoint_temperature", "convective_inhibition",
        "surface_latent_heat_flux", "instantaneous_moisture_flux",
        "mean_total_precipitation_rate", "mean_sea_level_pressure",
        "friction_velocity", "lake_cover", "high_vegetation_cover",
        "land_sea_mask", "k_index", "forecast_surface_roughness",
        "northward_turbulent_surface_stress",
        "eastward_turbulent_surface_stress",
        "sea_surface_temperature",
    ]

    # variables available on multiple pressure levels
    LEVEL_VARS: ClassVar[list] = [
        'u_component_of_wind', 'v_component_of_wind', 'geopotential',
        'temperature', 'relative_humidity', 'specific_humidity', 'divergence',
        'vertical_velocity', 'pressure', 'potential_vorticity'
    ]

    NAME_MAP: ClassVar[dict] = {
        'u10': 'u_10m',
        'v10': 'v_10m',
        'u100': 'u_100m',
        'v100': 'v_100m',
        't': 'temperature',
        't2m': 'temperature_2m',
        'sp': 'pressure_0m',
        'r': 'relativehumidity',
        'relative_humidity': 'relativehumidity',
        'q': 'specifichumidity',
        'd': 'divergence',
    }

    SHORT_NAME_MAP: ClassVar[dict] = {
        'convective_inhibition': 'cin',
        '2m_dewpoint_temperature': 'd2m',
        'potential_vorticity': 'pv',
        'vertical_velocity': 'w',
        'surface_latent_heat_flux': 'slhf',
        'instantaneous_moisture_flux': 'ie',
        'divergence': 'd',
        'total_precipitation': 'tp',
        'relative_humidity': 'relativehumidity',
        'convective_available_potential_energy': 'cape',
        'mean_total_precipitation_rate': 'mtpr',
        'u_component_of_wind': 'u',
        'v_component_of_wind': 'v'
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
                 variables=None,
                 check_files=False,
                 product_type='reanalysis'):
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
        variables : list | None
            Variables to download. If None this defaults to just gepotential
            and wind components.
        check_files : bool
            Check existing files. Remove and redownload if checks fail.
        product_type : str
            Can be 'reanalysis', 'ensemble_mean', 'ensemble_spread',
            'ensemble_members'
        """
        self.year = year
        self.month = month
        self.area = area
        self.levels = levels
        self.run_interp = run_interp
        self.overwrite = overwrite
        self.combined_out_pattern = combined_out_pattern
        self.interp_out_pattern = interp_out_pattern
        self.check_files = check_files
        self.required_shape = None
        self._interp_file = None
        self._combined_file = None
        self._variables = variables
        self.hours = [str(n).zfill(2) + ":00" for n in range(0, 24)]
        self.sfc_file_variables = ['geopotential']
        self.level_file_variables = ['geopotential']
        self.prep_var_lists(self.variables)
        self.product_type = product_type
        self.hours = self.get_hours()

        msg = ('Initialized EraDownloader with: '
               f'year={self.year}, month={self.month}, area={self.area}, '
               f'levels={self.levels}, variables={self.variables}, '
               f'product_type={self.product_type}')
        logger.info(msg)

    def get_hours(self):
        """ERA5 is hourly and EDA is 3-hourly. Check and warn for incompatible
        requests."""
        if self.product_type == 'reanalysis':
            hours = [str(n).zfill(2) + ":00" for n in range(0, 24)]
        else:
            hours = [str(n).zfill(2) + ":00" for n in range(0, 24, 3)]
        return hours

    @property
    def variables(self):
        """Get list of requested variables"""
        if self._variables is None:
            raise OSError('Received empty variable list.')
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
        if (self._interp_file is None and self.interp_out_pattern is not None
                and self.run_interp):
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
        basename = f'sfc_{self.year}_'
        basename += f'{str(self.month).zfill(2)}.nc'
        return os.path.join(basedir, basename)

    @property
    def level_file(self):
        """Get name of file with variables from pressure level download"""
        basedir = os.path.dirname(self.combined_file)
        basename = f'levels_{self.year}_'
        basename += f'{str(self.month).zfill(2)}.nc'
        return os.path.join(basedir, basename)

    @classmethod
    def get_tmp_file(cls, file):
        """Get temp file for given file. Then only needed variables will be
        written to the given file.
        """
        tmp_file = file.replace(".nc", "_tmp.nc")
        return tmp_file

    def _prep_var_lists(self, variables):
        """Add all downloadable variables for the generic requested variables.
        e.g. if variable = 'u' add all downloadable u variables to list.
        """
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
        variables.
        """
        variables = self._prep_var_lists(variables)
        for var in variables:
            if var in self.SFC_VARS and var not in self.sfc_file_variables:
                self.sfc_file_variables.append(var)
            elif (var in self.LEVEL_VARS
                  and var not in self.level_file_variables):
                self.level_file_variables.append(var)
            elif var not in self.SFC_VARS and var not in self.LEVEL_VARS:
                msg = f'Requested {var} is not available for download.'
                logger.warning(msg)
                warn(msg)

    @staticmethod
    def get_cds_client():
        """Get the copernicus climate data store (CDS) API object for ERA
        downloads."""
        try:
            import cdsapi
            cds_api_client = cdsapi.Client()
        except ImportError as e:
            msg = f'Could not import cdsapi package. {e}'
            logger.error(msg)
            raise ImportError(msg) from e

        msg = ('To download ERA5 data you need to have a ~/.cdsapirc file '
               'with a valid url and api key. Follow the instructions here: '
               'https://cds.climate.copernicus.eu/api-how-to')
        req_file = os.path.join(os.path.expanduser('~'), '.cdsapirc')
        assert os.path.exists(req_file), msg

        return cds_api_client

    def download_process_combine(self):
        """Run the download routine."""
        sfc_check = len(self.sfc_file_variables) > 0
        level_check = (len(self.level_file_variables) > 0
                       and self.levels is not None
                       and len(self.levels) > 0)
        if self.level_file_variables:
            msg = (f'{self.level_file_variables} requested but no levels'
                   ' were provided.')
            if self.levels is None:
                logger.warning(msg)
                warn(msg)

        time_dict = {'year': self.year, 'month': self.month, 'day': self.days,
                     'time': self.hours}
        if sfc_check:
            self.download_file(self.sfc_file_variables, time_dict=time_dict,
                               area=self.area, out_file=self.surface_file,
                               level_type='single', overwrite=self.overwrite,
                               product_type=self.product_type)
        if level_check:
            self.download_file(self.level_file_variables, time_dict=time_dict,
                               area=self.area, out_file=self.level_file,
                               level_type='pressure', levels=self.levels,
                               overwrite=self.overwrite,
                               product_type=self.product_type)
        if sfc_check or level_check:
            self.process_and_combine()

    @classmethod
    def download_file(cls, variables, time_dict, area, out_file, level_type,
                      levels=None, product_type='reanalysis', overwrite=False):
        """Download either single-level or pressure-level file

        Parameters
        ----------
        variables : list
            List of variables to download
        time_dict : dict
            Dictionary with year, month, day, time entries.
        area : list
            List of bounding box coordinates.
            e.g. [max_lat, min_lon, min_lat, max_lon]
        out_file : str
            Name of output file
        level_type : str
            Either 'single' or 'pressure'
        levels : list
            List of pressure levels to download, if level_type == 'pressure'
        product_type : str
            Can be 'reanalysis', 'ensemble_mean', 'ensemble_spread',
            'ensemble_members'
        overwrite : bool
            Whether to overwrite existing file
        """
        if not os.path.exists(out_file) or overwrite:
            msg = (f'Downloading {variables} to '
                   f'{out_file} with levels = {levels}.')
            logger.info(msg)
            entry = {
                'product_type': product_type,
                'format': 'netcdf',
                'variable': variables,
                'area': area}
            entry.update(time_dict)
            if level_type == 'pressure':
                entry['pressure_level'] = levels
            logger.info(f'Calling CDS-API with {entry}.')
            cds_api_client = cls.get_cds_client()
            cds_api_client.retrieve(
                f'reanalysis-era5-{level_type}-levels',
                entry, out_file)
        else:
            logger.info(f'File already exists: {out_file}.')

    def process_surface_file(self):
        """Rename variables and convert geopotential to geopotential height."""
        tmp_file = self.get_tmp_file(self.surface_file)
        with xr.open_dataset(self.surface_file, mode='a') as ds:
            new_ds = self.convert_z(ds, name='orog')
            new_ds = self.map_vars(new_ds)
            new_ds.to_netcdf(tmp_file)
        os.system(f'mv {tmp_file} {self.surface_file}')
        logger.info(f'Finished processing {self.surface_file}. Moved '
                    f'{tmp_file} to {self.surface_file}.')

    def map_vars(self, ds):
        """Map variables from old dataset to new dataset

        Parameters
        ----------
        ds : Dataset
            xr.Dataset() object for which to rename variables

        Returns
        -------
        new_ds : Dataset
            xr.Dataset() object with new variables written.
        """
        for old_name in ds.data_vars:
            new_name = self.NAME_MAP.get(old_name, old_name)
            ds = ds.rename({old_name: new_name})
        return ds

    def shift_temp(self, ds):
        """Shift temperature to celsius

        Parameters
        ----------
        ds : Dataset
            xr.Dataset() object for which to shift temperature

        Returns
        -------
        ds : Dataset
        """
        for var in ds.data_vars:
            if 'units' in ds[var].attrs and ds[var].attrs['units'] == 'K':
                ds[var] = (ds[var].dims, ds[var].values - 273.15)
                ds[var].attrs['units'] = 'C'
        return ds

    def add_pressure(self, ds):
        """Add pressure to dataset

        Parameters
        ----------
        ds : Dataset
            xr.Dataset() object for which to add pressure

        Returns
        -------
        ds : Dataset
        """
        if ('pressure' in self.variables
                and 'pressure' not in ds.data_vars):
            expand_axes = (0, 2, 3)
            pres = np.zeros(ds['zg'].values.shape)
            if 'number' in ds.dims:
                expand_axes = (0, 1, 3, 4)
            pres[:] = np.expand_dims(100 * ds['level'].values,
                                     axis=expand_axes)
            ds['pressure'] = (ds['zg'].dims, pres)
            ds['pressure'].attrs['units'] = 'Pa'
        return ds

    def convert_z(self, ds, name):
        """Convert z to given height variable

        Parameters
        ----------
        ds : Dataset
            xr.Dataset() object for new file
        name : str
            Variable name. e.g. zg or orog, typically

        Returns
        -------
        ds : Dataset
            xr.Dataset() object for new file with new height variable written.
        """
        if name not in ds.data_vars:
            ds['z'] = (ds['z'].dims, ds['z'].values / 9.81)
            ds = ds.rename({'z': name})
        return ds

    def process_level_file(self):
        """Convert geopotential to geopotential height."""
        tmp_file = self.get_tmp_file(self.level_file)
        with xr.open_dataset(self.level_file, mode='a') as ds:
            new_ds = self.convert_z(ds, name='zg')
            new_ds = self.map_vars(new_ds)
            new_ds = self.shift_temp(new_ds)
            new_ds = self.add_pressure(new_ds)
            new_ds.to_netcdf(tmp_file)

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

            logger.info(f'Combining {files} to {self.combined_file}.')
            with xr.open_mfdataset(files, compat='override') as ds:
                ds.to_netcdf(self.combined_file)
            logger.info(f'Finished writing {self.combined_file}')

            if os.path.exists(self.level_file):
                os.remove(self.level_file)
            if os.path.exists(self.surface_file):
                os.remove(self.surface_file)

    def good_file(self, file, required_shape=None):
        """Check if file has the required shape and variables.

        Parameters
        ----------
        file : str
            Name of file to check for required variables and shape
        required_shape : tuple | None
            Required shape of data to download. Used to check downloaded data.
            Should be (n_levels, n_lats, n_lons).  If None, no check is
            performed.

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
        good_vars, good_shape, good_hgts, _ = out
        return bool(good_vars and good_shape and good_hgts)

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
        valid_vars = list(self.LEVEL_VARS) + list(self.SFC_VARS)
        good = all(var in valid_vars for var in variables)
        if not good:
            msg = (f'Received variables {variables} not in valid variables '
                   f'list {valid_vars}')
            logger.error(msg)
            raise OSError(msg)

    def check_existing_files(self, required_shape=None):
        """If files exist already check them for good shape and required
        variables. Remove them if there was a problem so we can continue with
        routine from scratch.
        """
        if os.path.exists(self.combined_file):
            try:
                check = self.good_file(self.combined_file, required_shape)
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
        max_log_height (usually 100m) and linear interpolation above this.
        """
        variables = [var for var in self.variables if var in self.LEVEL_VARS]
        for var in self.variables:
            if var in self.NAME_MAP:
                variables.append(self.NAME_MAP[var])
            elif (var in self.SHORT_NAME_MAP
                    and var not in self.NAME_MAP.values()):
                variables.append(self.SHORT_NAME_MAP[var])
            else:
                variables.append(var)
        LogLinInterpolator.run(infile=self.combined_file,
                               outfile=self.interp_file,
                               max_workers=max_workers,
                               variables=variables,
                               overwrite=self.overwrite,
                               **kwargs)

    def get_monthly_file(self, interp_workers=None, prune_variables=False,
                         **interp_kwargs):
        """Download level and surface files, process variables, and combine
        processed files. Includes checks for shape and variables and option to
        interpolate.
        """
        if os.path.exists(self.combined_file) and self.overwrite:
            os.remove(self.combined_file)

        if self.check_files:
            self.check_existing_files()

        if not os.path.exists(self.combined_file):
            self.download_process_combine()

        if self.run_interp:
            self.run_interpolation(max_workers=interp_workers, **interp_kwargs)

        if self.interp_file is not None and os.path.exists(self.interp_file):
            if self.already_pruned(self.interp_file, prune_variables):
                logger.info(f'{self.interp_file} pruned already.')
            else:
                self.prune_output(self.interp_file, prune_variables)

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
    def already_pruned(cls, infile, prune_variables):
        """Check if file has been pruned already."""
        if not prune_variables:
            logger.info('Received prune_variables=False. Skipping pruning.')
            return
        with xr.open_dataset(infile) as ds:
            check_variables = [var for var in ds.data_vars
                               if 'level' in ds[var].dims]
            pruned = len(check_variables) == 0
        return pruned

    @classmethod
    def prune_output(cls, infile, prune_variables=False):
        """Prune output file to keep just single level variables"""
        if not prune_variables:
            logger.info('Received prune_variables=False. Skipping pruning.')
            return
        else:
            logger.info(f'Pruning {infile}.')
            tmp_file = cls.get_tmp_file(infile)
            with xr.open_dataset(infile) as ds:
                keep_vars = {k: v for k, v in dict(ds.data_vars)
                             if 'level' not in ds[k].dims}
                new_coords = {k: v for k, v in dict(ds.coords).items()
                              if 'level' not in k}
                new_ds = xr.Dataset(coords=new_coords, data_vars=keep_vars)
                new_ds.to_netcdf(tmp_file)
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
                  interp_workers=None,
                  variables=None,
                  prune_variables=False,
                  check_files=False,
                  product_type='reanalysis',
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
        interp_workers : int | None
            Max number of workers to use for interpolation.
        variables : list | None
            Variables to download. If None this defaults to just gepotential
            and wind components.
        prune_variables : bool
            Whether to remove 4D variables from data after interpolation. e.g.
            height interpolation could give u_10m, u_100m, u_120m from a 4D u
            array. If we only need these heights we could remove the 4D u array
            from the final data file.
        check_files : bool
            Check existing files. Remove and redownload if checks fail.
        product_type : str
            Can be 'reanalysis', 'ensemble_mean', 'ensemble_spread',
            'ensemble_members'
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
                         variables=variables,
                         check_files=check_files,
                         product_type=product_type)
        downloader.get_monthly_file(interp_workers=interp_workers,
                                    prune_variables=prune_variables,
                                    **interp_kwargs)

    @classmethod
    def run_year(cls,
                 year,
                 area,
                 levels,
                 combined_out_pattern,
                 combined_yearly_file=None,
                 interp_out_pattern=None,
                 interp_yearly_file=None,
                 run_interp=True,
                 overwrite=False,
                 max_workers=None,
                 interp_workers=None,
                 variables=None,
                 prune_variables=False,
                 check_files=False,
                 product_type='reanalysis',
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
        max_workers : int
            Max number of workers to use for downloading and processing monthly
            files.
        interp_workers : int | None
            Max number of workers to use for interpolation.
        variables : list | None
            Variables to download. If None this defaults to just gepotential
            and wind components.
        prune_variables : bool
            Whether to remove 4D variables from data after interpolation. e.g.
            height interpolation could give u_10m, u_100m, u_120m from a 4D u
            array. If we only need these heights we could remove the 4D u array
            from the final data file.
        check_files : bool
            Check existing files. Remove and redownload if checks fail.
        product_type : str
            Can be 'reanalysis', 'ensemble_mean', 'ensemble_spread',
            'ensemble_members'
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
                              interp_workers=interp_workers,
                              variables=variables,
                              prune_variables=prune_variables,
                              check_files=check_files,
                              product_type=product_type,
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
                        interp_workers=interp_workers,
                        prune_variables=prune_variables,
                        variables=variables,
                        check_files=check_files,
                        product_type=product_type,
                        **interp_kwargs)
                    futures[future] = {'year': year, 'month': month}
                    logger.info(f'Submitted future for year {year} and month '
                                f'{month}.')
            for future in as_completed(futures):
                future.result()
                v = futures[future]
                logger.info(f'Finished future for year {v["year"]} and month '
                            f'{v["month"]}.')

            if combined_yearly_file is not None:
                cls.make_yearly_file(year, combined_out_pattern,
                                     combined_yearly_file)

            if run_interp and interp_yearly_file is not None:
                cls.make_yearly_file(year, interp_out_pattern,
                                     interp_yearly_file)

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
            with xr.open_mfdataset(files, parallel=True) as res:
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
                    max_workers=height_check_workers)
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
                                        max_workers=height_check_workers)
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
