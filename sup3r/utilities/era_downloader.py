"""Download ERA5 file for the given year and month

NOTE: To use this you need to have a ~/.cdsapirc file with a url and api key.
Follow the instructions here: https://cds.climate.copernicus.eu/api-how-to
"""

import logging
import os
from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor, as_completed

import cdsapi
import numpy as np
import xarray as xr
from netCDF4 import Dataset

from sup3r.utilities.log_interpolator import LogLinInterpolator

logger = logging.getLogger(__name__)

c = cdsapi.Client()


class EraDownloader:
    """Class to handle ERA5 downloading, variable renaming, file combination,
    and interpolation."""

    def __init__(self, year, month, area, levels, combined_out_pattern,
                 interp_out_pattern=None, run_interp=True, overwrite=False,
                 required_shape=None):
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
            month format keys.  e.g. 'era5_uv_{year}_{month}_combined.nc'
        interp_out_pattern : str | None
            Pattern for interpolated monthly output file. Must include year and
            month format keys.  e.g. 'era5_uv_{year}_{month}_interp.nc'
        run_interp : bool
            Whether to run interpolation after downloading and combining files.
        overwrite : bool
            Whether to overwrite existing files.
        required_shape : tuple | None
            Required shape of data to download. Used to check downloaded data.
            If None, no check is performed.
        """
        self.year = year
        self.month = month
        self.area = area
        self.levels = levels
        self.run_interp = run_interp
        self.overwrite = overwrite
        self.required_shape = (None if not required_shape
                               else (1, len(levels), *required_shape))
        self.days = [str(n).zfill(2)
                     for n in np.arange(1, monthrange(year, month)[1] + 1)]
        self.hours = [str(n).zfill(2) + ":00" for n in np.arange(0, 24)]
        self.combined_file = combined_out_pattern.format(
            year=year, month=str(month).zfill(2))
        basedir = os.path.dirname(self.combined_file)
        self.surface_file = os.path.join(
            basedir, f'sfc_{year}_{str(month).zfill(2)}.nc')
        self.level_file = os.path.join(
            basedir, f'levels_{year}_{str(month).zfill(2)}.nc')
        if interp_out_pattern is not None and run_interp:
            self.interp_file = interp_out_pattern.format(
                year=year, month=str(month).zfill(2))

    def process_surface_file(self):
        """Rename variables and convert geopotential to geopotential height."""

        with Dataset(self.surface_file, "a") as ds:
            if 'z' in ds.variables:
                vals = ds.variables['z']
                ds.renameVariable('z', 'orog')
                ds.renameVariable('u10', 'u_10m')
                ds.renameVariable('v10', 'v_10m')
                ds.renameVariable('u100', 'u_100m')
                ds.renameVariable('v100', 'v_100m')
                ds.variables['orog'][:] = vals[:] / 9.81
                ds.variables['orog'].long_name = 'Orography'
                ds.variables['orog'].standard_name = 'orog'
                ds.variables['orog'].units = 'm'

    def process_level_file(self):
        """Convert geopotential to geopotential height."""

        with Dataset(self.level_file, "a") as ds:
            if 'z' in ds.variables:
                vals = ds.variables['z']
                ds.renameVariable('z', 'zg')
                ds.variables['zg'][:] = vals[:] / 9.81
                ds.variables['zg'].long_name = 'Geopotential Height'
                ds.variables['zg'].standard_name = 'zg'
                ds.variables['zg'].units = 'm'

    def download_levels_file(self):
        """Download file with requested pressure levels"""
        if not os.path.exists(self.level_file):
            c.retrieve(
                'reanalysis-era5-pressure-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': [
                        'u_component_of_wind', 'v_component_of_wind',
                        'geopotential'
                    ],
                    'pressure_level': self.levels,
                    'year': self.year,
                    'month': self.month,
                    'day': self.days,
                    'time': self.hours,
                    'area': self.area,
                },
                self.level_file)
        else:
            logger.info(f'File already exists: {self.level_file}.')

    def download_surface_file(self):
        """Download surface file"""
        if not os.path.exists(self.surface_file):
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': [
                        'geopotential',
                        '10m_u_component_of_wind',
                        '10m_v_component_of_wind',
                        '100m_u_component_of_wind',
                        '100m_v_component_of_wind'
                    ],
                    'year': self.year,
                    'month': self.month,
                    'day': self.days,
                    'time': self.hours,
                    'area': self.area,
                },
                self.surface_file)
        else:
            logger.info(f'File already exists: {self.surface_file}.')

    def good_file(self, file, shape=None):
        """Check if file has the required shape and variables."""
        tmp = xr.open_dataset(file)
        check = True

        if shape is not None:
            if tmp['u'].shape[1:] != shape[1:]:
                check = False
                logger.info(f'Bad shape: {file} {tmp["u"].shape} {shape}')

        vars = ['u', 'v', 'orog', 'zg']
        for h in [10, 100]:
            vars.append(f'u_{h}m')
            vars.append(f'v_{h}m')
        for var in vars:
            if var not in tmp.variables:
                check = False
                logger.info(f'Missing variable: {file} ({var})')
        return check

    def download_process_combine(self):
        """Download surface and level files, process variables, and combine."""
        if not os.path.exists(self.combined_file):
            self.download_surface_file()
            self.download_levels_file()
            logger.info(f'Processing {self.level_file}')
            self.process_level_file()
            logger.info(f'Processing {self.surface_file}')
            self.process_surface_file()

            logger.info(f'Combining {self.level_file} and {self.surface_file}')
            with xr.open_mfdataset([self.level_file, self.surface_file]) as ds:
                ds.to_netcdf(self.combined_file)
            logger.info(f'Finished writing {self.combined_file}')
            os.remove(self.level_file)
            os.remove(self.surface_file)

    def check_existing_files(self):
        """If files exist already check them for good shape and required
        variables. Remove them if there was a problem so we can continue with
        routine from scratch."""
        if os.path.exists(self.combined_file):
            try:
                check = self.good_file(self.combined_file,
                                       self.required_shape)
                if not check:
                    logger.info(f'Bad file: {self.combined_file}')
                    os.remove(self.combined_file)
                    if os.path.exists(self.interp_file):
                        os.remove(self.interp_file)
                else:
                    if os.path.exists(self.level_file):
                        os.remove(self.level_file)
                    if os.path.exists(self.surface_file):
                        os.remove(self.surface_file)
                    logger.info(f'{self.combined_file} already exists.')
            except Exception as e:
                logger.info(f'Something wrong with {self.combined_file}. {e}')
                os.remove(self.combined_file)
                os.remove(self.interp_file)

    def run_interpolation(self):
        """Run interpolation to get final final. Runs log interpolation up to
        max_log_height (usually 100m) and linear interpolation above this."""
        if not os.path.exists(self.interp_file) and self.run_interp:
            LogLinInterpolator.run(infile=self.combined_file,
                                   outfile=self.interp_file,
                                   input_heights=[10, 100],
                                   output_heights=[40, 80, 120, 160, 200],
                                   max_workers=32,
                                   max_log_height=100,
                                   overwrite=self.overwrite)
        elif self.run_interp and os.path.exists(self.interp_file):
            logger.info(f'{self.interp_file} already exists.')

    def get_monthly_file(self):
        """Download level and surface files, process variables, and combine
        processed files. Includes checks for shape and variables and option to
        interpolate."""

        if os.path.exists(self.combined_file) and self.overwrite:
            os.remove(self.combined_file)

        self.check_existing_files()

        self.download_process_combine()

        self.run_interpolation()

    @classmethod
    def all_months_exist(cls, year, file_pattern):
        """Check if all months in the requested year exist.

        Parameters
        ----------
        year : int
            Year of data to download.
        file_pattern : str
            Pattern for monthly output file. Must include year and month format
            keys. e.g. 'era5_uv_{year}_{month}_combined.nc'

        Returns
        -------
        bool
            True if all months in the requested year exist.
        """
        return all(os.path.exists(file_pattern.format(
            year=year, month=str(month).zfill(2))) for month in range(1, 13))

    @classmethod
    def run_month(cls, year, month, area, levels, combined_out_pattern,
                  interp_out_pattern=None, run_interp=True, overwrite=False,
                  required_shape=None):
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
            month format keys.  e.g. 'era5_uv_{year}_{month}_combined.nc'
        interp_out_pattern : str | None
            Pattern for interpolated monthly output file. Must include year and
            month format keys.  e.g. 'era5_uv_{year}_{month}_interp.nc'
        run_interp : bool
            Whether to run interpolation after downloading and combining files.
        overwrite : bool
            Whether to overwrite existing files.
        required_shape : tuple | None
            Required shape of data to download. Used to check downloaded data.
            If None, no check is performed.
        """
        downloader = cls(year=year, month=month, area=area, levels=levels,
                         combined_out_pattern=combined_out_pattern,
                         interp_out_pattern=interp_out_pattern,
                         run_interp=run_interp, overwrite=overwrite,
                         required_shape=required_shape)
        downloader.get_monthly_file()

    @classmethod
    def run_year(cls, year, area, levels, combined_out_pattern,
                 combined_yearly_file, interp_out_pattern=None,
                 interp_yearly_file=None, run_interp=True, overwrite=False,
                 required_shape=None, max_workers=12):
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
            month format keys.  e.g. 'era5_uv_{year}_{month}_combined.nc'
        combined_yearly_file : str
            Name of yearly file made from monthly combined files.
        interp_out_pattern : str | None
            Pattern for interpolated monthly output file. Must include year and
            month format keys.  e.g. 'era5_uv_{year}_{month}_interp.nc'
        interp_yearly_file : str
            Name of yearly file made from monthly interp files.
        run_interp : bool
            Whether to run interpolation after downloading and combining files.
        overwrite : bool
            Whether to overwrite existing files.
        required_shape : tuple | None
            Required shape of data to download. Used to check downloaded data.
            If None, no check is performed.
        max_workers : int
            Max number of workers to use for downloading and processing monthly
            files.
        """
        if max_workers == 1:
            for month in range(1, 13):
                cls.run_month(year=year, month=month, area=area, levels=levels,
                              combined_out_pattern=combined_out_pattern,
                              interp_out_pattern=interp_out_pattern,
                              run_interp=run_interp, overwrite=overwrite,
                              required_shape=required_shape)
        else:
            futures = {}
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                for month in range(1, 13):
                    future = exe.submit(
                        cls.run_month, year=year,
                        month=month, area=area, levels=levels,
                        combined_out_pattern=combined_out_pattern,
                        interp_out_pattern=interp_out_pattern,
                        run_interp=run_interp, overwrite=overwrite,
                        required_shape=required_shape)
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

        files = [file_pattern.format(year=year, month=str(month).zfill(2))
                 for month in range(1, 13)]

        if not os.path.exists(yearly_file):
            with xr.open_mfdataset(files) as res:
                logger.info(f'Combining {files}')
                os.makedirs(os.path.dirname(yearly_file), exist_ok=True)
                res.to_netcdf(yearly_file)
                logger.info(f'Saved {yearly_file}')
        else:
            logger.info(f'{yearly_file} already exists.')
