"""Download ERA5 file for the given year and month

Note
----
To use this you need to have cdsapi package installed and a ~/.cdsapirc file
with a url and api key. Follow the instructions here:
https://cds.climate.copernicus.eu/api-how-to
"""

import logging
import os
from calendar import monthrange
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from warnings import warn

import dask.array as da
import numpy as np

from sup3r.preprocessing import Loader
from sup3r.preprocessing.loaders.utilities import standardize_names
from sup3r.preprocessing.names import (
    ERA_NAME_MAP,
    LEVEL_VARS,
    SFC_VARS,
    Dimension,
)
from sup3r.preprocessing.utilities import log_args

logger = logging.getLogger(__name__)


class EraDownloader:
    """Class to handle ERA5 downloading, variable renaming, and file
    combinations."""

    @log_args
    def __init__(
        self,
        year,
        month,
        area,
        levels,
        combined_out_pattern,
        overwrite=False,
        variables=None,
        product_type='reanalysis',
    ):
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
        overwrite : bool
            Whether to overwrite existing files.
        variables : list | None
            Variables to download. If None this defaults to just gepotential
            and wind components.
        product_type : str
            Can be 'reanalysis', 'ensemble_mean', 'ensemble_spread',
            'ensemble_members'
        """
        self.year = year
        self.month = month
        self.area = area
        self.levels = levels
        self.overwrite = overwrite
        self.combined_out_pattern = combined_out_pattern
        self._combined_file = None
        self._variables = variables
        self.sfc_file_variables = []
        self.level_file_variables = []
        self.prep_var_lists(self.variables)
        self.product_type = product_type
        self.hours = self.get_hours()

    def get_hours(self):
        """ERA5 is hourly and EDA is 3-hourly. Check and warn for incompatible
        requests."""
        if self.product_type == 'reanalysis':
            hours = [str(n).zfill(2) + ':00' for n in range(0, 24)]
        else:
            hours = [str(n).zfill(2) + ':00' for n in range(0, 24, 3)]
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
            for n in np.arange(1, monthrange(self.year, self.month)[1] + 1)
        ]

    @property
    def combined_file(self):
        """Get name of file from combined surface and level files"""
        if self._combined_file is None:
            if '{var}' in self.combined_out_pattern:
                self._combined_file = self.combined_out_pattern.format(
                    year=self.year,
                    month=str(self.month).zfill(2),
                    var='_'.join(self.variables),
                )
            else:
                self._combined_file = self.combined_out_pattern.format(
                    year=self.year, month=str(self.month).zfill(2)
                )
            os.makedirs(os.path.dirname(self._combined_file), exist_ok=True)
        return self._combined_file

    @property
    def surface_file(self):
        """Get name of file with variables from single level download"""
        basedir = os.path.dirname(self.combined_file)
        basename = ''
        if '{var}' in self.combined_out_pattern:
            basename += '_'.join(self.variables) + '_'
        basename += f'sfc_{self.year}_'
        basename += f'{str(self.month).zfill(2)}.nc'
        return os.path.join(basedir, basename)

    @property
    def level_file(self):
        """Get name of file with variables from pressure level download"""
        basedir = os.path.dirname(self.combined_file)
        basename = ''
        if '{var}' in self.combined_out_pattern:
            basename += '_'.join(self.variables) + '_'
        basename += f'levels_{self.year}_'
        basename += f'{str(self.month).zfill(2)}.nc'
        return os.path.join(basedir, basename)

    @classmethod
    def get_tmp_file(cls, file):
        """Get temp file for given file. Then only needed variables will be
        written to the given file.
        """
        tmp_file = file.replace('.nc', '_tmp.nc')
        return tmp_file

    def _prep_var_lists(self, variables):
        """Add all downloadable variables for the generic requested variables.
        e.g. if variable = 'u' add all downloadable u variables to list.
        """
        d_vars = []
        var_list = variables.copy()
        for i, v in enumerate(var_list):
            if v in ('u', 'v'):
                var_list[i] = f'{v}_'

        all_vars = SFC_VARS + LEVEL_VARS + ['zg', 'orog']
        for var in var_list:
            d_vars.extend([d_var for d_var in all_vars if var in d_var])
        return d_vars

    def prep_var_lists(self, variables):
        """Create surface and level variable lists based on requested
        variables.
        """
        variables = self._prep_var_lists(variables)
        for var in variables:
            if var in SFC_VARS and var not in self.sfc_file_variables:
                self.sfc_file_variables.append(var)
            elif var in LEVEL_VARS and var not in self.level_file_variables:
                self.level_file_variables.append(var)
            elif var not in SFC_VARS + LEVEL_VARS + ['zg', 'orog']:
                msg = f'Requested {var} is not available for download.'
                logger.warning(msg)
                warn(msg)

        sfc_and_level_check = (
            len(self.sfc_file_variables) > 0
            and len(self.level_file_variables) > 0
            and 'orog' not in variables
            and 'zg' not in variables
        )
        if sfc_and_level_check:
            msg = (
                'Both surface and pressure level variables were requested '
                'without requesting "orog" and "zg". Adding these to the '
                'download'
            )
            logger.info(msg)
            self.sfc_file_variables.append('geopotential')
            self.level_file_variables.append('geopotential')

        else:
            if 'orog' in variables:
                self.sfc_file_variables.append('geopotential')
            if 'zg' in variables:
                self.level_file_variables.append('geopotential')

    @staticmethod
    def get_cds_client():
        """Get the copernicus climate data store (CDS) API object for ERA
        downloads."""

        try:
            import cdsapi  # noqa
        except ImportError as e:
            msg = f'Could not import cdsapi package. {e}'
            raise ImportError(msg) from e

        msg = (
            'To download ERA5 data you need to have a ~/.cdsapirc file '
            'with a valid url and api key. Follow the instructions here: '
            'https://cds.climate.copernicus.eu/api-how-to'
        )
        req_file = os.path.join(os.path.expanduser('~'), '.cdsapirc')
        assert os.path.exists(req_file), msg

        return cdsapi.Client()

    def download_process_combine(self):
        """Run the download routine."""
        sfc_check = len(self.sfc_file_variables) > 0
        level_check = (
            len(self.level_file_variables) > 0
            and self.levels is not None
            and len(self.levels) > 0
        )
        if self.level_file_variables:
            msg = (
                f'{self.level_file_variables} requested but no levels'
                ' were provided.'
            )
            if self.levels is None:
                logger.warning(msg)
                warn(msg)

        time_dict = {
            'year': self.year,
            'month': self.month,
            'day': self.days,
            'time': self.hours,
        }
        if sfc_check:
            self.download_file(
                self.sfc_file_variables,
                time_dict=time_dict,
                area=self.area,
                out_file=self.surface_file,
                level_type='single',
                overwrite=self.overwrite,
                product_type=self.product_type,
            )
        if level_check:
            self.download_file(
                self.level_file_variables,
                time_dict=time_dict,
                area=self.area,
                out_file=self.level_file,
                level_type='pressure',
                levels=self.levels,
                overwrite=self.overwrite,
                product_type=self.product_type,
            )
        if sfc_check or level_check:
            self.process_and_combine()

    @classmethod
    def download_file(
        cls,
        variables,
        time_dict,
        area,
        out_file,
        level_type,
        levels=None,
        product_type='reanalysis',
        overwrite=False,
    ):
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
            msg = (
                f'Downloading {variables} to {out_file} with levels '
                f'= {levels}.'
            )
            logger.info(msg)
            entry = {
                'product_type': product_type,
                'format': 'netcdf',
                'variable': variables,
                'area': area,
            }
            entry.update(time_dict)
            if level_type == 'pressure':
                entry['pressure_level'] = levels
            logger.info(f'Calling CDS-API with {entry}.')
            cds_api_client = cls.get_cds_client()
            cds_api_client.retrieve(
                f'reanalysis-era5-{level_type}-levels', entry, out_file
            )
        else:
            logger.info(f'File already exists: {out_file}.')

    def process_surface_file(self):
        """Rename variables and convert geopotential to geopotential height."""
        tmp_file = self.get_tmp_file(self.surface_file)
        ds = Loader(self.surface_file)
        logger.info('Converting "z" var to "orog" for %s', self.surface_file)
        ds = self.convert_z(ds, name='orog')
        ds = standardize_names(ds, ERA_NAME_MAP)
        ds.to_netcdf(tmp_file)
        os.replace(tmp_file, self.surface_file)
        logger.info(
            f'Finished processing {self.surface_file}. Moved {tmp_file} to '
            f'{self.surface_file}.'
        )

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
        if 'pressure' in self.variables and 'pressure' not in ds.data_vars:
            logger.info('Adding pressure variable.')
            pres = 100 * ds[Dimension.PRESSURE_LEVEL].values.astype(np.float32)
            ds['pressure'] = (
                ds['zg'].dims,
                da.broadcast_to(pres, ds['zg'].shape),
            )
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
        if name not in ds.data_vars and 'z' in ds.data_vars:
            ds['z'] = (ds['z'].dims, ds['z'].values / 9.81)
            ds = ds.rename({'z': name})
        return ds

    def process_level_file(self):
        """Convert geopotential to geopotential height."""
        tmp_file = self.get_tmp_file(self.level_file)
        ds = Loader(self.level_file)
        logger.info('Converting "z" var to "zg" for %s', self.level_file)
        ds = self.convert_z(ds, name='zg')
        ds = standardize_names(ds, ERA_NAME_MAP)
        ds = self.add_pressure(ds)
        ds.to_netcdf(tmp_file)
        os.replace(tmp_file, self.level_file)
        logger.info(
            f'Finished processing {self.level_file}. Moved '
            f'{tmp_file} to {self.level_file}.'
        )

    @classmethod
    def _write_dsets(cls, files, out_file, kwargs):
        """Write data vars to out_file one dset at a time."""
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        added_features = []
        tmp_file = cls.get_tmp_file(out_file)
        for file in files:
            ds = Loader(file, res_kwargs=kwargs)
            for f in set(ds.data_vars) - set(added_features):
                mode = 'w' if not os.path.exists(tmp_file) else 'a'
                logger.info('Adding %s to %s.', f, tmp_file)
                ds.to_netcdf(tmp_file, mode=mode)
                logger.info('Added %s to %s.', f, tmp_file)
                added_features.append(f)
        logger.info(f'Finished writing {tmp_file}')
        os.replace(tmp_file, out_file)
        logger.info('Moved %s to %s.', tmp_file, out_file)

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
            kwargs = {'compat': 'override'}
            try:
                self._write_dsets(
                    files, out_file=self.combined_file, kwargs=kwargs
                )
            except Exception as e:
                msg = f'Error combining {files}.'
                logger.error(msg)
                raise RuntimeError(msg) from e

            if os.path.exists(self.level_file):
                os.remove(self.level_file)
            if os.path.exists(self.surface_file):
                os.remove(self.surface_file)
        else:
            logger.info(f'{self.combined_file} already exists.')

    def get_monthly_file(self):
        """Download level and surface files, process variables, and combine
        processed files. Includes checks for shape and variables."""
        if os.path.exists(self.combined_file) and self.overwrite:
            os.remove(self.combined_file)

        if not os.path.exists(self.combined_file):
            self.download_process_combine()

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
                file_pattern.replace('_{var}', '').format(
                    year=year, month=str(month).zfill(2)
                )
            )
            for month in range(1, 13)
        )

    @classmethod
    def all_vars_exist(cls, year, month, file_pattern, variables):
        """Check if all monthly variable files for the requested year and month
        exist.

        Parameters
        ----------
        year : int
            Year used for data download.
        month : int
            Month used for data download
        file_pattern : str
            Pattern for monthly variable file. Must include year, month, and
            var format keys. e.g. 'era5_{year}_{month}_{var}_combined.nc'
        variables : list
            Variables that should have been downloaded

        Returns
        -------
        bool
            True if all monthly variable files for the requested year and month
            exist.
        """
        return all(
            os.path.exists(
                file_pattern.format(
                    year=year, month=str(month).zfill(2), var=var
                )
            )
            for var in variables
        )

    @classmethod
    def run_month(
        cls,
        year,
        month,
        area,
        levels,
        combined_out_pattern,
        overwrite=False,
        variables=None,
        product_type='reanalysis',
    ):
        """Run routine for the given month and year.

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
        overwrite : bool
            Whether to overwrite existing files.
        variables : list | None
            Variables to download. If None this defaults to just gepotential
            and wind components.
        product_type : str
            Can be 'reanalysis', 'ensemble_mean', 'ensemble_spread',
            'ensemble_members'
        """
        variables = variables if isinstance(variables, list) else [variables]
        for var in variables:
            downloader = cls(
                year=year,
                month=month,
                area=area,
                levels=levels,
                combined_out_pattern=combined_out_pattern,
                overwrite=overwrite,
                variables=[var],
                product_type=product_type,
            )
            downloader.get_monthly_file()

    @classmethod
    def run_year(
        cls,
        year,
        area,
        levels,
        combined_out_pattern,
        combined_yearly_file=None,
        overwrite=False,
        max_workers=None,
        variables=None,
        product_type='reanalysis',
    ):
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
        overwrite : bool
            Whether to overwrite existing files.
        max_workers : int
            Max number of workers to use for downloading and processing monthly
            files.
        variables : list | None
            Variables to download. If None this defaults to just gepotential
            and wind components.
        product_type : str
            Can be 'reanalysis', 'ensemble_mean', 'ensemble_spread',
            'ensemble_members'
        """
        msg = (
            'combined_out_pattern must have {year}, {month}, and {var} '
            'format keys'
        )
        assert all(
            key in combined_out_pattern
            for key in ('{year}', '{month}', '{var}')
        ), msg

        if max_workers == 1:
            for month in range(1, 13):
                for var in variables:
                    cls.run_month(
                        year=year,
                        month=month,
                        area=area,
                        levels=levels,
                        combined_out_pattern=combined_out_pattern,
                        overwrite=overwrite,
                        variables=[var],
                        product_type=product_type,
                    )
        else:
            futures = {}
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                for month in range(1, 13):
                    for var in variables:
                        future = exe.submit(
                            cls.run_month,
                            year=year,
                            month=month,
                            area=area,
                            levels=levels,
                            combined_out_pattern=combined_out_pattern,
                            overwrite=overwrite,
                            variables=[var],
                            product_type=product_type,
                        )
                        futures[future] = {
                            'year': year,
                            'month': month,
                            'var': var,
                        }
                        logger.info(
                            f'Submitted future for year {year} and '
                            f'month {month} and variable {var}.'
                        )
            for future in as_completed(futures):
                future.result()
                v = futures[future]
                logger.info(
                    f'Finished future for year {v["year"]} and month '
                    f'{v["month"]} and variable {v["var"]}.'
                )

        for month in range(1, 13):
            cls.make_monthly_file(year, month, combined_out_pattern, variables)

        if combined_yearly_file is not None:
            cls.make_yearly_file(
                year, combined_out_pattern, combined_yearly_file
            )

    @classmethod
    def make_monthly_file(cls, year, month, file_pattern, variables):
        """Combine monthly variable files into a single monthly file.

        Parameters
        ----------
        year : int
            Year used to download data
        month : int
            Month used to download data
        file_pattern : str
            File pattern for monthly variable files. Must have year, month, and
            var format keys. e.g. './era_{year}_{month}_{var}_combined.nc'
        variables : list
            List of variables downloaded.
        """
        msg = (
            f'Not all variable files with file_patten {file_pattern} for '
            f'year {year} and month {month} exist.'
        )
        assert cls.all_vars_exist(year, month, file_pattern, variables), msg

        files = [
            file_pattern.format(year=year, month=str(month).zfill(2), var=var)
            for var in variables
        ]

        outfile = file_pattern.replace('_{var}', '').format(
            year=year, month=str(month).zfill(2)
        )

        if not os.path.exists(outfile):
            logger.info(f'Combining {files} into {outfile}.')
            kwargs = {'chunks': 'auto'}
            try:
                cls._write_dsets(files, out_file=outfile, kwargs=kwargs)
            except Exception as e:
                msg = f'Error combining {files}.'
                logger.error(msg)
                raise RuntimeError(msg) from e
        else:
            logger.info(f'{outfile} already exists.')

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
        msg = (
            f'Not all monthly files with file_patten {file_pattern} for '
            f'year {year} exist.'
        )
        assert cls.all_months_exist(year, file_pattern), msg

        files = [
            file_pattern.replace('_{var}', '').format(
                year=year, month=str(month).zfill(2)
            )
            for month in range(1, 13)
        ]

        if not os.path.exists(yearly_file):
            kwargs = {
                'combine': 'nested',
                'concat_dim': 'time',
                'chunks': 'auto',
            }
            try:
                cls._write_dsets(files, out_file=yearly_file, kwargs=kwargs)
            except Exception as e:
                msg = f'Error combining {files}'
                logger.error(msg)
                raise RuntimeError(msg) from e
        else:
            logger.info(f'{yearly_file} already exists.')
