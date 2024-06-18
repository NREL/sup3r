"""pytests for general utilities"""

import os

import numpy as np
import xarray as xr

from sup3r.utilities.era_downloader import EraDownloader
from sup3r.utilities.pytest.helpers import (
    execute_pytest,
    make_fake_dset,
)


class TestEraDownloader(EraDownloader):
    """Testing version of era downloader with download_file method overridden
    since we wont include a cdsapi key in tests."""

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
        shape = (10, 10, 100)
        if levels is not None:
            shape = (*shape, len(levels))

        features = []

        if level_type == 'single':
            if 'geopotential' in variables:
                features.append('z')
            if '10m_u_component_of_wind' in variables:
                features.extend(['u10'])
            if '10m_v_component_of_wind' in variables:
                features.extend(['v10'])
            if '100m_u_component_of_wind' in variables:
                features.extend(['u100'])
            if '100m_v_component_of_wind' in variables:
                features.extend(['v100'])
            nc = make_fake_dset(
                shape=shape,
                features=features,
            )
            if 'z' in nc:
                nc['z'] = (nc['z'].dims, np.zeros(nc['z'].shape))
            nc.to_netcdf(out_file)
        else:
            if 'geopotential' in variables:
                features.append('z')
            if 'u_component_of_wind' in variables:
                features.append('u')
            if 'v_component_of_wind' in variables:
                features.append('v')
            nc = make_fake_dset(
                shape=shape,
                features=features
            )
            if 'z' in nc:
                arr = np.zeros(nc['z'].shape)
                for i in range(nc['z'].shape[1]):
                    arr[:, i, ...] = i * 100
                nc['z'] = (nc['z'].dims, arr)
            nc.to_netcdf(out_file)


def test_era_dl(tmpdir_factory):
    """Test basic post proc for era downloader."""

    variables = ['zg', 'orog', 'u', 'v']
    combined_out_pattern = os.path.join(
        tmpdir_factory.mktemp('tmp'), 'era5_{year}_{month}_{var}.nc'
    )
    year = 2000
    month = 1
    area = [50, -130, 23, -65]
    levels = [1000, 900, 800]
    TestEraDownloader.run_month(
        year=year,
        month=month,
        area=area,
        levels=levels,
        combined_out_pattern=combined_out_pattern,
        variables=variables,
    )
    for v in variables:
        tmp = xr.open_dataset(
            combined_out_pattern.format(year=2000, month='01', var=v)
        )
        assert v in tmp


def test_era_dl_log_interp(tmpdir_factory):
    """Test post proc for era downloader, including log interpolation."""

    combined_out_pattern = os.path.join(
        tmpdir_factory.mktemp('tmp'), 'era5_{year}_{month}_{var}.nc'
    )
    interp_out_pattern = os.path.join(
        tmpdir_factory.mktemp('tmp'), 'era5_{year}_{month}_interp.nc'
    )
    TestEraDownloader.run_month(
        year=2000,
        month=1,
        area=[50, -130, 23, -65],
        levels=[1000, 900, 800],
        variables=['zg', 'orog', 'u', 'v'],
        combined_out_pattern=combined_out_pattern,
        interp_out_pattern=interp_out_pattern,
    )


def test_era_dl_year(tmpdir_factory):
    """Test post proc for era downloader, including log interpolation, for full
    year."""

    combined_out_pattern = os.path.join(
        tmpdir_factory.mktemp('tmp'), 'era5_{year}_{month}_{var}.nc'
    )
    interp_out_pattern = os.path.join(
        tmpdir_factory.mktemp('tmp'), 'era5_{year}_{month}_interp.nc'
    )
    yearly_file = os.path.join(tmpdir_factory.mktemp('tmp'), 'era5_final.nc')
    TestEraDownloader.run_year(
        year=2000,
        area=[50, -130, 23, -65],
        levels=[1000, 900, 800],
        variables=['zg', 'orog', 'u', 'v'],
        combined_out_pattern=combined_out_pattern,
        interp_out_pattern=interp_out_pattern,
        combined_yearly_file=yearly_file,
        max_workers=1,
        interp_workers=1
    )


if __name__ == '__main__':
    execute_pytest(__file__)
