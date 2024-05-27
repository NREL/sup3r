# -*- coding: utf-8 -*-
"""pytests for data handling"""

import os
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from rex import init_logger

from sup3r import TEST_DATA_DIR
from sup3r.containers import LoaderH5, LoaderNC
from sup3r.utilities.pytest.helpers import (
    execute_pytest,
    make_fake_dset,
    make_fake_nc_file,
)

h5_files = [
    os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5'),
    os.path.join(TEST_DATA_DIR, 'test_wtk_co_2013.h5'),
]
nc_files = [os.path.join(TEST_DATA_DIR, 'test_era5_co_2012.nc')]
cc_files = [os.path.join(TEST_DATA_DIR, 'uas_test.nc')]

features = ['windspeed_100m', 'winddirection_100m']

init_logger('sup3r', log_level='DEBUG')


def test_time_independent_loading():
    """Make sure loaders work with time independent files."""
    with TemporaryDirectory() as td:
        out_file = os.path.join(td, 'topo.nc')
        nc = make_fake_dset((20, 20, 1), features=['topography'])
        nc = nc.isel(time=0)
        nc = nc.drop('time')
        assert 'time' not in nc.dims
        assert 'time' not in nc.coords
        nc.to_netcdf(out_file)
        loader = LoaderNC(out_file)
        assert loader.dims == ('south_north', 'west_east')


def test_dim_ordering():
    """Make sure standard reordering works with dimensions not in the standard
    list."""
    input_files = [
        os.path.join(TEST_DATA_DIR, 'ua_test.nc'),
        os.path.join(TEST_DATA_DIR, 'va_test.nc'),
        os.path.join(TEST_DATA_DIR, 'orog_test.nc'),
        os.path.join(TEST_DATA_DIR, 'zg_test.nc'),
    ]
    loader = LoaderNC(input_files)
    assert loader.dims == ('south_north', 'west_east', 'time', 'level', 'nbnd')


def test_lat_inversion():
    """Write temp file with ascending lats and load. Needs to be corrected to
    descending lats."""
    with TemporaryDirectory() as td:
        nc = make_fake_dset((20, 20, 100, 5), features=['u', 'v'])
        nc['latitude'] = (nc['latitude'].dims, nc['latitude'].data[::-1])
        out_file = os.path.join(td, 'inverted.nc')
        nc.to_netcdf(out_file)
        loader = LoaderNC(out_file)
        assert nc['latitude'][0, 0] < nc['latitude'][-1, 0]
        assert loader.lat_lon[-1, 0, 0] < loader.lat_lon[0, 0, 0]

        assert np.array_equal(
            nc['u']
            .transpose('south_north', 'west_east', 'time', 'level')
            .data[::-1],
            loader['u'],
        )


def test_load_cc():
    """Test simple era5 file loading."""
    chunks = (5, 5, 5)
    loader = LoaderNC(cc_files, chunks=chunks)
    assert all(
        loader.data[f].chunksize == chunks
        for f in loader.features
        if len(loader.data[f].shape) == 3
    )
    assert isinstance(loader.time_index, pd.DatetimeIndex)
    assert loader.dims[:3] == ('south_north', 'west_east', 'time')


def test_load_era5():
    """Test simple era5 file loading."""
    chunks = (5, 5, 5)
    loader = LoaderNC(nc_files, chunks=chunks)
    assert all(
        loader.data[f].chunksize == chunks
        for f in loader.features
        if len(loader.data[f].shape) == 3
    )
    assert isinstance(loader.time_index, pd.DatetimeIndex)
    assert loader.dims[:3] == ('south_north', 'west_east', 'time')


def test_load_nc():
    """Test simple netcdf file loading."""
    with TemporaryDirectory() as td:
        temp_file = os.path.join(td, 'test.nc')
        make_fake_nc_file(
            temp_file, shape=(10, 10, 20), features=['u_100m', 'v_100m']
        )
        chunks = (5, 5, 5)
        loader = LoaderNC(temp_file, chunks=chunks)
        assert loader.shape == (10, 10, 20, 2)
        assert all(loader.data[f].chunksize == chunks for f in loader.features)


def test_load_h5():
    """Test simple netcdf file loading. Also checks renaming elevation ->
    topography."""

    chunks = (5, 5)
    loader = LoaderH5(h5_files[0], chunks=chunks)
    feats = [
        'pressure_100m',
        'temperature_100m',
        'winddirection_100m',
        'winddirection_80m',
        'windspeed_100m',
        'windspeed_80m',
        'topography',
    ]
    assert loader.data.shape == (400, 8784, len(feats))
    assert sorted(loader.features) == sorted(feats)
    assert all(loader[f].chunksize == chunks for f in feats[:-1])


def test_multi_file_load_nc():
    """Test multi file loading with all features the same shape."""
    with TemporaryDirectory() as td:
        wind_file = os.path.join(td, 'wind.nc')
        make_fake_nc_file(
            wind_file, shape=(10, 10, 20), features=['u_100m', 'v_100m']
        )
        press_file = os.path.join(td, 'press.nc')
        make_fake_nc_file(
            press_file,
            shape=(10, 10, 20),
            features=['pressure_0m', 'pressure_100m'],
        )
        loader = LoaderNC([wind_file, press_file])
        assert loader.shape == (10, 10, 20, 4)


def test_5d_load_nc():
    """Test loading netcdf data with some multi level features. This also
    check renaming of orog -> topography"""
    with TemporaryDirectory() as td:
        wind_file = os.path.join(td, 'wind.nc')
        make_fake_nc_file(
            wind_file,
            shape=(10, 10, 20),
            features=['orog', 'u_100m', 'v_100m'],
        )
        level_file = os.path.join(td, 'wind_levs.nc')
        make_fake_nc_file(
            level_file, shape=(10, 10, 20, 3), features=['zg', 'u']
        )
        loader = LoaderNC([wind_file, level_file])

        assert loader.shape == (10, 10, 20, 3, 5)
        assert sorted(loader.features) == sorted(
            ['topography', 'u_100m', 'v_100m', 'zg', 'u']
        )
        assert loader['u_100m'].shape == (10, 10, 20)
        assert loader['u'].shape == (10, 10, 20, 3)
        assert loader[['u', 'topography']].shape == (10, 10, 20, 3, 2)
        assert loader.data.dtype == np.float32


if __name__ == '__main__':
    execute_pytest(__file__)
