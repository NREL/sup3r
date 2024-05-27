# -*- coding: utf-8 -*-
"""pytests for data handling"""

import os
from tempfile import TemporaryDirectory

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from rex import init_logger

from sup3r import TEST_DATA_DIR
from sup3r.containers import (
    Deriver,
    DirectExtracterH5,
    DirectExtracterNC,
)
from sup3r.utilities.pytest.helpers import execute_pytest, make_fake_nc_file
from sup3r.utilities.utilities import (
    transform_rotate_wind,
)

h5_files = [
    os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5'),
    os.path.join(TEST_DATA_DIR, 'test_wtk_co_2013.h5'),
]
nc_files = [os.path.join(TEST_DATA_DIR, 'test_era5_co_2012.nc')]

features = ['windspeed_100m', 'winddirection_100m']
h5_target = (39.01, -105.15)
nc_target = (37.25, -107)
h5_shape = (20, 20)
nc_shape = (10, 10)

init_logger('sup3r', log_level='DEBUG')


def make_5d_nc_file(td, features):
    """Make netcdf file with variables needed for tests. some 4d some 5d."""
    wind_file = os.path.join(td, 'wind.nc')
    make_fake_nc_file(
        wind_file, shape=(60, 60, 100), features=['orog', *features]
    )
    level_file = os.path.join(td, 'wind_levs.nc')
    make_fake_nc_file(level_file, shape=(60, 60, 100, 3), features=['zg', 'u'])
    out_file = os.path.join(td, 'nc_5d.nc')
    xr.open_mfdataset([wind_file, level_file]).to_netcdf(out_file)
    return out_file


@pytest.mark.parametrize(
    [
        'input_files',
        'DirectExtracter',
        'Deriver',
        'shape',
        'target',
    ],
    [
        (None, DirectExtracterNC, Deriver, nc_shape, nc_target),
    ],
)
def test_unneeded_uv_transform(
    input_files, DirectExtracter, Deriver, shape, target
):
    """Test that output of deriver is the same as extracter when no derivation
    is needed."""

    with TemporaryDirectory() as td:
        if input_files is None:
            input_files = [make_5d_nc_file(td, ['u_100m', 'v_100m'])]
        derive_features = ['U_100m', 'V_100m']
        extracter = DirectExtracter(
            input_files[0],
            target=target,
            shape=shape,
        )
    deriver = Deriver(extracter.data, features=derive_features)

    assert da.map_blocks(
        lambda x, y: x == y, extracter['U_100m'], deriver['U_100m']
    ).all()
    assert da.map_blocks(
        lambda x, y: x == y, extracter['V_100m'], deriver['V_100m']
    ).all()


@pytest.mark.parametrize(
    [
        'input_files',
        'DirectExtracter',
        'Deriver',
        'shape',
        'target',
    ],
    [
        (None, DirectExtracterNC, Deriver, nc_shape, nc_target),
        (h5_files, DirectExtracterH5, Deriver, h5_shape, h5_target),
    ],
)
def test_uv_transform(input_files, DirectExtracter, Deriver, shape, target):
    """Test that ws/wd -> u/v transform is done correctly"""

    with TemporaryDirectory() as td:
        if input_files is None:
            input_files = [
                make_5d_nc_file(td, ['windspeed_100m', 'winddirection_100m'])
            ]
        derive_features = ['U_100m', 'V_100m']
        extracter = DirectExtracter(
            input_files[0],
            target=target,
            shape=shape,
        )
    deriver = Deriver(extracter.data, features=derive_features)
    u, v = transform_rotate_wind(
        extracter['windspeed_100m'],
        extracter['winddirection_100m'],
        extracter['lat_lon'],
    )
    assert np.array_equal(u, deriver['U_100m'])
    assert np.array_equal(v, deriver['V_100m'])


@pytest.mark.parametrize(
    [
        'input_files',
        'DirectExtracter',
        'Deriver',
        'shape',
        'target',
    ],
    [
        (
            h5_files,
            DirectExtracterH5,
            Deriver,
            h5_shape,
            h5_target,
        ),
        (None, DirectExtracterNC, Deriver, nc_shape, nc_target),
    ],
)
def test_hr_coarsening(input_files, DirectExtracter, Deriver, shape, target):
    """Test spatial coarsening of the high res field"""

    features = ['windspeed_100m', 'winddirection_100m']
    with TemporaryDirectory() as td:
        if input_files is None:
            input_files = [make_5d_nc_file(td, features=features)]
        extracter = DirectExtracter(
            input_files[0],
            target=target,
            shape=shape,
        )
    deriver = Deriver(extracter.data, features=features, hr_spatial_coarsen=2)
    assert deriver.data.shape == (
        shape[0] // 2,
        shape[1] // 2,
        deriver.data.shape[2],
        len(features),
    )
    assert deriver.lat_lon.shape == (shape[0] // 2, shape[1] // 2, 2)
    assert extracter.lat_lon.shape == (shape[0], shape[1], 2)
    assert deriver.data.dtype == np.dtype(np.float32)


if __name__ == '__main__':
    execute_pytest(__file__)
