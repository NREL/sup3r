# -*- coding: utf-8 -*-
"""pytests for data handling"""

import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from rex import init_logger

from sup3r import TEST_DATA_DIR
from sup3r.containers import (
    Deriver,
    ExtracterNC,
)
from sup3r.utilities.interpolation import Interpolator
from sup3r.utilities.pytest.helpers import execute_pytest, make_fake_nc_file

h5_files = [
    os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5'),
    os.path.join(TEST_DATA_DIR, 'test_wtk_co_2013.h5'),
]
nc_files = [os.path.join(TEST_DATA_DIR, 'test_era5_co_2012.nc')]

features = ['windspeed_100m', 'winddirection_100m']

init_logger('sup3r', log_level='DEBUG')


@pytest.mark.parametrize(
    ['DirectExtracter', 'Deriver', 'shape', 'target'],
    [
        (ExtracterNC, Deriver, (10, 10), (37.25, -107)),
    ],
)
def test_height_interp_nc(DirectExtracter, Deriver, shape, target):
    """Test that variables can be interpolated with height correctly"""

    with TemporaryDirectory() as td:
        wind_file = os.path.join(td, 'wind.nc')
        make_fake_nc_file(wind_file, shape=(10, 10, 20), features=['orog'])
        level_file = os.path.join(td, 'wind_levs.nc')
        make_fake_nc_file(
            level_file, shape=(10, 10, 20, 3), features=['zg', 'u']
        )

        derive_features = ['U_100m']
        no_transform = DirectExtracter(
            [wind_file, level_file], target=target, shape=shape
        )

        transform = Deriver(no_transform.data, derive_features)

        hgt_array = no_transform['zg'] - no_transform['topography'][..., None]
        out = Interpolator.interp_to_level(hgt_array, no_transform['u'], [100])

    assert np.array_equal(out, transform.data['u_100m'])


@pytest.mark.parametrize(
    ['DirectExtracter', 'Deriver', 'shape', 'target'],
    [
        (ExtracterNC, Deriver, (10, 10), (37.25, -107)),
    ],
)
def test_height_interp_with_single_lev_data_nc(
    DirectExtracter, Deriver, shape, target
):
    """Test that variables can be interpolated with height correctly"""

    with TemporaryDirectory() as td:
        wind_file = os.path.join(td, 'wind.nc')
        make_fake_nc_file(
            wind_file, shape=(10, 10, 20), features=['orog', 'u_10m']
        )
        level_file = os.path.join(td, 'wind_levs.nc')
        make_fake_nc_file(
            level_file, shape=(10, 10, 20, 3), features=['zg', 'u']
        )

        derive_features = ['U_100m']
        no_transform = DirectExtracter(
            [wind_file, level_file], target=target, shape=shape
        )

        transform = Deriver(
            no_transform.data,
            derive_features,
        )

    hgt_array = no_transform['zg'] - no_transform['topography'][..., None]
    h10 = np.zeros(hgt_array.shape[:-1])[..., None]
    h10[:] = 10
    hgt_array = np.concatenate([hgt_array, h10], axis=-1)
    u = np.concatenate(
        [no_transform['u'], no_transform['u_10m'][..., None]], axis=-1
    )
    out = Interpolator.interp_to_level(hgt_array, u, [100])

    assert np.array_equal(out, transform.data['u_100m'])


if __name__ == '__main__':
    execute_pytest(__file__)
