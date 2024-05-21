# -*- coding: utf-8 -*-
"""pytests for data handling"""

import os
from tempfile import TemporaryDirectory

import dask.array as da
import numpy as np
import pytest
from rex import init_logger

from sup3r import TEST_DATA_DIR
from sup3r.containers import (
    DeriverNC,
    DirectExtracterNC,
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


def _height_interp(u, orog, zg):
    hgt_array = zg - orog
    u_100m = Interpolator.interp_to_level(
        np.transpose(u, axes=(3, 0, 1, 2)),
        np.transpose(hgt_array, axes=(3, 0, 1, 2)),
        levels=[100],
    )[..., None]
    return np.transpose(u_100m, axes=(1, 2, 0, 3))


def height_interp(container):
    """Interpolate u to u_100m."""
    return _height_interp(container['u'], container['orog'], container['zg'])


@pytest.mark.parametrize(
    ['DirectExtracter', 'Deriver', 'shape', 'target'],
    [
        (DirectExtracterNC, DeriverNC, (10, 10), (37.25, -107)),
    ],
)
def test_height_interp_nc(DirectExtracter, Deriver, shape, target):
    """Test that variables can be interpolated with height correctly"""

    with TemporaryDirectory() as td:
        wind_file = os.path.join(td, 'wind.nc')
        make_fake_nc_file(
            wind_file,
            shape=(20, 10, 10),
            features=['orog', 'u_100m', 'v_100m'],
        )
        level_file = os.path.join(td, 'wind_levs.nc')
        make_fake_nc_file(
            level_file, shape=(20, 3, 10, 10), features=['zg', 'u']
        )

        derive_features = ['U_100m']
        raw_features = ['orog', 'zg', 'u']
        no_transform = DirectExtracter(
            [wind_file, level_file],
            target=target,
            shape=shape)

        transform = Deriver(no_transform, derive_features)

        out = _height_interp(
            orog=no_transform['orog'],
            zg=no_transform['zg'],
            u=no_transform['u'],
        )

    assert da.map_blocks(lambda x, y: x == y, out, transform.data).all()


if __name__ == '__main__':
    execute_pytest(__file__)
