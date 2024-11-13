"""Test single level feature derivations by :class:`Deriver` objects"""

import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from sup3r.preprocessing import Deriver, Rasterizer
from sup3r.preprocessing.derivers.utilities import (
    transform_rotate_wind,
)
from sup3r.utilities.pytest.helpers import make_fake_nc_file
from sup3r.utilities.utilities import xr_open_mfdataset

features = ['windspeed_100m', 'winddirection_100m']
h5_target = (39.01, -105.15)
nc_target = (37.25, -107)
h5_shape = (20, 20)
nc_shape = (10, 10)


def make_5d_nc_file(td, features):
    """Make netcdf file with variables needed for tests. some 4d some 5d."""
    wind_file = os.path.join(td, 'wind.nc')
    make_fake_nc_file(
        wind_file, shape=(60, 60, 100), features=['orog', *features]
    )
    level_file = os.path.join(td, 'wind_levs.nc')
    make_fake_nc_file(level_file, shape=(60, 60, 100, 3), features=['zg', 'u'])
    out_file = os.path.join(td, 'nc_5d.nc')
    xr_open_mfdataset([wind_file, level_file]).to_netcdf(out_file)
    return out_file


@pytest.mark.parametrize(
    ['input_files', 'shape', 'target'], [(None, nc_shape, nc_target)]
)
def test_unneeded_uv_transform(input_files, shape, target):
    """Test that output of deriver is the same as rasterizer when no derivation
    is needed."""

    with TemporaryDirectory() as td:
        if input_files is None:
            input_files = [make_5d_nc_file(td, ['u_100m', 'v_100m'])]
        derive_features = ['U_100m', 'V_100m']
        rasterizer = Rasterizer(input_files[0], target=target, shape=shape)

    # upper case features warning
    with pytest.warns(match='Received some upper case features'):
        deriver = Deriver(rasterizer.data, features=derive_features)

        assert np.array_equal(
            rasterizer['U_100m'].data.compute(),
            deriver['U_100m'].data.compute(),
        )
        assert np.array_equal(
            rasterizer['V_100m'].data.compute(),
            deriver['V_100m'].data.compute(),
        )


@pytest.mark.parametrize(
    ['input_files', 'shape', 'target'],
    [(None, nc_shape, nc_target), (pytest.FPS_WTK, h5_shape, h5_target)],
)
def test_uv_transform(input_files, shape, target):
    """Test that ws/wd -> u/v transform is done correctly"""

    with TemporaryDirectory() as td:
        if input_files is None:
            input_files = [
                make_5d_nc_file(td, ['windspeed_100m', 'winddirection_100m'])
            ]
        derive_features = ['U_100m', 'V_100m']
        rasterizer = Rasterizer(
            input_files[0],
            target=target,
            shape=shape,
        )

    # warning about upper case features
    with pytest.warns(match='Received some upper case features'):
        deriver = Deriver(rasterizer.data, features=derive_features)
    u, v = transform_rotate_wind(
        rasterizer['windspeed_100m'],
        rasterizer['winddirection_100m'],
        rasterizer.lat_lon,
    )
    assert np.array_equal(u, deriver['U_100m'])
    assert np.array_equal(v, deriver['V_100m'])


@pytest.mark.parametrize(
    ['input_files', 'shape', 'target'],
    [(pytest.FPS_WTK, h5_shape, h5_target), (None, nc_shape, nc_target)],
)
def test_hr_coarsening(input_files, shape, target):
    """Test spatial coarsening of the high res field"""

    features = ['windspeed_100m', 'winddirection_100m']
    with TemporaryDirectory() as td:
        if input_files is None:
            input_files = [make_5d_nc_file(td, features=features)]
        rasterizer = Rasterizer(input_files[0], target=target, shape=shape)
    deriver = Deriver(rasterizer.data, features=features, hr_spatial_coarsen=2)
    assert deriver.data.shape == (
        shape[0] // 2,
        shape[1] // 2,
        deriver.data.shape[2],
        len(features),
    )
    assert deriver.lat_lon.shape == (shape[0] // 2, shape[1] // 2, 2)
    assert rasterizer.lat_lon.shape == (shape[0], shape[1], 2)
    assert deriver.data.dtype == np.dtype(np.float32)
