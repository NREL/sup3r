"""Test pressure and height level interpolation for feature derivations"""

import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from sup3r.preprocessing import Deriver, Rasterizer
from sup3r.utilities.interpolation import Interpolator
from sup3r.utilities.pytest.helpers import make_fake_nc_file


@pytest.mark.parametrize(
    ['shape', 'target', 'height'],
    [
        ((10, 10), (37.25, -107), 20),
        ((10, 10), (37.25, -107), 2),
        ((10, 10), (37.25, -107), 1000),
    ],
)
def test_plevel_height_interp_nc(shape, target, height):
    """Test that variables on pressure levels can be interpolated and
    extrapolated with height correctly"""

    with TemporaryDirectory() as td:
        wind_file = os.path.join(td, 'wind.nc')
        make_fake_nc_file(wind_file, shape=(10, 10, 20), features=['orog'])
        level_file = os.path.join(td, 'wind_levs.nc')
        make_fake_nc_file(
            level_file, shape=(10, 10, 20, 3), features=['zg', 'u']
        )

        derive_features = [f'U_{height}m']
        no_transform = Rasterizer(
            [wind_file, level_file], target=target, shape=shape
        )

        # warning about upper case features
        with pytest.warns():
            transform = Deriver(
                no_transform.data, derive_features, interp_method='linear'
            )

        hgt_array = (
            no_transform['zg'].data
            - no_transform['topography'].data[..., None]
        )
        out = Interpolator.interp_to_level(
            hgt_array, no_transform['u'].data, [np.float32(height)]
        )
    assert transform.data[f'u_{height}m'].data.dtype == np.float32
    assert np.array_equal(out, transform.data[f'u_{height}m'].data)


def test_single_levels_height_interp_nc(shape=(10, 10), target=(37.25, -107)):
    """Test that features can be interpolated from only single level
    variables"""

    with TemporaryDirectory() as td:
        level_file = os.path.join(td, 'wind_levs.nc')
        make_fake_nc_file(
            level_file, shape=(10, 10, 20), features=['u_10m', 'u_100m']
        )

        derive_features = ['u_30m']
        no_transform = Rasterizer([level_file], target=target, shape=shape)

        transform = Deriver(
            no_transform.data, derive_features, interp_method='linear'
        )

    h10 = np.zeros(transform.shape[:3], dtype=np.float32)[..., None]
    h10[:] = 10
    h100 = np.zeros(transform.shape[:3], dtype=np.float32)[..., None]
    h100[:] = 100
    hgt_array = np.concatenate([h10, h100], axis=-1)
    u = np.concatenate(
        [
            no_transform['u_10m'].data[..., None],
            no_transform['u_100m'].data[..., None],
        ],
        axis=-1,
    )
    out = Interpolator.interp_to_level(hgt_array, u, [np.float32(30)])

    assert transform.data['u_30m'].data.dtype == np.float32
    assert np.array_equal(out, transform.data['u_30m'].data)


def test_plevel_height_interp_with_single_lev_data_nc(
    shape=(10, 10), target=(37.25, -107)
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

        derive_features = ['u_100m']
        no_transform = Rasterizer(
            [wind_file, level_file], target=target, shape=shape
        )

        transform = Deriver(no_transform.data, derive_features)

    hgt_array = (
        no_transform['zg'].data - no_transform['topography'].data[..., None]
    )
    h10 = np.zeros(hgt_array.shape[:-1], dtype=np.float32)[..., None]
    h10[:] = 10
    hgt_array = np.concatenate([hgt_array, h10], axis=-1)
    u = np.concatenate(
        [no_transform['u'].data, no_transform['u_10m'].data[..., None]],
        axis=-1,
    )
    out = Interpolator.interp_to_level(hgt_array, u, [np.float32(100)])

    assert transform.data['u_100m'].data.dtype == np.float32
    assert np.array_equal(out, transform.data['u_100m'].data)


def test_log_interp(shape=(10, 10), target=(37.25, -107)):
    """Test that wind is successfully interpolated with log profile when the
    requested height is under 100 meters."""

    with TemporaryDirectory() as td:
        wind_file = os.path.join(td, 'wind.nc')
        make_fake_nc_file(
            wind_file, shape=(10, 10, 20), features=['orog', 'u_10m', 'u_100m']
        )
        level_file = os.path.join(td, 'wind_levs.nc')
        make_fake_nc_file(
            level_file, shape=(10, 10, 20, 3), features=['zg', 'u']
        )

        derive_features = ['u_40m']
        no_transform = Rasterizer(
            [wind_file, level_file], target=target, shape=shape
        )

        transform = Deriver(
            no_transform.data, derive_features, interp_method='log'
        )

    hgt_array = (
        no_transform['zg'].data - no_transform['topography'].data[..., None]
    )
    h10 = np.zeros(hgt_array.shape[:-1], dtype=np.float32)[..., None]
    h10[:] = 10
    h100 = np.zeros(hgt_array.shape[:-1], dtype=np.float32)[..., None]
    h100[:] = 100
    hgt_array = np.concatenate([hgt_array, h10, h100], axis=-1)
    u = np.concatenate(
        [
            no_transform['u'].data,
            no_transform['u_10m'].data[..., None],
            no_transform['u_100m'].data[..., None],
        ],
        axis=-1,
    )
    out = Interpolator.interp_to_level(
        hgt_array, u, [np.float32(40)], interp_method='log'
    )
    assert transform.data['u_40m'].data.dtype == np.float32
    assert np.array_equal(
        np.asarray(out), np.asarray(transform.data['u_40m'].data)
    )
