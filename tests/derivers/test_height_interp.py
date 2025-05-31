"""Test pressure and height level interpolation for feature derivations"""

import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from scipy.interpolate import interp1d

from sup3r.preprocessing import DataHandler, Deriver, Rasterizer
from sup3r.utilities.interpolation import Interpolator
from sup3r.utilities.pytest.helpers import make_fake_nc_file


@pytest.mark.parametrize(
    ['shape', 'target', 'height', 'chunks'],
    [
        ((10, 10), (37.25, -107), 20, 'auto'),
        ((10, 10), (37.25, -107), 2, 'auto'),
        ((10, 10), (37.25, -107), 1000, 'auto'),
        ((10, 10), (37.25, -107), 0, 'auto'),
        ((10, 10), (37.25, -107), 20, None),
        ((10, 10), (37.25, -107), 2, None),
        ((10, 10), (37.25, -107), 1000, None),
        ((10, 10), (37.25, -107), 0, None),
    ],
)
def test_plevel_height_interp_nc(shape, target, height, chunks):
    """Test that variables on pressure levels can be interpolated and
    extrapolated with height correctly. Also check that chunks=None works with
    height interpolation"""

    with TemporaryDirectory() as td:
        wind_file = os.path.join(td, 'wind.nc')
        make_fake_nc_file(wind_file, shape=(10, 10, 20), features=['orog'])
        level_file = os.path.join(td, 'wind_levs.nc')
        make_fake_nc_file(
            level_file, shape=(10, 10, 20, 3), features=['zg', 'u']
        )

        derive_features = [f'U_{height}m']
        no_transform = Rasterizer(
            [wind_file, level_file], target=target, shape=shape, chunks=chunks
        )

        if chunks is None:
            assert no_transform.loaded

        # warning about upper case features
        with pytest.warns(match='Received some upper case features'):
            transform = Deriver(
                no_transform.data,
                derive_features,
                interp_kwargs={'method': 'linear'},
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


def test_plevel_height_interp_and_derivation():
    """Test that u and v can be interpolated and then used to derive ws"""

    with TemporaryDirectory() as td:
        wind_file = os.path.join(td, 'wind.nc')
        make_fake_nc_file(wind_file, shape=(10, 10, 20), features=['orog'])
        level_file = os.path.join(td, 'wind_levs.nc')
        make_fake_nc_file(
            level_file, shape=(10, 10, 20, 3), features=['zg', 'u', 'v']
        )

        cache_pattern = td + '/{feature}.nc'
        derive_features = ['windspeed_20m']
        ws = DataHandler(
            [wind_file, level_file],
            cache_kwargs={'cache_pattern': cache_pattern},
            features=derive_features,
        )
        derive_features = ['u_20m', 'v_20m']
        uv = DataHandler(
            [wind_file, level_file],
            cache_kwargs={'cache_pattern': cache_pattern},
            features=derive_features,
        )
        ws = ws['windspeed_20m'].values
        ws_uv = np.hypot(uv['u_20m'].values, uv['v_20m'].values)
        assert np.allclose(ws, ws_uv, atol=1e-4)


def test_plevel_height_interp_nc_with_cache():
    """Test that height interpolated data can be cached correctly"""

    with TemporaryDirectory() as td:
        wind_file = os.path.join(td, 'wind.nc')
        make_fake_nc_file(wind_file, shape=(10, 10, 20), features=['orog'])
        level_file = os.path.join(td, 'wind_levs.nc')
        make_fake_nc_file(
            level_file, shape=(10, 10, 20, 3), features=['zg', 'u']
        )

        cache_pattern = td + '/{feature}.nc'
        derive_features = ['u_20m']
        _ = DataHandler(
            [wind_file, level_file],
            cache_kwargs={'cache_pattern': cache_pattern},
            features=derive_features,
        )


def test_plevel_height_interp_with_filtered_load_features():
    """Test that filtering load features can be used to control the features
    used in the derivations."""

    with TemporaryDirectory() as td:
        orog_file = os.path.join(td, 'orog.nc')
        make_fake_nc_file(orog_file, shape=(10, 10, 20), features=['orog'])
        sfc_file = os.path.join(td, 'u_10m.nc')
        make_fake_nc_file(sfc_file, shape=(10, 10, 20), features=['u_10m'])
        level_file = os.path.join(td, 'wind_levs.nc')
        make_fake_nc_file(
            level_file, shape=(10, 10, 20, 3), features=['zg', 'u']
        )
        derive_features = ['u_20m']
        dh_filt = DataHandler(
            [orog_file, sfc_file, level_file],
            features=derive_features,
            load_features=['topography', 'zg', 'u'],
        )
        dh_no_filt = DataHandler(
            [orog_file, sfc_file, level_file],
            features=derive_features,
        )
        dh = DataHandler(
            [orog_file, level_file],
            features=derive_features,
        )
        assert np.array_equal(
            dh_filt.data['u_20m'].data, dh.data['u_20m'].data
        )
        assert not np.array_equal(
            dh_filt.data['u_20m'].data, dh_no_filt.data['u_20m'].data
        )


def test_only_interp_method():
    """Test that interp method alone returns the right values"""
    hgt = np.zeros((10, 10, 5, 3))
    ws = np.zeros((10, 10, 5, 3))
    hgt[..., 0] = 10
    hgt[..., 1] = 40
    hgt[..., 2] = 100
    ws[..., 0] = 5
    ws[..., 1] = 8
    ws[..., 2] = 12

    out = np.asarray(Interpolator.interp_to_level(hgt, ws, [50]))
    func = interp1d(hgt[0, 0, 0], ws[0, 0, 0], fill_value='extrapolate')
    correct = func(50)
    assert np.allclose(out, correct)

    out = np.asarray(Interpolator.interp_to_level(hgt, ws, [60]))
    func = interp1d(hgt[0, 0, 0], ws[0, 0, 0], fill_value='extrapolate')
    correct = func(60)
    assert np.allclose(out, correct)


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
            no_transform.data,
            derive_features,
            interp_kwargs={'method': 'linear'},
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

        transform = Deriver(
            no_transform.data,
            derive_features,
        )

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
            no_transform.data,
            derive_features,
            interp_kwargs={'method': 'log'},
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
        hgt_array,
        u,
        [np.float32(40)],
        interp_kwargs={'method': 'log'},
    )
    assert transform.data['u_40m'].data.dtype == np.float32
    assert np.array_equal(
        np.asarray(out), np.asarray(transform.data['u_40m'].data)
    )
