# -*- coding: utf-8 -*-
"""pytests for data handling"""

import os

import numpy as np
import pytest
import xarray as xr
from rex import Resource, init_logger

from sup3r import TEST_DATA_DIR
from sup3r.containers.loaders import LoaderH5, LoaderNC
from sup3r.containers.wranglers import WranglerH5, WranglerNC
from sup3r.utilities.interpolation import Interpolator
from sup3r.utilities.utilities import spatial_coarsening, transform_rotate_wind

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


def height_interp(self, data):
    """Interpolate u to u_100m."""
    orog_idx = self.container.features.index('orog')
    zg_idx = self.container.features.index('zg')
    u_idx = self.container.features.index('u')
    zg = data[..., zg_idx]
    orog = data[..., orog_idx]
    u = data[..., u_idx]
    return _height_interp(u, orog, zg)


def ws_wd_transform(self, data):
    """Transform function for wrangler ws/wd -> u/v"""
    data[..., 0], data[..., 1] = transform_rotate_wind(
        ws=data[..., 0], wd=data[..., 1], lat_lon=self.lat_lon
    )
    return data


def coarse_transform(self, data):
    """Corasen high res wrangled data."""
    data = spatial_coarsening(data, s_enhance=2, obs_axis=False)
    self._lat_lon = spatial_coarsening(
        self.lat_lon, s_enhance=2, obs_axis=False
    )
    return data


def test_get_full_domain_nc():
    """Test data handling without target, shape, or raster_file input"""

    wrangler = WranglerNC(LoaderNC(nc_files, features))
    nc_res = xr.open_mfdataset(nc_files)
    shape = (len(nc_res['latitude']), len(nc_res['longitude']))
    target = (
        nc_res['latitude'].values.min(),
        nc_res['longitude'].values.min(),
    )
    assert wrangler.grid_shape == shape
    assert wrangler.target == target


def test_get_target_nc():
    """Test data handling without target or raster_file input"""
    wrangler = WranglerNC(LoaderNC(nc_files, features), shape=(4, 4))
    nc_res = xr.open_mfdataset(nc_files)
    target = (
        nc_res['latitude'].values.min(),
        nc_res['longitude'].values.min(),
    )
    assert wrangler.grid_shape == (4, 4)
    assert wrangler.target == target


@pytest.mark.parametrize(
    ['input_files', 'Loader', 'Wrangler', 'shape', 'target'],
    [
        (h5_files, LoaderH5, WranglerH5, (20, 20), (39.01, -105.15)),
        (nc_files, LoaderNC, WranglerNC, (10, 10), (37.25, -107)),
    ],
)
def test_data_extraction(input_files, Loader, Wrangler, shape, target):
    """Test extraction of raw features"""
    features = ['windspeed_100m', 'winddirection_100m']
    with Loader(input_files[0], features) as loader:
        wrangler = Wrangler(loader, features, target=target, shape=shape)
    assert wrangler.data.shape == (
        shape[0],
        shape[1],
        wrangler.data.shape[2],
        len(features),
    )
    assert wrangler.data.dtype == np.dtype(np.float32)


@pytest.mark.parametrize(
    ['input_files', 'Loader', 'Wrangler', 'shape', 'target'],
    [
        (h5_files, LoaderH5, WranglerH5, (20, 20), (39.01, -105.15)),
        (nc_files, LoaderNC, WranglerNC, (10, 10), (37.25, -107)),
    ],
)
def test_uv_transform(input_files, Loader, Wrangler, shape, target):
    """Test that ws/wd -> u/v transform is done correctly."""

    extract_features = ['U_100m', 'V_100m']
    raw_features = ['windspeed_100m', 'winddirection_100m']
    wrangler_no_transform = Wrangler(
        Loader(input_files[0], features=raw_features),
        raw_features,
        target=target,
        shape=shape,
    )
    wrangler = Wrangler(
        Loader(input_files[0], features=raw_features),
        extract_features,
        target=target,
        shape=shape,
        transform_function=ws_wd_transform,
    )
    out = wrangler_no_transform.data
    u, v = transform_rotate_wind(out[..., 0], out[..., 1], wrangler.lat_lon)
    out = np.concatenate([u[..., None], v[..., None]], axis=-1)
    assert np.array_equal(out, wrangler.data)


def test_topography_h5():
    """Test that topography is extracted correctly"""

    features = ['windspeed_100m', 'elevation']
    with (
        LoaderH5(h5_files[0], features=features) as loader,
        Resource(h5_files[0]) as res,
    ):
        wrangler = WranglerH5(
            loader, features, target=(39.01, -105.15), shape=(20, 20)
        )
        ri = wrangler.raster_index
        topo = res.get_meta_arr('elevation')[(ri.flatten(),)]
        topo = topo.reshape((ri.shape[0], ri.shape[1]))
        topo_idx = wrangler.features.index('elevation')
    assert np.allclose(topo, wrangler.data[..., 0, topo_idx])


@pytest.mark.parametrize(
    ['input_files', 'Loader', 'Wrangler', 'shape', 'target'],
    [
        (nc_files, LoaderNC, WranglerNC, (10, 10), (37.25, -107)),
    ],
)
def test_height_interp_nc(input_files, Loader, Wrangler, shape, target):
    """Test that variables can be interpolated with height correctly"""

    extract_features = ['U_100m']
    raw_features = ['orog', 'zg', 'u']
    wrangler_no_transform = Wrangler(
        Loader(input_files[0], features=raw_features),
        raw_features,
        target=target,
        shape=shape,
    )
    wrangler = Wrangler(
        Loader(input_files[0], features=raw_features),
        extract_features,
        target=target,
        shape=shape,
        transform_function=height_interp,
    )

    out = _height_interp(
        orog=wrangler_no_transform.data[..., 0],
        zg=wrangler_no_transform.data[..., 1],
        u=wrangler_no_transform.data[..., 2],
    )
    assert np.array_equal(out, wrangler.data)


@pytest.mark.parametrize(
    ['input_files', 'Loader', 'Wrangler', 'shape', 'target'],
    [
        (h5_files, LoaderH5, WranglerH5, (20, 20), (39.01, -105.15)),
        (nc_files, LoaderNC, WranglerNC, (10, 10), (37.25, -107)),
    ],
)
def test_hr_coarsening(input_files, Loader, Wrangler, shape, target):
    """Test spatial coarsening of the high res field"""

    features = ['windspeed_100m', 'winddirection_100m']
    with Loader(input_files[0], features) as loader:
        wrangler = Wrangler(
            loader,
            features,
            target=target,
            shape=shape,
            transform_function=coarse_transform,
        )

        assert wrangler.data.shape == (
            shape[0] // 2,
            shape[1] // 2,
            wrangler.data.shape[2],
            len(features),
        )
        assert wrangler.data.dtype == np.dtype(np.float32)


def execute_pytest(capture='all', flags='-rapP'):
    """Execute module as pytest with detailed summary report.

    Parameters
    ----------
    capture : str
        Log or stdout/stderr capture option. ex: log (only logger),
        all (includes stdout/stderr)
    flags : str
        Which tests to show logs and results for.
    """

    fname = os.path.basename(__file__)
    pytest.main(['-q', '--show-capture={}'.format(capture), fname, flags])


if __name__ == '__main__':
    execute_pytest()
