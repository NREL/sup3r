# -*- coding: utf-8 -*-
"""pytests for data handling"""

import os
import tempfile
from glob import glob

import numpy as np
import pytest
from rex import Resource, init_logger

from sup3r import TEST_DATA_DIR
from sup3r.containers.loaders import LoaderH5
from sup3r.containers.wranglers import WranglerH5
from sup3r.utilities.utilities import spatial_coarsening, transform_rotate_wind

input_files = [
    os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5'),
    os.path.join(TEST_DATA_DIR, 'test_wtk_co_2013.h5'),
]
target = (39.01, -105.15)
shape = (20, 20)
kwargs = {
    'target': target,
    'shape': shape,
    'max_delta': 20,
    'time_slice': slice(None, None, 1),
}
features = ['windspeed_100m', 'winddirection_100m']

init_logger('sup3r', log_level='DEBUG')


def ws_wd_transform(self, data):
    """Transform function for wrangler ws/wd -> u/v"""
    data[..., 0], data[..., 1] = transform_rotate_wind(
        ws=data[..., 0], wd=data[..., 1], lat_lon=self.lat_lon
    )
    return data


def coarse_transform(self, data):
    """Corasen high res wrangled data."""
    data = spatial_coarsening(data, s_enhance=2, obs_axis=False)
    self._lat_lon = spatial_coarsening(self.lat_lon, s_enhance=2,
                                       obs_axis=False)
    return data


def test_data_extraction():
    """Test extraction of raw features"""
    features = ['windspeed_100m', 'winddirection_100m']
    with LoaderH5(input_files[0], features) as loader:
        wrangler = WranglerH5(loader, features, **kwargs)
    assert wrangler.data.shape == (
        shape[0],
        shape[1],
        wrangler.data.shape[2],
        len(features),
    )
    assert wrangler.data.dtype == np.dtype(np.float32)


def test_uv_transform():
    """Test that ws/wd -> u/v transform is done correctly."""

    features = ['U_100m', 'V_100m']
    with LoaderH5(
        input_files[0], features=['windspeed_100m', 'winddirection_100m']
    ) as loader:
        wrangler_no_transform = WranglerH5(loader, features, **kwargs)
        wrangler = WranglerH5(
            loader, features, **kwargs, transform_function=ws_wd_transform
        )
    out = wrangler_no_transform.data
    ws, wd = out[..., 0], out[..., 1]
    u, v = transform_rotate_wind(ws, wd, wrangler.lat_lon)
    assert np.array_equal(u, wrangler.data[..., 0])
    assert np.array_equal(v, wrangler.data[..., 1])


def test_topography():
    """Test that topography is extracted correctly"""

    features = ['windspeed_100m', 'elevation']
    with (
        LoaderH5(input_files[0], features=features) as loader,
        Resource(input_files[0]) as res,
    ):
        wrangler = WranglerH5(loader, features, **kwargs)
        ri = wrangler.raster_index
        topo = res.get_meta_arr('elevation')[(ri.flatten(),)]
        topo = topo.reshape((ri.shape[0], ri.shape[1]))
        topo_idx = wrangler.features.index('elevation')
    assert np.allclose(topo, wrangler.data[..., 0, topo_idx])


def test_raster_index_caching():
    """Test raster index caching by saving file and then loading"""

    # saving raster file
    with tempfile.TemporaryDirectory() as td, LoaderH5(
        input_files[0], features
    ) as loader:
        raster_file = os.path.join(td, 'raster.txt')
        wrangler = WranglerH5(
            loader, features, raster_file=raster_file, **kwargs
        )
        # loading raster file
        wrangler = WranglerH5(
            loader, features, raster_file=raster_file
        )
        assert np.allclose(wrangler.target, target, atol=1)
        assert wrangler.data.shape == (
            shape[0],
            shape[1],
            wrangler.data.shape[2],
            len(features),
        )
        assert wrangler.shape[:2] == (shape[0], shape[1])


def test_hr_coarsening():
    """Test spatial coarsening of the high res field"""

    features = ['windspeed_100m', 'winddirection_100m']
    with LoaderH5(input_files[0], features) as loader:
        wrangler = WranglerH5(
            loader, features, **kwargs, transform_function=coarse_transform
        )

        assert wrangler.data.shape == (
            shape[0] // 2,
            shape[1] // 2,
            wrangler.data.shape[2],
            len(features),
        )
        assert wrangler.data.dtype == np.dtype(np.float32)


def test_data_caching():
    """Test data extraction with caching/loading"""

    with tempfile.TemporaryDirectory() as td:
        cache_pattern = os.path.join(td, 'cached_{feature}.h5')
        with LoaderH5(input_files[0], features) as loader:
            wrangler = WranglerH5(
                loader,
                features,
                cache_kwargs={'cache_pattern': cache_pattern},
                **kwargs,
            )

        assert wrangler.data.shape == (
            shape[0],
            shape[1],
            wrangler.data.shape[2],
            len(features),
        )
        assert wrangler.data.dtype == np.dtype(np.float32)

        loader = LoaderH5(glob(cache_pattern.format(feature='*')), features)

        assert np.array_equal(loader.data, wrangler.data)


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
