# -*- coding: utf-8 -*-
"""pytests for data handling"""

import os
import tempfile

import numpy as np
import pytest
from rex import Resource

from sup3r import TEST_DATA_DIR
from sup3r.containers.loaders import LoaderH5
from sup3r.containers.wranglers import WranglerH5
from sup3r.utilities.utilities import transform_rotate_wind

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


def ws_wd_transform(self, data):
    """Transform function for wrangler ws/wd -> u/v"""
    data[..., 0], data[..., 1] = transform_rotate_wind(
        ws=data[..., 0], wd=data[..., 1], lat_lon=self.lat_lon
    )
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
    """Test that topography is batched and extracted correctly"""

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
    handler = WranglerH5(
        input_files[0], features, hr_spatial_coarsen=2, **kwargs
    )
    assert handler.data.shape == (
        shape[0] // 2,
        shape[1] // 2,
        handler.data.shape[2],
        len(features),
    )
    assert handler.data.dtype == np.dtype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        cache_pattern = os.path.join(td, 'cached_features_h5')
        if os.path.exists(cache_pattern):
            os.system(f'rm {cache_pattern}')
        handler = WranglerH5(
            input_files[0],
            features,
            hr_spatial_coarsen=2,
            cache_pattern=cache_pattern,
            overwrite_cache=True,
            **kwargs,
        )
        assert handler.data is None
        handler.load_cached_data()
        assert handler.data.shape == (
            shape[0] // 2,
            shape[1] // 2,
            handler.data.shape[2],
            len(features),
        )
        assert handler.data.dtype == np.dtype(np.float32)


def test_data_caching():
    """Test data extraction class with data caching/loading"""

    with tempfile.TemporaryDirectory() as td:
        cache_pattern = os.path.join(td, 'cached_features_h5')
        handler = WranglerH5(
            input_files[0],
            features,
            cache_pattern=cache_pattern,
            overwrite_cache=True,
            **kwargs,
        )

        assert handler.data is None
        handler.load_cached_data()
        assert handler.data.shape == (
            shape[0],
            shape[1],
            handler.data.shape[2],
            len(features),
        )
        assert handler.data.dtype == np.dtype(np.float32)

        # test cache data but keep in memory
        cache_pattern = os.path.join(td, 'new_1_cache')
        handler = WranglerH5(
            input_files[0],
            features,
            cache_pattern=cache_pattern,
            overwrite_cache=True,
            load_cached=True,
            **kwargs,
        )
        assert handler.data is not None
        assert handler.data.dtype == np.dtype(np.float32)

        # test cache data but keep in memory, with no val split
        cache_pattern = os.path.join(td, 'new_2_cache')

        kwargs_new = kwargs.copy()
        kwargs_new['val_split'] = 0
        handler = WranglerH5(
            input_files[0],
            features,
            cache_pattern=cache_pattern,
            overwrite_cache=False,
            load_cached=True,
            **kwargs_new,
        )
        assert handler.data is not None
        assert handler.data.dtype == np.dtype(np.float32)


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
