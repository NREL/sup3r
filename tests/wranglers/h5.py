# -*- coding: utf-8 -*-
"""pytests for data handling"""

import os
import tempfile

import numpy as np
import pytest
import xarray as xr
from rex import Resource

from sup3r import TEST_DATA_DIR
from sup3r.containers.wranglers import WranglerH5 as DataHandlerH5
from sup3r.preprocessing import (
    DataHandlerNC,
)
from sup3r.utilities import utilities

input_files = [
    os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5'),
    os.path.join(TEST_DATA_DIR, 'test_wtk_co_2013.h5'),
]
target = (39.01, -105.15)
shape = (20, 20)
features = ['U_100m', 'V_100m', 'BVF2_200m']
dh_kwargs = {
    'target': target,
    'shape': shape,
    'max_delta': 20,
    'temporal_slice': slice(None, None, 1)
}


def test_topography():
    """Test that topography is batched and extracted correctly"""

    features = ['U_100m', 'V_100m', 'topography']
    data_handler = DataHandlerH5(input_files[0], features, **dh_kwargs)
    ri = data_handler.raster_index
    with Resource(input_files[0]) as res:
        topo = res.get_meta_arr('elevation')[(ri.flatten(),)]
        topo = topo.reshape((ri.shape[0], ri.shape[1]))
    topo_idx = data_handler.features.index('topography')
    assert np.allclose(topo, data_handler.data[..., 0, topo_idx])


def test_data_caching():
    """Test data extraction class with data caching/loading"""

    with tempfile.TemporaryDirectory() as td:
        cache_pattern = os.path.join(td, 'cached_features_h5')
        handler = DataHandlerH5(
            input_files[0],
            features,
            cache_pattern=cache_pattern,
            overwrite_cache=True,
            **dh_kwargs,
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
        handler = DataHandlerH5(
            input_files[0],
            features,
            cache_pattern=cache_pattern,
            overwrite_cache=True,
            load_cached=True,
            **dh_kwargs,
        )
        assert handler.data is not None
        assert handler.data.dtype == np.dtype(np.float32)

        # test cache data but keep in memory, with no val split
        cache_pattern = os.path.join(td, 'new_2_cache')

        dh_kwargs_new = dh_kwargs.copy()
        dh_kwargs_new['val_split'] = 0
        handler = DataHandlerH5(
            input_files[0],
            features,
            cache_pattern=cache_pattern,
            overwrite_cache=False,
            load_cached=True,
            **dh_kwargs_new,
        )
        assert handler.data is not None
        assert handler.data.dtype == np.dtype(np.float32)


def test_netcdf_data_caching():
    """Test caching of extracted data to netcdf files"""

    with tempfile.TemporaryDirectory() as td:
        nc_cache_file = os.path.join(td, 'nc_cache_file.nc')
        if os.path.exists(nc_cache_file):
            os.system(f'rm {nc_cache_file}')
        handler = DataHandlerH5(
            input_files[0],
            features,
            overwrite_cache=True,
            load_cached=True,
            **dh_kwargs,
        )
        target = tuple(handler.lat_lon[-1, 0, :])
        shape = handler.shape
        handler.to_netcdf(nc_cache_file)

        with xr.open_dataset(nc_cache_file) as res:
            assert all(f in res for f in features)

        nc_dh = DataHandlerNC(nc_cache_file, features)

        assert nc_dh.target == target
        assert nc_dh.shape == shape


def test_feature_handler():
    """Make sure compute feature is returning float32"""

    handler = DataHandlerH5(input_files[0], features, **dh_kwargs)
    tmp = handler.run_all_data_init()
    assert tmp.dtype == np.dtype(np.float32)

    vars = {}
    var_names = {
        'temperature_100m': 'T_bottom',
        'temperature_200m': 'T_top',
        'pressure_100m': 'P_bottom',
        'pressure_200m': 'P_top',
    }
    for k, v in var_names.items():
        tmp = handler.extract_feature(
            [input_files[0]], handler.raster_index, k
        )
        assert tmp.dtype == np.dtype(np.float32)
        vars[v] = tmp

    pt_top = utilities.potential_temperature(vars['T_top'], vars['P_top'])
    pt_bottom = utilities.potential_temperature(
        vars['T_bottom'], vars['P_bottom']
    )
    assert pt_top.dtype == np.dtype(np.float32)
    assert pt_bottom.dtype == np.dtype(np.float32)

    pt_diff = utilities.potential_temperature_difference(
        vars['T_top'], vars['P_top'], vars['T_bottom'], vars['P_bottom']
    )
    pt_mid = utilities.potential_temperature_average(
        vars['T_top'], vars['P_top'], vars['T_bottom'], vars['P_bottom']
    )

    assert pt_diff.dtype == np.dtype(np.float32)
    assert pt_mid.dtype == np.dtype(np.float32)

    bvf_squared = utilities.bvf_squared(
        vars['T_top'], vars['T_bottom'], vars['P_top'], vars['P_bottom'], 100
    )
    assert bvf_squared.dtype == np.dtype(np.float32)


def test_raster_index_caching():
    """Test raster index caching by saving file and then loading"""

    # saving raster file
    with tempfile.TemporaryDirectory() as td:
        raster_file = os.path.join(td, 'raster.txt')
        handler = DataHandlerH5(
            input_files[0], features, raster_file=raster_file, **dh_kwargs
        )
        # loading raster file
        handler = DataHandlerH5(
            input_files[0], features, raster_file=raster_file
        )
        assert np.allclose(handler.target, target, atol=1)
        assert handler.data.shape == (
            shape[0],
            shape[1],
            handler.data.shape[2],
            len(features),
        )
        assert handler.grid_shape == (shape[0], shape[1])


def test_data_extraction():
    """Test data extraction class"""
    handler = DataHandlerH5(
        input_files[0], features, **dh_kwargs
    )
    assert handler.data.shape == (
        shape[0],
        shape[1],
        handler.data.shape[2],
        len(features),
    )
    assert handler.data.dtype == np.dtype(np.float32)


def test_hr_coarsening():
    """Test spatial coarsening of the high res field"""
    handler = DataHandlerH5(
        input_files[0], features, hr_spatial_coarsen=2, **dh_kwargs
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
        handler = DataHandlerH5(
            input_files[0],
            features,
            hr_spatial_coarsen=2,
            cache_pattern=cache_pattern,
            overwrite_cache=True,
            **dh_kwargs,
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
