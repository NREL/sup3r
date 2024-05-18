# -*- coding: utf-8 -*-
"""pytests for data handling"""

import os

import numpy as np
import pytest
import xarray as xr
from rex import Resource, init_logger

from sup3r import TEST_DATA_DIR
from sup3r.containers import ExtracterH5, ExtracterNC, LoaderH5, LoaderNC
from sup3r.utilities.pytest.helpers import execute_pytest

h5_files = [
    os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5'),
    os.path.join(TEST_DATA_DIR, 'test_wtk_co_2013.h5'),
]
nc_files = [os.path.join(TEST_DATA_DIR, 'test_era5_co_2012.nc')]

features = ['windspeed_100m', 'winddirection_100m']

init_logger('sup3r', log_level='DEBUG')


def test_get_full_domain_nc():
    """Test data handling without target, shape, or raster_file input"""

    extracter = ExtracterNC(LoaderNC(nc_files, features))
    nc_res = xr.open_mfdataset(nc_files)
    shape = (len(nc_res['latitude']), len(nc_res['longitude']))
    target = (
        nc_res['latitude'].values.min(),
        nc_res['longitude'].values.min(),
    )
    assert extracter.grid_shape == shape
    assert np.array_equal(extracter.target, target)
    extracter.close()


def test_get_target_nc():
    """Test data handling without target or raster_file input"""
    extracter = ExtracterNC(
        LoaderNC(nc_files, features), shape=(4, 4)
    )
    nc_res = xr.open_mfdataset(nc_files)
    target = (
        nc_res['latitude'].values.min(),
        nc_res['longitude'].values.min(),
    )
    assert extracter.grid_shape == (4, 4)
    assert np.array_equal(extracter.target, target)
    extracter.close()


@pytest.mark.parametrize(
    ['input_files', 'Loader', 'Extracter', 'shape', 'target'],
    [
        (h5_files, LoaderH5, ExtracterH5, (20, 20), (39.01, -105.15)),
        (nc_files, LoaderNC, ExtracterNC, (10, 10), (37.25, -107)),
    ],
)
def test_data_extraction(input_files, Loader, Extracter, shape, target):
    """Test extraction of raw features"""
    features = ['windspeed_100m', 'winddirection_100m']
    extracter = Extracter(
        Loader(input_files[0], features), target=target, shape=shape
    )
    assert extracter.data.shape == (
        shape[0],
        shape[1],
        extracter.data.shape[2],
        len(features),
    )
    assert extracter.data.dtype == np.dtype(np.float32)
    extracter.close()


def test_topography_h5():
    """Test that topography is extracted correctly"""

    features = ['windspeed_100m', 'elevation']
    with Resource(h5_files[0]) as res:
        extracter = ExtracterH5(
            LoaderH5(h5_files[0], features),
            target=(39.01, -105.15),
            shape=(20, 20),
        )
        ri = extracter.raster_index
        topo = res.get_meta_arr('elevation')[(ri.flatten(),)]
        topo = topo.reshape((ri.shape[0], ri.shape[1]))
        topo_idx = extracter.features.index('elevation')
    assert np.allclose(topo, extracter.data[..., 0, topo_idx])


if __name__ == '__main__':
    execute_pytest()
