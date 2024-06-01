# -*- coding: utf-8 -*-
"""pytests for data handling"""

import os

import numpy as np
import pytest
import xarray as xr
from rex import Resource, init_logger

from sup3r import TEST_DATA_DIR
from sup3r.preprocessing import ExtracterH5, ExtracterNC
from sup3r.preprocessing.common import Dimension
from sup3r.utilities.pytest.helpers import execute_pytest

h5_files = [
    os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5'),
    os.path.join(TEST_DATA_DIR, 'test_wtk_co_2013.h5'),
]
nc_files = [os.path.join(TEST_DATA_DIR, 'test_era5_co_2012.nc')]

features = ['windspeed_100m', 'winddirection_100m']

init_logger('sup3r', log_level='DEBUG')


def test_get_just_coords_nc():
    """Test data handling without features, target, shape, or raster_file
    input"""

    extracter = ExtracterNC(file_paths=nc_files, features=[])
    nc_res = xr.open_mfdataset(nc_files)
    shape = (len(nc_res[Dimension.LATITUDE]), len(nc_res[Dimension.LONGITUDE]))
    target = (
        nc_res[Dimension.LATITUDE].values.min(),
        nc_res[Dimension.LONGITUDE].values.min(),
    )
    assert np.array_equal(
        extracter.lat_lon[-1, 0, :],
        (
            extracter.loader[Dimension.LATITUDE].min(),
            extracter.loader[Dimension.LONGITUDE].min(),
        ),
    )
    assert extracter.grid_shape == shape
    assert np.array_equal(extracter.target, target)


def test_get_full_domain_nc():
    """Test data handling without target, shape, or raster_file input"""

    extracter = ExtracterNC(file_paths=nc_files)
    nc_res = xr.open_mfdataset(nc_files)
    shape = (len(nc_res[Dimension.LATITUDE]), len(nc_res[Dimension.LONGITUDE]))
    target = (
        nc_res[Dimension.LATITUDE].values.min(),
        nc_res[Dimension.LONGITUDE].values.min(),
    )
    assert np.array_equal(
        extracter.lat_lon[-1, 0, :],
        (
            extracter.loader[Dimension.LATITUDE].min(),
            extracter.loader[Dimension.LONGITUDE].min(),
        ),
    )
    dim_order = (Dimension.LATITUDE, Dimension.LONGITUDE, Dimension.TIME)
    assert np.array_equal(
        extracter['u_100m'],
        nc_res['u_100m'].transpose(*dim_order).data.astype(np.float32),
    )
    assert np.array_equal(
        extracter['v_100m'],
        nc_res['v_100m'].transpose(*dim_order).data.astype(np.float32),
    )
    assert extracter.grid_shape == shape
    assert np.array_equal(extracter.target, target)


def test_get_target_nc():
    """Test data handling without target or raster_file input"""
    extracter = ExtracterNC(file_paths=nc_files, shape=(4, 4))
    nc_res = xr.open_mfdataset(nc_files)
    target = (
        nc_res[Dimension.LATITUDE].values.min(),
        nc_res[Dimension.LONGITUDE].values.min(),
    )
    assert extracter.grid_shape == (4, 4)
    assert np.array_equal(extracter.target, target)


@pytest.mark.parametrize(
    ['input_files', 'Extracter', 'shape', 'target'],
    [
        (
            h5_files,
            ExtracterH5,
            (20, 20),
            (39.01, -105.15),
        ),
        (
            nc_files,
            ExtracterNC,
            (10, 10),
            (37.25, -107),
        ),
    ],
)
def test_data_extraction(input_files, Extracter, shape, target):
    """Test extraction of raw features"""
    extracter = Extracter(
        file_paths=input_files[0],
        target=target,
        shape=shape,
    )
    assert extracter.shape[:3] == (
        shape[0],
        shape[1],
        extracter.shape[2],
    )
    assert extracter.data.dtype == np.dtype(np.float32)


def test_topography_h5():
    """Test that topography is extracted correctly"""

    with Resource(h5_files[0]) as res:
        extracter = ExtracterH5(
            file_paths=h5_files[0],
            target=(39.01, -105.15),
            shape=(20, 20),
            features='topography',
        )
        ri = extracter.raster_index
        topo = res.get_meta_arr('elevation')[(ri.flatten(),)]
        topo = topo.reshape((ri.shape[0], ri.shape[1]))
    assert np.allclose(topo, extracter['topography'])


if __name__ == '__main__':
    execute_pytest(__file__)
