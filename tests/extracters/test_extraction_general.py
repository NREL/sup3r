"""Tests across general functionality of :class:`Extracter` objects"""

import numpy as np
import pytest
import xarray as xr
from rex import Resource

from sup3r.preprocessing import ExtracterH5, ExtracterNC
from sup3r.preprocessing.utilities import Dimension

features = ['windspeed_100m', 'winddirection_100m']


def test_get_full_domain_nc():
    """Test data handling without target, shape, or raster_file input"""

    extracter = ExtracterNC(file_paths=pytest.FP_ERA)
    nc_res = xr.open_mfdataset(pytest.FP_ERA)
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

    # raise warning about upper case features
    with pytest.warns():
        assert np.array_equal(
            extracter['U_100m'],
            nc_res['u_100m'].transpose(*dim_order).data.astype(np.float32),
        )
        assert np.array_equal(
            extracter['V_100m'],
            nc_res['v_100m'].transpose(*dim_order).data.astype(np.float32),
        )
    assert extracter.grid_shape == shape
    assert np.array_equal(extracter.target, target)


def test_get_target_nc():
    """Test data handling without target or raster_file input"""
    extracter = ExtracterNC(file_paths=pytest.FP_ERA, shape=(4, 4))
    nc_res = xr.open_mfdataset(pytest.FP_ERA)
    target = (
        nc_res[Dimension.LATITUDE].values.min(),
        nc_res[Dimension.LONGITUDE].values.min(),
    )
    assert extracter.grid_shape == (4, 4)
    assert np.array_equal(extracter.target, target)


@pytest.mark.parametrize(
    ['input_files', 'Extracter', 'shape', 'target'],
    [
        (pytest.FP_WTK, ExtracterH5, (20, 20), (39.01, -105.15)),
        (pytest.FP_ERA, ExtracterNC, (10, 10), (37.25, -107)),
    ],
)
def test_data_extraction(input_files, Extracter, shape, target):
    """Test extraction of raw features"""
    extracter = Extracter(file_paths=input_files, target=target, shape=shape)
    assert extracter.shape[:3] == (shape[0], shape[1], extracter.shape[2])
    assert extracter.data.dtype == np.dtype(np.float32)


def test_topography_h5():
    """Test that topography is extracted correctly"""

    with Resource(pytest.FP_WTK) as res:
        extracter = ExtracterH5(
            file_paths=pytest.FP_WTK, target=(39.01, -105.15), shape=(20, 20)
        )
        ri = extracter.raster_index
        topo = res.get_meta_arr('elevation')[(ri.flatten(),)]
        topo = topo.reshape((ri.shape[0], ri.shape[1]))
    assert np.allclose(topo, extracter['topography', ..., 0])
