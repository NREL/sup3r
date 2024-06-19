"""Test data handler for netcdf climate change data"""

import os

import numpy as np
import pytest
import xarray as xr
from rex import Resource, init_logger
from scipy.spatial import KDTree

from sup3r import TEST_DATA_DIR
from sup3r.preprocessing import (
    DataHandlerNCforCC,
    DataHandlerNCforCCwithPowerLaw,
    LoaderNC,
)
from sup3r.preprocessing.derivers.methods import UWindPowerLaw
from sup3r.preprocessing.utilities import Dimension
from sup3r.utilities.pytest.helpers import execute_pytest

init_logger('sup3r', log_level='DEBUG')


def test_get_just_coords_nc():
    """Test data handling without features, target, shape, or raster_file
    input"""

    input_files = [os.path.join(TEST_DATA_DIR, 'uas_test.nc')]
    handler = DataHandlerNCforCC(file_paths=input_files, features=[])
    nc_res = LoaderNC(input_files)
    shape = (len(nc_res[Dimension.LATITUDE]), len(nc_res[Dimension.LONGITUDE]))
    target = (
        nc_res[Dimension.LATITUDE].min(),
        nc_res[Dimension.LONGITUDE].min(),
    )
    assert np.array_equal(
        handler.lat_lon[-1, 0, :],
        (
            handler.loader[Dimension.LATITUDE].min(),
            handler.loader[Dimension.LONGITUDE].min(),
        ),
    )
    assert not handler.data_vars
    assert handler.grid_shape == shape
    assert np.array_equal(handler.target, target)


def test_data_handling_nc_cc_power_law(hh=100):
    """Make sure the power law extrapolation of wind operates correctly"""
    input_files = [os.path.join(TEST_DATA_DIR, 'uas_test.nc')]

    with xr.open_mfdataset(input_files) as fh:
        scalar = (hh / UWindPowerLaw.NEAR_SFC_HEIGHT) ** UWindPowerLaw.ALPHA
        u_hh = fh['uas'].values * scalar
        u_hh = np.transpose(u_hh, axes=(1, 2, 0))
        features = [f'u_{hh}m']
        dh = DataHandlerNCforCCwithPowerLaw(input_files, features=features)
        if fh['lat'][-1] > fh['lat'][0]:
            u_hh = u_hh[::-1]
        mask = np.isnan(dh.data[features[0], ...])
        masked_u = dh.data[features[0], ...][~mask].compute_chunk_sizes()
        np.array_equal(masked_u, u_hh[~mask])


def test_data_handling_nc_cc():
    """Make sure the netcdf cc data handler operates correctly"""

    input_files = [
        os.path.join(TEST_DATA_DIR, 'ua_test.nc'),
        os.path.join(TEST_DATA_DIR, 'va_test.nc'),
        os.path.join(TEST_DATA_DIR, 'orog_test.nc'),
        os.path.join(TEST_DATA_DIR, 'zg_test.nc'),
    ]

    with xr.open_mfdataset(input_files) as fh:
        min_lat = np.min(fh.lat.values.astype(np.float32))
        min_lon = np.min(fh.lon.values.astype(np.float32))
        target = (min_lat, min_lon)
        plevel = fh.plev[-1]
        ua = np.transpose(fh['ua'][:, -1, ...].values, (1, 2, 0))
        va = np.transpose(fh['va'][:, -1, ...].values, (1, 2, 0))

    handler = DataHandlerNCforCC(
        input_files,
        features=['U_100m', 'V_100m'],
        target=target,
        shape=(20, 20),
    )
    assert handler.data.shape == (20, 20, 20, 2)

    handler = DataHandlerNCforCC(
        input_files,
        features=[f'U_{int(plevel)}pa', f'V_{int(plevel)}pa'],
        target=target,
        shape=(20, 20),
    )

    assert handler.data.shape == (20, 20, 20, 2)
    assert np.allclose(ua[::-1], handler.data[..., 0])
    assert np.allclose(va[::-1], handler.data[..., 1])


@pytest.mark.parametrize('agg', (1, 4))
def test_solar_cc(agg):
    """Test solar data handling from CC data file with clearsky ratio
    calculated using clearsky ratio from NSRDB h5 file."""

    features = ['clearsky_ratio', 'rsds', 'clearsky_ghi']
    input_files = [os.path.join(TEST_DATA_DIR, 'rsds_test.nc')]
    nsrdb_source_fp = os.path.join(TEST_DATA_DIR, 'test_nsrdb_co_2018.h5')

    with xr.open_mfdataset(input_files) as fh:
        min_lat = np.min(fh.lat.values.astype(np.float32))
        min_lon = np.min(fh.lon.values.astype(np.float32)) - 360
        target = (min_lat, min_lon)
        shape = (len(fh.lat.values), len(fh.lon.values))

    with pytest.raises(AssertionError):
        handler = DataHandlerNCforCC(
            input_files, features=features, target=target, shape=shape
        )

    handler = DataHandlerNCforCC(
        input_files,
        features=features,
        nsrdb_source_fp=nsrdb_source_fp,
        nsrdb_agg=agg,
        target=target,
        shape=shape,
        time_slice=slice(0, 1),
    )

    cs_ratio = handler.data[..., 0]
    ghi = handler.data[..., 1]
    cs_ghi = handler.data[..., 2]
    cs_ratio_truth = ghi / cs_ghi

    assert cs_ratio.max() < 1
    assert cs_ratio.min() > 0
    assert (ghi < cs_ghi).all()
    assert np.allclose(cs_ratio, cs_ratio_truth)

    with Resource(nsrdb_source_fp) as res:
        meta = res.meta
        tree = KDTree(meta[[Dimension.LATITUDE, Dimension.LONGITUDE]])
        cs_ghi_true = res['clearsky_ghi']

    # check a few sites against NSRDB source file
    for i in range(4):
        for j in range(4):
            test_coord = handler.lat_lon[i, j]
            _, inn = tree.query(test_coord, k=agg)

            assert np.allclose(cs_ghi_true[0:48, inn].mean(), cs_ghi[i, j])


if __name__ == '__main__':
    execute_pytest(__file__)
