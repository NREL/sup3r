"""Test data handler for netcdf climate change data"""
import os

import numpy as np
import pytest
import xarray as xr
from rex import Resource
from scipy.spatial import KDTree

from sup3r import TEST_DATA_DIR
from sup3r.preprocessing.data_handling import (
    DataHandlerNCforCC,
    DataHandlerNCforCCwithPowerLaw,
)
from sup3r.preprocessing.feature_handling import UWindPowerLaw


def test_data_handling_nc_cc_power_law(hh=100):
    """Make sure the power law extrapolation of wind operates correctly"""
    input_files = [os.path.join(TEST_DATA_DIR, 'uas_test.nc')]

    with xr.open_mfdataset(input_files) as fh:
        scalar = (hh / UWindPowerLaw.NEAR_SFC_HEIGHT)**UWindPowerLaw.ALPHA
        u_hh = fh['uas'].values * scalar
        u_hh = np.transpose(u_hh, axes=(1, 2, 0))
        dh = DataHandlerNCforCCwithPowerLaw(input_files, features=[f'u_{hh}m'])
        if dh.invert_lat:
            dh.data = dh.data[::-1]
        mask = np.isnan(dh.data[..., 0])
        assert np.allclose(dh.data[~mask, 0], u_hh[~mask])


def test_data_handling_nc_cc():
    """Make sure the netcdf cc data handler operates correctly"""

    input_files = [
        os.path.join(TEST_DATA_DIR, 'ua_test.nc'),
        os.path.join(TEST_DATA_DIR, 'va_test.nc'),
        os.path.join(TEST_DATA_DIR, 'orog_test.nc'),
        os.path.join(TEST_DATA_DIR, 'zg_test.nc')
    ]

    with xr.open_mfdataset(input_files) as fh:
        min_lat = np.min(fh.lat.values.astype(np.float32))
        min_lon = np.min(fh.lon.values.astype(np.float32))
        target = (min_lat, min_lon)
        plevel = fh.plev[-1]
        ua = np.transpose(fh['ua'][:, -1, ...].values, (1, 2, 0))
        va = np.transpose(fh['va'][:, -1, ...].values, (1, 2, 0))

    handler = DataHandlerNCforCC(input_files,
                                 features=['U_100m', 'V_100m'],
                                 target=target,
                                 shape=(20, 20),
                                 val_split=0.0,
                                 worker_kwargs=dict(max_workers=1))

    assert handler.data.shape == (20, 20, 20, 2)

    handler = DataHandlerNCforCC(
        input_files,
        features=[f'U_{int(plevel)}pa', f'V_{int(plevel)}pa'],
        target=target,
        shape=(20, 20),
        val_split=0.0,
        worker_kwargs=dict(max_workers=1))

    if handler.invert_lat:
        handler.data = handler.data[::-1]
    assert handler.data.shape == (20, 20, 20, 2)
    assert np.allclose(ua, handler.data[..., 0])
    assert np.allclose(va, handler.data[..., 1])


def test_solar_cc():
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
        handler = DataHandlerNCforCC(input_files,
                                     features=features,
                                     target=target,
                                     shape=shape,
                                     val_split=0.0,
                                     worker_kwargs=dict(max_workers=1))

    handler = DataHandlerNCforCC(input_files,
                                 features=features,
                                 nsrdb_source_fp=nsrdb_source_fp,
                                 target=target,
                                 shape=shape,
                                 temporal_slice=slice(0, 1),
                                 val_split=0.0,
                                 worker_kwargs=dict(max_workers=1))

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
        tree = KDTree(meta[['latitude', 'longitude']])
        cs_ghi_true = res['clearsky_ghi']

    # check a few sites against NSRDB source file
    for i in range(4):
        for j in range(4):
            test_coord = handler.lat_lon[i, j]
            _, inn = tree.query(test_coord)

            assert np.allclose(cs_ghi_true[0:48, inn].mean(), cs_ghi[i, j])
