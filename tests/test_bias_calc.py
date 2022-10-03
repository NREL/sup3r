# -*- coding: utf-8 -*-
"""pytests bias correction calculations"""
import os
import numpy as np
import xarray as xr

from sup3r import TEST_DATA_DIR
from sup3r.bias.bias_calc import LinearCorrection, MonthlyLinearCorrection


FP_NSRDB = os.path.join(TEST_DATA_DIR, 'test_nsrdb_co_2018.h5')
FP_CC = os.path.join(TEST_DATA_DIR, 'rsds_test.nc')


def test_linear_bc():
    """Test linear bias correction"""
    with xr.open_mfdataset(FP_CC) as fh:
        min_lat = np.min(fh.lat.values)
        min_lon = np.min(fh.lon.values) - 360
        target = (min_lat, min_lon)
        shape = (len(fh.lat.values), len(fh.lon.values))

    calc = LinearCorrection(FP_NSRDB, FP_CC, 'ghi', 'rsds',
                            target, shape, bias_handler='DataHandlerNCforCC')

    # test a known in-bounds gid
    bias_gid = 5
    dist, base_gid = calc.get_base_gid(bias_gid, 1)
    bias_data = calc.get_bias_data(bias_gid)
    base_data, base_ti = calc.get_base_data(calc.base_fps, calc.base_dset,
                                            base_gid, calc.base_handler,
                                            daily_avg=True)
    bias_coord = calc.bias_meta.loc[bias_gid, ['latitude', 'longitude']]
    base_coord = calc.base_meta.loc[base_gid, ['latitude', 'longitude']]
    true_dist = bias_coord.values - base_coord.values
    true_dist = np.hypot(true_dist[0], true_dist[1])
    assert np.allclose(true_dist, dist)
    assert true_dist < 0.1
    true_scalar = base_data.std() / bias_data.std()
    true_adder = base_data.mean() - bias_data.mean() * true_scalar

    scalar, adder = calc.run(knn=1, threshold=0.6, fill_extend=False,
                             max_workers=1)

    assert len(scalar.shape) == 3
    assert len(adder.shape) == 3
    assert scalar.shape[-1] == 1
    assert adder.shape[-1] == 1

    iloc = np.where(calc.bias_gid_raster == bias_gid)
    assert np.allclose(true_scalar, scalar[iloc])
    assert np.allclose(true_adder, adder[iloc])

    corners = ((0, 0, 0), (-1, 0, 0), (0, -1, 0), (-1, -1, 0))
    for corner in corners:
        assert np.isnan(scalar[corner])
        assert np.isnan(adder[corner])
    nan_mask = np.isnan(scalar)
    assert np.isnan(adder[nan_mask]).all()

    # make sure the NN fill works for out-of-bounds pixels
    scalar, adder = calc.run(knn=1, threshold=0.6, fill_extend=True,
                             max_workers=1)

    iloc = np.where(calc.bias_gid_raster == bias_gid)
    assert np.allclose(true_scalar, scalar[iloc])
    assert np.allclose(true_adder, adder[iloc])

    assert not np.isnan(scalar[nan_mask]).any()
    assert not np.isnan(adder[nan_mask]).any()

    # make sure smoothing affects the out-of-bounds pixels but not the in-bound
    smooth_scalar, smooth_adder = calc.run(knn=1, threshold=0.6,
                                           fill_extend=True, smooth_extend=2,
                                           max_workers=1)
    assert np.allclose(smooth_scalar[~nan_mask], scalar[~nan_mask])
    assert np.allclose(smooth_adder[~nan_mask], adder[~nan_mask])
    assert not np.allclose(smooth_scalar[nan_mask], scalar[nan_mask])
    assert not np.allclose(smooth_adder[nan_mask], adder[nan_mask])


def test_monthly_linear_bc():
    """Test linear bias correction on a month-by-month basis"""
    with xr.open_mfdataset(FP_CC) as fh:
        min_lat = np.min(fh.lat.values)
        min_lon = np.min(fh.lon.values) - 360
        target = (min_lat, min_lon)
        shape = (len(fh.lat.values), len(fh.lon.values))

    calc = MonthlyLinearCorrection(FP_NSRDB, FP_CC, 'ghi', 'rsds',
                                   target, shape,
                                   bias_handler='DataHandlerNCforCC')

    # test a known in-bounds gid
    bias_gid = 5
    dist, base_gid = calc.get_base_gid(bias_gid, 1)
    bias_data = calc.get_bias_data(bias_gid)
    base_data, base_ti = calc.get_base_data(calc.base_fps, calc.base_dset,
                                            base_gid, calc.base_handler,
                                            daily_avg=True)
    bias_coord = calc.bias_meta.loc[bias_gid, ['latitude', 'longitude']]
    base_coord = calc.base_meta.loc[base_gid, ['latitude', 'longitude']]
    true_dist = bias_coord.values - base_coord.values
    true_dist = np.hypot(true_dist[0], true_dist[1])
    assert np.allclose(true_dist, dist)
    assert true_dist < 0.1
    base_data = base_data[:31]  # just take Jan for testing
    bias_data = bias_data[:31]  # just take Jan for testing
    true_scalar = base_data.std() / bias_data.std()
    true_adder = base_data.mean() - bias_data.mean() * true_scalar

    scalar, adder = calc.run(knn=1, threshold=0.6, fill_extend=True,
                             max_workers=1)

    assert len(scalar.shape) == 3
    assert len(adder.shape) == 3
    assert scalar.shape[-1] == 12
    assert adder.shape[-1] == 12

    iloc = np.where(calc.bias_gid_raster == bias_gid)
    iloc += (0, )
    assert np.allclose(true_scalar, scalar[iloc])
    assert np.allclose(true_adder, adder[iloc])

    last_mon = base_ti.month[-1]
    assert np.isnan(scalar[..., last_mon:]).all()
    assert np.isnan(adder[..., last_mon:]).all()
