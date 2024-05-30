# -*- coding: utf-8 -*-
"""pytests for data handling with NSRDB files"""

import os
import shutil
import tempfile

import numpy as np
import pytest
from rex import Outputs, Resource, init_logger

from sup3r import TEST_DATA_DIR
from sup3r.preprocessing import (
    DataHandlerH5SolarCC,
    DataHandlerH5WindCC,
)
from sup3r.utilities.pytest.helpers import execute_pytest
from sup3r.utilities.utilities import nsrdb_sub_daily_sampler, pd_date_range

SHAPE = (20, 20)

INPUT_FILE_S = os.path.join(TEST_DATA_DIR, 'test_nsrdb_co_2018.h5')
FEATURES_S = ['clearsky_ratio', 'ghi', 'clearsky_ghi']
TARGET_S = (39.01, -105.13)

INPUT_FILE_W = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
FEATURES_W = ['U_100m', 'V_100m', 'temperature_100m']
TARGET_W = (39.01, -105.15)

INPUT_FILE_SURF = os.path.join(TEST_DATA_DIR, 'test_wtk_surface_vars.h5')
TARGET_SURF = (39.1, -105.4)

dh_kwargs = {
    'target': TARGET_S,
    'shape': SHAPE,
    'time_slice': slice(None, None, 2),
    'time_roll': -7,
}

np.random.seed(42)


init_logger('sup3r', log_level='DEBUG')


def test_solar_handler():
    """Test loading irrad data from NSRDB file and calculating clearsky ratio
    with NaN values for nighttime."""

    with pytest.raises(KeyError):
        handler = DataHandlerH5SolarCC(
            INPUT_FILE_S,
            features=['clearsky_ratio'],
            target=TARGET_S,
            shape=SHAPE,
        )
    dh_kwargs_new = dh_kwargs.copy()
    dh_kwargs_new['val_split'] = 0
    handler = DataHandlerH5SolarCC(
        INPUT_FILE_S, features=FEATURES_S, **dh_kwargs_new
    )

    assert handler.data.shape[2] % 24 == 0

    # some of the raw clearsky ghi and clearsky ratio data should be loaded in
    # the handler as NaN
    assert np.isnan(handler.data).any()


def test_solar_handler_w_wind():
    """Test loading irrad data from NSRDB file and calculating clearsky ratio
    with NaN values for nighttime. Also test the inclusion of wind features"""

    features_s = ['clearsky_ratio', 'U_200m', 'V_200m', 'ghi', 'clearsky_ghi']

    with tempfile.TemporaryDirectory() as td:
        res_fp = os.path.join(td, 'solar_w_wind.h5')
        shutil.copy(INPUT_FILE_S, res_fp)

        with Outputs(res_fp, mode='a') as res:
            res.write_dataset(
                'windspeed_200m',
                np.random.uniform(0, 20, res.shape),
                np.float32,
            )
            res.write_dataset(
                'winddirection_200m',
                np.random.uniform(0, 359.9, res.shape),
                np.float32,
            )

        handler = DataHandlerH5SolarCC(res_fp, features_s, **dh_kwargs)

        assert handler.data.shape[2] % 24 == 0


def test_solar_ancillary_vars():
    """Test the handling of the "final" feature set from the NSRDB including
    windspeed components and air temperature near the surface."""
    features = [
        'clearsky_ratio',
        'U',
        'V',
        'air_temperature',
        'ghi',
        'clearsky_ghi',
    ]
    dh_kwargs_new = dh_kwargs.copy()
    handler = DataHandlerH5SolarCC(INPUT_FILE_S, features, **dh_kwargs_new)

    assert handler.data.shape[-1] == 4

    assert np.allclose(np.min(handler.data[:, :, :, 1]), -6.1, atol=1)
    assert np.allclose(np.max(handler.data[:, :, :, 1]), 9.7, atol=1)

    assert np.allclose(np.min(handler.data[:, :, :, 2]), -9.8, atol=1)
    assert np.allclose(np.max(handler.data[:, :, :, 2]), 9.3, atol=1)

    assert np.allclose(np.min(handler.data[:, :, :, 3]), -18.3, atol=1)
    assert np.allclose(np.max(handler.data[:, :, :, 3]), 22.9, atol=1)

    with Resource(INPUT_FILE_S) as res:
        ws_source = res['wind_speed']

    ws_true = np.roll(ws_source[::2, 0], -7, axis=0)
    ws_test = np.sqrt(
        handler.data[0, 0, :, 1] ** 2 + handler.data[0, 0, :, 2] ** 2
    )
    assert np.allclose(ws_true, ws_test)

    ws_true = np.roll(ws_source[::2], -7, axis=0)
    ws_true = np.mean(ws_true, axis=1)
    ws_test = np.sqrt(handler.data[..., 1] ** 2 + handler.data[..., 2] ** 2)
    ws_test = np.mean(ws_test, axis=(0, 1))
    assert np.allclose(ws_true, ws_test)


def test_nsrdb_sub_daily_sampler():
    """Test the nsrdb data sampler which does centered sampling on daylight
    hours."""
    handler = DataHandlerH5SolarCC(INPUT_FILE_S, FEATURES_S, **dh_kwargs)
    ti = pd_date_range('20220101', '20230101', freq='1h', inclusive='left')
    ti = ti[0 : handler.data.shape[2]]

    for _ in range(100):
        tslice = nsrdb_sub_daily_sampler(handler.data, 4, ti)
        # with only 4 samples, there should never be any NaN data
        assert not np.isnan(handler.data[0, 0, tslice, 0]).any()

    for _ in range(100):
        tslice = nsrdb_sub_daily_sampler(handler.data, 8, ti)
        # with only 8 samples, there should never be any NaN data
        assert not np.isnan(handler.data[0, 0, tslice, 0]).any()

    for _ in range(100):
        tslice = nsrdb_sub_daily_sampler(handler.data, 20, ti)
        # there should be ~8 hours of non-NaN data
        # the beginning and ending timesteps should be nan
        assert (~np.isnan(handler.data[0, 0, tslice, 0])).sum() > 7
        assert np.isnan(handler.data[0, 0, tslice, 0])[:3].all()
        assert np.isnan(handler.data[0, 0, tslice, 0])[-3:].all()


def test_wind_handler():
    """Test the wind climinate change data handler object."""
    dh_kwargs_new = dh_kwargs.copy()
    dh_kwargs_new['target'] = TARGET_W
    handler = DataHandlerH5WindCC(INPUT_FILE_W, FEATURES_W, **dh_kwargs_new)

    assert handler.data.shape[2] % 24 == 0
    assert handler.val_data is None
    assert not np.isnan(handler.data).any()

    assert handler.daily_data.shape[2] == handler.data.shape[2] / 24

    for i, islice in enumerate(handler.daily_data_slices):
        hourly = handler.data[:, :, islice, :]
        truth = np.mean(hourly, axis=2)
        daily = handler.daily_data[:, :, i, :]
        assert np.allclose(daily, truth, atol=1e-6)


def test_surf_min_max_vars():
    """Test data handling of min/max training only variables"""
    surf_features = [
        'temperature_2m',
        'relativehumidity_2m',
        'temperature_min_2m',
        'temperature_max_2m',
        'relativehumidity_min_2m',
        'relativehumidity_max_2m',
    ]

    dh_kwargs_new = dh_kwargs.copy()
    dh_kwargs_new['target'] = TARGET_SURF
    handler = DataHandlerH5WindCC(
        INPUT_FILE_SURF, surf_features, **dh_kwargs_new
    )

    # all of the source hi-res hourly temperature data should be the same
    assert np.allclose(handler.data[..., 0], handler.data[..., 2])
    assert np.allclose(handler.data[..., 0], handler.data[..., 3])
    assert np.allclose(handler.data[..., 1], handler.data[..., 4])
    assert np.allclose(handler.data[..., 1], handler.data[..., 5])


if __name__ == '__main__':
    execute_pytest(__file__)
