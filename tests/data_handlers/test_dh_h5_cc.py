"""pytests for data handling with NSRDB files"""

import os
import shutil
import tempfile

import numpy as np
import pytest
from rex import Outputs, Resource

from sup3r.preprocessing import (
    DataHandlerH5SolarCC,
    DataHandlerH5WindCC,
)
from sup3r.preprocessing.utilities import lowered
from sup3r.utilities.utilities import RANDOM_GENERATOR

SHAPE = (20, 20)

FEATURES_S = ['clearsky_ratio', 'ghi', 'clearsky_ghi']
TARGET_S = (39.01, -105.13)

FEATURES_W = ['u_100m', 'v_100m', 'temperature_100m']
TARGET_W = (39.01, -105.15)

TARGET_SURF = (39.1, -105.4)

dh_kwargs = {
    'target': TARGET_S,
    'shape': SHAPE,
    'time_slice': slice(None, None, 2),
    'time_roll': -7,
}


def test_daily_handler():
    """Make sure the daily handler is performing averages correctly."""

    dh_kwargs_new = dh_kwargs.copy()
    dh_kwargs_new['target'] = TARGET_W
    handler = DataHandlerH5WindCC(pytest.FP_WTK, FEATURES_W, **dh_kwargs_new)
    daily_og = handler.daily
    tstep = handler.time_slice.step
    daily = handler.hourly.coarsen(time=int(24 / tstep)).mean()

    assert np.array_equal(
        daily[lowered(FEATURES_W)].to_array(),
        daily_og[lowered(FEATURES_W)].to_array(),
    )
    assert handler.hourly.name == 'hourly'
    assert handler.daily.name == 'daily'


def test_solar_handler():
    """Test loading irrad data from NSRDB file and calculating clearsky ratio
    with NaN values for nighttime."""

    handler = DataHandlerH5SolarCC(
        pytest.FP_NSRDB,
        features=['clearsky_ratio'],
        target=TARGET_S,
        shape=SHAPE,
    )
    assert 'clearsky_ratio' in handler
    assert ['clearsky_ghi', 'ghi'] not in handler
    handler = DataHandlerH5SolarCC(
        pytest.FP_NSRDB, features=FEATURES_S, **dh_kwargs
    )

    assert handler.data.shape[2] % 24 == 0

    # some of the raw clearsky ghi and clearsky ratio data should be loaded in
    # the handler as NaN
    assert np.isnan(handler.hourly.as_array()).any()


def test_solar_handler_w_wind():
    """Test loading irrad data from NSRDB file and calculating clearsky ratio
    with NaN values for nighttime. Also test the inclusion of wind features"""

    features_s = ['clearsky_ratio', 'U_200m', 'V_200m', 'ghi', 'clearsky_ghi']

    with tempfile.TemporaryDirectory() as td:
        res_fp = os.path.join(td, 'solar_w_wind.h5')
        shutil.copy(pytest.FP_NSRDB, res_fp)

        with Outputs(res_fp, mode='a') as res:
            res.write_dataset(
                'windspeed_200m',
                RANDOM_GENERATOR.uniform(0, 20, res.shape),
                np.float32,
            )
            res.write_dataset(
                'winddirection_200m',
                RANDOM_GENERATOR.uniform(0, 359.9, res.shape),
                np.float32,
            )

        handler = DataHandlerH5SolarCC(res_fp, features_s, **dh_kwargs)

        assert handler.data.shape[2] % 24 == 0
        assert features_s in handler.data


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
    handler = DataHandlerH5SolarCC(pytest.FP_NSRDB, features, **dh_kwargs)

    assert np.allclose(np.min(handler.hourly['U'][...]), -6.1, atol=1)
    assert np.allclose(np.max(handler.hourly['U'][...]), 9.7, atol=1)

    assert np.allclose(np.min(handler.hourly['V'][...]), -9.8, atol=1)
    assert np.allclose(np.max(handler.hourly['V'][...]), 9.3, atol=1)

    assert np.allclose(
        np.min(handler.hourly['air_temperature'][...]), -18.3, atol=1
    )
    assert np.allclose(
        np.max(handler.hourly['air_temperature'][...]), 22.9, atol=1
    )

    with Resource(pytest.FP_NSRDB) as res:
        ws_source = res['wind_speed']

    ws_true = np.roll(ws_source[::2, 0], -7, axis=0)
    ws_test = np.sqrt(
        handler.hourly['U'][0, 0] ** 2 + handler.hourly['V'][0, 0] ** 2
    )
    assert np.allclose(ws_true, ws_test)

    ws_true = np.roll(ws_source[::2], -7, axis=0)
    ws_true = np.mean(ws_true, axis=1)
    ws_test = np.sqrt(
        handler.hourly['U'][...] ** 2 + handler.hourly['V'][...] ** 2
    )
    ws_test = np.mean(ws_test, axis=(0, 1))
    assert np.allclose(ws_true, ws_test)


def test_wind_handler():
    """Test the wind climate change data handler object."""
    dh_kwargs_new = dh_kwargs.copy()
    dh_kwargs_new['target'] = TARGET_W
    handler = DataHandlerH5WindCC(pytest.FP_WTK, FEATURES_W, **dh_kwargs_new)

    tstep = handler.time_slice.step
    assert handler.data.hourly.shape[2] % (24 // tstep) == 0
    assert not np.isnan(handler.daily.as_array()).any()
    assert handler.daily.shape[2] == handler.hourly.shape[2] / (24 // tstep)
    n_hours = handler.hourly.sizes['time']
    n_days = handler.daily.sizes['time']
    daily_data_slices = [
        slice(x[0], x[-1] + 1)
        for x in np.array_split(np.arange(n_hours), n_days)
    ]
    for i in range(0, n_days, 10):
        islice = daily_data_slices[i]
        hourly = handler.hourly.isel(time=islice)
        truth = hourly.mean(dim='time')
        daily = handler.daily.isel(time=i)
        assert np.allclose(daily.as_array(), truth.as_array(), atol=1e-6)


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
        pytest.FP_WTK_SURF, surf_features, **dh_kwargs_new
    )

    # all of the source hi-res hourly temperature data should be the same
    assert np.allclose(
        handler.hourly[surf_features[0]], handler.hourly[surf_features[2]]
    )
    assert np.allclose(
        handler.hourly[surf_features[0]], handler.hourly[surf_features[3]]
    )
    assert np.allclose(
        handler.hourly[surf_features[1]], handler.hourly[surf_features[4]]
    )
    assert np.allclose(
        handler.hourly[surf_features[1]], handler.hourly[surf_features[5]]
    )
