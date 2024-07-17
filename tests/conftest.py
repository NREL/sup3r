"""Global pytest fixtures."""

import os

import pytest
from rex import init_logger

from sup3r import CONFIG_DIR, TEST_DATA_DIR


@pytest.hookimpl
def pytest_configure(config):  # pylint: disable=unused-argument # noqa: ARG001
    """Global pytest config."""
    init_logger('sup3r', log_level='DEBUG')

    pytest.FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
    pytest.FP_NSRDB = os.path.join(TEST_DATA_DIR, 'test_nsrdb_co_2018.h5')
    pytest.FPS_WTK = [
        pytest.FP_WTK,
        os.path.join(TEST_DATA_DIR, 'test_wtk_co_2013.h5'),
    ]
    pytest.FP_WTK_SURF = os.path.join(
        TEST_DATA_DIR, 'test_wtk_surface_vars.h5'
    )
    pytest.FP_ERA = os.path.join(TEST_DATA_DIR, 'test_era5_co_2012.nc')
    pytest.FP_WRF = os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_00_00_00')
    pytest.ST_FP_GEN = os.path.join(
        CONFIG_DIR, 'spatiotemporal', 'gen_3x_4x_2f.json'
    )
    pytest.S_FP_GEN = os.path.join(CONFIG_DIR, 'spatial', 'gen_2x_2f.json')
    pytest.ST_FP_DISC = os.path.join(CONFIG_DIR, 'spatiotemporal', 'disc.json')
    pytest.S_FP_DISC = os.path.join(CONFIG_DIR, 'spatial', 'disc.json')
    pytest.FPS_GCM = [
        os.path.join(TEST_DATA_DIR, 'ua_test.nc'),
        os.path.join(TEST_DATA_DIR, 'va_test.nc'),
        os.path.join(TEST_DATA_DIR, 'orog_test.nc'),
        os.path.join(TEST_DATA_DIR, 'zg_test.nc'),
    ]
    pytest.FP_UAS = os.path.join(TEST_DATA_DIR, 'uas_test.nc')
    pytest.FP_RSDS = os.path.join(TEST_DATA_DIR, 'rsds_test.nc')
