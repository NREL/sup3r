# -*- coding: utf-8 -*-
"""pytests for data handling with NSRDB files"""

import os
import shutil
import tempfile

import numpy as np
from rex import Outputs, init_logger

from sup3r import TEST_DATA_DIR
from sup3r.preprocessing import (
    DataHandlerH5SolarCC,
    DualSamplerCC,
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
sample_shape = (20, 20, 24)

np.random.seed(42)


init_logger('sup3r', log_level='DEBUG')


def test_solar_handler_sampling(plot=False):
    """Test loading irrad data from NSRDB file and calculating clearsky ratio
    with NaN values for nighttime."""

    handler = DataHandlerH5SolarCC(
            INPUT_FILE_S,
            features=['clearsky_ratio'],
            target=TARGET_S,
            shape=SHAPE,
        )
    assert ['clearsky_ghi', 'ghi'] not in handler
    assert 'clearsky_ratio' in handler

    handler = DataHandlerH5SolarCC(
        INPUT_FILE_S, features=FEATURES_S, **dh_kwargs)
    assert ['clearsky_ghi', 'ghi', 'clearsky_ratio'] in handler

    sampler = DualSamplerCC(handler, sample_shape)

    assert handler.data.shape[2] % 24 == 0
    assert sampler.data.shape[2] % 24 == 0

    # some of the raw clearsky ghi and clearsky ratio data should be loaded in
    # the handler as NaN but the daily data should not have any NaN values
    assert np.isnan(handler.data[...]).any()
    assert np.isnan(sampler.data[...][1]).any()
    assert not np.isnan(handler.daily_data[...]).any()
    assert not np.isnan(sampler.data[...][0]).any()

    for _ in range(10):
        obs_ind_daily, obs_ind_hourly = sampler.get_sample_index()
        assert obs_ind_hourly[2].start / 24 == obs_ind_daily[2].start
        assert obs_ind_hourly[2].stop / 24 == obs_ind_daily[2].stop

        obs_daily, obs_hourly = sampler.get_next()
        assert obs_hourly.shape[2] == 24
        assert obs_daily.shape[2] == 1


'''
        cs_ratio_profile = obs_hourly[0, 0, :, 0]
        assert np.isnan(cs_ratio_profile[0]) & np.isnan(cs_ratio_profile[-1])
        nan_mask = np.isnan(cs_ratio_profile)
        assert all((cs_ratio_profile <= 1)[~nan_mask.compute()])
        assert all((cs_ratio_profile >= 0)[~nan_mask.compute()])
        # new feature engineering so that whenever sunset starts, all
        # clearsky_ratio data is NaN
        for i in range(obs_hourly.shape[2]):
            if np.isnan(obs_hourly[:, :, i, 0]).any():
                assert np.isnan(obs_hourly[:, :, i, 0]).all()

    if plot:
        for p in range(2):
            obs_hourly, obs_daily = sampler.get_next()
            for i in range(obs_hourly.shape[2]):
                _, axes = plt.subplots(1, 2, figsize=(15, 8))

                a = axes[0].imshow(obs_hourly[:, :, i, 0], vmin=0, vmax=1)
                plt.colorbar(a, ax=axes[0])
                axes[0].set_title('Clearsky Ratio')

                tmp = obs_daily[:, :, 0, 0]
                a = axes[1].imshow(tmp, vmin=tmp.min(), vmax=tmp.max())
                plt.colorbar(a, ax=axes[1])
                axes[1].set_title('Daily Average Clearsky Ratio')

                plt.title(i)
                plt.savefig(
                    './test_nsrdb_handler_{}_{}.png'.format(p, i),
                    dpi=300,
                    bbox_inches='tight',
                )
                plt.close()
'''


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
        sampler = DualSamplerCC(handler, sample_shape=sample_shape)
        assert handler.data.shape[2] % 24 == 0

        # some of the raw clearsky ghi and clearsky ratio data should be loaded
        # in the handler as NaN
        assert np.isnan(handler.data).any()

        for _ in range(10):
            obs_ind_hourly, obs_ind_daily = sampler.get_sample_index()
            assert obs_ind_hourly[2].start / 24 == obs_ind_daily[2].start
            assert obs_ind_hourly[2].stop / 24 == obs_ind_daily[2].stop

            obs_hourly, obs_daily = sampler.get_next()
            assert obs_hourly.shape[2] == 24
            assert obs_daily.shape[2] == 1

            for idf in (1, 2):
                msg = f'Wind feature "{features_s[idf]}" got messed up'
                assert not (obs_daily[..., idf] == 0).any(), msg
                assert not (np.abs(obs_daily[..., idf]) > 20).any(), msg


def test_nsrdb_sub_daily_sampler():
    """Test the nsrdb data sampler which does centered sampling on daylight
    hours."""
    handler = DataHandlerH5SolarCC(INPUT_FILE_S, FEATURES_S, **dh_kwargs)
    ti = pd_date_range(
        '20220101',
        '20230101',
        freq='1h',
        inclusive='left',
    )
    ti = ti[0 : len(handler.time_index)]

    for _ in range(100):
        tslice = nsrdb_sub_daily_sampler(handler.data, 4, ti)
        # with only 4 samples, there should never be any NaN data
        assert not np.isnan(handler['clearsky_ratio'][0, 0, tslice]).any()

    for _ in range(100):
        tslice = nsrdb_sub_daily_sampler(handler.data, 8, ti)
        # with only 8 samples, there should never be any NaN data
        assert not np.isnan(handler['clearsky_ratio'][0, 0, tslice]).any()

    for _ in range(100):
        tslice = nsrdb_sub_daily_sampler(handler.data, 20, ti)
        # there should be ~8 hours of non-NaN data
        # the beginning and ending timesteps should be nan
        assert (~np.isnan(handler['clearsky_ratio'][0, 0, tslice])).sum() > 7
        assert np.isnan(handler['clearsky_ratio'][0, 0, tslice])[:3].all()
        assert np.isnan(handler['clearsky_ratio'][0, 0, tslice])[-3:].all()


if __name__ == '__main__':
    test_solar_handler_sampling()
    if False:
        execute_pytest(__file__)
