"""pytests for data handling with NSRDB files"""

import os
import shutil
import tempfile

import numpy as np
import pytest
from rex import Outputs

from sup3r.preprocessing import (
    DataHandlerH5SolarCC,
    DualSamplerCC,
)
from sup3r.preprocessing.samplers.utilities import nsrdb_sub_daily_sampler
from sup3r.utilities.pytest.helpers import DualSamplerTesterCC
from sup3r.utilities.utilities import RANDOM_GENERATOR, pd_date_range

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
sample_shape = (20, 20, 24)


def test_solar_handler_sampling():
    """Test sampling from solar cc handler for spatiotemporal models."""

    handler = DataHandlerH5SolarCC(
        pytest.FP_NSRDB, features=['clearsky_ratio'], **dh_kwargs
    )
    assert ['clearsky_ghi', 'ghi'] not in handler
    assert 'clearsky_ratio' in handler

    handler = DataHandlerH5SolarCC(
        pytest.FP_NSRDB, features=FEATURES_S, **dh_kwargs
    )
    assert ['clearsky_ghi', 'ghi', 'clearsky_ratio'] in handler

    sampler = DualSamplerTesterCC(
        data=handler.data, sample_shape=sample_shape, batch_size=1
    )

    assert handler.data.shape[2] % 24 == 0
    assert sampler.data.shape[2] % 24 == 0

    # some of the raw clearsky ghi and clearsky ratio data should be loaded in
    # the handler as NaN but the low_res data should not have any NaN values
    assert np.isnan(handler.data.hourly.as_array()).any()
    assert np.isnan(sampler.data.high_res.as_array()).any()
    assert not np.isnan(handler.data.daily.as_array()).any()
    assert not np.isnan(sampler.data.low_res.as_array()).any()

    assert np.array_equal(
        handler.data.daily.as_array(), sampler.data.low_res.as_array()
    )
    assert np.allclose(
        handler.data.hourly.as_array(),
        sampler.data.high_res.as_array(),
        equal_nan=True,
    )

    for i in range(10):
        obs_low_res, obs_high_res = next(sampler)
        assert obs_high_res[0].shape[2] == 24
        assert obs_low_res[0].shape[2] == 1

        obs_ind_low_res, obs_ind_high_res = sampler.index_record[i]
        assert obs_ind_high_res[2].start / 24 == obs_ind_low_res[2].start
        assert obs_ind_high_res[2].stop / 24 == obs_ind_low_res[2].stop

        assert np.array_equal(
            obs_low_res[0], handler.data.daily.sample(obs_ind_low_res)
        )
        mask = np.isnan(handler.data.hourly.sample(obs_ind_high_res).compute())
        assert np.array_equal(
            obs_high_res[0][~mask],
            handler.data.hourly.sample(obs_ind_high_res).compute()[~mask],
        )

        cs_ratio_profile = handler.data.hourly.as_array()[0, 0, :, 0].compute()
        assert np.isnan(cs_ratio_profile[0]) & np.isnan(cs_ratio_profile[-1])
        nan_mask = np.isnan(cs_ratio_profile)
        assert all((cs_ratio_profile <= 1)[~nan_mask])
        assert all((cs_ratio_profile >= 0)[~nan_mask])
        # new feature engineering so that whenever sunset starts, all
        # clearsky_ratio data is NaN
        for i in range(obs_high_res.shape[2]):
            if np.isnan(obs_high_res[:, :, i, 0]).any():
                assert np.isnan(obs_high_res[:, :, i, 0]).all()


def test_solar_handler_sampling_spatial_only():
    """Test sampling from solar cc handler for a spatial only model
    (sample_shape[-1] = 1)"""

    handler = DataHandlerH5SolarCC(
        pytest.FP_NSRDB, features=['clearsky_ratio'], **dh_kwargs
    )

    sampler = DualSamplerTesterCC(
        data=handler.data, sample_shape=(20, 20, 1), t_enhance=1, batch_size=1
    )

    assert handler.data.shape[2] % 24 == 0

    # some of the raw clearsky ghi and clearsky ratio data should be loaded in
    # the handler as NaN but the low_res data should not have any NaN values
    assert np.isnan(handler.data.hourly.as_array()).any()
    assert not np.isnan(sampler.data.high_res.as_array()).any()
    assert not np.isnan(handler.data.daily.as_array()).any()
    assert not np.isnan(sampler.data.low_res.as_array()).any()

    assert np.allclose(
        handler.data.daily.as_array(),
        sampler.data.high_res.as_array(),
    )

    for i in range(10):
        low_res, high_res = next(sampler)
        assert high_res[0].shape[2] == 1
        assert low_res[0].shape[2] == 1

        obs_ind_low_res, obs_ind_high_res = sampler.index_record[i]
        assert obs_ind_high_res[2].start == obs_ind_low_res[2].start
        assert obs_ind_high_res[2].stop == obs_ind_low_res[2].stop

        assert np.array_equal(
            low_res[0], handler.data.daily.sample(obs_ind_low_res)
        )
        assert np.allclose(
            high_res[0], handler.data.daily.sample(obs_ind_low_res)
        )


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
        sampler = DualSamplerCC(
            handler, sample_shape=sample_shape, batch_size=1
        )
        assert handler.data.shape[2] % 24 == 0

        # some of the raw clearsky ghi and clearsky ratio data should be loaded
        # in the handler as NaN
        assert np.isnan(handler.data.hourly[...]).any()

        for _ in range(10):
            obs_ind_daily, obs_ind_hourly = sampler.get_sample_index()
            assert obs_ind_hourly[2].start / 24 == obs_ind_daily[2].start
            assert obs_ind_hourly[2].stop / 24 == obs_ind_daily[2].stop

            obs_daily, obs_hourly = next(sampler)
            assert obs_hourly[0].shape[2] == 24
            assert obs_daily[0].shape[2] == 1

            for idf in (1, 2):
                msg = f'Wind feature "{features_s[idf]}" got messed up'
                assert not (obs_daily[0][..., idf] == 0).any(), msg
                assert not (np.abs(obs_daily[0][..., idf]) > 20).any(), msg


def test_nsrdb_sub_daily_sampler():
    """Test the nsrdb data sampler which does centered sampling on daylight
    hours."""
    handler = DataHandlerH5SolarCC(pytest.FP_NSRDB, FEATURES_S, **dh_kwargs)
    ti = pd_date_range(
        '20220101',
        '20230101',
        freq='1h',
        inclusive='left',
    )
    ti = ti[0 : len(handler.hourly.time_index)]

    for _ in range(20):
        tslice = nsrdb_sub_daily_sampler(handler.hourly, 4, ti)
        # with only 4 samples, there should never be any NaN data
        assert not np.isnan(
            handler.hourly['clearsky_ratio'][0, 0, tslice]
        ).any()

    for _ in range(20):
        tslice = nsrdb_sub_daily_sampler(handler.hourly, 8, ti)
        # with only 8 samples, there should never be any NaN data
        assert not np.isnan(
            handler.hourly['clearsky_ratio'][0, 0, tslice]
        ).any()

    for _ in range(20):
        tslice = nsrdb_sub_daily_sampler(handler.hourly, 20, ti)
        # there should be ~8 hours of non-NaN data
        # the beginning and ending timesteps should be nan
        assert (
            ~np.isnan(handler.hourly['clearsky_ratio'][0, 0, tslice])
        ).sum() > 7
        assert np.isnan(handler.hourly['clearsky_ratio'][0, 0, tslice])[
            :3
        ].all()
        assert np.isnan(handler.hourly['clearsky_ratio'][0, 0, tslice])[
            -3:
        ].all()
