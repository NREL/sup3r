# -*- coding: utf-8 -*-
"""pytests for data handling with NSRDB files"""
import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytest
from rex import Outputs, Resource

from sup3r import TEST_DATA_DIR
from sup3r.preprocessing.batch_handling import (
    BatchHandlerCC,
    SpatialBatchHandlerCC,
)
from sup3r.preprocessing.data_handling import (
    DataHandlerH5SolarCC,
    DataHandlerH5WindCC,
)
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

dh_kwargs = dict(target=TARGET_S, shape=SHAPE,
                 temporal_slice=slice(None, None, 2),
                 time_roll=-7, val_split=0.1, sample_shape=(20, 20, 24),
                 worker_kwargs=dict(worker_kwargs=1))


def test_solar_handler(plot=False):
    """Test loading irrad data from NSRDB file and calculating clearsky ratio
    with NaN values for nighttime."""

    with pytest.raises(KeyError):
        handler = DataHandlerH5SolarCC(INPUT_FILE_S, ['clearsky_ratio'],
                                       target=TARGET_S, shape=SHAPE)
    dh_kwargs_new = dh_kwargs.copy()
    dh_kwargs_new['val_split'] = 0
    handler = DataHandlerH5SolarCC(INPUT_FILE_S, FEATURES_S,
                                   **dh_kwargs_new)

    assert handler.data.shape[2] % 24 == 0
    assert handler.val_data is None

    # some of the raw clearsky ghi and clearsky ratio data should be loaded in
    # the handler as NaN
    assert np.isnan(handler.data).any()

    for _ in range(10):
        obs_ind_hourly, obs_ind_daily = handler.get_observation_index()
        assert obs_ind_hourly[2].start / 24 == obs_ind_daily[2].start
        assert obs_ind_hourly[2].stop / 24 == obs_ind_daily[2].stop

        obs_hourly, obs_daily = handler.get_next()
        assert obs_hourly.shape[2] == 24
        assert obs_daily.shape[2] == 1

        cs_ratio_profile = obs_hourly[0, 0, :, 0]
        assert np.isnan(cs_ratio_profile[0]) & np.isnan(cs_ratio_profile[-1])

        nan_mask = np.isnan(cs_ratio_profile)
        assert all((cs_ratio_profile <= 1)[~nan_mask])
        assert all((cs_ratio_profile >= 0)[~nan_mask])

        # new feature engineering so that whenever sunset starts, all
        # clearsky_ratio data is NaN
        for i in range(obs_hourly.shape[2]):
            if np.isnan(obs_hourly[:, :, i, 0]).any():
                assert np.isnan(obs_hourly[:, :, i, 0]).all()

    if plot:
        for p in range(2):
            obs_hourly, obs_daily = handler.get_next()
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
                plt.savefig('./test_nsrdb_handler_{}_{}.png'.format(p, i),
                            dpi=300, bbox_inches='tight')
                plt.close()


def test_solar_handler_w_wind():
    """Test loading irrad data from NSRDB file and calculating clearsky ratio
    with NaN values for nighttime. Also test the inclusion of wind features"""

    features_s = ['clearsky_ratio', 'U_200m', 'V_200m', 'ghi', 'clearsky_ghi']

    with tempfile.TemporaryDirectory() as td:
        res_fp = os.path.join(td, 'solar_w_wind.h5')
        shutil.copy(INPUT_FILE_S, res_fp)

        with Outputs(res_fp, mode='a') as res:
            res.write_dataset('windspeed_200m',
                              np.random.uniform(0, 20, res.shape),
                              np.float32)
            res.write_dataset('winddirection_200m',
                              np.random.uniform(0, 359.9, res.shape),
                              np.float32)

        handler = DataHandlerH5SolarCC(res_fp, features_s, **dh_kwargs)

        assert handler.data.shape[2] % 24 == 0
        assert handler.val_data is None

        # some of the raw clearsky ghi and clearsky ratio data should be loaded
        # in the handler as NaN
        assert np.isnan(handler.data).any()

        for _ in range(10):
            obs_ind_hourly, obs_ind_daily = handler.get_observation_index()
            assert obs_ind_hourly[2].start / 24 == obs_ind_daily[2].start
            assert obs_ind_hourly[2].stop / 24 == obs_ind_daily[2].stop

            obs_hourly, obs_daily = handler.get_next()
            assert obs_hourly.shape[2] == 24
            assert obs_daily.shape[2] == 1

            for idf in (1, 2):
                msg = f'Wind feature "{features_s[idf]}" got messed up'
                assert not (obs_daily[..., idf] == 0).any(), msg
                assert not (np.abs(obs_daily[..., idf]) > 20).any(), msg


def test_solar_batching(plot=False):
    """Test batching of nsrdb data against hand-calc coarsening"""
    dh_kwargs_new = dh_kwargs.copy()
    dh_kwargs_new['sample_shape'] = (20, 20, 72)
    handler = DataHandlerH5SolarCC(INPUT_FILE_S, FEATURES_S,
                                   **dh_kwargs_new)

    batcher = BatchHandlerCC([handler], batch_size=1, n_batches=10,
                             s_enhance=1, sub_daily_shape=8)

    for batch in batcher:
        assert batch.high_res.shape[3] == 8
        assert batch.low_res.shape[3] == 3

        # make sure the high res sample is found in the source handler data
        found = False
        high_res_source = handler.data[:, :, handler.current_obs_index[2], :]
        for i in range(high_res_source.shape[2]):
            check = high_res_source[:, :, i:i + 8, :]
            if np.allclose(batch.high_res, check):
                found = True
                break
        assert found

        # make sure the daily avg data corresponds to the high res data slice
        day_start = int(handler.current_obs_index[2].start / 24)
        day_stop = int(handler.current_obs_index[2].stop / 24)
        check = handler.daily_data[:, :, slice(day_start, day_stop)]
        assert np.allclose(batch.low_res, check)

    if plot:
        handler = DataHandlerH5SolarCC(INPUT_FILE_S, FEATURES_S,
                                       **dh_kwargs)
        batcher = BatchHandlerCC([handler], batch_size=1, n_batches=10,
                                 s_enhance=1, sub_daily_shape=8)
        for p, batch in enumerate(batcher):
            for i in range(batch.high_res.shape[3]):
                _, axes = plt.subplots(1, 4, figsize=(20, 4))

                tmp = (batch.high_res[0, :, :, i, 0] * batcher.stds[0]
                       + batcher.means[0])
                a = axes[0].imshow(tmp, vmin=0, vmax=1)
                plt.colorbar(a, ax=axes[0])
                axes[0].set_title('Batch high res cs ratio')

                tmp = (batch.low_res[0, :, :, 0, 0] * batcher.stds[0]
                       + batcher.means[0])
                a = axes[1].imshow(tmp, vmin=tmp.min(), vmax=tmp.max())
                plt.colorbar(a, ax=axes[1])
                axes[1].set_title('Batch low res cs ratio')

                tmp = (batch.high_res[0, :, :, i, 1] * batcher.stds[1]
                       + batcher.means[1])
                a = axes[2].imshow(tmp, vmin=0, vmax=1100)
                plt.colorbar(a, ax=axes[2])
                axes[2].set_title('GHI')

                tmp = (batch.high_res[0, :, :, i, 2] * batcher.stds[2]
                       + batcher.means[2])
                a = axes[3].imshow(tmp, vmin=0, vmax=1100)
                plt.colorbar(a, ax=axes[3])
                axes[3].set_title('Clear GHI')

                plt.savefig('./test_nsrdb_batch_{}_{}.png'.format(p, i),
                            dpi=300, bbox_inches='tight')
                plt.close()

            if p > 4:
                break


def test_solar_batching_spatial(plot=False):
    """Test batching of nsrdb data with spatial only enhancement"""
    dh_kwargs_new = dh_kwargs.copy()
    dh_kwargs_new['sample_shape'] = (20, 20)
    handler = DataHandlerH5SolarCC(INPUT_FILE_S, FEATURES_S,
                                   **dh_kwargs_new)

    batcher = SpatialBatchHandlerCC([handler], batch_size=8, n_batches=10,
                                    s_enhance=2)

    for batch in batcher:
        assert batch.high_res.shape == (8, 20, 20, 1)
        assert batch.low_res.shape == (8, 10, 10, 1)

    if plot:
        for p, batch in enumerate(batcher):
            for i in range(batch.high_res.shape[3]):
                _, axes = plt.subplots(1, 2, figsize=(10, 4))

                tmp = (batch.high_res[i, :, :, 0] * batcher.stds[0]
                       + batcher.means[0])
                a = axes[0].imshow(tmp, vmin=tmp.min(), vmax=tmp.max())
                plt.colorbar(a, ax=axes[0])
                axes[0].set_title('Batch high res cs ratio')

                tmp = (batch.low_res[i, :, :, 0] * batcher.stds[0]
                       + batcher.means[0])
                a = axes[1].imshow(tmp, vmin=tmp.min(), vmax=tmp.max())
                plt.colorbar(a, ax=axes[1])
                axes[1].set_title('Batch low res cs ratio')

                plt.savefig('./test_nsrdb_batch_{}_{}.png'.format(p, i),
                            dpi=300, bbox_inches='tight')
                plt.close()

            if p > 4:
                break


def test_solar_batch_nan_stats():
    """Test that the batch handler calculates the correct statistics even with
    NaN data present"""
    handler = DataHandlerH5SolarCC(INPUT_FILE_S, FEATURES_S,
                                   **dh_kwargs)

    true_csr_mean = np.nanmean(handler.data[..., 0])
    true_csr_stdev = np.nanstd(handler.data[..., 0])

    orig_daily_mean = handler.daily_data[..., 0].mean()

    batcher = BatchHandlerCC([handler], batch_size=1, n_batches=10,
                             s_enhance=1, sub_daily_shape=9)

    assert np.allclose(batcher.means[FEATURES_S[0]], true_csr_mean)
    assert np.allclose(batcher.stds[FEATURES_S[0]], true_csr_stdev)

    # make sure the daily means were also normalized by same values
    new = (orig_daily_mean - true_csr_mean) / true_csr_stdev
    assert np.allclose(new, handler.daily_data[..., 0].mean(), atol=1e-4)

    handler1 = DataHandlerH5SolarCC(INPUT_FILE_S, FEATURES_S,
                                    **dh_kwargs)

    handler2 = DataHandlerH5SolarCC(INPUT_FILE_S, FEATURES_S,
                                    **dh_kwargs)

    batcher = BatchHandlerCC([handler1, handler2], batch_size=1,
                             n_batches=10, s_enhance=1, sub_daily_shape=9)

    assert np.allclose(true_csr_mean, batcher.means[FEATURES_S[0]])
    assert np.allclose(true_csr_stdev, batcher.stds[FEATURES_S[0]])


def test_solar_val_data():
    """Validation data is not enabled for solar CC model, test that the batch
    handler does not have validation data."""
    handler = DataHandlerH5SolarCC(INPUT_FILE_S, FEATURES_S,
                                   **dh_kwargs)

    batcher = BatchHandlerCC([handler], batch_size=1, n_batches=10,
                             s_enhance=2, sub_daily_shape=8)

    n = 0
    for _ in batcher.val_data:
        n += 1

    assert n == 0
    assert not batcher.val_data.any()


def test_solar_ancillary_vars():
    """Test the handling of the "final" feature set from the NSRDB including
    windspeed components and air temperature near the surface."""
    features = ['clearsky_ratio', 'U', 'V', 'air_temperature', 'ghi',
                'clearsky_ghi']
    dh_kwargs_new = dh_kwargs.copy()
    dh_kwargs_new['val_split'] = 0.001
    handler = DataHandlerH5SolarCC(INPUT_FILE_S, features,
                                   **dh_kwargs_new)

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
    ws_test = np.sqrt(handler.data[0, 0, :, 1]**2
                      + handler.data[0, 0, :, 2]**2)
    assert np.allclose(ws_true, ws_test)

    ws_true = np.roll(ws_source[::2], -7, axis=0)
    ws_true = np.mean(ws_true, axis=1)
    ws_test = np.sqrt(handler.data[..., 1]**2 + handler.data[..., 2]**2)
    ws_test = np.mean(ws_test, axis=(0, 1))
    assert np.allclose(ws_true, ws_test)


def test_nsrdb_sub_daily_sampler():
    """Test the nsrdb data sampler which does centered sampling on daylight
    hours."""
    handler = DataHandlerH5SolarCC(INPUT_FILE_S, FEATURES_S,
                                   **dh_kwargs)
    ti = pd_date_range('20220101', '20230101', freq='1h', inclusive='left')
    ti = ti[0:handler.data.shape[2]]

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
        assert ((~np.isnan(handler.data[0, 0, tslice, 0])).sum() > 7)
        assert np.isnan(handler.data[0, 0, tslice, 0])[:3].all()
        assert np.isnan(handler.data[0, 0, tslice, 0])[-3:].all()


def test_solar_multi_day_coarse_data():
    """Test a multi day sample with only 9 hours of high res data output"""
    dh_kwargs_new = dh_kwargs.copy()
    dh_kwargs_new['sample_shape'] = (20, 20, 72)
    handler = DataHandlerH5SolarCC(INPUT_FILE_S, FEATURES_S,
                                   **dh_kwargs_new)

    batcher = BatchHandlerCC([handler], batch_size=4, n_batches=10,
                             s_enhance=4, sub_daily_shape=9)

    for batch in batcher:
        assert batch.low_res.shape == (4, 5, 5, 3, 1)
        assert batch.high_res.shape == (4, 20, 20, 9, 1)

    for batch in batcher.val_data:
        assert batch.low_res.shape == (4, 5, 5, 3, 1)
        assert batch.high_res.shape == (4, 20, 20, 9, 1)

    # run another test with u/v on low res side but not high res
    features = ['clearsky_ratio', 'u', 'v', 'ghi', 'clearsky_ghi']
    dh_kwargs_new['lr_only_features'] = ['u', 'v']
    handler = DataHandlerH5SolarCC(INPUT_FILE_S, features,
                                   **dh_kwargs_new)

    batcher = BatchHandlerCC([handler], batch_size=4, n_batches=10,
                             s_enhance=4, sub_daily_shape=9)

    for batch in batcher:
        assert batch.low_res.shape == (4, 5, 5, 3, 3)
        assert batch.high_res.shape == (4, 20, 20, 9, 1)

    for batch in batcher.val_data:
        assert batch.low_res.shape == (4, 5, 5, 3, 3)
        assert batch.high_res.shape == (4, 20, 20, 9, 1)


def test_wind_handler():
    """Test the wind climinate change data handler object."""
    dh_kwargs_new = dh_kwargs.copy()
    dh_kwargs_new['target'] = TARGET_W
    handler = DataHandlerH5WindCC(INPUT_FILE_W, FEATURES_W,
                                  **dh_kwargs_new)

    assert handler.data.shape[2] % 24 == 0
    assert handler.val_data is None
    assert not np.isnan(handler.data).any()

    assert handler.daily_data.shape[2] == handler.data.shape[2] / 24

    for i, islice in enumerate(handler.daily_data_slices):
        hourly = handler.data[:, :, islice, :]
        truth = np.mean(hourly, axis=2)
        daily = handler.daily_data[:, :, i, :]
        assert np.allclose(daily, truth, atol=1e-6)


def test_wind_batching():
    """Test the wind climate change data batching object."""
    dh_kwargs_new = dh_kwargs.copy()
    dh_kwargs_new['target'] = TARGET_W
    dh_kwargs_new['sample_shape'] = (20, 20, 72)
    dh_kwargs_new['val_split'] = 0
    handler = DataHandlerH5WindCC(INPUT_FILE_W, FEATURES_W,
                                  **dh_kwargs_new)

    batcher = BatchHandlerCC([handler], batch_size=1, n_batches=10,
                             s_enhance=1, sub_daily_shape=None)

    for batch in batcher:
        assert batch.high_res.shape[3] == 72
        assert batch.low_res.shape[3] == 3

        assert batch.high_res.shape[-1] == len(FEATURES_W)
        assert batch.low_res.shape[-1] == len(FEATURES_W)

        slices = [slice(0, 24), slice(24, 48), slice(48, 72)]
        for i, islice in enumerate(slices):
            hourly = batch.high_res[:, :, :, islice, :]
            truth = np.mean(hourly, axis=3)
            daily = batch.low_res[:, :, :, i, :]
            assert np.allclose(daily, truth, atol=1e-6)


def test_wind_batching_spatial(plot=False):
    """Test batching of wind data with spatial only enhancement"""
    dh_kwargs_new = dh_kwargs.copy()
    dh_kwargs_new['target'] = TARGET_W
    dh_kwargs_new['sample_shape'] = (20, 20)
    handler = DataHandlerH5WindCC(INPUT_FILE_W, FEATURES_W,
                                  **dh_kwargs_new)

    batcher = SpatialBatchHandlerCC([handler], batch_size=8, n_batches=10,
                                    s_enhance=5)

    for batch in batcher:
        assert batch.high_res.shape == (8, 20, 20, 3)
        assert batch.low_res.shape == (8, 4, 4, 3)

    if plot:
        for p, batch in enumerate(batcher):
            for i in range(batch.high_res.shape[3]):
                _, axes = plt.subplots(1, 2, figsize=(10, 4))

                tmp = (batch.high_res[i, :, :, 0] * batcher.stds[0]
                       + batcher.means[0])
                a = axes[0].imshow(tmp, vmin=tmp.min(), vmax=tmp.max())
                plt.colorbar(a, ax=axes[0])
                axes[0].set_title('Batch high res cs ratio')

                tmp = (batch.low_res[i, :, :, 0] * batcher.stds[0]
                       + batcher.means[0])
                a = axes[1].imshow(tmp, vmin=tmp.min(), vmax=tmp.max())
                plt.colorbar(a, ax=axes[1])
                axes[1].set_title('Batch low res cs ratio')

                plt.savefig('./test_wind_batch_{}_{}.png'.format(p, i),
                            dpi=300, bbox_inches='tight')
                plt.close()

            if p > 4:
                break


def test_surf_min_max_vars():
    """Test data handling of min/max training only variables"""
    surf_features = ['temperature_2m',
                     'relativehumidity_2m',
                     'temperature_min_2m',
                     'temperature_max_2m',
                     'relativehumidity_min_2m',
                     'relativehumidity_max_2m']

    dh_kwargs_new = dh_kwargs.copy()
    dh_kwargs_new['target'] = TARGET_SURF
    dh_kwargs_new['sample_shape'] = (20, 20, 72)
    dh_kwargs_new['val_split'] = 0
    dh_kwargs_new['temporal_slice'] = slice(None, None, 1)
    dh_kwargs_new['lr_only_features'] = ['*_min_*', '*_max_*']
    handler = DataHandlerH5WindCC(INPUT_FILE_SURF, surf_features,
                                  **dh_kwargs_new)

    # all of the source hi-res hourly temperature data should be the same
    assert np.allclose(handler.data[..., 0], handler.data[..., 2])
    assert np.allclose(handler.data[..., 0], handler.data[..., 3])
    assert np.allclose(handler.data[..., 1], handler.data[..., 4])
    assert np.allclose(handler.data[..., 1], handler.data[..., 5])

    batcher = BatchHandlerCC([handler], batch_size=1, n_batches=10,
                             s_enhance=1, sub_daily_shape=None)

    for batch in batcher:
        assert batch.high_res.shape[3] == 72
        assert batch.low_res.shape[3] == 3

        assert batch.high_res.shape[-1] == len(surf_features) - 4
        assert batch.low_res.shape[-1] == len(surf_features)

        # compare daily avg temp vs min and max
        assert (batch.low_res[..., 0] > batch.low_res[..., 2]).all()
        assert (batch.low_res[..., 0] < batch.low_res[..., 3]).all()

        # compare daily avg rh vs min and max
        assert (batch.low_res[..., 1] > batch.low_res[..., 4]).all()
        assert (batch.low_res[..., 1] < batch.low_res[..., 5]).all()
