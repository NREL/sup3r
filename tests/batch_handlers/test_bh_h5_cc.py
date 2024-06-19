"""pytests for H5 climate change data batch handlers"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
from rex import init_logger

from sup3r import TEST_DATA_DIR
from sup3r.preprocessing import (
    BatchHandlerCC,
    DataHandlerH5SolarCC,
    DataHandlerH5WindCC,
)
from sup3r.utilities.pytest.helpers import TestBatchHandlerCC, execute_pytest

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


@pytest.mark.parametrize(
    ('hr_tsteps', 't_enhance', 'features'),
    [
        (72, 24, ['clearsky_ratio']),
        (24, 8, ['clearsky_ratio']),
        (72, 24, FEATURES_S),
        (24, 8, FEATURES_S),
    ],
)
def test_solar_batching(hr_tsteps, t_enhance, features, plot=False):
    """Test batching of nsrdb data with and without down sampling to day
    hours"""
    handler = DataHandlerH5SolarCC(
        INPUT_FILE_S, features=features, **dh_kwargs
    )
    batcher = TestBatchHandlerCC(
        [handler],
        val_containers=[],
        batch_size=1,
        n_batches=10,
        s_enhance=1,
        t_enhance=t_enhance,
        means=dict.fromkeys(features, 0),
        stds=dict.fromkeys(features, 1),
        sample_shape=(20, 20, hr_tsteps),
    )

    assert not np.isnan(handler.data.hourly[...]).all()
    assert not np.isnan(handler.data.daily[...]).any()
    high_res_source = handler.data.hourly[...].compute()
    for counter, batch in enumerate(batcher):
        assert batch.high_res.shape[3] == hr_tsteps
        assert batch.low_res.shape[3] == 3

        # make sure the high res sample is found in the source handler data
        daily_idx, hourly_idx = batcher.containers[0].index_record[counter]
        hr_source = high_res_source[:, :, hourly_idx[2], :]
        found = False
        for i in range(hr_source.shape[2] - hr_tsteps + 1):
            check = hr_source[..., i : i + hr_tsteps, :]
            mask = np.isnan(check)
            if np.allclose(batch.high_res[0][~mask], check[~mask]):
                found = True
                break
        assert found

        # make sure the daily avg data corresponds to the high res data slice
        day_start = int(hourly_idx[2].start / 24)
        day_stop = int(hourly_idx[2].stop / 24)
        check = handler.data.daily[:, :, slice(day_start, day_stop)]
        assert np.allclose(batch.low_res[0].numpy(), check)
        check = handler.data.daily[:, :, daily_idx[2]]
        assert np.allclose(batch.low_res[0].numpy(), check)
    batcher.stop()

    if plot:
        handler = DataHandlerH5SolarCC(INPUT_FILE_S, FEATURES_S, **dh_kwargs)
        batcher = BatchHandlerCC(
            [handler],
            [],
            batch_size=1,
            n_batches=10,
            s_enhance=1,
            t_enhance=8,
            sample_shape=(20, 20, 24),
        )
        for p, batch in enumerate(batcher):
            for i in range(batch.high_res.shape[3]):
                _, axes = plt.subplots(1, 4, figsize=(20, 4))

                tmp = (
                    batch.high_res[0, :, :, i, 0] * batcher.stds[0]
                    + batcher.means[0]
                )
                a = axes[0].imshow(tmp, vmin=0, vmax=1)
                plt.colorbar(a, ax=axes[0])
                axes[0].set_title('Batch high res cs ratio')

                tmp = (
                    batch.low_res[0, :, :, 0, 0] * batcher.stds[0]
                    + batcher.means[0]
                )
                a = axes[1].imshow(tmp, vmin=tmp.min(), vmax=tmp.max())
                plt.colorbar(a, ax=axes[1])
                axes[1].set_title('Batch low res cs ratio')

                tmp = (
                    batch.high_res[0, :, :, i, 1] * batcher.stds[1]
                    + batcher.means[1]
                )
                a = axes[2].imshow(tmp, vmin=0, vmax=1100)
                plt.colorbar(a, ax=axes[2])
                axes[2].set_title('GHI')

                tmp = (
                    batch.high_res[0, :, :, i, 2] * batcher.stds[2]
                    + batcher.means[2]
                )
                a = axes[3].imshow(tmp, vmin=0, vmax=1100)
                plt.colorbar(a, ax=axes[3])
                axes[3].set_title('Clear GHI')

                plt.savefig(
                    './test_nsrdb_batch_{}_{}.png'.format(p, i),
                    dpi=300,
                    bbox_inches='tight',
                )
                plt.close()

            if p > 4:
                break


def test_solar_batching_spatial(plot=False):
    """Test batching of nsrdb data with spatial only enhancement"""
    handler = DataHandlerH5SolarCC(INPUT_FILE_S, FEATURES_S, **dh_kwargs)

    batcher = TestBatchHandlerCC(
        [handler],
        val_containers=[],
        batch_size=8,
        n_batches=10,
        s_enhance=2,
        t_enhance=1,
        sample_shape=(20, 20, 1),
        feature_sets={'lr_only_features': ['clearsky_ghi', 'ghi']},
    )

    for batch in batcher:
        assert batch.high_res.shape == (8, 20, 20, 1)
        assert batch.low_res.shape == (8, 10, 10, len(FEATURES_S))

    if plot:
        for p, batch in enumerate(batcher):
            for i in range(batch.high_res.shape[3]):
                _, axes = plt.subplots(1, 2, figsize=(10, 4))

                tmp = (
                    batch.high_res[i, :, :, 0] * batcher.stds[0]
                    + batcher.means[0]
                )
                a = axes[0].imshow(tmp, vmin=tmp.min(), vmax=tmp.max())
                plt.colorbar(a, ax=axes[0])
                axes[0].set_title('Batch high res cs ratio')

                tmp = (
                    batch.low_res[i, :, :, 0] * batcher.stds[0]
                    + batcher.means[0]
                )
                a = axes[1].imshow(tmp, vmin=tmp.min(), vmax=tmp.max())
                plt.colorbar(a, ax=axes[1])
                axes[1].set_title('Batch low res cs ratio')

                plt.savefig(
                    './test_nsrdb_batch_{}_{}.png'.format(p, i),
                    dpi=300,
                    bbox_inches='tight',
                )
                plt.close()

            if p > 4:
                break
    batcher.stop()


def test_solar_batch_nan_stats():
    """Test that the batch handler calculates the correct statistics even with
    NaN data present"""
    handler = DataHandlerH5SolarCC(INPUT_FILE_S, FEATURES_S, **dh_kwargs)

    true_csr_mean = np.nanmean(handler.data.hourly['clearsky_ratio', ...])
    true_csr_stdev = np.nanstd(handler.data.hourly['clearsky_ratio', ...])

    batcher = TestBatchHandlerCC(
        [handler],
        [],
        batch_size=1,
        n_batches=10,
        s_enhance=1,
        t_enhance=24,
        sample_shape=(10, 10, 9),
    )

    assert np.allclose(batcher.means[FEATURES_S[0]], true_csr_mean)
    assert np.allclose(batcher.stds[FEATURES_S[0]], true_csr_stdev)

    batcher = BatchHandlerCC(
        [handler, handler],
        [],
        batch_size=1,
        n_batches=10,
        s_enhance=1,
        t_enhance=24,
        sample_shape=(10, 10, 9),
    )

    assert np.allclose(true_csr_mean, batcher.means[FEATURES_S[0]])
    assert np.allclose(true_csr_stdev, batcher.stds[FEATURES_S[0]])


def test_solar_multi_day_coarse_data():
    """Test a multi day sample with only 9 hours of high res data output"""
    handler = DataHandlerH5SolarCC(INPUT_FILE_S, FEATURES_S, **dh_kwargs)

    batcher = TestBatchHandlerCC(
        train_containers=[handler],
        val_containers=[handler],
        batch_size=4,
        n_batches=10,
        s_enhance=4,
        t_enhance=3,
        sample_shape=(20, 20, 9),
        feature_sets={'lr_only_features': ['clearsky_ghi', 'ghi']},
    )

    for batch in batcher:
        assert batch.low_res.shape == (4, 5, 5, 3, len(FEATURES_S))
        assert batch.high_res.shape == (4, 20, 20, 9, 1)

    for batch in batcher.val_data:
        assert batch.low_res.shape == (4, 5, 5, 3, len(FEATURES_S))
        assert batch.high_res.shape == (4, 20, 20, 9, 1)
    batcher.stop()

    # run another test with u/v on low res side but not high res
    features = ['clearsky_ratio', 'u', 'v', 'ghi', 'clearsky_ghi']
    feature_sets = {'lr_only_features': ['u', 'v', 'clearsky_ghi', 'ghi']}
    handler = DataHandlerH5SolarCC(INPUT_FILE_S, features, **dh_kwargs)

    batcher = TestBatchHandlerCC(
        train_containers=[handler],
        val_containers=[handler],
        batch_size=4,
        n_batches=10,
        s_enhance=4,
        t_enhance=3,
        sample_shape=(20, 20, 9),
        feature_sets=feature_sets,
        mode='eager'
    )

    for batch in batcher:
        assert batch.low_res.shape == (4, 5, 5, 3, len(features))
        assert batch.high_res.shape == (4, 20, 20, 9, 1)

    for batch in batcher.val_data:
        assert batch.low_res.shape == (4, 5, 5, 3, len(features))
        assert batch.high_res.shape == (4, 20, 20, 9, 1)
    batcher.stop()


def test_wind_batching():
    """Test the wind climate change data batching object."""
    dh_kwargs_new = dh_kwargs.copy()
    dh_kwargs_new['target'] = TARGET_W
    dh_kwargs_new['time_slice'] = slice(None)
    handler = DataHandlerH5WindCC(INPUT_FILE_W, FEATURES_W, **dh_kwargs_new)

    batcher = TestBatchHandlerCC(
        [handler],
        [],
        batch_size=1,
        n_batches=10,
        s_enhance=1,
        t_enhance=24,
        sample_shape=(20, 20, 72),
    )

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
    batcher.stop()


def test_wind_batching_spatial(plot=False):
    """Test batching of wind data with spatial only enhancement"""
    dh_kwargs_new = dh_kwargs.copy()
    dh_kwargs_new['target'] = TARGET_W
    dh_kwargs_new['time_slice'] = slice(None)
    handler = DataHandlerH5WindCC(INPUT_FILE_W, FEATURES_W, **dh_kwargs_new)

    batcher = TestBatchHandlerCC(
        [handler],
        [],
        batch_size=8,
        n_batches=10,
        s_enhance=5,
        t_enhance=1,
        sample_shape=(20, 20),
        mode='eager'
    )

    for batch in batcher:
        assert batch.high_res.shape == (8, 20, 20, 3)
        assert batch.low_res.shape == (8, 4, 4, 3)

    if plot:
        for p, batch in enumerate(batcher):
            for i in range(batch.high_res.shape[3]):
                _, axes = plt.subplots(1, 2, figsize=(10, 4))

                tmp = (
                    batch.high_res[i, :, :, 0] * batcher.stds[0]
                    + batcher.means[0]
                )
                a = axes[0].imshow(tmp, vmin=tmp.min(), vmax=tmp.max())
                plt.colorbar(a, ax=axes[0])
                axes[0].set_title('Batch high res cs ratio')

                tmp = (
                    batch.low_res[i, :, :, 0] * batcher.stds[0]
                    + batcher.means[0]
                )
                a = axes[1].imshow(tmp, vmin=tmp.min(), vmax=tmp.max())
                plt.colorbar(a, ax=axes[1])
                axes[1].set_title('Batch low res cs ratio')

                plt.savefig(
                    './test_wind_batch_{}_{}.png'.format(p, i),
                    dpi=300,
                    bbox_inches='tight',
                )
                plt.close()

            if p > 4:
                break
    batcher.stop()


def test_surf_min_max_vars():
    """Test data handling of min / max training only variables"""
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
    dh_kwargs_new['time_slice'] = slice(None, None, 1)
    handler = DataHandlerH5WindCC(
        INPUT_FILE_SURF, surf_features, **dh_kwargs_new
    )

    batcher = TestBatchHandlerCC(
        [handler],
        [],
        batch_size=1,
        n_batches=10,
        s_enhance=1,
        t_enhance=24,
        sample_shape=(20, 20, 72),
        feature_sets={'lr_only_features': ['*_min_*', '*_max_*']},
        mode='eager'
    )

    assert (
        batcher.low_res['temperature_2m'].data
        > batcher.low_res['temperature_min_2m'].data
    ).all()
    assert (
        batcher.low_res['temperature_2m'].data
        < batcher.low_res['temperature_max_2m'].data
    ).all()
    assert (
        batcher.low_res['relativehumidity_2m'].data
        > batcher.low_res['relativehumidity_min_2m'].data
    ).all()
    assert (
        batcher.low_res['relativehumidity_2m'].data
        < batcher.low_res['relativehumidity_max_2m'].data
    ).all()

    assert (
        batcher.means['temperature_2m']
        == batcher.means['temperature_min_2m']
        == batcher.means['temperature_max_2m']
    )
    assert (
        batcher.stds['temperature_2m']
        == batcher.stds['temperature_min_2m']
        == batcher.stds['temperature_max_2m']
    )

    for _, batch in enumerate(batcher):

        assert batch.high_res.shape[3] == 72
        assert batch.low_res.shape[3] == 3

        assert batch.high_res.shape[-1] == len(surf_features) - 4
        assert batch.low_res.shape[-1] == len(surf_features)

        # compare daily avg temp vs min and max
        assert (batch.low_res[..., 0] > batch.low_res[..., 2]).numpy().all()
        assert (batch.low_res[..., 0] < batch.low_res[..., 3]).numpy().all()

        # compare daily avg rh vs min and max
        assert (batch.low_res[..., 1] > batch.low_res[..., 4]).numpy().all()
        assert (batch.low_res[..., 1] < batch.low_res[..., 5]).numpy().all()
    batcher.stop()


if __name__ == '__main__':
    execute_pytest(__file__)
