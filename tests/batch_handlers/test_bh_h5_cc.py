"""pytests for H5 climate change data batch handlers"""

import os

import matplotlib.pyplot as plt
import numpy as np
from rex import init_logger

from sup3r import TEST_DATA_DIR
from sup3r.preprocessing import (
    BatchHandlerCC,
    DataHandlerH5SolarCC,
    DataHandlerH5WindCC,
)
from sup3r.utilities.pytest.helpers import TestDualSamplerCC, execute_pytest
from sup3r.utilities.utilities import nn_fill_array

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


class TestBatchHandlerCC(BatchHandlerCC):
    """Wrapper for tracking observation indices for testing."""

    SAMPLER = TestDualSamplerCC

    @property
    def current_obs_index(self):
        """Track observation index as it is sampled."""
        return self.containers[0].current_obs_index


def test_solar_batching_no_subsample():
    """Test batching of nsrdb data without down sampling to day hours"""
    handler = DataHandlerH5SolarCC(
        INPUT_FILE_S, features=['clearsky_ratio'], **dh_kwargs
    )

    batcher = TestBatchHandlerCC(
        [handler],
        val_containers=[],
        batch_size=1,
        n_batches=10,
        s_enhance=1,
        t_enhance=24,
        means={'clearsky_ratio': 0},
        stds={'clearsky_ratio': 1},
        sample_shape=(20, 20, 72),
        sub_daily_shape=None,
    )

    assert not np.isnan(handler.data.hourly[...]).all()
    assert not np.isnan(handler.data.daily[...]).all()
    for batch in batcher:
        assert batch.high_res.shape[3] == 72
        assert batch.low_res.shape[3] == 3

        # make sure the high res sample is found in the source handler data
        _, hourly_idx = batcher.current_obs_index
        high_res_source = nn_fill_array(
            handler.data.hourly[:, :, hourly_idx[2], :].compute()
        )
        assert np.allclose(batch.high_res[0], high_res_source)

        # make sure the daily avg data corresponds to the high res data slice
        day_start = int(hourly_idx[2].start / 24)
        day_stop = int(hourly_idx[2].stop / 24)
        check = handler.data.daily[:, :, slice(day_start, day_stop)]
        assert np.allclose(batch.low_res[0], check)
    batcher.stop()


def test_solar_batching(plot=False):
    """Make sure batches are coming from correct sample indices."""
    handler = DataHandlerH5SolarCC(
        INPUT_FILE_S, ['clearsky_ratio'], **dh_kwargs
    )

    batcher = TestBatchHandlerCC(
        train_containers=[handler],
        val_containers=[],
        batch_size=1,
        n_batches=10,
        s_enhance=1,
        t_enhance=24,
        means={'clearsky_ratio': 0},
        stds={'clearsky_ratio': 1},
        sample_shape=(20, 20, 72),
        sub_daily_shape=8,
    )

    for batch in batcher:
        assert batch.high_res.shape[3] == 8
        assert batch.low_res.shape[3] == 3

        # make sure the high res sample is found in the source handler data
        found = False
        _, hourly_idx = batcher.current_obs_index
        high_res_source = handler.data.hourly[:, :, hourly_idx[2], :]
        for i in range(high_res_source.shape[2] - 8):
            check = high_res_source[:, :, i : i + 8]
            if np.allclose(batch.high_res[0], check):
                found = True
                break
        assert found

        # make sure the daily avg data corresponds to the high res data slice
        day_start = int(hourly_idx[2].start / 24)
        day_stop = int(hourly_idx[2].stop / 24)
        check = handler.data.daily[:, :, slice(day_start, day_stop)]
        assert np.nansum(batch.low_res - check) == 0
    batcher.stop()

    if plot:
        handler = DataHandlerH5SolarCC(INPUT_FILE_S, FEATURES_S, **dh_kwargs)
        batcher = BatchHandlerCC(
            [handler],
            batch_size=1,
            n_batches=10,
            s_enhance=1,
            sub_daily_shape=8,
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
    dh_kwargs_new = dh_kwargs.copy()
    dh_kwargs_new['sample_shape'] = (20, 20)
    handler = DataHandlerH5SolarCC(INPUT_FILE_S, FEATURES_S, **dh_kwargs_new)

    batcher = BatchHandlerCC(
        [handler], batch_size=8, n_batches=10, s_enhance=2, t_enhance=1
    )

    for batch in batcher:
        assert batch.high_res.shape == (8, 20, 20, 1)
        assert batch.low_res.shape == (8, 10, 10, 1)

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


def test_solar_batch_nan_stats():
    """Test that the batch handler calculates the correct statistics even with
    NaN data present"""
    handler = DataHandlerH5SolarCC(INPUT_FILE_S, FEATURES_S, **dh_kwargs)

    true_csr_mean = np.nanmean(handler.data[..., 0])
    true_csr_stdev = np.nanstd(handler.data[..., 0])

    orig_daily_mean = handler.daily_data[..., 0].mean()

    batcher = BatchHandlerCC(
        [handler], batch_size=1, n_batches=10, s_enhance=1, sub_daily_shape=9
    )

    assert np.allclose(batcher.means[FEATURES_S[0]], true_csr_mean)
    assert np.allclose(batcher.stds[FEATURES_S[0]], true_csr_stdev)

    # make sure the daily means were also normalized by same values
    new = (orig_daily_mean - true_csr_mean) / true_csr_stdev
    assert np.allclose(new, handler.daily_data[..., 0].mean(), atol=1e-4)

    handler1 = DataHandlerH5SolarCC(INPUT_FILE_S, FEATURES_S, **dh_kwargs)

    handler2 = DataHandlerH5SolarCC(INPUT_FILE_S, FEATURES_S, **dh_kwargs)

    batcher = BatchHandlerCC(
        [handler1, handler2],
        batch_size=1,
        n_batches=10,
        s_enhance=1,
        sub_daily_shape=9,
    )

    assert np.allclose(true_csr_mean, batcher.means[FEATURES_S[0]])
    assert np.allclose(true_csr_stdev, batcher.stds[FEATURES_S[0]])


def test_solar_val_data():
    """Validation data is not enabled for solar CC model, test that the batch
    handler does not have validation data."""
    handler = DataHandlerH5SolarCC(INPUT_FILE_S, FEATURES_S, **dh_kwargs)

    batcher = BatchHandlerCC(
        [handler], batch_size=1, n_batches=10, s_enhance=2, sub_daily_shape=8
    )

    n = 0
    for _ in batcher.val_data:
        n += 1

    assert n == 0
    assert not batcher.val_data.any()


def test_solar_multi_day_coarse_data():
    """Test a multi day sample with only 9 hours of high res data output"""
    dh_kwargs_new = dh_kwargs.copy()
    dh_kwargs_new['sample_shape'] = (20, 20, 72)
    handler = DataHandlerH5SolarCC(INPUT_FILE_S, FEATURES_S, **dh_kwargs_new)

    batcher = BatchHandlerCC(
        [handler], batch_size=4, n_batches=10, s_enhance=4, sub_daily_shape=9
    )

    for batch in batcher:
        assert batch.low_res.shape == (4, 5, 5, 3, 1)
        assert batch.high_res.shape == (4, 20, 20, 9, 1)

    for batch in batcher.val_data:
        assert batch.low_res.shape == (4, 5, 5, 3, 1)
        assert batch.high_res.shape == (4, 20, 20, 9, 1)

    # run another test with u/v on low res side but not high res
    features = ['clearsky_ratio', 'u', 'v', 'ghi', 'clearsky_ghi']
    dh_kwargs_new['lr_only_features'] = ['u', 'v']
    handler = DataHandlerH5SolarCC(INPUT_FILE_S, features, **dh_kwargs_new)

    batcher = BatchHandlerCC(
        [handler], batch_size=4, n_batches=10, s_enhance=4, sub_daily_shape=9
    )

    for batch in batcher:
        assert batch.low_res.shape == (4, 5, 5, 3, 3)
        assert batch.high_res.shape == (4, 20, 20, 9, 1)

    for batch in batcher.val_data:
        assert batch.low_res.shape == (4, 5, 5, 3, 3)
        assert batch.high_res.shape == (4, 20, 20, 9, 1)


def test_wind_batching():
    """Test the wind climate change data batching object."""
    dh_kwargs_new = dh_kwargs.copy()
    dh_kwargs_new['target'] = TARGET_W
    dh_kwargs_new['sample_shape'] = (20, 20, 72)
    dh_kwargs_new['val_split'] = 0
    handler = DataHandlerH5WindCC(INPUT_FILE_W, FEATURES_W, **dh_kwargs_new)

    batcher = BatchHandlerCC(
        [handler],
        batch_size=1,
        n_batches=10,
        s_enhance=1,
        sub_daily_shape=None,
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


def test_wind_batching_spatial(plot=False):
    """Test batching of wind data with spatial only enhancement"""
    dh_kwargs_new = dh_kwargs.copy()
    dh_kwargs_new['target'] = TARGET_W
    dh_kwargs_new['sample_shape'] = (20, 20)
    handler = DataHandlerH5WindCC(INPUT_FILE_W, FEATURES_W, **dh_kwargs_new)

    batcher = BatchHandlerCC(
        [handler], batch_size=8, n_batches=10, s_enhance=5, t_enhance=1
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
    dh_kwargs_new['sample_shape'] = (20, 20, 72)
    dh_kwargs_new['val_split'] = 0
    dh_kwargs_new['time_slice'] = slice(None, None, 1)
    dh_kwargs_new['lr_only_features'] = ['*_min_*', '*_max_*']
    handler = DataHandlerH5WindCC(
        INPUT_FILE_SURF, surf_features, **dh_kwargs_new
    )

    batcher = BatchHandlerCC(
        [handler],
        batch_size=1,
        n_batches=10,
        s_enhance=1,
        sub_daily_shape=None,
    )

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


if __name__ == '__main__':
    execute_pytest(__file__)
