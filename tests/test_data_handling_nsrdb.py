# -*- coding: utf-8 -*-
"""pytests for data handling with NSRDB files"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rex import Resource

from sup3r import TEST_DATA_DIR
from sup3r.preprocessing.data_handling import DataHandlerNsrdb
from sup3r.preprocessing.batch_handling import BatchHandlerNsrdb
from sup3r.utilities.utilities import nsrdb_sampler


INPUT_FILE = os.path.join(TEST_DATA_DIR, 'test_nsrdb_co_2018.h5')
TARGET = (39.01, -105.13)
SHAPE = (20, 20)
FEATURES = ['clearsky_ratio', 'ghi', 'clearsky_ghi']


def coarsen_alternate_calc(batch, s_enhance):
    """alternate calculation to coarsen solar daily data"""
    truth = batch.high_res[0, :, :, :, 0].copy()
    night_mask = np.isnan(batch.high_res[0, :, :, :, 2])
    truth[night_mask] = np.nan
    truth = np.nansum(truth, axis=2) / 24
    if s_enhance > 1:
        truth = truth.reshape(truth.shape[0] // s_enhance, s_enhance,
                              truth.shape[1] // s_enhance, s_enhance,
                              ).sum((1, 3)) / s_enhance**2
    return truth


def test_handler(plot=False):
    """Test loading irrad data from NSRDB file and calculating clearsky ratio
    with NaN values for nighttime."""
    handler = DataHandlerNsrdb(INPUT_FILE, FEATURES,
                               target=TARGET, shape=SHAPE,
                               temporal_slice=slice(None, None, 2),
                               time_roll=-7,
                               val_split=0.1,
                               sample_shape=(20, 20, 24),
                               extract_workers=1,
                               compute_workers=1)

    assert handler.data.shape[2] % 24 == 0
    assert handler.val_data.shape[2] % 24 == 0

    # some of the raw clearsky ghi and clearsky ratio data should be loaded in
    # the handler as NaN
    assert np.isnan(handler.data).any()

    for _ in range(10):
        obs = handler.get_next()
        cs_ratio_profile = obs[0, 0, :, 0]
        cs_ghi_profile = obs[0, 0, :, 2]
        assert np.isnan(cs_ratio_profile[0]) & np.isnan(cs_ratio_profile[-1])
        assert np.isnan(cs_ghi_profile[0]) & np.isnan(cs_ghi_profile[-1])

        nan_mask = np.isnan(cs_ratio_profile)
        assert all((cs_ratio_profile <= 1)[~nan_mask])
        assert all((cs_ratio_profile >= 0)[~nan_mask])

        nan_mask = np.isnan(cs_ghi_profile)
        assert all((cs_ghi_profile <= 1200)[~nan_mask])
        assert all((cs_ghi_profile >= 0)[~nan_mask])

        # new feature engineering so that whenever sunset starts, all
        # clearsky_ratio data is NaN
        for i in range(obs.shape[2]):
            if np.isnan(obs[:, :, i, 0]).any():
                assert np.isnan(obs[:, :, i, 0]).all()
            if (obs[:, :, i, -1] <= 1).any():
                assert np.isnan(obs[:, :, i, 0]).all()

    if plot:
        obs = handler.get_next()
        for i in range(obs.shape[2]):
            _, axes = plt.subplots(1, 3, figsize=(15, 4))

            a = axes[0].imshow(obs[:, :, i, 0], vmin=0, vmax=1)
            plt.colorbar(a, ax=axes[0])
            axes[0].set_title('Clearsky Ratio')

            a = axes[1].imshow(obs[:, :, i, 1], vmin=0, vmax=1100)
            plt.colorbar(a, ax=axes[1])
            axes[1].set_title('GHI')

            a = axes[2].imshow(obs[:, :, i, 2], vmin=0, vmax=1100)
            plt.colorbar(a, ax=axes[2])
            axes[2].set_title('Clearsky GHI')

            plt.title(i)
            plt.savefig('./test_nsrdb_handler_{}.png'.format(i), dpi=300,
                        bbox_inches='tight')
            plt.close()


def test_batching(plot=False):
    """Test batching of nsrdb data against hand-calc coarsening"""
    handler = DataHandlerNsrdb(INPUT_FILE, FEATURES,
                               target=TARGET, shape=SHAPE,
                               temporal_slice=slice(None, None, 2),
                               time_roll=-7,
                               val_split=0.1,
                               sample_shape=(20, 20, 24),
                               extract_workers=1,
                               compute_workers=1)

    batcher = BatchHandlerNsrdb([handler], batch_size=1, n_batches=10,
                                s_enhance=1)

    for batch in batcher:
        truth = coarsen_alternate_calc(batch, 1)
        assert np.allclose(batch.low_res[0, :, :, 0, 0], truth, rtol=1e-4)

    batcher = BatchHandlerNsrdb([handler], batch_size=1, n_batches=10,
                                s_enhance=2)

    for batch in batcher:
        truth = coarsen_alternate_calc(batch, 2)
        assert np.allclose(batch.low_res[0, :, :, 0, 0], truth)

    if plot:
        for batch in batcher:
            truth = coarsen_alternate_calc(batch, 2)
            for i in range(batch.high_res.shape[3]):
                _, axes = plt.subplots(1, 5, figsize=(25, 4))

                a = axes[0].imshow(batch.high_res[0, :, :, i, 0]
                                   * batcher.stds[0] + batcher.means[0],
                                   vmin=0, vmax=1)
                plt.colorbar(a, ax=axes[0])
                axes[0].set_title('Batch high res')

                a = axes[1].imshow(batch.low_res[0, :, :, 0, 0]
                                   * batcher.stds[0] + batcher.means[0],
                                   vmin=0, vmax=1)
                plt.colorbar(a, ax=axes[1])
                axes[1].set_title('Batch low res')

                a = axes[2].imshow(truth
                                   * batcher.stds[0] + batcher.means[0],
                                   vmin=0, vmax=1)
                plt.colorbar(a, ax=axes[2])
                axes[2].set_title('Hand calc low res')

                a = axes[3].imshow(batch.high_res[0, :, :, i, 1]
                                   * batcher.stds[1] + batcher.means[1],
                                   vmin=0, vmax=1100)
                plt.colorbar(a, ax=axes[3])
                axes[3].set_title('GHI')

                a = axes[4].imshow(batch.high_res[0, :, :, i, 2]
                                   * batcher.stds[2] + batcher.means[2],
                                   vmin=0, vmax=1100)
                plt.colorbar(a, ax=axes[4])
                axes[4].set_title('Clear GHI')

                plt.savefig('./test_nsrdb_batch_{}.png'.format(i), dpi=300,
                            bbox_inches='tight')
                plt.close()


def test_batch_nan_stats():
    """Test that the batch handler calculates the correct statistics even with
    NaN data present"""
    handler = DataHandlerNsrdb(INPUT_FILE, FEATURES,
                               target=TARGET, shape=SHAPE,
                               temporal_slice=slice(None, None, 2),
                               time_roll=-7,
                               val_split=0.1,
                               sample_shape=(20, 20, 12),
                               max_extract_workers=1,
                               max_compute_workers=1)

    true_means = [np.nanmean(handler.data[..., i])
                  for i in range(len(FEATURES))]
    true_stdevs = [np.nanstd(handler.data[..., i])
                   for i in range(len(FEATURES))]

    batcher = BatchHandlerNsrdb([handler], batch_size=1, n_batches=10,
                                s_enhance=1)

    assert np.allclose(true_means, batcher.means)
    assert np.allclose(true_stdevs, batcher.stds)

    handler1 = DataHandlerNsrdb(INPUT_FILE, FEATURES,
                                target=TARGET, shape=SHAPE,
                                temporal_slice=slice(None, None, 2),
                                time_roll=-7,
                                val_split=0.1,
                                sample_shape=(20, 20, 12),
                                max_extract_workers=1,
                                max_compute_workers=1)

    handler2 = DataHandlerNsrdb(INPUT_FILE, FEATURES,
                                target=TARGET, shape=SHAPE,
                                temporal_slice=slice(None, None, 2),
                                time_roll=-7,
                                val_split=0.1,
                                sample_shape=(20, 20, 12),
                                max_extract_workers=1,
                                max_compute_workers=1)

    batcher = BatchHandlerNsrdb([handler1, handler2], batch_size=1,
                                n_batches=10, s_enhance=1)

    assert np.allclose(true_means, batcher.means)
    assert np.allclose(true_stdevs, batcher.stds)


def test_val_data():
    """Test basic properties of the nsrdb validation dataset"""
    handler = DataHandlerNsrdb(INPUT_FILE, FEATURES,
                               target=TARGET, shape=SHAPE,
                               temporal_slice=slice(None, None, 2),
                               time_roll=-7,
                               val_split=0.1,
                               sample_shape=(20, 20, 24),
                               extract_workers=1,
                               compute_workers=1)

    batcher = BatchHandlerNsrdb([handler],
                                batch_size=1, n_batches=10,
                                s_enhance=2, t_enhance=24,
                                temporal_coarsening_method='average')

    for batch in batcher.val_data:

        obs = batch.high_res[0]
        cs_ratio_profile = obs[0, 0, :, 0] * batcher.stds[0] + batcher.means[0]
        cs_ghi_profile = obs[0, 0, :, 2] * batcher.stds[2] + batcher.means[2]
        assert np.isnan(cs_ghi_profile[0]) & np.isnan(cs_ghi_profile[-1])

        nan_mask = np.isnan(cs_ratio_profile)
        assert all((cs_ratio_profile <= 1)[~nan_mask])
        assert all((cs_ratio_profile >= 0)[~nan_mask])

        nan_mask = np.isnan(cs_ghi_profile)
        assert all((cs_ghi_profile <= 1200)[~nan_mask])
        assert all((cs_ghi_profile >= 0)[~nan_mask])

        truth = coarsen_alternate_calc(batch, 2)
        assert np.allclose(batch.low_res[0, :, :, 0, 0], truth)


def test_ancillary_vars():
    """Test the handling of the "final" feature set from the NSRDB including
    windspeed components and air temperature near the surface."""
    features = ['clearsky_ratio', 'U', 'V', 'air_temperature']
    handler = DataHandlerNsrdb(INPUT_FILE, features,
                               target=TARGET, shape=SHAPE,
                               temporal_slice=slice(None, None, 2),
                               time_roll=-7,
                               val_split=0.0,
                               sample_shape=(20, 20, 24),
                               extract_workers=1,
                               compute_workers=1)

    assert handler.data.shape[-1] == 4

    assert np.allclose(np.min(handler.data[:, :, :, 1]), -9.3, atol=1)
    assert np.allclose(np.max(handler.data[:, :, :, 1]), 9.7, atol=1)

    assert np.allclose(np.min(handler.data[:, :, :, 2]), -9.8, atol=1)
    assert np.allclose(np.max(handler.data[:, :, :, 2]), 6.1, atol=1)

    assert np.allclose(np.min(handler.data[:, :, :, 3]), -18.3, atol=1)
    assert np.allclose(np.max(handler.data[:, :, :, 3]), 22.9, atol=1)

    with Resource(INPUT_FILE) as res:
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


def test_nsrdb_sampler():
    """Test the nsrdb data sampler which does centered sampling on daylight
    hours."""
    handler = DataHandlerNsrdb(INPUT_FILE, FEATURES,
                               target=TARGET, shape=SHAPE,
                               temporal_slice=slice(None, None, 2),
                               time_roll=-7,
                               val_split=0.1,
                               sample_shape=(20, 20, 24),
                               max_extract_workers=1,
                               max_compute_workers=1)
    ti = pd.date_range('20220101', '20230101', freq='1h', inclusive='left')
    ti = ti[0:handler.data.shape[2]]

    for _ in range(100):
        tslice = nsrdb_sampler(handler.data, 4, ti)
        # with only 4 samples, there should never be any NaN data
        assert not np.isnan(handler.data[0, 0, tslice, 0]).any()

    for _ in range(100):
        tslice = nsrdb_sampler(handler.data, 8, ti)
        # with only 8 samples, there should never be any NaN data
        assert not np.isnan(handler.data[0, 0, tslice, 0]).any()

    for _ in range(100):
        tslice = nsrdb_sampler(handler.data, 20, ti)
        # there should be ~8 hours of non-NaN data
        # the beginning and ending timesteps should be nan
        assert ((~np.isnan(handler.data[0, 0, tslice, 0])).sum() > 7)
        assert np.isnan(handler.data[0, 0, tslice, 0])[:3].all()
        assert np.isnan(handler.data[0, 0, tslice, 0])[-3:].all()
