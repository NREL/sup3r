# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN"""
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from rex import init_logger

from sup3r import TEST_DATA_DIR
from sup3r.preprocessing.data_handling.dual_data_handling import (
    DualDataHandler, )
from sup3r.preprocessing.data_handling.h5_data_handling import DataHandlerH5
from sup3r.preprocessing.data_handling.nc_data_handling import DataHandlerNC
from sup3r.preprocessing.dual_batch_handling import (DualBatchHandler,
                                                     SpatialDualBatchHandler,
                                                     )
from sup3r.utilities.utilities import spatial_coarsening

FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
FP_ERA = os.path.join(TEST_DATA_DIR, 'test_era5_co_2012.nc')
TARGET_COORD = (39.01, -105.15)
FEATURES = ['U_100m', 'V_100m']


def test_dual_data_handler(log=True,
                           full_shape=(20, 20),
                           sample_shape=(10, 10, 1),
                           plot=True):
    """Test basic spatial model training with only gen content loss."""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    # need to reduce the number of temporal examples to test faster
    hr_handler = DataHandlerH5(FP_WTK,
                               FEATURES,
                               target=TARGET_COORD,
                               shape=full_shape,
                               sample_shape=sample_shape,
                               temporal_slice=slice(None, None, 10),
                               worker_kwargs=dict(max_workers=1),
                               )
    lr_handler = DataHandlerNC(FP_ERA,
                               FEATURES,
                               sample_shape=(sample_shape[0] // 2,
                                             sample_shape[1] // 2, 1),
                               temporal_slice=slice(None, None, 10),
                               worker_kwargs=dict(max_workers=1),
                               )

    dual_handler = DualDataHandler(hr_handler,
                                   lr_handler,
                                   s_enhance=2,
                                   t_enhance=1,
                                   val_split=0.1)

    batch_handler = SpatialDualBatchHandler([dual_handler],
                                            batch_size=2,
                                            s_enhance=2,
                                            n_batches=10)

    if plot:
        for i, batch in enumerate(batch_handler):
            fig, ax = plt.subplots(1, 2, figsize=(5, 10))
            fig.suptitle(f'High vs Low Res ({dual_handler.features[-1]})')
            ax[0].set_title('High Res')
            ax[0].imshow(np.mean(batch.high_res[..., -1], axis=0))
            ax[1].set_title('Low Res')
            ax[1].imshow(np.mean(batch.low_res[..., -1], axis=0))
            fig.savefig(f'./high_vs_low_{str(i).zfill(3)}.png',
                        bbox_inches='tight')


def test_regrid_caching(log=True,
                        full_shape=(20, 20),
                        sample_shape=(10, 10, 1)):
    """Test caching and loading of regridded data"""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    # need to reduce the number of temporal examples to test faster
    with tempfile.TemporaryDirectory() as td:
        hr_handler = DataHandlerH5(FP_WTK,
                                   FEATURES,
                                   target=TARGET_COORD,
                                   shape=full_shape,
                                   sample_shape=sample_shape,
                                   temporal_slice=slice(None, None, 10),
                                   worker_kwargs=dict(max_workers=1),
                                   )
        lr_handler = DataHandlerNC(FP_ERA,
                                   FEATURES,
                                   sample_shape=(sample_shape[0] // 2,
                                                 sample_shape[1] // 2, 1),
                                   temporal_slice=slice(None, None, 10),
                                   worker_kwargs=dict(max_workers=1),
                                   )
        old_dh = DualDataHandler(hr_handler,
                                 lr_handler,
                                 s_enhance=2,
                                 t_enhance=1,
                                 val_split=0.1,
                                 regrid_cache_pattern=f'{td}/cache.pkl',
                                 )

        # Load handlers again
        hr_handler = DataHandlerH5(FP_WTK,
                                   FEATURES,
                                   target=TARGET_COORD,
                                   shape=full_shape,
                                   sample_shape=sample_shape,
                                   temporal_slice=slice(None, None, 10),
                                   worker_kwargs=dict(max_workers=1),
                                   )
        lr_handler = DataHandlerNC(FP_ERA,
                                   FEATURES,
                                   sample_shape=(sample_shape[0] // 2,
                                                 sample_shape[1] // 2, 1),
                                   temporal_slice=slice(None, None, 10),
                                   worker_kwargs=dict(max_workers=1),
                                   )
        new_dh = DualDataHandler(hr_handler,
                                 lr_handler,
                                 s_enhance=2,
                                 t_enhance=1,
                                 val_split=0.1,
                                 regrid_cache_pattern=f'{td}/cache.pkl',
                                 )
        assert np.array_equal(old_dh.lr_data, new_dh.lr_data)
        assert np.array_equal(old_dh.hr_data, new_dh.hr_data)


def test_regrid_caching_in_steps(log=True,
                                 full_shape=(20, 20),
                                 sample_shape=(10, 10, 1)):
    """Test caching and loading of regridded data"""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    # need to reduce the number of temporal examples to test faster
    with tempfile.TemporaryDirectory() as td:
        hr_handler = DataHandlerH5(FP_WTK,
                                   FEATURES[0],
                                   target=TARGET_COORD,
                                   shape=full_shape,
                                   sample_shape=sample_shape,
                                   temporal_slice=slice(None, None, 10),
                                   worker_kwargs=dict(max_workers=1),
                                   )
        lr_handler = DataHandlerNC(FP_ERA,
                                   FEATURES[0],
                                   sample_shape=(sample_shape[0] // 2,
                                                 sample_shape[1] // 2, 1),
                                   temporal_slice=slice(None, None, 10),
                                   worker_kwargs=dict(max_workers=1),
                                   )
        dh_step1 = DualDataHandler(hr_handler,
                                   lr_handler,
                                   s_enhance=2,
                                   t_enhance=1,
                                   val_split=0.1,
                                   regrid_cache_pattern=f'{td}/cache.pkl',
                                   )

        # Load handlers again with one cached feature and one noncached feature
        hr_handler = DataHandlerH5(FP_WTK,
                                   FEATURES,
                                   target=TARGET_COORD,
                                   shape=full_shape,
                                   sample_shape=sample_shape,
                                   temporal_slice=slice(None, None, 10),
                                   worker_kwargs=dict(max_workers=1),
                                   )
        lr_handler = DataHandlerNC(FP_ERA,
                                   FEATURES,
                                   sample_shape=(sample_shape[0] // 2,
                                                 sample_shape[1] // 2, 1),
                                   temporal_slice=slice(None, None, 10),
                                   worker_kwargs=dict(max_workers=1),
                                   )
        dh_step2 = DualDataHandler(hr_handler,
                                   lr_handler,
                                   s_enhance=2,
                                   t_enhance=1,
                                   val_split=0.1,
                                   regrid_cache_pattern=f'{td}/cache.pkl')

        assert np.array_equal(dh_step2.lr_data[..., 0:1], dh_step1.lr_data)
        assert np.array_equal(dh_step2.noncached_features, FEATURES[1:])
        assert np.array_equal(dh_step2.cached_features, FEATURES[0:1])


def test_st_dual_batch_handler(log=False,
                               full_shape=(20, 20),
                               sample_shape=(10, 10, 4)):
    """Test spatiotemporal dual batch handler."""
    t_enhance = 2
    s_enhance = 2

    if log:
        init_logger('sup3r', log_level='DEBUG')

    # need to reduce the number of temporal examples to test faster
    hr_handler = DataHandlerH5(FP_WTK,
                               FEATURES,
                               target=TARGET_COORD,
                               shape=full_shape,
                               sample_shape=sample_shape,
                               temporal_slice=slice(None, None, 10),
                               worker_kwargs=dict(max_workers=1))
    lr_handler = DataHandlerNC(FP_ERA,
                               FEATURES,
                               sample_shape=(sample_shape[0] // s_enhance,
                                             sample_shape[1] // s_enhance,
                                             sample_shape[2] // t_enhance,
                                             ),
                               temporal_slice=slice(None, None,
                                                    t_enhance * 10),
                               worker_kwargs=dict(max_workers=1))

    dual_handler = DualDataHandler(hr_handler,
                                   lr_handler,
                                   s_enhance=s_enhance,
                                   t_enhance=t_enhance,
                                   val_split=0.1)

    batch_handler = DualBatchHandler([dual_handler, dual_handler],
                                     batch_size=2,
                                     s_enhance=s_enhance,
                                     t_enhance=t_enhance,
                                     n_batches=10)
    assert np.allclose(batch_handler.handler_weights, 0.5)

    for batch in batch_handler:

        handler_index = batch_handler.current_handler_index
        handler = batch_handler.data_handlers[handler_index]

        for i, index in enumerate(batch_handler.current_batch_indices):
            hr_index = index['hr_index']
            lr_index = index['lr_index']

            coarse_lat_lon = spatial_coarsening(
                handler.hr_lat_lon[hr_index[:2]], obs_axis=False)
            lr_lat_lon = handler.lr_lat_lon[lr_index[:2]]
            assert np.array_equal(coarse_lat_lon, lr_lat_lon)

            coarse_ti = handler.hr_time_index[hr_index[2]][::t_enhance]
            lr_ti = handler.lr_time_index[lr_index[2]]
            assert np.array_equal(coarse_ti.values, lr_ti.values)

            assert np.array_equal(batch.high_res[i], handler.hr_data[hr_index])
            assert np.array_equal(batch.low_res[i], handler.lr_data[lr_index])


def test_spatial_dual_batch_handler(log=False,
                                    full_shape=(20, 20),
                                    sample_shape=(10, 10, 1),
                                    plot=True):
    """Test spatial dual batch handler."""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    # need to reduce the number of temporal examples to test faster
    hr_handler = DataHandlerH5(FP_WTK,
                               FEATURES,
                               target=TARGET_COORD,
                               shape=full_shape,
                               hr_spatial_coarsen=2,
                               sample_shape=sample_shape,
                               temporal_slice=slice(None, None, 10),
                               worker_kwargs=dict(max_workers=1))
    lr_handler = DataHandlerNC(FP_ERA,
                               FEATURES,
                               sample_shape=(sample_shape[0] // 2,
                                             sample_shape[1] // 2, 1),
                               temporal_slice=slice(None, None, 10),
                               worker_kwargs=dict(max_workers=1))

    dual_handler = DualDataHandler(hr_handler,
                                   lr_handler,
                                   s_enhance=2,
                                   t_enhance=1,
                                   val_split=0.0,
                                   shuffle_time=True)

    batch_handler = SpatialDualBatchHandler([dual_handler],
                                            batch_size=2,
                                            s_enhance=2,
                                            t_enhance=1,
                                            n_batches=10)

    for i, batch in enumerate(batch_handler):
        for j, index in enumerate(batch_handler.current_batch_indices):
            hr_index = index['hr_index']
            lr_index = index['lr_index']

            assert np.array_equal(batch.high_res[j, :, :],
                                  dual_handler.hr_data[hr_index][..., 0, :])
            assert np.array_equal(batch.low_res[j, :, :],
                                  dual_handler.lr_data[lr_index][..., 0, :])

            coarse_lat_lon = spatial_coarsening(
                dual_handler.hr_lat_lon[hr_index[:2]], obs_axis=False)
            lr_lat_lon = dual_handler.lr_lat_lon[lr_index[:2]]
            assert np.array_equal(coarse_lat_lon, lr_lat_lon)

        if plot:
            for ifeature in range(batch.high_res.shape[-1]):
                data_fine = batch.high_res[0, :, :, ifeature]
                data_coarse = batch.low_res[0, :, :, ifeature]
                fig = plt.figure(figsize=(10, 5))
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                ax1.imshow(data_fine)
                ax2.imshow(data_coarse)
                plt.savefig(f'./{i}_{ifeature}.png', bbox_inches='tight')
                plt.close()


def test_validation_batching(log=False,
                             full_shape=(20, 20),
                             sample_shape=(10, 10, 4)):
    """Test batching of validation data for dual batch handler"""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    s_enhance = 2
    t_enhance = 2

    hr_handler = DataHandlerH5(FP_WTK,
                               FEATURES,
                               target=TARGET_COORD,
                               shape=full_shape,
                               sample_shape=sample_shape,
                               temporal_slice=slice(None, None, 10),
                               worker_kwargs=dict(max_workers=1))
    lr_handler = DataHandlerNC(FP_ERA,
                               FEATURES,
                               sample_shape=(sample_shape[0] // s_enhance,
                                             sample_shape[1] // s_enhance,
                                             sample_shape[2] // t_enhance),
                               temporal_slice=slice(None, None,
                                                    t_enhance * 10),
                               worker_kwargs=dict(max_workers=1))

    dual_handler = DualDataHandler(hr_handler,
                                   lr_handler,
                                   s_enhance=s_enhance,
                                   t_enhance=t_enhance,
                                   val_split=0.1)

    batch_handler = DualBatchHandler([dual_handler],
                                     batch_size=2,
                                     s_enhance=s_enhance,
                                     t_enhance=t_enhance,
                                     n_batches=10)

    for batch in batch_handler.val_data:
        assert batch.high_res.dtype == np.dtype(np.float32)
        assert batch.low_res.dtype == np.dtype(np.float32)
        assert batch.low_res.shape[0] == batch.high_res.shape[0]
        assert batch.low_res.shape == (batch.low_res.shape[0],
                                       sample_shape[0] // s_enhance,
                                       sample_shape[1] // s_enhance,
                                       sample_shape[2] // t_enhance,
                                       len(FEATURES))
        assert batch.high_res.shape == (batch.high_res.shape[0],
                                        sample_shape[0], sample_shape[1],
                                        sample_shape[2], len(FEATURES))

        for j, index in enumerate(
                batch_handler.val_data.current_batch_indices):
            hr_index = index['hr_index']
            lr_index = index['lr_index']

            assert np.array_equal(batch.high_res[j],
                                  dual_handler.hr_val_data[hr_index])
            assert np.array_equal(batch.low_res[j],
                                  dual_handler.lr_val_data[lr_index])

            coarse_lat_lon = spatial_coarsening(
                dual_handler.hr_lat_lon[hr_index[:2]], obs_axis=False)
            lr_lat_lon = dual_handler.lr_lat_lon[lr_index[:2]]

            assert np.array_equal(coarse_lat_lon, lr_lat_lon)

            coarse_ti = dual_handler.hr_val_time_index[
                hr_index[2]][::t_enhance]
            lr_ti = dual_handler.lr_val_time_index[lr_index[2]]
            assert np.array_equal(coarse_ti.values, lr_ti.values)


def test_normalization(log=False,
                       full_shape=(20, 20),
                       sample_shape=(10, 10, 4)):
    """Test correct normalization"""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    s_enhance = 2
    t_enhance = 2

    hr_handler = DataHandlerH5(FP_WTK,
                               FEATURES,
                               target=TARGET_COORD,
                               shape=full_shape,
                               sample_shape=sample_shape,
                               temporal_slice=slice(None, None, 10),
                               worker_kwargs=dict(max_workers=1))
    lr_handler = DataHandlerNC(FP_ERA,
                               FEATURES,
                               sample_shape=(sample_shape[0] // s_enhance,
                                             sample_shape[1] // s_enhance,
                                             sample_shape[2] // t_enhance),
                               temporal_slice=slice(None, None,
                                                    t_enhance * 10),
                               worker_kwargs=dict(max_workers=1))

    dual_handler = DualDataHandler(hr_handler,
                                   lr_handler,
                                   s_enhance=s_enhance,
                                   t_enhance=t_enhance,
                                   val_split=0.1)

    means = [
        np.nanmean(dual_handler.lr_data[..., i])
        for i in range(dual_handler.lr_data.shape[-1])
    ]
    stdevs = [
        np.nanstd(dual_handler.lr_data[..., i] - means[i])
        for i in range(dual_handler.lr_data.shape[-1])
    ]

    batch_handler = DualBatchHandler([dual_handler],
                                     batch_size=2,
                                     s_enhance=s_enhance,
                                     t_enhance=t_enhance,
                                     n_batches=10)
    assert np.allclose(batch_handler.means, means)
    assert np.allclose(batch_handler.stds, stdevs)
    stacked_data = np.concatenate(
        [d.data for d in batch_handler.data_handlers], axis=2)

    for i in range(len(FEATURES)):
        std = np.std(stacked_data[..., i])
        if std == 0:
            std = 1
        mean = np.mean(stacked_data[..., i])
        assert np.allclose(std, 1, atol=1e-3)
        assert np.allclose(mean, 0, atol=1e-3)
