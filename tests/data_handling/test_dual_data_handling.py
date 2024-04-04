# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN"""
import copy
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from rex import init_logger
import pytest

from sup3r import TEST_DATA_DIR
from sup3r.preprocessing.data_handling.dual_data_handling import (
    DualDataHandler,
)
from sup3r.preprocessing.data_handling.h5_data_handling import DataHandlerH5
from sup3r.preprocessing.data_handling.nc_data_handling import DataHandlerNC
from sup3r.preprocessing.dual_batch_handling import (
    DualBatchHandler,
    SpatialDualBatchHandler,
)
from sup3r.utilities.utilities import spatial_coarsening

FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
FP_ERA = os.path.join(TEST_DATA_DIR, 'test_era5_co_2012.nc')
TARGET_COORD = (39.01, -105.15)
FEATURES = ['U_100m', 'V_100m']


def test_dual_data_handler(log=False,
                           full_shape=(20, 20),
                           sample_shape=(10, 10, 1),
                           plot=False):
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


def test_regrid_caching(log=False,
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
                                 cache_pattern=f'{td}/cache.pkl',
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
                                 cache_pattern=f'{td}/cache.pkl',
                                 )
        assert np.array_equal(old_dh.lr_data, new_dh.lr_data)
        assert np.array_equal(old_dh.hr_data, new_dh.hr_data)


def test_regrid_caching_in_steps(log=False,
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
                                   cache_pattern=f'{td}/cache.pkl',
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
                                   cache_pattern=f'{td}/cache.pkl')

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

            # hr_data is a view of hr_dh.data
            assert np.array_equal(batch.high_res[i], handler.hr_data[hr_index])
            assert np.allclose(batch.low_res[i], handler.lr_data[lr_index])


def test_spatial_dual_batch_handler(log=False,
                                    full_shape=(20, 20),
                                    sample_shape=(10, 10, 1),
                                    plot=False):
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

            # hr_data is a view of hr_dh.data
            assert np.array_equal(batch.high_res[j, :, :],
                                  dual_handler.hr_data[hr_index][..., 0, :])
            assert np.allclose(batch.low_res[j, :, :],
                               dual_handler.lr_data[lr_index][..., 0, :])

            coarse_lat_lon = spatial_coarsening(
                dual_handler.hr_lat_lon[hr_index[:2]], obs_axis=False)
            lr_lat_lon = dual_handler.lr_lat_lon[lr_index[:2]]
            assert np.allclose(coarse_lat_lon, lr_lat_lon)

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


@pytest.mark.parametrize(('cache', 'val_split'),
                         ([True, 1.0], [True, 0.0], [False, 0.0]))
def test_normalization(cache,
                       val_split,
                       log=False,
                       full_shape=(20, 20),
                       sample_shape=(10, 10, 4)):
    """Test correct normalization"""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    s_enhance = 2
    t_enhance = 2

    with tempfile.TemporaryDirectory() as td:
        hr_cache = None
        lr_cache = None
        dual_cache = None
        if cache:
            hr_cache = os.path.join(td, 'hr_cache_{feature}.pkl')
            lr_cache = os.path.join(td, 'lr_cache_{feature}.pkl')
            dual_cache = os.path.join(td, 'dual_cache_{feature}.pkl')

        hr_handler = DataHandlerH5(FP_WTK,
                                   FEATURES,
                                   target=TARGET_COORD,
                                   shape=full_shape,
                                   sample_shape=sample_shape,
                                   temporal_slice=slice(None, None, 10),
                                   cache_pattern=hr_cache,
                                   worker_kwargs=dict(max_workers=1),
                                   val_split=0.0)
        lr_handler = DataHandlerNC(FP_ERA,
                                   FEATURES,
                                   sample_shape=(sample_shape[0] // s_enhance,
                                                 sample_shape[1] // s_enhance,
                                                 sample_shape[2] // t_enhance),
                                   temporal_slice=slice(None, None,
                                                        t_enhance * 10),
                                   cache_pattern=lr_cache,
                                   worker_kwargs=dict(max_workers=1),
                                   val_split=0.0)

        dual_handler = DualDataHandler(hr_handler,
                                       lr_handler,
                                       s_enhance=s_enhance,
                                       t_enhance=t_enhance,
                                       cache_pattern=dual_cache,
                                       val_split=val_split)

    if val_split == 0.0:
        assert id(dual_handler.hr_data.base) == id(dual_handler.hr_dh.data)

    assert hr_handler.data.dtype == np.float32
    assert lr_handler.data.dtype == np.float32
    assert dual_handler.lr_data.dtype == np.float32
    assert dual_handler.hr_data.dtype == np.float32
    assert dual_handler.lr_data.dtype == np.float32
    assert dual_handler.hr_data.dtype == np.float32

    hr_means0 = np.mean(hr_handler.data, axis=(0, 1, 2))
    lr_means0 = np.mean(lr_handler.data, axis=(0, 1, 2))
    ddh_hr_means0 = np.mean(dual_handler.hr_data, axis=(0, 1, 2))
    ddh_lr_means0 = np.mean(dual_handler.lr_data, axis=(0, 1, 2))

    means = copy.deepcopy(dual_handler.means)
    stdevs = copy.deepcopy(dual_handler.stds)
    assert all(v.dtype == np.float32 for v in means.values())
    assert all(v.dtype == np.float32 for v in stdevs.values())

    batch_handler = DualBatchHandler([dual_handler],
                                     batch_size=2,
                                     s_enhance=s_enhance,
                                     t_enhance=t_enhance,
                                     n_batches=10,
                                     norm=True)

    if val_split == 0.0:
        assert id(dual_handler.hr_data.base) == id(dual_handler.hr_dh.data)

    assert hr_handler.data.dtype == np.float32
    assert lr_handler.data.dtype == np.float32
    assert dual_handler.lr_data.dtype == np.float32
    assert dual_handler.hr_data.dtype == np.float32

    hr_means1 = np.mean(hr_handler.data, axis=(0, 1, 2))
    lr_means1 = np.mean(lr_handler.data, axis=(0, 1, 2))
    ddh_hr_means1 = np.mean(dual_handler.hr_data, axis=(0, 1, 2))
    ddh_lr_means1 = np.mean(dual_handler.lr_data, axis=(0, 1, 2))

    assert all(means[k] == v for k, v in batch_handler.means.items())
    assert all(stdevs[k] == v for k, v in batch_handler.stds.items())

    assert all(v.dtype == np.float32 for v in batch_handler.means.values())
    assert all(v.dtype == np.float32 for v in batch_handler.stds.values())

    # normalization stats retrieved from LR data before re-gridding
    for idf in range(lr_handler.shape[-1]):
        std = dual_handler.data[..., idf].std()
        mean = dual_handler.data[..., idf].mean()
        assert np.allclose(std, 1, atol=1e-3), str(std)
        assert np.allclose(mean, 0, atol=1e-3), str(mean)

        fn = FEATURES[idf]
        true_hr_mean0 = (hr_means0[idf] - means[fn]) / stdevs[fn]
        true_lr_mean0 = (lr_means0[idf] - means[fn]) / stdevs[fn]
        true_ddh_hr_mean0 = (ddh_hr_means0[idf] - means[fn]) / stdevs[fn]
        true_ddh_lr_mean0 = (ddh_lr_means0[idf] - means[fn]) / stdevs[fn]

        rtol, atol = 1e-6, 1e-5
        assert np.allclose(true_hr_mean0, hr_means1[idf], rtol=rtol, atol=atol)
        assert np.allclose(true_lr_mean0, lr_means1[idf],
                           rtol=rtol, atol=atol)
        assert np.allclose(true_ddh_hr_mean0, ddh_hr_means1[idf],
                           rtol=rtol, atol=atol)
        assert np.allclose(true_ddh_lr_mean0, ddh_lr_means1[idf],
                           rtol=rtol, atol=atol)


def test_no_regrid(log=False, full_shape=(20, 20), sample_shape=(10, 10, 4)):
    """Test no regridding of the LR data with correct normalization and
    view/slice of the lr dataset"""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    s_enhance = 2
    t_enhance = 2

    hr_dh = DataHandlerH5(FP_WTK, FEATURES[0], target=TARGET_COORD,
                          shape=full_shape, sample_shape=sample_shape,
                          temporal_slice=slice(None, None, 10),
                          worker_kwargs=dict(max_workers=1),
                          val_split=0.0)
    lr_handler = DataHandlerH5(FP_WTK, FEATURES[1], target=TARGET_COORD,
                               shape=full_shape,
                               sample_shape=(sample_shape[0] // s_enhance,
                                             sample_shape[1] // s_enhance,
                                             sample_shape[2] // t_enhance),
                               temporal_slice=slice(None, -10,
                                                    t_enhance * 10),
                               hr_spatial_coarsen=2, cache_pattern=None,
                               worker_kwargs=dict(max_workers=1),
                               val_split=0.0)

    hr_dh0 = copy.deepcopy(hr_dh)
    hr_dh1 = copy.deepcopy(hr_dh)
    lr_dh0 = copy.deepcopy(lr_handler)
    lr_dh1 = copy.deepcopy(lr_handler)

    ddh0 = DualDataHandler(hr_dh0, lr_dh0, s_enhance=s_enhance,
                           t_enhance=t_enhance, regrid_lr=True)
    ddh1 = DualDataHandler(hr_dh1, lr_dh1, s_enhance=s_enhance,
                           t_enhance=t_enhance, regrid_lr=False)

    _ = DualBatchHandler([ddh0], norm=True)
    _ = DualBatchHandler([ddh1], norm=True)

    hr_m0 = np.mean(ddh0.hr_data, axis=(0, 1, 2))
    lr_m0 = np.mean(ddh0.lr_data, axis=(0, 1, 2))
    hr_m1 = np.mean(ddh1.hr_data, axis=(0, 1, 2))
    lr_m1 = np.mean(ddh1.lr_data, axis=(0, 1, 2))
    assert np.allclose(hr_m0, hr_m1)
    assert np.allclose(lr_m0, lr_m1)
    assert np.allclose(hr_m0, 0, atol=1e-3)
    assert np.allclose(lr_m0, 0, atol=1e-6)

    hr_s0 = np.std(ddh0.hr_data, axis=(0, 1, 2))
    lr_s0 = np.std(ddh0.lr_data, axis=(0, 1, 2))
    hr_s1 = np.std(ddh1.hr_data, axis=(0, 1, 2))
    lr_s1 = np.std(ddh1.lr_data, axis=(0, 1, 2))
    assert np.allclose(hr_s0, hr_s1)
    assert np.allclose(lr_s0, lr_s1)
    assert np.allclose(hr_s0, 1, atol=1e-3)
    assert np.allclose(lr_s0, 1, atol=1e-6)


@pytest.mark.parametrize(['lr_features', 'hr_features', 'hr_exo_features'],
                         [(['U_100m'], ['U_100m', 'V_100m'], ['V_100m']),
                          (['U_100m'], ['U_100m', 'V_100m'], ('V_100m',)),
                          (['U_100m'], ['V_100m', 'BVF2_200m'], ['BVF2_200m']),
                          (['U_100m'], ('V_100m', 'BVF2_200m'), ['BVF2_200m']),
                          (['U_100m'], ['V_100m', 'BVF2_200m'], [])])
def test_mixed_lr_hr_features(lr_features, hr_features, hr_exo_features):
    """Test weird mixes of low-res and high-res features that should work with
    the dual dh"""
    lr_handler = DataHandlerNC(FP_ERA,
                               lr_features,
                               sample_shape=(5, 5, 4),
                               temporal_slice=slice(None, None, 1),
                               worker_kwargs=dict(max_workers=1),
                               )
    hr_handler = DataHandlerH5(FP_WTK,
                               hr_features,
                               hr_exo_features=hr_exo_features,
                               target=TARGET_COORD,
                               shape=(20, 20),
                               sample_shape=(5, 5, 4),
                               temporal_slice=slice(None, None, 1),
                               worker_kwargs=dict(max_workers=1),
                               )

    dual_handler = DualDataHandler(hr_handler,
                                   lr_handler,
                                   s_enhance=1,
                                   t_enhance=1,
                                   val_split=0.0)

    batch_handler = DualBatchHandler(dual_handler, batch_size=2,
                                     s_enhance=1, t_enhance=1,
                                     n_batches=10,
                                     worker_kwargs={'max_workers': 2})

    n_hr_features = (len(batch_handler.hr_out_features)
                     + len(batch_handler.hr_exo_features))
    hr_only_features = [fn for fn in hr_features if fn not in lr_features]
    hr_out_true = [fn for fn in hr_features if fn not in hr_exo_features]
    assert batch_handler.features == lr_features + hr_only_features
    assert batch_handler.lr_features == list(lr_features)
    assert batch_handler.hr_exo_features == list(hr_exo_features)
    assert batch_handler.hr_out_features == list(hr_out_true)

    for batch in batch_handler:
        assert batch.high_res.shape[-1] == n_hr_features
        assert batch.low_res.shape[-1] == len(batch_handler.lr_features)

        if batch_handler.lr_features == lr_features + hr_only_features:
            assert np.allclose(batch.low_res, batch.high_res)
        elif batch_handler.lr_features != lr_features + hr_only_features:
            assert not np.allclose(batch.low_res, batch.high_res)


def test_bad_cache_load():
    """This tests good errors when load_cached gets messed up with dual data
    handling and stats normalization."""
    s_enhance = 2
    t_enhance = 2
    full_shape = (20, 20)
    sample_shape = (10, 10, 4)

    with tempfile.TemporaryDirectory() as td:
        lr_cache = f'{td}/lr_cache_' + '{feature}.pkl'
        hr_cache = f'{td}/hr_cache_' + '{feature}.pkl'
        dual_cache = f'{td}/dual_cache_' + '{feature}.pkl'

        hr_handler = DataHandlerH5(FP_WTK,
                                   FEATURES,
                                   target=TARGET_COORD,
                                   shape=full_shape,
                                   sample_shape=sample_shape,
                                   temporal_slice=slice(None, None, 10),
                                   cache_pattern=hr_cache,
                                   load_cached=False,
                                   worker_kwargs=dict(max_workers=1))

        lr_handler = DataHandlerNC(FP_ERA,
                                   FEATURES,
                                   sample_shape=(sample_shape[0] // s_enhance,
                                                 sample_shape[1] // s_enhance,
                                                 sample_shape[2] // t_enhance),
                                   temporal_slice=slice(None, None,
                                                        t_enhance * 10),
                                   cache_pattern=lr_cache,
                                   load_cached=False,
                                   worker_kwargs=dict(max_workers=1))

        # because load_cached is False
        assert hr_handler.data is None
        assert lr_handler.data is None

        dual_handler = DualDataHandler(hr_handler,
                                       lr_handler,
                                       s_enhance=s_enhance,
                                       t_enhance=t_enhance,
                                       cache_pattern=dual_cache,
                                       load_cached=False,
                                       val_split=0.0)

        # because load_cached is False
        assert hr_handler.data is None
        assert lr_handler.data is not None

        good_err = "DataHandler.data=None!"
        with pytest.raises(RuntimeError) as ex:
            _ = copy.deepcopy(dual_handler.means)
        assert good_err in str(ex.value)

        with pytest.raises(RuntimeError) as ex:
            _ = copy.deepcopy(dual_handler.stds)
        assert good_err in str(ex.value)

        with pytest.raises(RuntimeError) as ex:
            dual_handler.normalize()
        assert good_err in str(ex.value)

        dual_handler = DualDataHandler(hr_handler,
                                       lr_handler,
                                       s_enhance=s_enhance,
                                       t_enhance=t_enhance,
                                       cache_pattern=dual_cache,
                                       load_cached=True,
                                       val_split=0.0)

        # because load_cached is True
        assert hr_handler.data is not None
        assert lr_handler.data is not None

        # should run without error now that load_cached=True
        _ = copy.deepcopy(dual_handler.means)
        _ = copy.deepcopy(dual_handler.stds)
        dual_handler.normalize()
