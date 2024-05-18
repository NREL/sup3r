# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN"""
import copy
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytest
from rex import init_logger

from sup3r import TEST_DATA_DIR
from sup3r.preprocessing import (
    DataHandlerH5,
    DataHandlerNC,
)

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
                               time_slice=slice(None, None, 10),
                               )
    lr_handler = DataHandlerNC(FP_ERA,
                               FEATURES,
                               time_slice=slice(None, None, 10),
                               )

    dual_handler = DualDataHandler(hr_handler,
                                   lr_handler,
                                   s_enhance=2,
                                   t_enhance=1)

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
                                   time_slice=slice(None, None, 10),
                                   )
        lr_handler = DataHandlerNC(FP_ERA,
                                   FEATURES,
                                   time_slice=slice(None, None, 10),
                                   )
        old_dh = DualDataHandler(hr_handler,
                                 lr_handler,
                                 s_enhance=2,
                                 t_enhance=1,
                                 cache_pattern=f'{td}/cache.pkl',
                                 )

        # Load handlers again
        hr_handler = DataHandlerH5(FP_WTK,
                                   FEATURES,
                                   target=TARGET_COORD,
                                   shape=full_shape,
                                   time_slice=slice(None, None, 10),
                                   )
        lr_handler = DataHandlerNC(FP_ERA,
                                   FEATURES,
                                   time_slice=slice(None, None, 10),
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
                                   time_slice=slice(None, None, 10),
                                   )
        lr_handler = DataHandlerNC(FP_ERA,
                                   FEATURES[0],
                                   time_slice=slice(None, None, 10),
                                   )
        dh_step1 = DualDataHandler(hr_handler,
                                   lr_handler,
                                   s_enhance=2,
                                   t_enhance=1,
                                   cache_pattern=f'{td}/cache.pkl',
                                   )

        # Load handlers again with one cached feature and one noncached feature
        hr_handler = DataHandlerH5(FP_WTK,
                                   FEATURES,
                                   target=TARGET_COORD,
                                   shape=full_shape,
                                   time_slice=slice(None, None, 10),
                                   )
        lr_handler = DataHandlerNC(FP_ERA,
                                   FEATURES,
                                   time_slice=slice(None, None, 10),
                                   )
        dh_step2 = DualDataHandler(hr_handler,
                                   lr_handler,
                                   s_enhance=2,
                                   t_enhance=1,
                                   cache_pattern=f'{td}/cache.pkl')

        assert np.array_equal(dh_step2.lr_data[..., 0:1], dh_step1.lr_data)
        assert np.array_equal(dh_step2.noncached_features, FEATURES[1:])
        assert np.array_equal(dh_step2.cached_features, FEATURES[0:1])


def test_no_regrid(log=False, full_shape=(20, 20), sample_shape=(10, 10, 4)):
    """Test no regridding of the LR data with correct normalization and
    view/slice of the lr dataset"""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    s_enhance = 2
    t_enhance = 2

    hr_dh = DataHandlerH5(FP_WTK, FEATURES[0], target=TARGET_COORD,
                          shape=full_shape, sample_shape=sample_shape,
                          time_slice=slice(None, None, 10))
    lr_handler = DataHandlerH5(FP_WTK, FEATURES[1], target=TARGET_COORD,
                               shape=full_shape,
                               sample_shape=(sample_shape[0] // s_enhance,
                                             sample_shape[1] // s_enhance,
                                             sample_shape[2] // t_enhance),
                               time_slice=slice(None, -10,
                                                    t_enhance * 10),
                               hr_spatial_coarsen=2, cache_pattern=None)

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
                               time_slice=slice(None, None, 1),
                               )
    hr_handler = DataHandlerH5(FP_WTK,
                               hr_features,
                               hr_exo_features=hr_exo_features,
                               target=TARGET_COORD,
                               shape=(20, 20),
                               time_slice=slice(None, None, 1),
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
                                   time_slice=slice(None, None, 10),
                                   cache_pattern=hr_cache,
                                   load_cached=False,
                                   worker_kwargs=dict(max_workers=1))

        lr_handler = DataHandlerNC(FP_ERA,
                                   FEATURES,
                                   sample_shape=(sample_shape[0] // s_enhance,
                                                 sample_shape[1] // s_enhance,
                                                 sample_shape[2] // t_enhance),
                                   time_slice=slice(None, None,
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
