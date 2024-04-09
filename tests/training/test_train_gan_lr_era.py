# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN with dual data handler"""
import json
import os
import tempfile

import numpy as np
import pytest
import tensorflow as tf
from rex import init_logger
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from sup3r import CONFIG_DIR, TEST_DATA_DIR
from sup3r.models import Sup3rGan
from sup3r.preprocessing.data_handling import (
    DataHandlerH5,
    DataHandlerNC,
    DualDataHandler,
)
from sup3r.preprocessing.dual_batch_handling import (
    DualBatchHandler,
    SpatialDualBatchHandler,
)

FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
FP_ERA = os.path.join(TEST_DATA_DIR, 'test_era5_co_2012.nc')
TARGET_COORD = (39.01, -105.15)
FEATURES = ['U_100m', 'V_100m']


def test_train_spatial(
    log=False, full_shape=(20, 20), sample_shape=(10, 10, 1), n_epoch=2
):
    """Test basic spatial model training with only gen content loss."""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(
        fp_gen, fp_disc, learning_rate=2e-5, loss='MeanAbsoluteError'
    )

    # need to reduce the number of temporal examples to test faster
    hr_handler = DataHandlerH5(
        FP_WTK,
        FEATURES,
        target=TARGET_COORD,
        shape=full_shape,
        sample_shape=sample_shape,
        temporal_slice=slice(None, None, 10),
        worker_kwargs=dict(max_workers=1),
    )
    lr_handler = DataHandlerNC(
        FP_ERA,
        FEATURES,
        sample_shape=(sample_shape[0] // 2, sample_shape[1] // 2, 1),
        temporal_slice=slice(None, None, 10),
        worker_kwargs=dict(max_workers=1),
    )

    dual_handler = DualDataHandler(
        hr_handler, lr_handler, s_enhance=2, t_enhance=1, val_split=0.1
    )

    batch_handler = SpatialDualBatchHandler(
        [dual_handler], batch_size=2, s_enhance=2, n_batches=2
    )

    with tempfile.TemporaryDirectory() as td:
        # test that training works and reduces loss
        model.train(
            batch_handler,
            input_resolution={'spatial': '30km', 'temporal': '60min'},
            n_epoch=n_epoch,
            weight_gen_advers=0.0,
            train_gen=True,
            train_disc=False,
            checkpoint_int=1,
            out_dir=os.path.join(td, 'test_{epoch}'),
        )

        assert len(model.history) == n_epoch
        vlossg = model.history['val_loss_gen'].values
        tlossg = model.history['train_loss_gen'].values
        assert np.sum(np.diff(vlossg)) < 0
        assert np.sum(np.diff(tlossg)) < 0
        assert 'test_0' in os.listdir(td)
        assert 'test_1' in os.listdir(td)
        assert 'model_gen.pkl' in os.listdir(td + '/test_1')
        assert 'model_disc.pkl' in os.listdir(td + '/test_1')

        # make an un-trained dummy model
        dummy = Sup3rGan(
            fp_gen, fp_disc, learning_rate=2e-5, loss='MeanAbsoluteError'
        )

        # test save/load functionality
        out_dir = os.path.join(td, 'spatial_gan')
        model.save(out_dir)
        loaded = model.load(out_dir)

        assert isinstance(dummy.loss_fun, tf.keras.losses.MeanAbsoluteError)
        assert isinstance(model.loss_fun, tf.keras.losses.MeanAbsoluteError)
        assert isinstance(loaded.loss_fun, tf.keras.losses.MeanAbsoluteError)

        for batch in batch_handler:
            out_og = model._tf_generate(batch.low_res)
            out_dummy = dummy._tf_generate(batch.low_res)
            out_loaded = loaded._tf_generate(batch.low_res)

            # make sure the loaded model generates the same data as the saved
            # model but different than the dummy
            tf.assert_equal(out_og, out_loaded)
            with pytest.raises(InvalidArgumentError):
                tf.assert_equal(out_og, out_dummy)

            # make sure the trained model has less loss than dummy
            loss_og = model.calc_loss(batch.high_res, out_og)[0]
            loss_dummy = dummy.calc_loss(batch.high_res, out_dummy)[0]
            assert loss_og.numpy() < loss_dummy.numpy()


def test_train_st(n_epoch=3, log=False):
    """Test basic spatiotemporal model training with only gen content loss."""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(
        fp_gen, fp_disc, learning_rate=1e-5, loss='MeanAbsoluteError'
    )

    hr_handler = DataHandlerH5(
        FP_WTK,
        FEATURES,
        target=TARGET_COORD,
        shape=(20, 20),
        sample_shape=(12, 12, 16),
        temporal_slice=slice(None, None, 10),
        worker_kwargs=dict(max_workers=1),
    )
    lr_handler = DataHandlerNC(
        FP_ERA,
        FEATURES,
        sample_shape=(4, 4, 4),
        temporal_slice=slice(None, None, 40),
        worker_kwargs=dict(max_workers=1),
    )

    dual_handler = DualDataHandler(
        hr_handler, lr_handler, s_enhance=3, t_enhance=4, val_split=0.1
    )

    batch_handler = DualBatchHandler(
        [dual_handler],
        batch_size=5,
        s_enhance=3,
        t_enhance=4,
        n_batches=5,
        worker_kwargs={'max_workers': 1},
    )

    assert batch_handler.norm_workers == 1
    assert batch_handler.stats_workers == 1
    assert batch_handler.load_workers == 1

    with tempfile.TemporaryDirectory() as td:
        # test that training works and reduces loss
        model.train(
            batch_handler,
            input_resolution={'spatial': '30km', 'temporal': '60min'},
            n_epoch=n_epoch,
            weight_gen_advers=0.0,
            train_gen=True,
            train_disc=False,
            checkpoint_int=1,
            out_dir=os.path.join(td, 'test_{epoch}'),
        )

        assert 'config_generator' in model.meta
        assert 'config_discriminator' in model.meta
        assert len(model.history) == n_epoch
        assert all(model.history['train_gen_trained_frac'] == 1)
        assert all(model.history['train_disc_trained_frac'] == 0)
        vlossg = model.history['val_loss_gen'].values
        tlossg = model.history['train_loss_gen'].values
        assert np.sum(np.diff(vlossg)) < 0
        assert np.sum(np.diff(tlossg)) < 0
        assert 'test_0' in os.listdir(td)
        assert 'test_1' in os.listdir(td)
        assert 'model_gen.pkl' in os.listdir(td + '/test_1')
        assert 'model_disc.pkl' in os.listdir(td + '/test_1')

        # test save/load functionality
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)
        loaded = model.load(out_dir)

        with open(os.path.join(out_dir, 'model_params.json')) as f:
            model_params = json.load(f)

        assert np.allclose(model_params['optimizer']['learning_rate'], 1e-5)
        assert np.allclose(
            model_params['optimizer_disc']['learning_rate'], 1e-5
        )
        assert 'learning_rate_gen' in model.history
        assert 'learning_rate_disc' in model.history

        assert 'config_generator' in loaded.meta
        assert 'config_discriminator' in loaded.meta
        assert model.meta['class'] == 'Sup3rGan'

        # make an un-trained dummy model
        dummy = Sup3rGan(
            fp_gen, fp_disc, learning_rate=1e-5, loss='MeanAbsoluteError'
        )

        for batch in batch_handler:
            out_og = model._tf_generate(batch.low_res)
            out_dummy = dummy._tf_generate(batch.low_res)
            out_loaded = loaded._tf_generate(batch.low_res)

            # make sure the loaded model generates the same data as the saved
            # model but different than the dummy
            tf.assert_equal(out_og, out_loaded)
            with pytest.raises(InvalidArgumentError):
                tf.assert_equal(out_og, out_dummy)

            # make sure the trained model has less loss than dummy
            loss_og = model.calc_loss(batch.high_res, out_og)[0]
            loss_dummy = dummy.calc_loss(batch.high_res, out_dummy)[0]
            assert loss_og.numpy() < loss_dummy.numpy()

        # test that a new shape can be passed through the generator
        test_data = np.ones((3, 10, 10, 4, len(FEATURES)), dtype=np.float32)
        y_test = model._tf_generate(test_data)
        assert y_test.shape[0] == test_data.shape[0]
        assert y_test.shape[1] == test_data.shape[1] * 3
        assert y_test.shape[2] == test_data.shape[2] * 3
        assert y_test.shape[3] == test_data.shape[3] * 4
        assert y_test.shape[4] == test_data.shape[4]
