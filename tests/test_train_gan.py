# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN"""
import os
import numpy as np
import pytest
import tempfile
import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from rex import init_logger

from sup3r import TEST_DATA_DIR
from sup3r import CONFIG_DIR
from sup3r.models.models import SpatialGan, SpatioTemporalGan
from sup3r.data_handling.preprocessing import (DataHandler,
                                               SpatialBatchHandler,
                                               BatchHandler)


FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
TARGET_COORD = (39.01, -105.15)
FEATURES = ['windspeed_100m', 'winddirection_100m']


def test_train_spatial(log=False, full_shape=(20, 20), sample_shape=(10, 10),
                       n_epoch=6):
    """Test basic spatial model training with only gen content loss."""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')

    SpatialGan.seed()
    model = SpatialGan(fp_gen, fp_disc, learning_rate=1e-6)

    # need to reduce the number of temporal examples to test faster
    handler = DataHandler(FP_WTK, FEATURES, target=TARGET_COORD,
                          shape=full_shape,
                          spatial_sample_shape=sample_shape,
                          time_pruning=10)

    batch_handler = SpatialBatchHandler([handler], batch_size=8, spatial_res=2,
                                        n_batches=10)

    with tempfile.TemporaryDirectory() as td:
        # test that training works and reduces loss
        model.train(batch_handler, n_epoch=n_epoch, weight_gen_advers=0.0,
                    train_gen=True, train_disc=False, checkpoint_int=2,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        assert len(model.history) == n_epoch
        vlossg = model.history['val_loss_gen'].values
        tlossg = model.history['train_loss_gen'].values
        assert (np.diff(vlossg) < 0).all()
        assert (np.diff(tlossg) < 0).sum() >= (n_epoch / 1.5)
        assert 'test_0' in os.listdir(td)
        assert 'test_2' in os.listdir(td)
        assert 'test_5' in os.listdir(td)
        assert 'model_gen.pkl' in os.listdir(td + '/test_5')
        assert 'model_disc.pkl' in os.listdir(td + '/test_5')

        # make an un-trained dummy model
        dummy = SpatialGan(fp_gen, fp_disc, learning_rate=1e-6)

        # test save/load functionality
        out_dir = os.path.join(td, 'spatial_gan')
        model.save(out_dir)
        loaded = model.load(out_dir)
        for batch in batch_handler:
            out_og = model.generate(batch.low_res)
            out_dummy = dummy.generate(batch.low_res)
            out_loaded = loaded.generate(batch.low_res)

            # make sure the loaded model generates the same data as the saved
            # model but different than the dummy
            tf.assert_equal(out_og, out_loaded)
            with pytest.raises(InvalidArgumentError):
                tf.assert_equal(out_og, out_dummy)

            # make sure the trained model has less loss than dummy
            loss_og = model.calc_loss(batch.high_res, out_og)[0]
            loss_dummy = dummy.calc_loss(batch.high_res, out_dummy)[0]
            assert loss_og.numpy() < loss_dummy.numpy()


def test_train_st(n_epoch=6, log=False):
    """Test basic spatiotemporal model training with only gen content loss."""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x.json')
    fp_disc_s = os.path.join(CONFIG_DIR, 'spatiotemporal/disc_space.json')
    fp_disc_t = os.path.join(CONFIG_DIR, 'spatiotemporal/disc_time.json')

    SpatioTemporalGan.seed()
    model = SpatioTemporalGan(fp_gen, fp_disc_s, fp_disc_t,
                              learning_rate=1e-4)

    handler = DataHandler(FP_WTK, FEATURES, target=TARGET_COORD,
                          shape=(20, 20),
                          temporal_sample_shape=24,
                          spatial_sample_shape=(18, 18),
                          time_pruning=1, val_split=0.005)

    batch_handler = BatchHandler([handler], batch_size=2,
                                 spatial_res=3, temporal_res=4,
                                 n_batches=4)

    with tempfile.TemporaryDirectory() as td:
        # test that training works and reduces loss
        model.train(batch_handler, n_epoch=n_epoch,
                    weight_gen_advers_s=0.0, weight_gen_advers_t=0.0,
                    train_gen=True, train_disc_s=False, train_disc_t=False,
                    checkpoint_int=2,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        assert 'config_generator' in model.meta
        assert 'config_spatial_disc' in model.meta
        assert 'config_temporal_disc' in model.meta

        assert len(model.history) == n_epoch
        assert all(model.history['train_gen_trained_frac'] == 1)
        assert all(model.history['train_disc_s_trained_frac'] == 0)
        assert all(model.history['train_disc_t_trained_frac'] == 0)
        vlossg = model.history['val_loss_gen'].values
        tlossg = model.history['train_loss_gen'].values
        assert (np.diff(vlossg) < 0).sum() >= (n_epoch / 1.5)
        assert (np.diff(tlossg) < 0).sum() >= (n_epoch / 1.5)
        assert 'test_0' in os.listdir(td)
        assert 'test_2' in os.listdir(td)
        assert 'test_5' in os.listdir(td)
        assert 'model_gen.pkl' in os.listdir(td + '/test_5')
        assert 'model_disc_s.pkl' in os.listdir(td + '/test_5')
        assert 'model_disc_t.pkl' in os.listdir(td + '/test_5')

        # make an un-trained dummy model
        dummy = SpatialGan(fp_gen, fp_disc_s, fp_disc_t)

        # test save/load functionality
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)
        loaded = model.load(out_dir)

        assert 'config_generator' in loaded.meta
        assert 'config_spatial_disc' in loaded.meta
        assert 'config_temporal_disc' in loaded.meta

        for batch in batch_handler:
            out_og = model.generate(batch.low_res)
            out_dummy = dummy.generate(batch.low_res)
            out_loaded = loaded.generate(batch.low_res)

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
        test_data = np.ones((3, 50, 50, 12, 2), dtype=np.float32)
        y_test = model.generate(test_data)
        assert y_test.shape[0] == test_data.shape[0]
        assert y_test.shape[1] == test_data.shape[1] * 3
        assert y_test.shape[2] == test_data.shape[2] * 3
        assert y_test.shape[3] == test_data.shape[3] * 4
        assert y_test.shape[4] == test_data.shape[4]


def test_optimizer_update():
    """Test updating optimizer method."""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x.json')
    fp_disc_s = os.path.join(CONFIG_DIR, 'spatiotemporal/disc_space.json')
    fp_disc_t = os.path.join(CONFIG_DIR, 'spatiotemporal/disc_time.json')

    SpatioTemporalGan.seed()
    model = SpatioTemporalGan(fp_gen, fp_disc_s, fp_disc_t,
                              learning_rate=1e-4)

    updated_model = SpatioTemporalGan(fp_gen, fp_disc_s, fp_disc_t,
                                      learning_rate=1e-2)
    updated_model.update_optimizer(learning_rate=1e-4)

    assert updated_model.optimizer_config == model.optimizer_config
