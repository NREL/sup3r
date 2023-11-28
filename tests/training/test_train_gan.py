# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN"""
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
from sup3r.models.data_centric import Sup3rGanDC, Sup3rGanSpatialDC
from sup3r.preprocessing.batch_handling import (
    BatchHandler,
    BatchHandlerDC,
    BatchHandlerSpatialDC,
    SpatialBatchHandler,
)
from sup3r.preprocessing.data_handling import DataHandlerDCforH5, DataHandlerH5
from sup3r.utilities.loss_metrics import MmdMseLoss

FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
TARGET_COORD = (39.01, -105.15)
FEATURES = ['U_100m', 'V_100m']


def test_train_spatial(log=False, full_shape=(20, 20),
                       sample_shape=(10, 10, 1), n_epoch=2):
    """Test basic spatial model training with only gen content loss."""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=2e-5,
                     loss='MeanAbsoluteError')

    # need to reduce the number of temporal examples to test faster
    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 10),
                            worker_kwargs=dict(max_workers=1), val_split=0.1)

    batch_handler = SpatialBatchHandler([handler], batch_size=2, s_enhance=2,
                                        n_batches=2)

    with tempfile.TemporaryDirectory() as td:
        # test that training works and reduces loss
        model.train(batch_handler,
                    input_resolution={'spatial': '8km', 'temporal': '30min'},
                    n_epoch=n_epoch,
                    weight_gen_advers=0.0,
                    train_gen=True,
                    train_disc=False,
                    checkpoint_int=1,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        assert len(model.history) == n_epoch
        vlossg = model.history['val_loss_gen'].values
        tlossg = model.history['train_loss_gen'].values
        assert np.sum(np.diff(vlossg)) < 0
        assert np.sum(np.diff(tlossg)) < 0
        assert 'test_0' in os.listdir(td)
        assert 'test_1' in os.listdir(td)
        assert 'model_gen.pkl' in os.listdir(td + '/test_1')
        assert 'model_disc.pkl' in os.listdir(td + '/test_1')
        assert model.means is not None
        assert model.stdevs is not None

        # make an un-trained dummy model
        dummy = Sup3rGan(fp_gen, fp_disc, learning_rate=2e-5,
                         loss='MeanAbsoluteError')

        # test save/load functionality
        out_dir = os.path.join(td, 'spatial_gan')
        model.save(out_dir)
        loaded = model.load(out_dir)

        assert isinstance(dummy.loss_fun, tf.keras.losses.MeanAbsoluteError)
        assert isinstance(model.loss_fun, tf.keras.losses.MeanAbsoluteError)
        assert isinstance(loaded.loss_fun, tf.keras.losses.MeanAbsoluteError)

        for batch in batch_handler:
            out_og = model.generate(batch.low_res, norm_in=True,
                                    un_norm_out=True)
            out_dummy = dummy.generate(batch.low_res, norm_in=True,
                                       un_norm_out=True)
            out_loaded = loaded.generate(batch.low_res, norm_in=True,
                                         un_norm_out=True)
            assert out_og.dtype == np.float32
            assert out_dummy.dtype == np.float32
            assert out_loaded.dtype == np.float32

            # make sure the loaded model generates the same data as the saved
            # model but different than the dummy
            tf.assert_equal(out_og, out_loaded)
            with pytest.raises(InvalidArgumentError):
                tf.assert_equal(out_og, out_dummy)

            # make sure the trained model has less loss than dummy
            out_og = model.generate(batch.low_res, norm_in=False,
                                    un_norm_out=False)
            out_dummy = dummy.generate(batch.low_res, norm_in=False,
                                       un_norm_out=False)
            loss_og = model.calc_loss(batch.high_res, out_og)[0]
            loss_dummy = dummy.calc_loss(batch.high_res, out_dummy)[0]
            assert loss_og.numpy() < loss_dummy.numpy()


def test_train_st_weight_update(n_epoch=2, log=False):
    """Test basic spatiotemporal model training with discriminators and
    adversarial loss updating."""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4,
                     learning_rate_disc=3e-4)

    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            shape=(20, 20),
                            sample_shape=(12, 12, 16),
                            temporal_slice=slice(None, None, 1),
                            val_split=0.005,
                            worker_kwargs=dict(max_workers=1))

    batch_handler = BatchHandler([handler], batch_size=2,
                                 s_enhance=3, t_enhance=4,
                                 n_batches=2)

    adaptive_update_bounds = (0.9, 0.99)
    with tempfile.TemporaryDirectory() as td:
        model.train(batch_handler,
                    input_resolution={'spatial': '12km', 'temporal': '60min'},
                    n_epoch=n_epoch,
                    weight_gen_advers=1e-6,
                    train_gen=True, train_disc=True,
                    checkpoint_int=10,
                    out_dir=os.path.join(td, 'test_{epoch}'),
                    adaptive_update_bounds=adaptive_update_bounds,
                    adaptive_update_fraction=0.05)

        # check that weight is changed
        check_lower = any(frac < adaptive_update_bounds[0] for frac in
                          model.history['train_disc_trained_frac'][:-1])
        check_higher = any(frac > adaptive_update_bounds[1] for frac in
                           model.history['train_disc_trained_frac'][:-1])
        assert check_lower or check_higher
        for e in range(0, n_epoch - 1):
            weight_old = model.history['weight_gen_advers'][e]
            weight_new = model.history['weight_gen_advers'][e + 1]
            if (model.history['train_disc_trained_frac'][e]
                    < adaptive_update_bounds[0]):
                assert weight_new > weight_old
            if (model.history['train_disc_trained_frac'][e]
                    > adaptive_update_bounds[1]):
                assert weight_new < weight_old


def test_train_spatial_dc(log=False, full_shape=(20, 20),
                          sample_shape=(10, 10, 1), n_epoch=2):
    """Test data-centric spatial model training. Check that the spatial
    weights give the correct number of observations from each spatial bin"""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')

    Sup3rGan.seed()
    model = Sup3rGanSpatialDC(fp_gen, fp_disc, learning_rate=1e-4,
                              learning_rate_disc=3e-4, loss='MmdMseLoss')

    handler = DataHandlerDCforH5(FP_WTK, FEATURES, target=TARGET_COORD,
                                 shape=full_shape,
                                 sample_shape=sample_shape,
                                 temporal_slice=slice(None, None, 1),
                                 val_split=0.005,
                                 worker_kwargs=dict(max_workers=1))
    batch_size = 2
    n_batches = 2
    total_count = batch_size * n_batches
    deviation = np.sqrt(1 / (total_count - 1))

    batch_handler = BatchHandlerSpatialDC([handler], batch_size=batch_size,
                                          s_enhance=2, n_batches=n_batches)

    with tempfile.TemporaryDirectory() as td:
        # test that the normalized number of samples from each bin is close
        # to the weight for that bin
        model.train(batch_handler,
                    input_resolution={'spatial': '8km', 'temporal': '30min'},
                    n_epoch=n_epoch,
                    weight_gen_advers=0.0,
                    train_gen=True, train_disc=False,
                    checkpoint_int=2,
                    out_dir=os.path.join(td, 'test_{epoch}'))
        assert np.allclose(batch_handler.old_spatial_weights,
                           batch_handler.norm_spatial_record,
                           atol=deviation)

        out_dir = os.path.join(td, 'dc_gan')
        model.save(out_dir)
        loaded = model.load(out_dir)

        assert isinstance(model.loss_fun, MmdMseLoss)
        assert isinstance(loaded.loss_fun, MmdMseLoss)
        assert model.meta['class'] == 'Sup3rGanSpatialDC'
        assert loaded.meta['class'] == 'Sup3rGanSpatialDC'


def test_train_st_dc(n_epoch=2, log=False):
    """Test data-centric spatiotemporal model training. Check that the temporal
    weights give the correct number of observations from each temporal bin"""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGanDC(fp_gen, fp_disc, learning_rate=1e-4,
                       learning_rate_disc=3e-4, loss='MmdMseLoss')

    handler = DataHandlerDCforH5(FP_WTK, FEATURES, target=TARGET_COORD,
                                 shape=(20, 20), sample_shape=(12, 12, 16),
                                 temporal_slice=slice(None, None, 1),
                                 val_split=0.005,
                                 worker_kwargs=dict(max_workers=1))
    batch_size = 4
    n_batches = 2
    total_count = batch_size * n_batches
    deviation = np.sqrt(1 / (total_count - 1))
    batch_handler = BatchHandlerDC([handler], batch_size=batch_size,
                                   s_enhance=3, t_enhance=4,
                                   n_batches=n_batches)

    with tempfile.TemporaryDirectory() as td:
        # test that the normalized number of samples from each bin is close
        # to the weight for that bin
        model.train(batch_handler,
                    input_resolution={'spatial': '12km', 'temporal': '60min'},
                    n_epoch=n_epoch,
                    weight_gen_advers=0.0,
                    train_gen=True, train_disc=False,
                    checkpoint_int=2,
                    out_dir=os.path.join(td, 'test_{epoch}'))
        assert np.allclose(batch_handler.old_temporal_weights,
                           batch_handler.norm_temporal_record,
                           atol=deviation)

        out_dir = os.path.join(td, 'dc_gan')
        model.save(out_dir)
        loaded = model.load(out_dir)

        assert isinstance(model.loss_fun, MmdMseLoss)
        assert isinstance(loaded.loss_fun, MmdMseLoss)
        assert model.meta['class'] == 'Sup3rGanDC'
        assert loaded.meta['class'] == 'Sup3rGanDC'


def test_train_st(n_epoch=2, log=False):
    """Test basic spatiotemporal model training with only gen content loss."""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=5e-5,
                     learning_rate_disc=2e-5)

    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            shape=(20, 20), sample_shape=(12, 12, 16),
                            temporal_slice=slice(None, None, 1),
                            val_split=0.005,
                            worker_kwargs=dict(max_workers=1))

    batch_handler = BatchHandler([handler], batch_size=2,
                                 s_enhance=3, t_enhance=4,
                                 n_batches=2,
                                 worker_kwargs=dict(max_workers=1))

    assert batch_handler.norm_workers == 1
    assert batch_handler.stats_workers == 1
    assert batch_handler.load_workers == 1

    with tempfile.TemporaryDirectory() as td:
        # test that training works and reduces loss
        model.train(batch_handler,
                    input_resolution={'spatial': '12km', 'temporal': '60min'},
                    n_epoch=n_epoch,
                    weight_gen_advers=0.0,
                    train_gen=True, train_disc=False,
                    checkpoint_int=1,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        assert 'config_generator' in model.meta
        assert 'config_discriminator' in model.meta
        assert len(model.history) == n_epoch
        assert all(model.history['train_gen_trained_frac'] == 1)
        assert all(model.history['train_disc_trained_frac'] == 0)
        vlossg = model.history['val_loss_gen'].values
        tlossg = model.history['train_loss_gen'].values
        assert (np.diff(vlossg) < 0).sum() >= (n_epoch / 2)
        assert (np.diff(tlossg) < 0).sum() >= (n_epoch / 2)
        assert 'test_0' in os.listdir(td)
        assert 'test_1' in os.listdir(td)
        assert 'model_gen.pkl' in os.listdir(td + '/test_1')
        assert 'model_disc.pkl' in os.listdir(td + '/test_1')
        assert model.means is not None
        assert model.stdevs is not None

        # test save/load functionality
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)
        loaded = model.load(out_dir)

        with open(os.path.join(out_dir, 'model_params.json')) as f:
            model_params = json.load(f)

        assert np.allclose(model_params['optimizer']['learning_rate'], 5e-5)
        assert np.allclose(model_params['optimizer_disc']['learning_rate'],
                           2e-5)
        assert 'learning_rate_gen' in model.history
        assert 'learning_rate_disc' in model.history

        assert 'config_generator' in loaded.meta
        assert 'config_discriminator' in loaded.meta
        assert model.meta['class'] == 'Sup3rGan'

        # make an un-trained dummy model
        dummy = Sup3rGan(fp_gen, fp_disc, learning_rate=5e-5,
                         learning_rate_disc=2e-5)

        for batch in batch_handler:
            out_og = model.generate(batch.low_res, norm_in=True,
                                    un_norm_out=True)
            out_dummy = dummy.generate(batch.low_res, norm_in=True,
                                       un_norm_out=True)
            out_loaded = loaded.generate(batch.low_res, norm_in=True,
                                         un_norm_out=True)
            assert out_og.dtype == np.float32
            assert out_dummy.dtype == np.float32
            assert out_loaded.dtype == np.float32

            # make sure the loaded model generates the same data as the saved
            # model but different than the dummy
            tf.assert_equal(out_og, out_loaded)
            with pytest.raises(InvalidArgumentError):
                tf.assert_equal(out_og, out_dummy)

            # make sure the trained model has less loss than dummy
            out_og = model.generate(batch.low_res, norm_in=False,
                                    un_norm_out=False)
            out_dummy = dummy.generate(batch.low_res, norm_in=False,
                                       un_norm_out=False)
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


def test_optimizer_update():
    """Test updating optimizer method."""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4,
                     learning_rate_disc=4e-4)

    assert model.optimizer.learning_rate == 1e-4
    assert model.optimizer_disc.learning_rate == 4e-4

    model.update_optimizer(option='generator', learning_rate=2)

    assert model.optimizer.learning_rate == 2
    assert model.optimizer_disc.learning_rate == 4e-4

    model.update_optimizer(option='discriminator', learning_rate=0.4)

    assert model.optimizer.learning_rate == 2
    assert model.optimizer_disc.learning_rate == 0.4

    model.update_optimizer(option='all', learning_rate=0.1)

    assert model.optimizer.learning_rate == 0.1
    assert model.optimizer_disc.learning_rate == 0.1


def test_input_res_check():
    """Make sure error is raised for invalid input resolution"""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4,
                     learning_rate_disc=4e-4)

    with pytest.raises(RuntimeError):
        model.set_model_params(
            input_resolution={'spatial': '22km', 'temporal': '9min'})


def test_enhancement_check():
    """Make sure error is raised for invalid enhancement factor inputs"""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4,
                     learning_rate_disc=4e-4)

    with pytest.raises(RuntimeError):
        model.set_model_params(
            input_resolution={'spatial': '12km', 'temporal': '60min'},
            s_enhance=7, t_enhance=3)
