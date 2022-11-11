# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN"""
import os
# import json
import numpy as np
import pytest
import tempfile
import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from rex import init_logger

from sup3r import TEST_DATA_DIR
from sup3r import CONFIG_DIR
from sup3r.models import Sup3r_cond_mom
from sup3r.preprocessing.data_handling import DataHandlerH5
from sup3r.preprocessing.batch_handling import (BatchHandler,
                                                SpatialBatchHandler)


FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
TARGET_COORD = (39.01, -105.15)
FEATURES = ['U_100m', 'V_100m']


def test_train_st(log=False, full_shape=(20, 20),
                  sample_shape=(12, 12, 24), n_epoch=4,
                  batch_size=4, n_batches=4,
                  out_dir_root=None):
    """Test basic spatiotemporal model training."""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')

    Sup3r_cond_mom.seed()
    model = Sup3r_cond_mom(fp_gen, learning_rate=5e-5)

    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 1),
                            val_split=0.005,
                            max_workers=1)

    batch_handler = BatchHandler([handler], batch_size=batch_size,
                                 s_enhance=3, t_enhance=4,
                                 n_batches=n_batches)

    with tempfile.TemporaryDirectory() as td:
        if out_dir_root is None:
            out_dir_root = td
        model.train(batch_handler, n_epoch=n_epoch,
                    checkpoint_int=10,
                    out_dir=os.path.join(out_dir_root, 'test_{epoch}'))


def test_train_spatial(log=False, full_shape=(20, 20),
                       sample_shape=(10, 10, 1), n_epoch=4,
                       batch_size=4, n_batches=4,
                       out_dir_root=None):
    """Test basic spatial model training."""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')

    Sup3r_cond_mom.seed()
    model = Sup3r_cond_mom(fp_gen, learning_rate=5e-5)

    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 10),
                            val_split=0.005,
                            max_workers=1)

    batch_handler = SpatialBatchHandler([handler], batch_size=batch_size,
                                        s_enhance=2, n_batches=n_batches)

    with tempfile.TemporaryDirectory() as td:
        if out_dir_root is None:
            out_dir_root = td
        model.train(batch_handler, n_epoch=n_epoch,
                    checkpoint_int=2,
                    out_dir=os.path.join(out_dir_root, 'test_{epoch}'))

        assert len(model.history) == n_epoch
        vlossg = model.history['val_loss_gen'].values
        tlossg = model.history['train_loss_gen'].values
        assert np.sum(np.diff(vlossg)) < 0
        assert np.sum(np.diff(tlossg)) < 0
        assert 'test_0' in os.listdir(out_dir_root)
        assert 'test_2' in os.listdir(out_dir_root)
        assert 'model_gen.pkl' in os.listdir(out_dir_root + '/test_3')

        # make an un-trained dummy model
        dummy = Sup3r_cond_mom(fp_gen, learning_rate=2e-5)

        # test save/load functionality
        out_dir = os.path.join(out_dir_root, 'spatial_cond_mom')
        model.save(out_dir)
        loaded = model.load(out_dir)

        assert isinstance(dummy.loss_fun, tf.keras.losses.MeanSquaredError)
        assert isinstance(model.loss_fun, tf.keras.losses.MeanSquaredError)
        assert isinstance(loaded.loss_fun, tf.keras.losses.MeanSquaredError)

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


if __name__ == "__main__":
    # test_train_st(n_epoch=4, log=True, full_shape=(20, 20),
    #               batch_size=4, n_batches=20,
    #               out_dir_root='st_model')
    test_train_spatial(n_epoch=4, log=True, full_shape=(20, 20),
                       sample_shape=(10, 10, 1),
                       batch_size=12, n_batches=40,
                       out_dir_root='s_model')
    # test_train_spatial(log=True, full_shape=(20, 20),
    #                    sample_shape=(10, 10, 1), n_epoch=4)
