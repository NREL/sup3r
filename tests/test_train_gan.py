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
from sup3r.models.models import SpatialGan
from sup3r.data_handling.preprocessing import DataHandler, SpatialBatchHandler


input_file = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
target = (39.01, -105.15)
full_shape = (20, 20)
sample_shape = (10, 10)
features = ['windspeed_100m', 'winddirection_100m']
n_epoch = 6


def test_train_spatial(log=False):
    """Test basic model training with only gen content loss."""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')

    SpatialGan.seed()
    model = SpatialGan(fp_gen, fp_disc, learning_rate=1e-6)

    # need to reduce the number of temporal examples to test faster
    handler = DataHandler(input_file, target, full_shape, features,
                          spatial_sample_shape=sample_shape,
                          time_step=10)

    batch_handler = SpatialBatchHandler([handler], batch_size=8, spatial_res=2,
                                        spatial_sample_shape=sample_shape,
                                        n_batches=10)

    with tempfile.TemporaryDirectory() as td:
        # test that training works and reduces loss
        model.train(batch_handler, n_epoch=n_epoch, weight_gen_advers=0.0,
                    train_gen=True, train_disc=False, checkpoint_int=2,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        assert len(model.history) == n_epoch
        assert (np.diff(model.history['validation_loss_gen'].values) < 0).all()
        assert 'test_0' in os.listdir(td)
        assert 'test_2' in os.listdir(td)
        assert 'model_gen.pkl' in os.listdir(td + '/test_2')
        assert 'model_disc.pkl' in os.listdir(td + '/test_2')

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
