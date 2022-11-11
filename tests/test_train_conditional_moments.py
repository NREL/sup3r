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
from sup3r.models import Sup3rCondMom
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

    Sup3rCondMom.seed()
    model = Sup3rCondMom(fp_gen, learning_rate=5e-5)

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

    Sup3rCondMom.seed()
    model = Sup3rCondMom(fp_gen, learning_rate=5e-5)

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
        dummy = Sup3rCondMom(fp_gen, learning_rate=2e-5)

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


def test_out_spatial(plot=False, full_shape=(20, 20),
                     sample_shape=(10, 10, 1),
                     batch_size=4, n_batches=4,
                     s_enhance=2, model_dir=None):
    """Test basic spatial model outputing."""
    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 10),
                            val_split=0,
                            max_workers=1)

    batch_handler = SpatialBatchHandler([handler],
                                        batch_size=batch_size,
                                        s_enhance=s_enhance,
                                        n_batches=n_batches)

    # Load Model
    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    model_unloaded = Sup3rCondMom(fp_gen, learning_rate=5e-5)
    if model_dir is None:
        model = model_unloaded
    else:
        model = model_unloaded.load(model_dir)

    # Check sizes
    for batch in batch_handler:
        assert batch.high_res.shape == (batch_size, sample_shape[0],
                                        sample_shape[1], 2)
        assert batch.low_res.shape == (batch_size,
                                       sample_shape[0] // s_enhance,
                                       sample_shape[1] // s_enhance, 2)
        out = model._tf_generate(batch.low_res)
        assert out.shape == (batch_size, sample_shape[0], sample_shape[1], 2)
        break
    if plot:
        import matplotlib.pyplot as plt
        from sup3r.utilities.plot_utilities import (plot_multi_contour,
                                                    makeMovie)
        figureFolder = 'Figures'
        os.makedirs(figureFolder, exist_ok=True)
        movieFolder = os.path.join(figureFolder, 'Movie')
        os.makedirs(movieFolder, exist_ok=True)
        n_snap = 0
        for p, batch in enumerate(batch_handler):
            out = model._tf_generate(batch.low_res).numpy()
            for i in range(batch.high_res.shape[0]):
                lr = (batch.low_res[i, :, :, 0] * batch_handler.stds[0]
                      + batch_handler.means[0])
                hr = (batch.high_res[i, :, :, 0] * batch_handler.stds[0]
                      + batch_handler.means[0])
                gen = (out[i, :, :, 0] * batch_handler.stds[0]
                       + batch_handler.means[0])
                fig = plot_multi_contour(
                    [lr, hr, gen],
                    [0, batch.high_res.shape[1]],
                    [0, batch.high_res.shape[2]],
                    ['U [m/s]', 'U [m/s]', 'U [m/s]'],
                    ['LR', 'HR', r'$\mathbb{E}$(HR|LR)'],
                    ['x [m]', 'x [m]', 'x [m]'],
                    ['y [m]', 'y [m]', 'y [m]'],
                    [np.amin(lr), np.amin(hr), np.amin(hr)],
                    [np.amax(lr), np.amax(hr), np.amax(hr)],
                )
                fig.savefig(os.path.join(movieFolder,
                                         "im_{}.png".format(n_snap)),
                            dpi=100, bbox_inches='tight')
                plt.close(fig)
                n_snap += 1
            if p > 4:
                break
        makeMovie(n_snap, movieFolder, os.path.join(figureFolder, 'mom1.gif'),
                  fps=6)


if __name__ == "__main__":
    # test_train_st(n_epoch=4, log=True, full_shape=(20, 20),
    #               batch_size=4, n_batches=20,
    #               out_dir_root='st_model')
    # test_train_spatial(n_epoch=10, log=True, full_shape=(20, 20),
    #                    sample_shape=(10, 10, 1),
    #                    batch_size=16, n_batches=100,
    #                    out_dir_root='s_model')

    test_out_spatial(plot=True, full_shape=(20, 20),
                     sample_shape=(10, 10, 1),
                     batch_size=4, n_batches=4,
                     s_enhance=2, model_dir='s_model_save/spatial_cond_mom')
