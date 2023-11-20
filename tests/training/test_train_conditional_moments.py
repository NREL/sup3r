# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN"""
import os
import tempfile

# import json
import numpy as np
import pytest
import tensorflow as tf
from rex import init_logger
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from sup3r import CONFIG_DIR, TEST_DATA_DIR
from sup3r.models import Sup3rCondMom
from sup3r.preprocessing.conditional_moment_batch_handling import (
    BatchHandlerMom1,
    BatchHandlerMom1SF,
    BatchHandlerMom2,
    BatchHandlerMom2Sep,
    BatchHandlerMom2SepSF,
    BatchHandlerMom2SF,
    SpatialBatchHandlerMom1,
    SpatialBatchHandlerMom1SF,
    SpatialBatchHandlerMom2,
    SpatialBatchHandlerMom2Sep,
    SpatialBatchHandlerMom2SepSF,
    SpatialBatchHandlerMom2SF,
)
from sup3r.preprocessing.data_handling import DataHandlerH5

FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
TARGET_COORD = (39.01, -105.15)
FEATURES = ['U_100m', 'V_100m']
TRAIN_FEATURES = None


@pytest.mark.parametrize('FEATURES, TRAIN_FEATURES,'
                         + 's_padding, t_padding',
                         [(['U_100m', 'V_100m'],
                           None,
                           None, None),
                          (['U_100m', 'V_100m', 'BVF2_200m'],
                           ['BVF2_200m'],
                           None, None),
                          (['U_100m', 'V_100m'],
                           None,
                           1, 1)])
def test_train_s_mom1(FEATURES, TRAIN_FEATURES,
                      s_padding, t_padding,
                      log=False, full_shape=(20, 20),
                      sample_shape=(10, 10, 1), n_epoch=2,
                      batch_size=2, n_batches=6,
                      out_dir_root=None):
    """Test basic spatial model training."""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')

    Sup3rCondMom.seed()
    model = Sup3rCondMom(fp_gen, learning_rate=1e-4)

    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            lr_only_features=TRAIN_FEATURES,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 10),
                            val_split=0.005,
                            worker_kwargs=dict(max_workers=1))

    batch_handler = SpatialBatchHandlerMom1([handler],
                                            batch_size=batch_size,
                                            s_enhance=2,
                                            n_batches=n_batches,
                                            s_padding=s_padding,
                                            t_padding=t_padding)

    with tempfile.TemporaryDirectory() as td:
        if out_dir_root is None:
            out_dir_root = td
        model.train(batch_handler,
                    input_resolution={'spatial': '8km', 'temporal': '30min'},
                    n_epoch=n_epoch,
                    checkpoint_int=2,
                    out_dir=os.path.join(out_dir_root, 'test_{epoch}'))

        assert len(model.history) == n_epoch
        vlossg = model.history['val_loss_gen'].values
        tlossg = model.history['train_loss_gen'].values
        assert np.sum(np.diff(vlossg)) < 0
        assert np.sum(np.diff(tlossg)) < 0
        assert 'test_0' in os.listdir(out_dir_root)
        assert 'model_gen.pkl' in os.listdir(out_dir_root
                                             + '/test_%d' % (n_epoch - 1))

        # make an un-trained dummy model
        dummy = Sup3rCondMom(fp_gen, learning_rate=1e-4)

        # test save/load functionality
        out_dir = os.path.join(out_dir_root, 's_cond_mom')
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
            loss_og = model.calc_loss(batch.output, out_og,
                                      batch.mask)[0]
            loss_dummy = dummy.calc_loss(batch.output, out_dummy,
                                         batch.mask)[0]
            assert loss_og.numpy() < loss_dummy.numpy()


@pytest.mark.parametrize('FEATURES, TRAIN_FEATURES,'
                         + 's_padding, t_padding',
                         [(['U_100m', 'V_100m'],
                           None,
                           None, None),
                          (['U_100m', 'V_100m', 'BVF2_200m'],
                           ['BVF2_200m'],
                           None, None),
                          (['U_100m', 'V_100m'],
                           None,
                           1, 1)])
def test_train_s_mom1_sf(FEATURES, TRAIN_FEATURES,
                         s_padding, t_padding,
                         log=False, full_shape=(20, 20),
                         sample_shape=(10, 10, 1), n_epoch=2,
                         batch_size=2, n_batches=2,
                         out_dir_root=None):
    """Test basic spatial model training."""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')

    Sup3rCondMom.seed()
    model = Sup3rCondMom(fp_gen, learning_rate=1e-4)

    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            lr_only_features=TRAIN_FEATURES,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 10),
                            val_split=0.005,
                            worker_kwargs=dict(max_workers=1))

    batch_handler = SpatialBatchHandlerMom1SF([handler],
                                              batch_size=batch_size,
                                              s_enhance=2,
                                              n_batches=n_batches,
                                              s_padding=s_padding,
                                              t_padding=t_padding)

    with tempfile.TemporaryDirectory() as td:
        if out_dir_root is None:
            out_dir_root = td
        model.train(batch_handler,
                    input_resolution={'spatial': '8km', 'temporal': '30min'},
                    n_epoch=n_epoch,
                    checkpoint_int=2,
                    out_dir=os.path.join(out_dir_root, 'test_{epoch}'))

        # test save/load functionality
        out_dir = os.path.join(out_dir_root, 's_cond_mom')
        model.save(out_dir)


@pytest.mark.parametrize('FEATURES, TRAIN_FEATURES,'
                         + 's_padding, t_padding',
                         [(['U_100m', 'V_100m'],
                           None,
                           None, None),
                          (['U_100m', 'V_100m', 'BVF2_200m'],
                           ['BVF2_200m'],
                           None, None),
                          (['U_100m', 'V_100m'],
                           None,
                           1, 1)])
def test_train_s_mom2(FEATURES, TRAIN_FEATURES,
                      s_padding, t_padding,
                      log=False, full_shape=(20, 20),
                      sample_shape=(10, 10, 1), n_epoch=2,
                      batch_size=2, n_batches=2,
                      out_dir_root=None,
                      model_mom1_dir=None):
    """Test basic spatial model training for second conditional moment"""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    # Load Mom 1 Model
    if model_mom1_dir is None:
        fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
        model_mom1 = Sup3rCondMom(fp_gen)
    else:
        fp_gen = os.path.join(model_mom1_dir, 'model_params.json')
        model_mom1 = Sup3rCondMom(fp_gen).load(model_mom1_dir)

    Sup3rCondMom.seed()
    fp_gen_mom2 = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    model_mom2 = Sup3rCondMom(fp_gen_mom2, learning_rate=1e-4)

    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            lr_only_features=TRAIN_FEATURES,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 10),
                            val_split=0.005,
                            worker_kwargs=dict(max_workers=1))

    batch_handler = SpatialBatchHandlerMom2([handler], batch_size=batch_size,
                                            s_enhance=2, n_batches=n_batches,
                                            model_mom1=model_mom1,
                                            s_padding=s_padding,
                                            t_padding=t_padding)

    with tempfile.TemporaryDirectory() as td:
        if out_dir_root is None:
            out_dir_root = td
        model_mom2.train(batch_handler,
                         input_resolution={'spatial': '8km',
                                           'temporal': '30min'},
                         n_epoch=n_epoch,
                         checkpoint_int=2,
                         out_dir=os.path.join(out_dir_root, 'test_{epoch}'))
        # test save/load functionality
        out_dir = os.path.join(out_dir_root, 's_cond_mom')
        model_mom2.save(out_dir)


@pytest.mark.parametrize('FEATURES, TRAIN_FEATURES,'
                         + 's_padding, t_padding',
                         [(['U_100m', 'V_100m'],
                           None,
                           None, None),
                          (['U_100m', 'V_100m', 'BVF2_200m'],
                           ['BVF2_200m'],
                           None, None),
                          (['U_100m', 'V_100m'],
                           None,
                           1, 1)])
def test_train_s_mom2_sf(FEATURES, TRAIN_FEATURES,
                         s_padding, t_padding,
                         log=False, full_shape=(20, 20),
                         sample_shape=(10, 10, 1), n_epoch=2,
                         batch_size=2, n_batches=2,
                         out_dir_root=None,
                         model_mom1_dir=None):
    """Test basic spatial model training for second conditional moment"""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    # Load Mom 1 Model
    if model_mom1_dir is None:
        fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
        model_mom1 = Sup3rCondMom(fp_gen)
    else:
        fp_gen = os.path.join(model_mom1_dir, 'model_params.json')
        model_mom1 = Sup3rCondMom(fp_gen).load(model_mom1_dir)

    Sup3rCondMom.seed()
    fp_gen_mom2 = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    model_mom2 = Sup3rCondMom(fp_gen_mom2, learning_rate=1e-4)

    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            lr_only_features=TRAIN_FEATURES,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 10),
                            val_split=0.005,
                            worker_kwargs=dict(max_workers=1))

    batch_handler = SpatialBatchHandlerMom2SF([handler],
                                              batch_size=batch_size,
                                              s_enhance=2,
                                              n_batches=n_batches,
                                              model_mom1=model_mom1,
                                              s_padding=s_padding,
                                              t_padding=t_padding)

    with tempfile.TemporaryDirectory() as td:
        if out_dir_root is None:
            out_dir_root = td
        model_mom2.train(batch_handler,
                         input_resolution={'spatial': '8km',
                                           'temporal': '30min'},
                         n_epoch=n_epoch,
                         checkpoint_int=2,
                         out_dir=os.path.join(out_dir_root, 'test_{epoch}'))
        # test save/load functionality
        out_dir = os.path.join(out_dir_root, 's_cond_mom')
        model_mom2.save(out_dir)


@pytest.mark.parametrize('FEATURES, TRAIN_FEATURES,'
                         + 's_padding, t_padding',
                         [(['U_100m', 'V_100m'],
                           None,
                           None, None),
                          (['U_100m', 'V_100m', 'BVF2_200m'],
                           ['BVF2_200m'],
                           None, None),
                          (['U_100m', 'V_100m'],
                           None,
                           1, 1)])
def test_train_s_mom2_sep(FEATURES, TRAIN_FEATURES,
                          s_padding, t_padding,
                          log=False, full_shape=(20, 20),
                          sample_shape=(10, 10, 1), n_epoch=2,
                          batch_size=2, n_batches=2,
                          out_dir_root=None):
    """Test basic spatial model training for second conditional moment
    separate from first moment"""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    Sup3rCondMom.seed()
    fp_gen_mom2 = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    model_mom2 = Sup3rCondMom(fp_gen_mom2, learning_rate=1e-4)

    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            lr_only_features=TRAIN_FEATURES,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 10),
                            val_split=0.005,
                            worker_kwargs=dict(max_workers=1))

    batch_handler = SpatialBatchHandlerMom2Sep([handler],
                                               batch_size=batch_size,
                                               s_enhance=2,
                                               n_batches=n_batches,
                                               s_padding=s_padding,
                                               t_padding=t_padding)

    with tempfile.TemporaryDirectory() as td:
        if out_dir_root is None:
            out_dir_root = td
        model_mom2.train(batch_handler,
                         input_resolution={'spatial': '8km',
                                           'temporal': '30min'},
                         n_epoch=n_epoch,
                         checkpoint_int=2,
                         out_dir=os.path.join(out_dir_root, 'test_{epoch}'))
        # test save/load functionality
        out_dir = os.path.join(out_dir_root, 's_cond_mom')
        model_mom2.save(out_dir)


@pytest.mark.parametrize('FEATURES, TRAIN_FEATURES,'
                         + 's_padding, t_padding',
                         [(['U_100m', 'V_100m'],
                           None,
                           None, None),
                          (['U_100m', 'V_100m', 'BVF2_200m'],
                           ['BVF2_200m'],
                           None, None),
                          (['U_100m', 'V_100m'],
                           None,
                           1, 1)])
def test_train_s_mom2_sep_sf(FEATURES, TRAIN_FEATURES,
                             s_padding, t_padding,
                             log=False, full_shape=(20, 20),
                             sample_shape=(10, 10, 1), n_epoch=2,
                             batch_size=2, n_batches=2,
                             out_dir_root=None):
    """Test basic spatial model training for second conditional moment
    of subfilter velocity separate from first moment"""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    Sup3rCondMom.seed()
    fp_gen_mom2 = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    model_mom2 = Sup3rCondMom(fp_gen_mom2, learning_rate=1e-4)

    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            lr_only_features=TRAIN_FEATURES,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 10),
                            val_split=0.005,
                            worker_kwargs=dict(max_workers=1))

    batch_handler = SpatialBatchHandlerMom2SepSF([handler],
                                                 batch_size=batch_size,
                                                 s_enhance=2,
                                                 n_batches=n_batches,
                                                 s_padding=s_padding,
                                                 t_padding=t_padding)

    with tempfile.TemporaryDirectory() as td:
        if out_dir_root is None:
            out_dir_root = td
        model_mom2.train(batch_handler,
                         input_resolution={'spatial': '8km',
                                           'temporal': '30min'},
                         n_epoch=n_epoch,
                         checkpoint_int=2,
                         out_dir=os.path.join(out_dir_root, 'test_{epoch}'))
        # test save/load functionality
        out_dir = os.path.join(out_dir_root, 's_cond_mom')
        model_mom2.save(out_dir)


@pytest.mark.parametrize('FEATURES, end_t_padding',
                         [(['U_100m', 'V_100m'], False),
                          (['U_100m', 'V_100m'], True)])
def test_train_st_mom1(FEATURES,
                       end_t_padding,
                       log=False, full_shape=(20, 20),
                       sample_shape=(12, 12, 24), n_epoch=2,
                       batch_size=2, n_batches=2,
                       out_dir_root=None):
    """Test basic spatiotemporal model training
    for first conditional moment."""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR,
                          'spatiotemporal',
                          'gen_3x_4x_2f.json')

    Sup3rCondMom.seed()
    model = Sup3rCondMom(fp_gen, learning_rate=1e-4)

    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 1),
                            val_split=0.005,
                            worker_kwargs=dict(max_workers=1))

    batch_handler = BatchHandlerMom1([handler], batch_size=batch_size,
                                     s_enhance=3, t_enhance=4,
                                     n_batches=n_batches,
                                     end_t_padding=end_t_padding)

    with tempfile.TemporaryDirectory() as td:
        if out_dir_root is None:
            out_dir_root = td
        model.train(batch_handler,
                    input_resolution={'spatial': '12km', 'temporal': '60min'},
                    n_epoch=n_epoch,
                    checkpoint_int=2,
                    out_dir=os.path.join(out_dir_root, 'test_{epoch}'))

        # test save/load functionality
        out_dir = os.path.join(out_dir_root, 'st_cond_mom')
        model.save(out_dir)


@pytest.mark.parametrize('FEATURES, t_enhance_mode',
                         [(['U_100m', 'V_100m'], 'constant'),
                          (['U_100m', 'V_100m'], 'linear')])
def test_train_st_mom1_sf(FEATURES,
                          t_enhance_mode,
                          end_t_padding=False,
                          log=False, full_shape=(20, 20),
                          sample_shape=(12, 12, 24), n_epoch=2,
                          batch_size=2, n_batches=2,
                          temporal_slice=slice(None, None, 1),
                          out_dir_root=None):
    """Test basic spatiotemporal model training for first conditional moment
    of the subfilter velocity."""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR,
                          'spatiotemporal',
                          'gen_3x_4x_2f.json')

    Sup3rCondMom.seed()
    model = Sup3rCondMom(fp_gen, learning_rate=1e-4)

    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=temporal_slice,
                            val_split=0.005,
                            worker_kwargs=dict(max_workers=1))

    batch_handler = BatchHandlerMom1SF(
        [handler], batch_size=batch_size,
        s_enhance=3, t_enhance=4,
        n_batches=n_batches,
        end_t_padding=end_t_padding,
        temporal_enhancing_method=t_enhance_mode)

    with tempfile.TemporaryDirectory() as td:
        if out_dir_root is None:
            out_dir_root = td
        model.train(batch_handler,
                    input_resolution={'spatial': '12km', 'temporal': '60min'},
                    n_epoch=n_epoch,
                    checkpoint_int=2,
                    out_dir=os.path.join(out_dir_root, 'test_{epoch}'))

        # test save/load functionality
        out_dir = os.path.join(out_dir_root, 'st_cond_mom')
        model.save(out_dir)


@pytest.mark.parametrize('FEATURES',
                         (['U_100m', 'V_100m'],))
def test_train_st_mom2(FEATURES,
                       end_t_padding=False,
                       log=False, full_shape=(20, 20),
                       sample_shape=(12, 12, 16), n_epoch=2,
                       batch_size=2, n_batches=2,
                       temporal_slice=slice(None, None, 1),
                       out_dir_root=None,
                       model_mom1_dir=None):
    """Test basic spatiotemporal model training
    for second conditional moment"""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    # Load Mom 1 Model
    if model_mom1_dir is None:
        fp_gen = os.path.join(CONFIG_DIR,
                              'spatiotemporal',
                              'gen_3x_4x_2f.json')
        model_mom1 = Sup3rCondMom(fp_gen)
    else:
        fp_gen = os.path.join(model_mom1_dir, 'model_params.json')
        model_mom1 = Sup3rCondMom(fp_gen).load(model_mom1_dir)

    Sup3rCondMom.seed()
    fp_gen_mom2 = os.path.join(CONFIG_DIR,
                               'spatiotemporal',
                               'gen_3x_4x_2f.json')
    model_mom2 = Sup3rCondMom(fp_gen_mom2, learning_rate=1e-4)

    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=temporal_slice,
                            val_split=0.005,
                            worker_kwargs=dict(max_workers=1))

    batch_handler = BatchHandlerMom2([handler], batch_size=batch_size,
                                     s_enhance=3, t_enhance=4,
                                     n_batches=n_batches,
                                     model_mom1=model_mom1,
                                     end_t_padding=end_t_padding)

    with tempfile.TemporaryDirectory() as td:
        if out_dir_root is None:
            out_dir_root = td
        model_mom2.train(batch_handler,
                         input_resolution={'spatial': '12km',
                                           'temporal': '60min'},
                         n_epoch=n_epoch,
                         checkpoint_int=2,
                         out_dir=os.path.join(out_dir_root, 'test_{epoch}'))
        # test save/load functionality
        out_dir = os.path.join(out_dir_root, 'st_cond_mom')
        model_mom2.save(out_dir)


@pytest.mark.parametrize('FEATURES',
                         (['U_100m', 'V_100m'],))
def test_train_st_mom2_sf(FEATURES,
                          t_enhance_mode='constant',
                          end_t_padding=False,
                          log=False, full_shape=(20, 20),
                          sample_shape=(12, 12, 16), n_epoch=2,
                          temporal_slice=slice(None, None, 1),
                          batch_size=2, n_batches=2,
                          out_dir_root=None,
                          model_mom1_dir=None):
    """Test basic spatial model training for second conditional moment
    of subfilter velocity"""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    # Load Mom 1 Model
    if model_mom1_dir is None:
        fp_gen = os.path.join(CONFIG_DIR,
                              'spatiotemporal',
                              'gen_3x_4x_2f.json')
        model_mom1 = Sup3rCondMom(fp_gen)
    else:
        fp_gen = os.path.join(model_mom1_dir, 'model_params.json')
        model_mom1 = Sup3rCondMom(fp_gen).load(model_mom1_dir)

    Sup3rCondMom.seed()
    fp_gen_mom2 = os.path.join(CONFIG_DIR,
                               'spatiotemporal',
                               'gen_3x_4x_2f.json')
    model_mom2 = Sup3rCondMom(fp_gen_mom2, learning_rate=1e-4)

    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=temporal_slice,
                            val_split=0.005,
                            worker_kwargs=dict(max_workers=1))

    batch_handler = BatchHandlerMom2SF(
        [handler], batch_size=batch_size,
        s_enhance=3, t_enhance=4,
        n_batches=n_batches,
        model_mom1=model_mom1,
        end_t_padding=end_t_padding,
        temporal_enhancing_method=t_enhance_mode)

    with tempfile.TemporaryDirectory() as td:
        if out_dir_root is None:
            out_dir_root = td
        model_mom2.train(batch_handler,
                         input_resolution={'spatial': '12km',
                                           'temporal': '60min'},
                         n_epoch=n_epoch,
                         checkpoint_int=2,
                         out_dir=os.path.join(out_dir_root, 'test_{epoch}'))
        # test save/load functionality
        out_dir = os.path.join(out_dir_root, 'st_cond_mom')
        model_mom2.save(out_dir)


@pytest.mark.parametrize('FEATURES',
                         (['U_100m', 'V_100m'],))
def test_train_st_mom2_sep(FEATURES,
                           end_t_padding=False,
                           log=False, full_shape=(20, 20),
                           sample_shape=(12, 12, 16), n_epoch=2,
                           temporal_slice=slice(None, None, 1),
                           batch_size=2, n_batches=2,
                           out_dir_root=None):
    """Test basic spatiotemporal model training
    for second conditional moment separate from
    first moment"""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    Sup3rCondMom.seed()
    fp_gen_mom2 = os.path.join(CONFIG_DIR,
                               'spatiotemporal',
                               'gen_3x_4x_2f.json')
    model_mom2 = Sup3rCondMom(fp_gen_mom2, learning_rate=1e-4)

    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=temporal_slice,
                            val_split=0.005,
                            worker_kwargs=dict(max_workers=1))

    batch_handler = BatchHandlerMom2Sep([handler],
                                        batch_size=batch_size,
                                        s_enhance=3,
                                        t_enhance=4,
                                        n_batches=n_batches,
                                        end_t_padding=end_t_padding)

    with tempfile.TemporaryDirectory() as td:
        if out_dir_root is None:
            out_dir_root = td
        model_mom2.train(batch_handler,
                         input_resolution={'spatial': '12km',
                                           'temporal': '60min'},
                         n_epoch=n_epoch,
                         checkpoint_int=2,
                         out_dir=os.path.join(out_dir_root, 'test_{epoch}'))
        # test save/load functionality
        out_dir = os.path.join(out_dir_root, 'st_cond_mom')
        model_mom2.save(out_dir)


@pytest.mark.parametrize('FEATURES',
                         (['U_100m', 'V_100m'],))
def test_train_st_mom2_sep_sf(FEATURES,
                              t_enhance_mode='constant',
                              end_t_padding=False,
                              log=False, full_shape=(20, 20),
                              sample_shape=(12, 12, 16), n_epoch=2,
                              batch_size=2, n_batches=2,
                              out_dir_root=None):
    """Test basic spatial model training for second conditional moment
    of subfilter velocity separate from first moment"""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    Sup3rCondMom.seed()
    fp_gen_mom2 = os.path.join(CONFIG_DIR,
                               'spatiotemporal',
                               'gen_3x_4x_2f.json')
    model_mom2 = Sup3rCondMom(fp_gen_mom2, learning_rate=1e-4)

    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 1),
                            val_split=0.005,
                            worker_kwargs=dict(max_workers=1))

    batch_handler = BatchHandlerMom2SepSF(
        [handler],
        batch_size=batch_size,
        s_enhance=3, t_enhance=4,
        n_batches=n_batches,
        end_t_padding=end_t_padding,
        temporal_enhancing_method=t_enhance_mode)

    with tempfile.TemporaryDirectory() as td:
        if out_dir_root is None:
            out_dir_root = td
        model_mom2.train(batch_handler,
                         input_resolution={'spatial': '12km',
                                           'temporal': '60min'},
                         n_epoch=n_epoch,
                         checkpoint_int=2,
                         out_dir=os.path.join(out_dir_root, 'test_{epoch}'))
        # test save/load functionality
        out_dir = os.path.join(out_dir_root, 'st_cond_mom')
        model_mom2.save(out_dir)
