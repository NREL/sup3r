# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN"""
import os
import tempfile

# import json
import pytest
from rex import init_logger

from sup3r import CONFIG_DIR, TEST_DATA_DIR
from sup3r.models import Sup3rCondMom
from sup3r.preprocessing import (
    BatchHandlerMom1,
    BatchHandlerMom1SF,
    BatchHandlerMom2,
    BatchHandlerMom2Sep,
    BatchHandlerMom2SepSF,
    BatchHandlerMom2SF,
    DataHandlerH5,
)

FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
TARGET_COORD = (39.01, -105.15)
FEATURES = ['U_100m', 'V_100m']
TRAIN_FEATURES = None


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
                            time_slice=slice(None, None, 1),
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
                          time_slice=slice(None, None, 1),
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
                            time_slice=time_slice,
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
                       time_slice=slice(None, None, 1),
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
                            time_slice=time_slice,
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
                          time_slice=slice(None, None, 1),
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
                            time_slice=time_slice,
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
                           time_slice=slice(None, None, 1),
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
                            time_slice=time_slice,
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
                            time_slice=slice(None, None, 1),
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
