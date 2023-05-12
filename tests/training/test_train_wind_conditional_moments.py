# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN for solar climate change
applications"""
import os
import pytest
import numpy as np
import tempfile

from rex import init_logger

from sup3r import CONFIG_DIR
from sup3r import TEST_DATA_DIR
from sup3r.models import WindCondMom
from sup3r.preprocessing.data_handling import DataHandlerH5
from sup3r.preprocessing.wind_conditional_moment_batch_handling import (
    WindSpatialBatchHandlerMom1,
    WindSpatialBatchHandlerMom1SF,
    WindSpatialBatchHandlerMom2,
    WindSpatialBatchHandlerMom2SF,
    WindSpatialBatchHandlerMom2Sep,
    WindSpatialBatchHandlerMom2SepSF,
    WindBatchHandlerMom1,
    WindBatchHandlerMom1SF,
    WindBatchHandlerMom2,
    WindBatchHandlerMom2SF,
    WindBatchHandlerMom2Sep,
    WindBatchHandlerMom2SepSF)


SHAPE = (20, 20)

INPUT_FILE_S = os.path.join(TEST_DATA_DIR, 'test_nsrdb_co_2018.h5')
FEATURES_S = ['clearsky_ratio', 'ghi', 'clearsky_ghi']
TARGET_S = (39.01, -105.13)

INPUT_FILE_W = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
FEATURES_W = ['U_100m', 'V_100m', 'temperature_100m', 'topography']
TARGET_W = (39.01, -105.15)

FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
TARGET_COORD = (39.01, -105.15)


def make_s_gen_model(custom_layer):
    """Make simple conditional moment model with
    flexible custom layer."""
    return [{"class": "FlexiblePadding",
             "paddings": [[0, 0], [3, 3], [3, 3], [0, 0]],
             "mode": "REFLECT"},
            {"class": "Conv2DTranspose", "filters": 64, "kernel_size": 3,
             "strides": 1, "activation": "relu"},
            {"class": "Cropping2D", "cropping": 4},

            {"class": "FlexiblePadding",
             "paddings": [[0, 0], [3, 3], [3, 3], [0, 0]],
             "mode": "REFLECT"},
            {"class": "Conv2DTranspose", "filters": 64,
             "kernel_size": 3, "strides": 1, "activation": "relu"},
            {"class": "Cropping2D", "cropping": 4},

            {"class": "FlexiblePadding",
             "paddings": [[0, 0], [3, 3], [3, 3], [0, 0]],
             "mode": "REFLECT"},
            {"class": "Conv2DTranspose", "filters": 64,
             "kernel_size": 3, "strides": 1, "activation": "relu"},
            {"class": "Cropping2D", "cropping": 4},
            {"class": "SpatialExpansion", "spatial_mult": 2},
            {"class": "Activation", "activation": "relu"},

            {"class": custom_layer},

            {"class": "FlexiblePadding",
             "paddings": [[0, 0], [3, 3], [3, 3], [0, 0]],
             "mode": "REFLECT"},
            {"class": "Conv2DTranspose", "filters": 2,
             "kernel_size": 3, "strides": 1, "activation": "relu"},
            {"class": "Cropping2D", "cropping": 4}]


@pytest.mark.parametrize('custom_layer, batch_class', [
                         ('Sup3rAdder', WindSpatialBatchHandlerMom1),
                         ('Sup3rConcat', WindSpatialBatchHandlerMom1),
                         ('Sup3rConcat', WindSpatialBatchHandlerMom1SF)])
def test_wind_non_cc_hi_res_topo_mom1(custom_layer, batch_class,
                                      log=False, out_dir_root=None,
                                      n_epoch=1, n_batches=2, batch_size=2):
    """Test spatial first conditional moment for wind model for non cc with
    the custom Sup3rAdder or Sup3rConcat layer that adds/concatenates hi-res
    topography in the middle of the network.
    Test for direct first moment or subfilter velocity."""

    if log:
        init_logger('sup3r', log_level='DEBUG')

    handler = DataHandlerH5(FP_WTK,
                            ('U_100m', 'V_100m', 'topography'),
                            target=TARGET_COORD, shape=SHAPE,
                            temporal_slice=slice(None, None, 10),
                            val_split=0.1,
                            sample_shape=(20, 20),
                            worker_kwargs=dict(max_workers=1),
                            train_only_features=tuple())

    batcher = batch_class([handler],
                          batch_size=batch_size,
                          n_batches=n_batches,
                          s_enhance=2)

    gen_model = make_s_gen_model(custom_layer)

    WindCondMom.seed()
    model = WindCondMom(gen_model, learning_rate=1e-4)

    with tempfile.TemporaryDirectory() as td:
        if out_dir_root is None:
            out_dir_root = td
        model.train(batcher, n_epoch=n_epoch,
                    checkpoint_int=None,
                    out_dir=os.path.join(out_dir_root, 'test_{epoch}'))

        assert 'test_0' in os.listdir(out_dir_root)
        assert model.meta['output_features'] == ['U_100m', 'V_100m']
        assert model.meta['class'] == 'WindCondMom'
        assert 'topography' in batcher.output_features
        assert 'topography' not in model.output_features

    x = np.random.uniform(0, 1, (4, 30, 30, 3))
    hi_res_topo = np.random.uniform(0, 1, (60, 60))

    y = model.generate(x, exogenous_data=(None, hi_res_topo))

    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == x.shape[1] * 2
    assert y.shape[2] == x.shape[2] * 2
    assert y.shape[3] == x.shape[3] - 1


@pytest.mark.parametrize('batch_class', [
                         WindBatchHandlerMom1,
                         WindBatchHandlerMom1SF])
def test_wind_non_cc_hi_res_st_topo_mom1(batch_class, log=False,
                                         out_dir_root=None,
                                         n_epoch=1, n_batches=2, batch_size=2):
    """Test spatiotemporal first conditional moment for wind model for non cc
    Sup3rConcat layer that concatenates hi-res topography in the middle of
    the network. Test for direct first moment or subfilter velocity."""

    if log:
        init_logger('sup3r', log_level='DEBUG')

    handler = DataHandlerH5(FP_WTK,
                            ('U_100m', 'V_100m', 'topography'),
                            target=TARGET_COORD, shape=SHAPE,
                            temporal_slice=slice(None, None, 1),
                            val_split=0.1,
                            sample_shape=(12, 12, 24),
                            worker_kwargs=dict(max_workers=1),
                            train_only_features=tuple())

    fp_gen = os.path.join(CONFIG_DIR,
                          'spatiotemporal',
                          'gen_3x_4x_topo_2f.json')

    WindCondMom.seed()
    model_mom1 = WindCondMom(fp_gen, learning_rate=1e-4)

    batcher = batch_class([handler],
                          batch_size=batch_size,
                          s_enhance=3, t_enhance=4,
                          model_mom1=model_mom1,
                          n_batches=n_batches)

    with tempfile.TemporaryDirectory() as td:
        if out_dir_root is None:
            out_dir_root = td
        model_mom1.train(batcher, n_epoch=n_epoch,
                         checkpoint_int=None,
                         out_dir=os.path.join(out_dir_root, 'test_{epoch}'))


@pytest.mark.parametrize('custom_layer, batch_class', [
                         ('Sup3rConcat', WindSpatialBatchHandlerMom2),
                         ('Sup3rConcat', WindSpatialBatchHandlerMom2Sep),
                         ('Sup3rConcat', WindSpatialBatchHandlerMom2SF),
                         ('Sup3rConcat', WindSpatialBatchHandlerMom2SepSF)])
def test_wind_non_cc_hi_res_topo_mom2(custom_layer, batch_class,
                                      log=False, out_dir_root=None,
                                      n_epoch=1, n_batches=2, batch_size=2):
    """Test spatial second conditional moment for wind model for non cc
    with the Sup3rConcat layer that concatenates hi-res topography in
    the middle of the network. Test for direct second moment or
    subfilter velocity.
    Test for separate or learning coupled with first moment."""

    if log:
        init_logger('sup3r', log_level='DEBUG')

    handler = DataHandlerH5(FP_WTK,
                            ('U_100m', 'V_100m', 'topography'),
                            target=TARGET_COORD, shape=SHAPE,
                            temporal_slice=slice(None, None, 10),
                            val_split=0.1,
                            sample_shape=(20, 20),
                            worker_kwargs=dict(max_workers=1),
                            train_only_features=tuple())

    gen_model = make_s_gen_model(custom_layer)

    WindCondMom.seed()
    model_mom1 = WindCondMom(gen_model, learning_rate=1e-4)
    model_mom2 = WindCondMom(gen_model, learning_rate=1e-4)

    batcher = batch_class([handler],
                          batch_size=batch_size,
                          model_mom1=model_mom1,
                          n_batches=n_batches,
                          s_enhance=2)

    with tempfile.TemporaryDirectory() as td:
        if out_dir_root is None:
            out_dir_root = td
        model_mom2.train(batcher, n_epoch=n_epoch,
                         checkpoint_int=None,
                         out_dir=os.path.join(out_dir_root, 'test_{epoch}'))


@pytest.mark.parametrize('batch_class', [
                         WindBatchHandlerMom2,
                         WindBatchHandlerMom2Sep,
                         WindBatchHandlerMom2SF,
                         WindBatchHandlerMom2SepSF])
def test_wind_non_cc_hi_res_st_topo_mom2(batch_class, log=False,
                                         out_dir_root=None,
                                         n_epoch=1, n_batches=2, batch_size=2):
    """Test spatiotemporal second conditional moment for wind model for non cc
    Sup3rConcat layer that concatenates hi-res topography in the middle of
    the network. Test for direct second moment or subfilter velocity.
    Test for separate or learning coupled with first moment."""

    if log:
        init_logger('sup3r', log_level='DEBUG')

    handler = DataHandlerH5(FP_WTK,
                            ('U_100m', 'V_100m', 'topography'),
                            target=TARGET_COORD, shape=SHAPE,
                            temporal_slice=slice(None, None, 1),
                            val_split=0.1,
                            sample_shape=(12, 12, 24),
                            worker_kwargs=dict(max_workers=1),
                            train_only_features=tuple())

    fp_gen = os.path.join(CONFIG_DIR,
                          'spatiotemporal',
                          'gen_3x_4x_topo_2f.json')

    WindCondMom.seed()
    model_mom1 = WindCondMom(fp_gen, learning_rate=1e-4)
    model_mom2 = WindCondMom(fp_gen, learning_rate=1e-4)

    batcher = batch_class([handler],
                          batch_size=batch_size,
                          s_enhance=3, t_enhance=4,
                          model_mom1=model_mom1,
                          n_batches=n_batches)

    with tempfile.TemporaryDirectory() as td:
        if out_dir_root is None:
            out_dir_root = td
        model_mom2.train(batcher, n_epoch=n_epoch,
                         checkpoint_int=None,
                         out_dir=os.path.join(out_dir_root,
                                              'test_{epoch}'))
