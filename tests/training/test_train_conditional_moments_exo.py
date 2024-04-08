# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN for solar climate change
applications"""
import os
import tempfile

import numpy as np
import pytest
from rex import init_logger

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

            {"class": custom_layer, "name": "topography"},

            {"class": "FlexiblePadding",
             "paddings": [[0, 0], [3, 3], [3, 3], [0, 0]],
             "mode": "REFLECT"},
            {"class": "Conv2DTranspose", "filters": 2,
             "kernel_size": 3, "strides": 1, "activation": "relu"},
            {"class": "Cropping2D", "cropping": 4}]


@pytest.mark.parametrize('custom_layer, batch_class', [
                         ('Sup3rAdder', SpatialBatchHandlerMom1),
                         ('Sup3rConcat', SpatialBatchHandlerMom1),
                         ('Sup3rConcat', SpatialBatchHandlerMom1SF)])
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
                            lr_only_features=tuple(),
                            hr_exo_features=('topography',))

    batcher = batch_class([handler],
                          batch_size=batch_size,
                          n_batches=n_batches,
                          s_enhance=2)

    gen_model = make_s_gen_model(custom_layer)

    Sup3rCondMom.seed()
    model = Sup3rCondMom(gen_model, learning_rate=1e-4)
    input_resolution = {'spatial': '8km', 'temporal': '60min'}
    with tempfile.TemporaryDirectory() as td:
        if out_dir_root is None:
            out_dir_root = td
        model.train(batcher,
                    input_resolution={'spatial': '8km', 'temporal': '60min'},
                    n_epoch=n_epoch,
                    checkpoint_int=None,
                    out_dir=os.path.join(out_dir_root, 'test_{epoch}'))
        assert f'test_{n_epoch-1}' in os.listdir(out_dir_root)
        assert model.meta['hr_out_features'] == ['U_100m', 'V_100m']
        assert model.meta['class'] == 'Sup3rCondMom'
        assert model.meta['input_resolution'] == input_resolution
        assert 'topography' in batcher.hr_exo_features
        assert 'topography' not in model.hr_out_features

    x = np.random.uniform(0, 1, (4, 30, 30, 3))
    hi_res_topo = np.random.uniform(0, 1, (4, 60, 60, 1))
    exo_tmp = {
        'topography': {
            'steps': [
                {'model': 0, 'combine_type': 'layer', 'data': hi_res_topo}]}}

    y = model.generate(x, exogenous_data=exo_tmp)

    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == x.shape[1] * 2
    assert y.shape[2] == x.shape[2] * 2
    assert y.shape[3] == x.shape[3] - 1


@pytest.mark.parametrize('batch_class', [
                         BatchHandlerMom1,
                         BatchHandlerMom1SF])
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
                            lr_only_features=tuple(),
                            hr_exo_features=('topography',))

    fp_gen = os.path.join(CONFIG_DIR,
                          'sup3rcc',
                          'gen_wind_3x_4x_2f.json')

    Sup3rCondMom.seed()
    model_mom1 = Sup3rCondMom(fp_gen, learning_rate=1e-4)

    batcher = batch_class([handler],
                          batch_size=batch_size,
                          s_enhance=3, t_enhance=4,
                          model_mom1=model_mom1,
                          n_batches=n_batches)

    with tempfile.TemporaryDirectory() as td:
        if out_dir_root is None:
            out_dir_root = td
        model_mom1.train(batcher,
                         input_resolution={'spatial': '12km',
                                           'temporal': '60min'},
                         n_epoch=n_epoch,
                         checkpoint_int=None,
                         out_dir=os.path.join(out_dir_root, 'test_{epoch}'))


@pytest.mark.parametrize('custom_layer, batch_class', [
                         ('Sup3rConcat', SpatialBatchHandlerMom2),
                         ('Sup3rConcat', SpatialBatchHandlerMom2Sep),
                         ('Sup3rConcat', SpatialBatchHandlerMom2SF),
                         ('Sup3rConcat', SpatialBatchHandlerMom2SepSF)])
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
                            lr_only_features=tuple(),
                            hr_exo_features=('topography',))

    gen_model = make_s_gen_model(custom_layer)

    Sup3rCondMom.seed()
    model_mom1 = Sup3rCondMom(gen_model, learning_rate=1e-4)
    model_mom2 = Sup3rCondMom(gen_model, learning_rate=1e-4)

    batcher = batch_class([handler],
                          batch_size=batch_size,
                          model_mom1=model_mom1,
                          n_batches=n_batches,
                          s_enhance=2)

    with tempfile.TemporaryDirectory() as td:
        if out_dir_root is None:
            out_dir_root = td
        model_mom2.train(batcher,
                         input_resolution={'spatial': '8km',
                                           'temporal': '60min'},
                         n_epoch=n_epoch,
                         checkpoint_int=None,
                         out_dir=os.path.join(out_dir_root, 'test_{epoch}'))


@pytest.mark.parametrize('batch_class', [
                         BatchHandlerMom2,
                         BatchHandlerMom2Sep,
                         BatchHandlerMom2SF,
                         BatchHandlerMom2SepSF])
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
                            lr_only_features=tuple(),
                            hr_exo_features=('topography',))

    fp_gen = os.path.join(CONFIG_DIR,
                          'sup3rcc',
                          'gen_wind_3x_4x_2f.json')

    Sup3rCondMom.seed()
    model_mom1 = Sup3rCondMom(fp_gen, learning_rate=1e-4)
    model_mom2 = Sup3rCondMom(fp_gen, learning_rate=1e-4)

    batcher = batch_class([handler],
                          batch_size=batch_size,
                          s_enhance=3, t_enhance=4,
                          model_mom1=model_mom1,
                          n_batches=n_batches)

    with tempfile.TemporaryDirectory() as td:
        if out_dir_root is None:
            out_dir_root = td
        model_mom2.train(batcher,
                         input_resolution={'spatial': '12km',
                                           'temporal': '60min'},
                         n_epoch=n_epoch,
                         checkpoint_int=None,
                         out_dir=os.path.join(out_dir_root,
                                              'test_{epoch}'))
