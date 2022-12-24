# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN for solar climate change
applications"""
import os
import pytest
import numpy as np
import tempfile

from rex import init_logger

from sup3r import TEST_DATA_DIR
from sup3r import CONFIG_DIR
from sup3r.models import WindGan
from sup3r.preprocessing.data_handling import (DataHandlerH5WindCC,
                                               DataHandlerH5)
from sup3r.preprocessing.batch_handling import (BatchHandlerCC,
                                                SpatialBatchHandlerCC,
                                                SpatialBatchHandler)


SHAPE = (20, 20)

INPUT_FILE_S = os.path.join(TEST_DATA_DIR, 'test_nsrdb_co_2018.h5')
FEATURES_S = ['clearsky_ratio', 'ghi', 'clearsky_ghi']
TARGET_S = (39.01, -105.13)

INPUT_FILE_W = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
FEATURES_W = ['U_100m', 'V_100m', 'temperature_100m', 'topography']
TARGET_W = (39.01, -105.15)

FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
TARGET_COORD = (39.01, -105.15)


def test_wind_cc_model(log=False):
    """Test the wind climate change wtk super res model.

    NOTE that the full 10x model is too big to train on the 20x20 test data.
    """

    handler = DataHandlerH5WindCC(INPUT_FILE_W, FEATURES_W,
                                  target=TARGET_W, shape=SHAPE,
                                  temporal_slice=slice(None, None, 2),
                                  time_roll=-7,
                                  sample_shape=(20, 20, 96),
                                  worker_kwargs=dict(max_workers=1),
                                  train_only_features=tuple())

    batcher = BatchHandlerCC([handler], batch_size=2, n_batches=2,
                             s_enhance=4, sub_daily_shape=None)

    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_4x_24x_3f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    WindGan.seed()
    model = WindGan(fp_gen, fp_disc, learning_rate=1e-4)

    with tempfile.TemporaryDirectory() as td:
        model.train(batcher, n_epoch=1,
                    weight_gen_advers=0.0,
                    train_gen=True, train_disc=False,
                    checkpoint_int=None,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        assert 'test_0' in os.listdir(td)
        assert model.meta['class'] == 'WindGan'
        assert 'topography' in batcher.output_features
        assert 'topography' not in model.output_features
        assert len(model.output_features) == len(FEATURES_W) - 1

    x = np.random.uniform(0, 1, (1, 4, 4, 4, 4))
    y = model.generate(x)
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == x.shape[1] * 4
    assert y.shape[2] == x.shape[2] * 4
    assert y.shape[3] == x.shape[3] * 24
    assert y.shape[4] == x.shape[4] - 1


def test_wind_cc_model_spatial(log=False):
    """Test the wind climate change wtk super res model with spatial
    enhancement only.
    """
    handler = DataHandlerH5WindCC(INPUT_FILE_W,
                                  ('U_100m', 'V_100m', 'topography'),
                                  target=TARGET_W, shape=SHAPE,
                                  temporal_slice=slice(None, None, 2),
                                  time_roll=-7,
                                  val_split=0.1,
                                  sample_shape=(20, 20),
                                  worker_kwargs=dict(max_workers=1),
                                  train_only_features=tuple())

    batcher = SpatialBatchHandlerCC([handler], batch_size=2, n_batches=2,
                                    s_enhance=2)

    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')

    WindGan.seed()
    model = WindGan(fp_gen, fp_disc, learning_rate=1e-4)

    with tempfile.TemporaryDirectory() as td:
        model.train(batcher, n_epoch=1,
                    weight_gen_advers=0.0,
                    train_gen=True, train_disc=False,
                    checkpoint_int=None,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        assert 'test_0' in os.listdir(td)
        assert model.meta['output_features'] == ['U_100m', 'V_100m']
        assert model.meta['class'] == 'WindGan'
        assert 'topography' in batcher.output_features
        assert 'topography' not in model.output_features

    x = np.random.uniform(0, 1, (4, 30, 30, 3))
    y = model.generate(x)
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == x.shape[1] * 2
    assert y.shape[2] == x.shape[2] * 2
    assert y.shape[3] == x.shape[3] - 1


@pytest.mark.parametrize('custom_layer', ['Sup3rAdder', 'Sup3rConcat'])
def test_wind_hi_res_topo(custom_layer, log=False):
    """Test a special wind cc model with the custom Sup3rAdder or Sup3rConcat
    layer that adds/concatenates hi-res topography in the middle of the
    network."""

    handler = DataHandlerH5WindCC(INPUT_FILE_W,
                                  ('U_100m', 'V_100m', 'topography'),
                                  target=TARGET_W, shape=SHAPE,
                                  temporal_slice=slice(None, None, 2),
                                  time_roll=-7,
                                  val_split=0.1,
                                  sample_shape=(20, 20),
                                  worker_kwargs=dict(max_workers=1),
                                  train_only_features=tuple())

    batcher = SpatialBatchHandlerCC([handler], batch_size=2, n_batches=2,
                                    s_enhance=2)

    if log:
        init_logger('sup3r', log_level='DEBUG')

    gen_model = [{"class": "FlexiblePadding",
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

    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')

    WindGan.seed()
    model = WindGan(gen_model, fp_disc, learning_rate=1e-4)

    with tempfile.TemporaryDirectory() as td:
        model.train(batcher, n_epoch=1,
                    weight_gen_advers=0.0,
                    train_gen=True, train_disc=False,
                    checkpoint_int=None,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        assert 'test_0' in os.listdir(td)
        assert model.meta['output_features'] == ['U_100m', 'V_100m']
        assert model.meta['class'] == 'WindGan'
        assert 'topography' in batcher.output_features
        assert 'topography' not in model.output_features

    x = np.random.uniform(0, 1, (4, 30, 30, 3))
    hi_res_topo = np.random.uniform(0, 1, (60, 60))

    with pytest.raises(RuntimeError):
        y = model.generate(x, exogenous_data=None)

    y = model.generate(x, exogenous_data=(None, hi_res_topo))

    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == x.shape[1] * 2
    assert y.shape[2] == x.shape[2] * 2
    assert y.shape[3] == x.shape[3] - 1


@pytest.mark.parametrize('custom_layer', ['Sup3rAdder', 'Sup3rConcat'])
def test_wind_non_cc_hi_res_topo(custom_layer, log=False):
    """Test a special wind model for non cc with the custom Sup3rAdder or
    Sup3rConcat layer that adds/concatenates hi-res topography in the middle of
    the network."""

    handler = DataHandlerH5(FP_WTK,
                            ('U_100m', 'V_100m', 'topography'),
                            target=TARGET_COORD, shape=SHAPE,
                            temporal_slice=slice(None, None, 10),
                            val_split=0.1,
                            sample_shape=(20, 20),
                            worker_kwargs=dict(max_workers=1),
                            train_only_features=tuple())

    batcher = SpatialBatchHandler([handler], batch_size=2, n_batches=2,
                                  s_enhance=2)

    if log:
        init_logger('sup3r', log_level='DEBUG')

    gen_model = [{"class": "FlexiblePadding",
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

    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')

    WindGan.seed()
    model = WindGan(gen_model, fp_disc, learning_rate=1e-4)

    with tempfile.TemporaryDirectory() as td:
        model.train(batcher, n_epoch=1,
                    weight_gen_advers=0.0,
                    train_gen=True, train_disc=False,
                    checkpoint_int=None,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        assert 'test_0' in os.listdir(td)
        assert model.meta['output_features'] == ['U_100m', 'V_100m']
        assert model.meta['class'] == 'WindGan'
        assert 'topography' in batcher.output_features
        assert 'topography' not in model.output_features

    x = np.random.uniform(0, 1, (4, 30, 30, 3))
    hi_res_topo = np.random.uniform(0, 1, (60, 60))

    with pytest.raises(RuntimeError):
        y = model.generate(x, exogenous_data=None)

    y = model.generate(x, exogenous_data=(None, hi_res_topo))

    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == x.shape[1] * 2
    assert y.shape[2] == x.shape[2] * 2
    assert y.shape[3] == x.shape[3] - 1
