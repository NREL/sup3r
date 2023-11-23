"""Test the basic training of super resolution GAN for solar climate change
applications"""
import os
import tempfile

import numpy as np
import pytest
from rex import init_logger

from sup3r import CONFIG_DIR, TEST_DATA_DIR
from sup3r.models import Sup3rGan
from sup3r.models.data_centric import Sup3rGanDC
from sup3r.preprocessing.batch_handling import (
    BatchHandlerDC,
    SpatialBatchHandler,
    SpatialBatchHandlerCC,
)
from sup3r.preprocessing.data_handling import (
    DataHandlerDCforH5,
    DataHandlerH5,
    DataHandlerH5WindCC,
)

SHAPE = (20, 20)

INPUT_FILE_S = os.path.join(TEST_DATA_DIR, 'test_nsrdb_co_2018.h5')
FEATURES_S = ['clearsky_ratio', 'ghi', 'clearsky_ghi']
TARGET_S = (39.01, -105.13)

INPUT_FILE_W = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
FEATURES_W = ['temperature_100m', 'U_100m', 'V_100m', 'topography']
TARGET_W = (39.01, -105.15)

FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
TARGET_COORD = (39.01, -105.15)


@pytest.mark.parametrize('CustomLayer', ['Sup3rAdder', 'Sup3rConcat'])
def test_wind_hi_res_topo_with_train_only(CustomLayer, log=False):
    """Test a special wind cc model with the custom Sup3rAdder or Sup3rConcat
    layer that adds/concatenates hi-res topography in the middle of the
    network. This also includes a train only feature"""

    handler = DataHandlerH5WindCC(INPUT_FILE_W,
                                  FEATURES_W,
                                  target=TARGET_W, shape=SHAPE,
                                  temporal_slice=slice(None, None, 2),
                                  time_roll=-7,
                                  val_split=0.1,
                                  sample_shape=(20, 20),
                                  worker_kwargs=dict(max_workers=1),
                                  lr_only_features=['temperature_100m'],
                                  hr_exo_features=['topography'])
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

                 {"class": CustomLayer, "name": "topography"},

                 {"class": "FlexiblePadding",
                  "paddings": [[0, 0], [3, 3], [3, 3], [0, 0]],
                  "mode": "REFLECT"},
                 {"class": "Conv2DTranspose", "filters": 2,
                  "kernel_size": 3, "strides": 1, "activation": "relu"},
                 {"class": "Cropping2D", "cropping": 4}]

    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(gen_model, fp_disc, learning_rate=1e-4)

    with tempfile.TemporaryDirectory() as td:
        model.train(batcher,
                    input_resolution={'spatial': '16km',
                                      'temporal': '3600min'},
                    n_epoch=1,
                    weight_gen_advers=0.0,
                    train_gen=True, train_disc=False,
                    checkpoint_int=None,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        assert model.lr_features == FEATURES_W
        assert model.hr_out_features == ['U_100m', 'V_100m']
        assert model.hr_exo_features == ['topography']
        assert 'test_0' in os.listdir(td)
        assert model.meta['hr_out_features'] == ['U_100m', 'V_100m']
        assert model.meta['class'] == 'Sup3rGan'
        assert 'topography' in batcher.hr_exo_features
        assert 'topography' not in model.hr_out_features

    x = np.random.uniform(0, 1, (4, 30, 30, 4))
    hi_res_topo = np.random.uniform(0, 1, (4, 60, 60, 1))

    with pytest.raises(RuntimeError):
        y = model.generate(x, exogenous_data=None)

    exo_tmp = {
        'topography': {
            'steps': [
                {'model': 0, 'combine_type': 'layer', 'data': hi_res_topo}]}}
    y = model.generate(x, exogenous_data=exo_tmp)

    assert y.dtype == np.float32
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == x.shape[1] * 2
    assert y.shape[2] == x.shape[2] * 2
    assert y.shape[3] == x.shape[3] - 2


@pytest.mark.parametrize('CustomLayer', ['Sup3rAdder', 'Sup3rConcat'])
def test_wind_hi_res_topo(CustomLayer, log=False):
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
                                  lr_only_features=(),
                                  hr_exo_features=('topography',))

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

                 {"class": CustomLayer, "name": "topography"},

                 {"class": "FlexiblePadding",
                  "paddings": [[0, 0], [3, 3], [3, 3], [0, 0]],
                  "mode": "REFLECT"},
                 {"class": "Conv2DTranspose", "filters": 2,
                  "kernel_size": 3, "strides": 1, "activation": "relu"},
                 {"class": "Cropping2D", "cropping": 4}]

    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(gen_model, fp_disc, learning_rate=1e-4)

    with tempfile.TemporaryDirectory() as td:
        model.train(batcher,
                    input_resolution={'spatial': '16km',
                                      'temporal': '3600min'},
                    n_epoch=1,
                    weight_gen_advers=0.0,
                    train_gen=True, train_disc=False,
                    checkpoint_int=None,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        assert 'test_0' in os.listdir(td)
        assert model.meta['hr_out_features'] == ['U_100m', 'V_100m']
        assert model.meta['class'] == 'Sup3rGan'
        assert 'topography' in batcher.hr_exo_features
        assert 'topography' not in model.hr_out_features

    x = np.random.uniform(0, 1, (4, 30, 30, 3))
    hi_res_topo = np.random.uniform(0, 1, (4, 60, 60, 1))

    with pytest.raises(RuntimeError):
        y = model.generate(x, exogenous_data=None)

    exo_tmp = {
        'topography': {
            'steps': [
                {'model': 0, 'combine_type': 'layer', 'data': hi_res_topo}]}}
    y = model.generate(x, exogenous_data=exo_tmp)

    assert y.dtype == np.float32
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == x.shape[1] * 2
    assert y.shape[2] == x.shape[2] * 2
    assert y.shape[3] == x.shape[3] - 1


@pytest.mark.parametrize('CustomLayer', ['Sup3rAdder', 'Sup3rConcat'])
def test_wind_non_cc_hi_res_topo(CustomLayer, log=False):
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
                            lr_only_features=tuple(),
                            hr_exo_features=('topography',))

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

                 {"class": CustomLayer, "name": "topography"},

                 {"class": "FlexiblePadding",
                  "paddings": [[0, 0], [3, 3], [3, 3], [0, 0]],
                  "mode": "REFLECT"},
                 {"class": "Conv2DTranspose", "filters": 2,
                  "kernel_size": 3, "strides": 1, "activation": "relu"},
                 {"class": "Cropping2D", "cropping": 4}]

    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(gen_model, fp_disc, learning_rate=1e-4)

    with tempfile.TemporaryDirectory() as td:
        model.train(batcher,
                    input_resolution={'spatial': '16km',
                                      'temporal': '3600min'},
                    n_epoch=1,
                    weight_gen_advers=0.0,
                    train_gen=True, train_disc=False,
                    checkpoint_int=None,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        assert 'test_0' in os.listdir(td)
        assert model.meta['hr_out_features'] == ['U_100m', 'V_100m']
        assert model.meta['class'] == 'Sup3rGan'
        assert 'topography' in batcher.hr_exo_features
        assert 'topography' not in model.hr_out_features

    x = np.random.uniform(0, 1, (4, 30, 30, 3))
    hi_res_topo = np.random.uniform(0, 1, (4, 60, 60, 1))

    with pytest.raises(RuntimeError):
        y = model.generate(x, exogenous_data=None)

    exo_tmp = {
        'topography': {
            'steps': [
                {'model': 0, 'combine_type': 'layer', 'data': hi_res_topo}]}}
    y = model.generate(x, exogenous_data=exo_tmp)

    assert y.dtype == np.float32
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == x.shape[1] * 2
    assert y.shape[2] == x.shape[2] * 2
    assert y.shape[3] == x.shape[3] - 1


@pytest.mark.parametrize('CustomLayer', ['Sup3rAdder', 'Sup3rConcat'])
def test_wind_dc_hi_res_topo(CustomLayer, log=False):
    """Test a special data centric wind model with the custom Sup3rAdder or
    Sup3rConcat layer that adds/concatenates hi-res topography in the middle of
    the network."""

    handler = DataHandlerDCforH5(INPUT_FILE_W,
                                 ('U_100m', 'V_100m', 'topography'),
                                 target=TARGET_W, shape=SHAPE,
                                 temporal_slice=slice(None, None, 2),
                                 val_split=0.0,
                                 sample_shape=(20, 20, 8),
                                 worker_kwargs=dict(max_workers=1),
                                 lr_only_features=tuple(),
                                 hr_exo_features=('topography',))

    batcher = BatchHandlerDC([handler], batch_size=2, n_batches=2,
                             s_enhance=2)

    if log:
        init_logger('sup3r', log_level='DEBUG')

    gen_model = [{"class": "FlexiblePadding",
                  "paddings": [[0, 0], [2, 2], [2, 2], [2, 2], [0, 0]],
                  "mode": "REFLECT"},
                 {"class": "Conv3D", "filters": 64, "kernel_size": 3,
                  "strides": 1, "activation": "relu"},
                 {"class": "Cropping3D", "cropping": 1},

                 {"class": "FlexiblePadding",
                  "paddings": [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]],
                  "mode": "REFLECT"},
                 {"class": "Conv3D", "filters": 64,
                  "kernel_size": 3, "strides": 1, "activation": "relu"},
                 {"class": "Cropping3D", "cropping": 2},

                 {"class": "FlexiblePadding",
                  "paddings": [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]],
                  "mode": "REFLECT"},
                 {"class": "Conv3D", "filters": 64,
                  "kernel_size": 3, "strides": 1, "activation": "relu"},
                 {"class": "Cropping3D", "cropping": 2},
                 {"class": "SpatioTemporalExpansion", "spatial_mult": 2},
                 {"class": "Activation", "activation": "relu"},

                 {"class": CustomLayer, "name": "topography"},

                 {"class": "FlexiblePadding",
                  "paddings": [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]],
                  "mode": "REFLECT"},
                 {"class": "Conv3D", "filters": 2,
                  "kernel_size": 3, "strides": 1, "activation": "relu"},
                 {"class": "Cropping3D", "cropping": 2}]

    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGanDC.seed()
    model = Sup3rGanDC(gen_model, fp_disc, learning_rate=1e-4)

    with tempfile.TemporaryDirectory() as td:
        model.train(batcher,
                    input_resolution={'spatial': '16km',
                                      'temporal': '3600min'},
                    n_epoch=1,
                    weight_gen_advers=0.0,
                    train_gen=True, train_disc=False,
                    checkpoint_int=None,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        assert 'test_0' in os.listdir(td)
        assert model.meta['hr_out_features'] == ['U_100m', 'V_100m']
        assert model.meta['class'] == 'Sup3rGanDC'
        assert 'topography' in batcher.hr_exo_features
        assert 'topography' not in model.hr_out_features

    x = np.random.uniform(0, 1, (1, 30, 30, 4, 3))
    hi_res_topo = np.random.uniform(0, 1, (1, 60, 60, 4, 1))

    with pytest.raises(RuntimeError):
        y = model.generate(x, exogenous_data=None)

    exo_tmp = {
        'topography': {
            'steps': [
                {'model': 0, 'combine_type': 'layer', 'data': hi_res_topo}]}}
    y = model.generate(x, exogenous_data=exo_tmp)

    assert y.dtype == np.float32
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == x.shape[1] * 2
    assert y.shape[2] == x.shape[2] * 2
    assert y.shape[3] == x.shape[3]
    assert y.shape[4] == x.shape[4] - 1
