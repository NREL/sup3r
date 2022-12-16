# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN for solar climate change
applications"""
import os
import pytest
import numpy as np
import tempfile

from rex import init_logger

from sup3r import TEST_DATA_DIR
from sup3r.models import WindCondMom
from sup3r.preprocessing.data_handling import DataHandlerH5
from sup3r.preprocessing.conditional_moment_batch_handling import (
    SpatialBatchHandlerMom1)


SHAPE = (20, 20)

INPUT_FILE_S = os.path.join(TEST_DATA_DIR, 'test_nsrdb_co_2018.h5')
FEATURES_S = ['clearsky_ratio', 'ghi', 'clearsky_ghi']
TARGET_S = (39.01, -105.13)

INPUT_FILE_W = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
FEATURES_W = ['U_100m', 'V_100m', 'temperature_100m', 'topography']
TARGET_W = (39.01, -105.15)

FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
TARGET_COORD = (39.01, -105.15)


@pytest.mark.parametrize('custom_layer', ['Sup3rAdder', 'Sup3rConcat'])
def test_wind_non_cc_hi_res_topo(custom_layer, log=False, out_dir_root=None,
                                 n_epoch=1, n_batches=10, batch_size=8):
    """Test a special wind model for non cc with the custom Sup3rAdder or
    Sup3rConcat layer that adds/concatenates hi-res topography in the middle of
    the network."""

    if log:
        init_logger('sup3r', log_level='DEBUG')

    handler = DataHandlerH5(FP_WTK,
                            ('U_100m', 'V_100m', 'topography'),
                            target=TARGET_COORD, shape=SHAPE,
                            temporal_slice=slice(None, None, 10),
                            val_split=0.1,
                            sample_shape=(20, 20),
                            max_workers=1,
                            train_only_features=tuple())

    batcher = SpatialBatchHandlerMom1([handler],
                                      batch_size=batch_size,
                                      n_batches=n_batches,
                                      s_enhance=2)

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

    WindCondMom.seed()
    model = WindCondMom(gen_model, learning_rate=1e-4)

    with tempfile.TemporaryDirectory() as td:
        if out_dir_root is None:
            out_dir_root = td
        model.train(batcher, n_epoch=n_epoch,
                    checkpoint_int=None,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        assert 'test_0' in os.listdir(td)
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
