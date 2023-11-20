# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN for solar climate change
applications"""
import os
import tempfile

import numpy as np
from rex import init_logger
from tensorflow.keras.losses import MeanAbsoluteError

from sup3r import CONFIG_DIR, TEST_DATA_DIR
from sup3r.models import SolarCC, Sup3rGan
from sup3r.preprocessing.batch_handling import (
    BatchHandlerCC,
    SpatialBatchHandlerCC,
)
from sup3r.preprocessing.data_handling import DataHandlerH5SolarCC

SHAPE = (20, 20)

INPUT_FILE_S = os.path.join(TEST_DATA_DIR, 'test_nsrdb_co_2018.h5')
FEATURES_S = ['clearsky_ratio', 'ghi', 'clearsky_ghi']
TARGET_S = (39.01, -105.13)

INPUT_FILE_W = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
FEATURES_W = ['U_100m', 'V_100m', 'temperature_100m', 'topography']
TARGET_W = (39.01, -105.15)


def test_solar_cc_model(log=False):
    """Test the solar climate change nsrdb super res model.

    NOTE that the full 10x model is too big to train on the 20x20 test data.
    """

    handler = DataHandlerH5SolarCC(INPUT_FILE_S, FEATURES_S,
                                   target=TARGET_S, shape=SHAPE,
                                   temporal_slice=slice(None, None, 2),
                                   time_roll=-7,
                                   sample_shape=(20, 20, 72),
                                   worker_kwargs=dict(max_workers=1))

    batcher = BatchHandlerCC([handler], batch_size=2, n_batches=2,
                             s_enhance=1, sub_daily_shape=24)

    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'sup3rcc/gen_solar_1x_8x_1f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4,
                     loss='MeanAbsoluteError')

    with tempfile.TemporaryDirectory() as td:
        model.train(batcher,
                    input_resolution={'spatial': '4km', 'temporal': '40min'},
                    n_epoch=1,
                    weight_gen_advers=0.0,
                    train_gen=True, train_disc=False,
                    checkpoint_int=None,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        assert 'test_0' in os.listdir(td)
        assert model.meta['hr_out_features'] == ['clearsky_ratio']
        assert model.meta['class'] == 'Sup3rGan'

        out_dir = os.path.join(td, 'cc_gan')
        model.save(out_dir)
        loaded = model.load(out_dir)

        assert isinstance(model.loss_fun, MeanAbsoluteError)
        assert isinstance(loaded.loss_fun, MeanAbsoluteError)
        assert model.meta['class'] == 'Sup3rGan'
        assert loaded.meta['class'] == 'Sup3rGan'

    x = np.random.uniform(0, 1, (1, 30, 30, 3, 1))
    y = model.generate(x)
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == x.shape[1]
    assert y.shape[2] == x.shape[2]
    assert y.shape[3] == x.shape[3] * 8
    assert y.shape[4] == x.shape[4]


def test_solar_cc_model_spatial(log=False):
    """Test the solar climate change nsrdb super res model with spatial
    enhancement only.
    """

    handler = DataHandlerH5SolarCC(INPUT_FILE_S, FEATURES_S,
                                   target=TARGET_S, shape=SHAPE,
                                   temporal_slice=slice(None, None, 2),
                                   time_roll=-7,
                                   val_split=0.1,
                                   sample_shape=(20, 20),
                                   worker_kwargs=dict(max_workers=1))

    batcher = SpatialBatchHandlerCC([handler], batch_size=2, n_batches=2,
                                    s_enhance=5)

    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'sup3rcc/gen_solar_5x_1x_1f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)

    with tempfile.TemporaryDirectory() as td:
        model.train(batcher,
                    input_resolution={'spatial': '25km', 'temporal': '15min'},
                    n_epoch=1,
                    weight_gen_advers=0.0,
                    train_gen=True, train_disc=False,
                    checkpoint_int=None,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        assert 'test_0' in os.listdir(td)
        assert model.meta['hr_out_features'] == ['clearsky_ratio']
        assert model.meta['class'] == 'Sup3rGan'

    x = np.random.uniform(0, 1, (4, 10, 10, 1))
    y = model.generate(x)
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == x.shape[1] * 5
    assert y.shape[2] == x.shape[2] * 5
    assert y.shape[3] == x.shape[3]


def test_solar_custom_loss(log=False):
    """Test custom solar loss with only disc and content over daylight hours"""
    handler = DataHandlerH5SolarCC(INPUT_FILE_S, FEATURES_S,
                                   target=TARGET_S, shape=SHAPE,
                                   temporal_slice=slice(None, None, 2),
                                   time_roll=-7,
                                   sample_shape=(5, 5, 72),
                                   worker_kwargs=dict(max_workers=1))

    batcher = BatchHandlerCC([handler], batch_size=1, n_batches=1,
                             s_enhance=1, sub_daily_shape=24)

    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'sup3rcc/gen_solar_1x_8x_1f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = SolarCC(fp_gen, fp_disc, learning_rate=1e-4,
                    loss='MeanAbsoluteError')

    with tempfile.TemporaryDirectory() as td:
        model.train(batcher,
                    input_resolution={'spatial': '4km', 'temporal': '40min'},
                    n_epoch=1,
                    weight_gen_advers=0.0,
                    train_gen=True, train_disc=False,
                    checkpoint_int=None,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        shape = (1, 4, 4, 72, 1)
        hi_res_true = np.random.uniform(0, 1, shape).astype(np.float32)
        hi_res_gen = np.random.uniform(0, 1, shape).astype(np.float32)
        loss1, _ = model.calc_loss(hi_res_true, hi_res_gen,
                                   weight_gen_advers=0.0,
                                   train_gen=True, train_disc=False)

        t_len = hi_res_true.shape[3]
        n_days = int(t_len // 24)
        day_slices = [slice(SolarCC.STARTING_HOUR + x,
                            SolarCC.STARTING_HOUR + x + SolarCC.DAYLIGHT_HOURS)
                      for x in range(0, 24 * n_days, 24)]

        for tslice in day_slices:
            hi_res_gen[:, :, :, tslice, :] = hi_res_true[:, :, :, tslice, :]

        loss2, _ = model.calc_loss(hi_res_true, hi_res_gen,
                                   weight_gen_advers=0.0,
                                   train_gen=True, train_disc=False)

        assert loss1 > loss2
        assert loss2 == 0
