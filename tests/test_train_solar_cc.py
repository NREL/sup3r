# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN for solar climate change
applications"""
import os
import tempfile

from rex import init_logger

from sup3r import TEST_DATA_DIR
from sup3r import CONFIG_DIR
from sup3r.models import Sup3rGan
from sup3r.preprocessing.data_handling import DataHandlerH5SolarCC
from sup3r.preprocessing.batch_handling import BatchHandlerSolarCC


INPUT_FILE = os.path.join(TEST_DATA_DIR, 'test_nsrdb_co_2018.h5')
TARGET = (39.01, -105.13)
SHAPE = (20, 20)

FEATURES = ['clearsky_ratio', 'U', 'V', 'air_temperature']

HANDLER = DataHandlerH5SolarCC(INPUT_FILE, FEATURES,
                               target=TARGET, shape=SHAPE,
                               temporal_slice=slice(None, None, 2),
                               time_roll=-7,
                               sample_shape=(20, 20, 72),
                               max_extract_workers=1,
                               max_compute_workers=1)

BATCHER = BatchHandlerSolarCC([HANDLER], batch_size=16, n_batches=2,
                              s_enhance=2, sub_daily_shape=9)


def test_solar_cc_model(log=False):
    """Test the tile solar climate change nsrdb super res model.

    NOTE that the full 10x model is too big to train on the 20x20 test data.
    """

    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'solar_cc/gen_2x_3x_1f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)

    with tempfile.TemporaryDirectory() as td:
        model.train(BATCHER, n_epoch=1,
                    weight_gen_advers=0.0,
                    train_gen=True, train_disc=False,
                    checkpoint_int=None,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        assert 'test_0' in os.listdir(td)
