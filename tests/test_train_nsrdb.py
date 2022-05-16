# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN"""
import os
import tempfile

from sup3r import TEST_DATA_DIR
from sup3r import CONFIG_DIR
from sup3r.models import Sup3rGan
from sup3r.preprocessing.data_handling import DataHandlerNsrdb
from sup3r.preprocessing.batch_handling import NsrdbBatchHandler


INPUT_FILE = os.path.join(TEST_DATA_DIR, 'test_nsrdb_co_2018.h5')
TARGET = (39.01, -105.13)
SHAPE = (20, 20)

FEATURES = ['clearsky_ratio', 'U', 'V', 'air_temperature']

handler = DataHandlerNsrdb(INPUT_FILE, FEATURES,
                           target=TARGET, shape=SHAPE,
                           temporal_slice=slice(None, None, 2),
                           time_roll=-7,
                           val_split=0.05,
                           sample_shape=(20, 20, 24),
                           max_extract_workers=1,
                           max_compute_workers=1)

batcher = NsrdbBatchHandler([handler],
                            batch_size=16, n_batches=2,
                            s_enhance=4, t_enhance=24,
                            temporal_coarsening_method='average')


def test_nsrdb_model():
    """Test the tile nsrdb super res model.

    NOTE that the full 10x model is too big to train on the 20x20 test data.
    """
    fp_gen = os.path.join(CONFIG_DIR, 'nsrdb/gen_4x_24x_1f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)

    with tempfile.TemporaryDirectory() as td:
        model.train(batcher, n_epoch=1,
                    weight_gen_advers=0.0,
                    train_gen=True, train_disc=False,
                    checkpoint_int=None,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        assert 'test_0' in os.listdir(td)
