# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN"""
import os
import tempfile

from sup3r import TEST_DATA_DIR
from sup3r import CONFIG_DIR
from sup3r.models import SpatioTemporalGan
from sup3r.preprocessing.data_handling import DataHandlerNsrdb
from sup3r.preprocessing.batch_handling import NsrdbBatchHandler


INPUT_FILE = os.path.join(TEST_DATA_DIR, 'test_nsrdb_co_2018.h5')
TARGET = (39.01, -105.13)
SHAPE = (20, 20)
FEATURES = ['clearsky_ratio', 'ghi', 'clearsky_ghi']

FEATURES = ['clearsky_ratio', 'U', 'V', 'air_temperature']

handler = DataHandlerNsrdb(INPUT_FILE, FEATURES,
                           target=TARGET, shape=SHAPE,
                           time_shape=slice(None, None, 2),
                           time_roll=-7,
                           val_split=0.05,
                           temporal_sample_shape=24,
                           spatial_sample_shape=(20, 20),
                           max_extract_workers=1,
                           max_compute_workers=1)

batcher = NsrdbBatchHandler([handler],
                            batch_size=16, n_batches=2,
                            s_enhance=4, t_enhance=24,
                            temporal_coarsening_method='average')


def test_hybrid_model():
    """Test the hybrid 4D/5D nsrdb super res model.

    NOTE that the full 10x model is too big to train on the 20x20 test data.
    """
    fp_gen = os.path.join(CONFIG_DIR, 'nsrdb/gen_hybrid_4x_24x.json')
    fp_disc_s = os.path.join(CONFIG_DIR, 'spatiotemporal/disc_space.json')
    fp_disc_t = os.path.join(CONFIG_DIR, 'spatiotemporal/disc_time.json')

    SpatioTemporalGan.seed()
    model = SpatioTemporalGan(fp_gen, fp_disc_s, fp_disc_t,
                              learning_rate=1e-4)

    with tempfile.TemporaryDirectory() as td:
        model.train(batcher, n_epoch=1,
                    weight_gen_advers_s=0.0, weight_gen_advers_t=0.0,
                    train_gen=True, train_disc_s=False, train_disc_t=False,
                    checkpoint_int=None,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        assert 'test_0' in os.listdir(td)


def test_tiled_model():
    """Test the tile nsrdb super res model.

    NOTE that the full 10x model is too big to train on the 20x20 test data.
    """
    fp_gen = os.path.join(CONFIG_DIR, 'nsrdb/gen_tile_4x_24x.json')
    fp_disc_s = os.path.join(CONFIG_DIR, 'spatiotemporal/disc_space.json')
    fp_disc_t = os.path.join(CONFIG_DIR, 'spatiotemporal/disc_time.json')

    SpatioTemporalGan.seed()
    model = SpatioTemporalGan(fp_gen, fp_disc_s, fp_disc_t,
                              learning_rate=1e-4)

    with tempfile.TemporaryDirectory() as td:
        model.train(batcher, n_epoch=1,
                    weight_gen_advers_s=0.0, weight_gen_advers_t=0.0,
                    train_gen=True, train_disc_s=False, train_disc_t=False,
                    checkpoint_int=None,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        assert 'test_0' in os.listdir(td)
