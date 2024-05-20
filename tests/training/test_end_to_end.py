"""Test data loading, extraction, batch building, and training workflows."""

import os
from tempfile import TemporaryDirectory

from rex import init_logger

from sup3r import CONFIG_DIR, TEST_DATA_DIR
from sup3r.containers import (
    BatchHandler,
    DataHandlerH5,
    LoaderH5,
    Sampler,
    StatsCollection,
)
from sup3r.models import Sup3rGan
from sup3r.utilities.pytest.helpers import execute_pytest

INPUT_FILES = [
    os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5'),
    os.path.join(TEST_DATA_DIR, 'test_wtk_co_2013.h5'),
]
FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
TARGET_COORD = (39.01, -105.15)
FEATURES = ['U_100m', 'V_100m']
target = (39.01, -105.15)
shape = (20, 20)
kwargs = {
    'target': target,
    'shape': shape,
    'max_delta': 20,
    'time_slice': slice(None, None, 1),
}

init_logger('sup3r', log_level='DEBUG')


def test_end_to_end():
    """Test data loading, extraction to h5 files with chunks, batch building,
    and training with validation end to end workflow."""

    derive_features = ['U_100m', 'V_100m']

    with TemporaryDirectory() as td:
        train_cache_pattern = os.path.join(td, 'train_{feature}.h5')
        val_cache_pattern = os.path.join(td, 'val_{feature}.h5')
        # get training data
        _ = DataHandlerH5(
            INPUT_FILES[0],
            features=derive_features,
            **kwargs,
            cache_kwargs={'cache_pattern': train_cache_pattern,
                          'chunks': {'U_100m': (50, 20, 20),
                                     'V_100m': (50, 20, 20)}},
        )
        # get val data
        _ = DataHandlerH5(
            INPUT_FILES[1],
            features=derive_features,
            **kwargs,
            cache_kwargs={'cache_pattern': val_cache_pattern,
                          'chunks': {'U_100m': (50, 20, 20),
                                     'V_100m': (50, 20, 20)}},
        )

        train_files = [
            train_cache_pattern.format(feature=f) for f in derive_features
        ]
        val_files = [
            val_cache_pattern.format(feature=f) for f in derive_features
        ]

        # init training data sampler
        train_sampler = Sampler(
            LoaderH5(train_files, features=derive_features),
            sample_shape=(12, 12, 16),
            feature_sets={'features': derive_features},
        )

        # init val data sampler
        val_sampler = Sampler(
            LoaderH5(val_files, features=derive_features),
            sample_shape=(12, 12, 16),
            feature_sets={'features': derive_features},
        )

        means_file = os.path.join(td, 'means.json')
        stds_file = os.path.join(td, 'stds.json')
        _ = StatsCollection(
            [train_sampler, val_sampler],
            means_file=means_file,
            stds_file=stds_file,
        )
        batcher = BatchHandler(
            train_containers=[LoaderH5(train_files, derive_features)],
            val_containers=[LoaderH5(val_files, derive_features)],
            n_batches=2,
            batch_size=10,
            s_enhance=3,
            t_enhance=4,
            means=means_file,
            stds=stds_file,
        )

        fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
        fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

        Sup3rGan.seed()
        model = Sup3rGan(
            fp_gen, fp_disc, learning_rate=2e-5, loss='MeanAbsoluteError'
        )
        model.train(
            batcher,
            input_resolution={'spatial': '30km', 'temporal': '60min'},
            n_epoch=2,
            weight_gen_advers=0.01,
            train_gen=True,
            train_disc=True,
            checkpoint_int=10,
            out_dir=os.path.join(td, 'test_{epoch}'),
        )
        batcher.stop()


if __name__ == '__main__':
    execute_pytest(__file__)
