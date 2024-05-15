"""Test data loading, extraction, batch building, and training workflows."""

import os
from tempfile import TemporaryDirectory

import pytest
from rex import init_logger

from sup3r import CONFIG_DIR, TEST_DATA_DIR
from sup3r.containers import (
    BatchQueueWithValidation,
    LoaderH5,
    Sampler,
    StatsCollection,
    WranglerH5,
)
from sup3r.models import Sup3rGan
from sup3r.utilities.utilities import transform_rotate_wind

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


def ws_wd_transform(self, data):
    """Transform function for wrangler ws/wd -> u/v"""
    data[..., 0], data[..., 1] = transform_rotate_wind(
        ws=data[..., 0], wd=data[..., 1], lat_lon=self.lat_lon
    )
    return data


def test_end_to_end():
    """Test data loading, extraction to h5 files with chunks, batch building,
    and training with validation end to end workflow."""

    extract_features = ['U_100m', 'V_100m']
    raw_features = ['windspeed_100m', 'winddirection_100m']

    with TemporaryDirectory() as td:
        train_cache_pattern = os.path.join(td, 'train_{feature}.h5')
        val_cache_pattern = os.path.join(td, 'val_{feature}.h5')
        # get training data
        _ = WranglerH5(
            LoaderH5(INPUT_FILES[0], raw_features),
            extract_features,
            **kwargs,
            transform_function=ws_wd_transform,
            cache_kwargs={'cache_pattern': train_cache_pattern,
                          'chunks': {'U_100m': (20, 10, 10),
                                     'V_100m': (20, 10, 10)}},
        )
        # get val data
        _ = WranglerH5(
            LoaderH5(INPUT_FILES[1], raw_features),
            extract_features,
            **kwargs,
            transform_function=ws_wd_transform,
            cache_kwargs={'cache_pattern': val_cache_pattern,
                          'chunks': {'U_100m': (20, 10, 10),
                                     'V_100m': (20, 10, 10)}},
        )

        train_files = [
            train_cache_pattern.format(feature=f) for f in extract_features
        ]
        val_files = [
            val_cache_pattern.format(feature=f) for f in extract_features
        ]

        # init training data sampler
        train_sampler = Sampler(
            LoaderH5(train_files, features=extract_features),
            sample_shape=(18, 18, 16),
            feature_sets={'features': extract_features},
        )

        # init val data sampler
        val_sampler = Sampler(
            LoaderH5(val_files, features=extract_features),
            sample_shape=(18, 18, 16),
            feature_sets={'features': extract_features},
        )

        means_file = os.path.join(td, 'means.json')
        stds_file = os.path.join(td, 'stds.json')
        _ = StatsCollection(
            [train_sampler, val_sampler],
            means_file=means_file,
            stds_file=stds_file,
        )
        batcher = BatchQueueWithValidation(
            [train_sampler],
            [val_sampler],
            n_batches=5,
            batch_size=100,
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
        batcher.start()
        model.train(
            batcher,
            input_resolution={'spatial': '30km', 'temporal': '60min'},
            n_epoch=5,
            weight_gen_advers=0.01,
            train_gen=True,
            train_disc=True,
            checkpoint_int=10,
            out_dir=os.path.join(td, 'test_{epoch}'),
        )
        batcher.stop()


def execute_pytest(capture='all', flags='-rapP'):
    """Execute module as pytest with detailed summary report.

    Parameters
    ----------
    capture : str
        Log or stdout/stderr capture option. ex: log (only logger),
        all (includes stdout/stderr)
    flags : str
        Which tests to show logs and results for.
    """

    fname = os.path.basename(__file__)
    pytest.main(['-q', '--show-capture={}'.format(capture), fname, flags])


if __name__ == '__main__':
    execute_pytest()
