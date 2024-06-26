"""Test data loading, extraction, batch building, and training workflows."""

import os
from tempfile import TemporaryDirectory

import pytest

from sup3r.models import Sup3rGan
from sup3r.preprocessing import (
    BatchHandler,
    DataHandlerH5,
    LoaderH5,
)

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


def test_end_to_end():
    """Test data loading, extraction to h5 files with chunks, batch building,
    and training with validation end to end workflow."""

    derive_features = ['U_100m', 'V_100m']

    with TemporaryDirectory() as td:
        train_cache_pattern = os.path.join(td, 'train_{feature}.h5')
        val_cache_pattern = os.path.join(td, 'val_{feature}.h5')
        # get training data
        _ = DataHandlerH5(
            pytest.FPS_WTK[0],
            features=derive_features,
            **kwargs,
            cache_kwargs={
                'cache_pattern': train_cache_pattern,
                'chunks': {'U_100m': (50, 20, 20), 'V_100m': (50, 20, 20)},
            },
        )
        # get val data
        _ = DataHandlerH5(
            pytest.FPS_WTK[1],
            features=derive_features,
            **kwargs,
            cache_kwargs={
                'cache_pattern': val_cache_pattern,
                'chunks': {'U_100m': (50, 20, 20), 'V_100m': (50, 20, 20)},
            },
        )

        train_files = [
            train_cache_pattern.format(feature=f.lower())
            for f in derive_features
        ]
        val_files = [
            val_cache_pattern.format(feature=f.lower())
            for f in derive_features
        ]

        means = os.path.join(td, 'means.json')
        stds = os.path.join(td, 'stds.json')

        train_containers = LoaderH5(train_files)
        train_containers.data = train_containers.data[derive_features]
        val_containers = LoaderH5(val_files)
        val_containers.data = val_containers.data[derive_features]

        batcher = BatchHandler(
            train_containers=[train_containers],
            val_containers=[val_containers],
            n_batches=2,
            batch_size=10,
            sample_shape=(12, 12, 16),
            s_enhance=3,
            t_enhance=4,
            means=means,
            stds=stds,
        )

        Sup3rGan.seed()
        model = Sup3rGan(
            pytest.ST_FP_GEN,
            pytest.ST_FP_DISC,
            learning_rate=2e-5,
            loss='MeanAbsoluteError',
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
