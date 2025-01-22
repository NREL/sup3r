"""Test the training of super resolution GANs with exo data."""

import os
import tempfile

import pytest

from sup3r.models import Sup3rGanFixedObs
from sup3r.preprocessing import (
    BatchHandler,
    DataHandler,
)

SHAPE = (20, 20)
FEATURES_W = ['u_10m', 'v_10m']
TARGET_W = (39.01, -105.15)


def test_fixed_wind_obs(gen_config_with_fixer):
    """Test a special model which fixes observations mid network with
    ``Sup3rFixer`` layer."""
    kwargs = {
        'file_paths': pytest.FP_WTK,
        'features': FEATURES_W,
        'target': TARGET_W,
        'shape': SHAPE,
    }

    train_handler = DataHandler(**kwargs, time_slice=slice(None, 3000, 10))

    val_handler = DataHandler(**kwargs, time_slice=slice(3000, None, 10))
    batcher = BatchHandler(
        [train_handler],
        [val_handler],
        batch_size=2,
        n_batches=1,
        s_enhance=2,
        t_enhance=1,
        sample_shape=(20, 20, 1),
    )

    Sup3rGanFixedObs.seed()

    model = Sup3rGanFixedObs(
        gen_config_with_fixer(),
        pytest.S_FP_DISC,
        obs_frac={'spatial': 0.1},
        learning_rate=1e-4,
    )
    assert model.obs_features == ['u_10m', 'v_10m']
    with tempfile.TemporaryDirectory() as td:
        model_kwargs = {
            'input_resolution': {'spatial': '16km', 'temporal': '3600min'},
            'n_epoch': 3,
            'weight_gen_advers': 0.0,
            'train_gen': True,
            'train_disc': False,
            'checkpoint_int': None,
            'out_dir': os.path.join(td, 'test_{epoch}'),
        }

        model.train(batcher, **model_kwargs)

        loaded = model.load(os.path.join(td, 'test_2'))
        loaded.train(batcher, **model_kwargs)
