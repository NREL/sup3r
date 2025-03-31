"""Test the training of data centric GAN models"""

import os
import tempfile

import numpy as np
import pytest

from sup3r.models import Sup3rGan, Sup3rGanDC
from sup3r.preprocessing import (
    DataHandler,
)
from sup3r.utilities.pytest.helpers import BatchHandlerTesterDC

TARGET_COORD = (39.01, -105.15)
FEATURES = ['u_100m', 'v_100m']


@pytest.mark.parametrize(
    ('n_space_bins', 'n_time_bins'), [(4, 1), (1, 4), (4, 4)]
)
def test_train_spatial_dc(
    n_space_bins,
    n_time_bins,
    full_shape=(20, 20),
    sample_shape=(8, 8, 1),
    n_epoch=4,
):
    """Test data-centric spatial model training. Check that the spatial
    weights give the correct number of observations from each spatial bin"""

    Sup3rGan.seed()
    model = Sup3rGanDC(
        pytest.S_FP_GEN,
        pytest.S_FP_DISC,
        learning_rate=1e-4,
        loss={'MmdLoss': {}, 'MeanSquaredError': {}},
    )

    handler = DataHandler(
        pytest.FP_WTK,
        FEATURES,
        target=TARGET_COORD,
        shape=full_shape,
        time_slice=slice(None, None, 10),
    )
    batch_size = 1
    n_batches = 10

    batcher = BatchHandlerTesterDC(
        train_containers=[handler],
        val_containers=[handler],
        n_space_bins=n_space_bins,
        n_time_bins=n_time_bins,
        batch_size=batch_size,
        s_enhance=2,
        n_batches=n_batches,
        sample_shape=sample_shape,
    )

    assert batcher.val_data.n_batches == n_space_bins * n_time_bins

    deviation = 1 / np.sqrt(batcher.n_batches * batcher.batch_size - 1)
    with tempfile.TemporaryDirectory() as td:
        # test that the normalized number of samples from each bin is close
        # to the weight for that bin
        model.train(
            batcher,
            input_resolution={'spatial': '8km', 'temporal': '30min'},
            n_epoch=n_epoch,
            weight_gen_advers=0.0,
            train_gen=True,
            train_disc=False,
            checkpoint_int=2,
            out_dir=os.path.join(td, 'test_{epoch}'),
        )
        assert np.allclose(
            batcher._mean_record_normed(batcher.space_bin_record),
            batcher._mean_record_normed(batcher.spatial_weights_record),
            atol=deviation,
        )
        assert np.allclose(
            batcher._mean_record_normed(batcher.time_bin_record),
            batcher._mean_record_normed(batcher.temporal_weights_record),
            atol=deviation,
        )

        out_dir = os.path.join(td, 'dc_gan')
        model.save(out_dir)
        loaded = model.load(out_dir)

        assert model.meta['class'] == 'Sup3rGanDC'
        assert loaded.meta['class'] == 'Sup3rGanDC'


@pytest.mark.parametrize(
    ('n_space_bins', 'n_time_bins'), [(4, 1), (1, 4), (4, 4)]
)
def test_train_st_dc(n_space_bins, n_time_bins, n_epoch=2):
    """Test data-centric spatiotemporal model training. Check that the temporal
    weights give the correct number of observations from each temporal bin"""

    Sup3rGan.seed()
    model = Sup3rGanDC(
        pytest.ST_FP_GEN,
        pytest.ST_FP_DISC,
        learning_rate=1e-4,
        loss={'MmdLoss': {}, 'MeanSquaredError': {}},
    )

    handler = DataHandler(
        pytest.FP_WTK,
        FEATURES,
        target=TARGET_COORD,
        shape=(20, 20),
        time_slice=slice(None, None, 10),
    )
    batch_size = 1
    n_batches = 30
    batcher = BatchHandlerTesterDC(
        train_containers=[handler],
        val_containers=[handler],
        batch_size=batch_size,
        sample_shape=(12, 12, 16),
        n_space_bins=n_space_bins,
        n_time_bins=n_time_bins,
        s_enhance=3,
        t_enhance=4,
        n_batches=n_batches,
    )

    deviation = 1 / np.sqrt(batcher.n_batches * batcher.batch_size - 1)

    with tempfile.TemporaryDirectory() as td:
        # test that the normalized number of samples from each bin is close
        # to the weight for that bin
        model.train(
            batcher,
            input_resolution={'spatial': '12km', 'temporal': '60min'},
            n_epoch=n_epoch,
            weight_gen_advers=0.0,
            train_gen=True,
            train_disc=False,
            checkpoint_int=2,
            out_dir=os.path.join(td, 'test_{epoch}'),
        )
        assert np.allclose(
            batcher._mean_record_normed(batcher.space_bin_record),
            batcher._mean_record_normed(batcher.spatial_weights_record),
            atol=deviation,
        )
        assert np.allclose(
            batcher._mean_record_normed(batcher.time_bin_record),
            batcher._mean_record_normed(batcher.temporal_weights_record),
            atol=deviation,
        )

        out_dir = os.path.join(td, 'dc_gan')
        model.save(out_dir)
        loaded = model.load(out_dir)

        assert model.meta['class'] == 'Sup3rGanDC'
        assert loaded.meta['class'] == 'Sup3rGanDC'
