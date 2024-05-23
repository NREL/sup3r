"""Test integration of batch queue with training routines and legacy data
containers."""

import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from rex import init_logger

from sup3r import CONFIG_DIR, TEST_DATA_DIR
from sup3r.containers import (
    BatchQueue,
    CroppedSampler,
    DirectExtracterH5,
)
from sup3r.models import Sup3rGan
from sup3r.utilities.pytest.helpers import execute_pytest

FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
TARGET_COORD = (39.01, -105.15)
FEATURES = ['windspeed_100m', 'winddirection_100m']

np.random.seed(42)


def get_val_queue_params(container, sample_shape):
    """Get train / test samplers and means / stds for batch queue inputs."""
    val_split = 0.1
    split_index = int(val_split * container.data.shape[2])
    val_slice = slice(0, split_index)
    train_slice = slice(split_index, container.data.shape[2])
    train_sampler = CroppedSampler(
        container, sample_shape, crop_slice=train_slice
    )
    val_sampler = CroppedSampler(container, sample_shape, crop_slice=val_slice)
    means = {f: container[f].mean() for f in FEATURES}
    stds = {f: container[f].std() for f in FEATURES}
    return train_sampler, val_sampler, means, stds


def test_train_spatial(
    log=True, full_shape=(20, 20), sample_shape=(10, 10, 1), n_epoch=5
):
    """Test basic spatial model training with only gen content loss."""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(
        fp_gen,
        fp_disc,
        learning_rate=2e-5,
        loss='MeanAbsoluteError',
    )

    # need to reduce the number of temporal examples to test faster
    extracter = DirectExtracterH5(
        FP_WTK,
        features=FEATURES,
        target=TARGET_COORD,
        shape=full_shape,
        time_slice=slice(None, None, 10),
    )
    train_sampler, val_sampler, means, stds = get_val_queue_params(
        extracter, sample_shape
    )
    batcher = BatchQueue(
        train_containers=[train_sampler],
        val_containers=[val_sampler],
        batch_size=2,
        s_enhance=2,
        t_enhance=1,
        n_batches=2,
        means=means,
        stds=stds,
    )

    # test that training works and reduces loss
    with TemporaryDirectory() as td:
        model.train(
            batcher,
            input_resolution={'spatial': '8km', 'temporal': '30min'},
            n_epoch=n_epoch,
            checkpoint_int=10,
            weight_gen_advers=0.0,
            train_gen=True,
            train_disc=False,
            out_dir=os.path.join(td, 'gan_{epoch}'),
        )

    assert len(model.history) == n_epoch
    vlossg = model.history['val_loss_gen'].values
    tlossg = model.history['train_loss_gen'].values
    assert np.sum(np.diff(vlossg)) < 0
    assert np.sum(np.diff(tlossg)) < 0
    assert model.means is not None
    assert model.stdevs is not None

    batcher.stop()


def test_train_st(
    log=True, full_shape=(20, 20), sample_shape=(12, 12, 16), n_epoch=5
):
    """Test basic spatiotemporal model training with only gen content loss."""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(
        fp_gen,
        fp_disc,
        learning_rate=2e-5,
        loss='MeanAbsoluteError',
    )

    # need to reduce the number of temporal examples to test faster
    extracter = DirectExtracterH5(
        FP_WTK,
        features=FEATURES,
        target=TARGET_COORD,
        shape=full_shape,
        time_slice=slice(None, None, 10),
    )

    train_sampler, val_sampler, means, stds = get_val_queue_params(
        extracter, sample_shape
    )
    batcher = BatchQueue(
        train_containers=[train_sampler],
        val_containers=[val_sampler],
        batch_size=2,
        n_batches=2,
        s_enhance=3,
        t_enhance=4,
        means=means,
        stds=stds,
    )

    with TemporaryDirectory() as td:
        with pytest.raises(RuntimeError):
            model.train(
                batcher,
                input_resolution={'spatial': '8km', 'temporal': '30min'},
                n_epoch=n_epoch,
                weight_gen_advers=0.0,
                train_gen=True,
                train_disc=False,
                out_dir=os.path.join(td, 'gan_{epoch}'),
            )

        model = Sup3rGan(
            fp_gen,
            fp_disc,
            learning_rate=2e-5,
            loss='MeanAbsoluteError',
        )

        model.train(
            batcher,
            input_resolution={'spatial': '12km', 'temporal': '60min'},
            n_epoch=n_epoch,
            checkpoint_int=10,
            weight_gen_advers=1e-6,
            train_gen=True,
            train_disc=True,
            out_dir=os.path.join(td, 'gan_{epoch}'),
        )

    assert len(model.history) == n_epoch
    vlossg = model.history['val_loss_gen'].values
    tlossg = model.history['train_loss_gen'].values
    assert np.sum(np.diff(vlossg)) < 0
    assert np.sum(np.diff(tlossg)) < 0
    assert model.means is not None
    assert model.stdevs is not None

    batcher.stop()


if __name__ == '__main__':
    execute_pytest(__file__)
