"""Test integration of batch queue with training routines and legacy data
handlers."""

import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from rex import init_logger

from sup3r import CONFIG_DIR, TEST_DATA_DIR
from sup3r.containers.batchers import BatchQueueWithValidation
from sup3r.containers.samplers import CroppedSampler
from sup3r.models import Sup3rGan
from sup3r.preprocessing import (
    DataHandlerH5,
)

FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
TARGET_COORD = (39.01, -105.15)
FEATURES = ['U_100m', 'V_100m']

np.random.seed(42)


def get_val_queue_params(handler, sample_shape):
    """Get train / test samplers and means / stds for batch queue inputs."""
    val_split = 0.1
    split_index = int(val_split * handler.data.shape[2])
    val_slice = slice(0, split_index)
    train_slice = slice(split_index, handler.data.shape[2])
    train_sampler = CroppedSampler(
        handler, sample_shape, crop_slice=train_slice
    )
    val_sampler = CroppedSampler(handler, sample_shape, crop_slice=val_slice)
    means = {
        FEATURES[i]: handler.data[..., i].mean() for i in range(len(FEATURES))
    }
    stds = {
        FEATURES[i]: handler.data[..., i].std() for i in range(len(FEATURES))
    }
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
        fp_gen, fp_disc, learning_rate=2e-5, loss='MeanAbsoluteError'
    )

    # need to reduce the number of temporal examples to test faster
    handler = DataHandlerH5(
        FP_WTK,
        FEATURES,
        target=TARGET_COORD,
        shape=full_shape,
        temporal_slice=slice(None, None, 10),
        worker_kwargs={'max_workers': 1},
        val_split=0.0,
    )

    train_sampler, val_sampler, means, stds = get_val_queue_params(
        handler, sample_shape
    )
    batch_handler = BatchQueueWithValidation(
        [train_sampler],
        [val_sampler],
        batch_size=2,
        s_enhance=2,
        t_enhance=1,
        n_batches=2,
        means=means,
        stds=stds,
    )

    batch_handler.start()
    # test that training works and reduces loss

    with TemporaryDirectory() as td:
        model.train(
            batch_handler,
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

    batch_handler.stop()


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
        fp_gen, fp_disc, learning_rate=2e-5, loss='MeanAbsoluteError'
    )

    # need to reduce the number of temporal examples to test faster
    handler = DataHandlerH5(
        FP_WTK,
        FEATURES,
        target=TARGET_COORD,
        shape=full_shape,
        temporal_slice=slice(None, None, 10),
        worker_kwargs={'max_workers': 1},
        val_split=0.0,
    )

    train_sampler, val_sampler, means, stds = get_val_queue_params(
        handler, sample_shape
    )
    batch_handler = BatchQueueWithValidation(
        [train_sampler],
        [val_sampler],
        batch_size=2,
        n_batches=2,
        s_enhance=3,
        t_enhance=4,
        means=means,
        stds=stds,
    )

    batch_handler.start()
    # test that training works and reduces loss

    with TemporaryDirectory() as td:
        with pytest.raises(RuntimeError):
            model.train(
                batch_handler,
                input_resolution={'spatial': '8km', 'temporal': '30min'},
                n_epoch=n_epoch,
                weight_gen_advers=0.0,
                train_gen=True,
                train_disc=False,
                out_dir=os.path.join(td, 'gan_{epoch}'),
            )

        model = Sup3rGan(
            fp_gen, fp_disc, learning_rate=2e-5, loss='MeanAbsoluteError'
        )

        model.train(
            batch_handler,
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

    batch_handler.stop()


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
