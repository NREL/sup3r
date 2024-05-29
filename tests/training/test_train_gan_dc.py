# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN"""

import os
import tempfile

import numpy as np
from rex import init_logger

from sup3r import CONFIG_DIR, TEST_DATA_DIR
from sup3r.models import Sup3rGan
from sup3r.models.data_centric import Sup3rGanDC, Sup3rGanSpatialDC
from sup3r.preprocessing import BatchHandlerDC, DataHandlerDCforH5
from sup3r.utilities.loss_metrics import MmdMseLoss

FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
TARGET_COORD = (39.01, -105.15)
FEATURES = ['U_100m', 'V_100m']


def test_train_spatial_dc(
    log=False, full_shape=(20, 20), sample_shape=(10, 10, 1), n_epoch=2
):
    """Test data-centric spatial model training. Check that the spatial
    weights give the correct number of observations from each spatial bin"""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')

    Sup3rGan.seed()
    model = Sup3rGanSpatialDC(
        fp_gen,
        fp_disc,
        learning_rate=1e-4,
        learning_rate_disc=3e-4,
        loss='MmdMseLoss',
    )

    handler = DataHandlerDCforH5(
        FP_WTK,
        FEATURES,
        target=TARGET_COORD,
        shape=full_shape,
        time_slice=slice(None, None, 1),
    )
    batch_size = 2
    n_batches = 2
    total_count = batch_size * n_batches
    deviation = np.sqrt(1 / (total_count - 1))

    batch_handler = BatchHandlerDC(
        [handler],
        batch_size=batch_size,
        s_enhance=2,
        n_batches=n_batches,
        sample_shape=sample_shape,
    )

    with tempfile.TemporaryDirectory() as td:
        # test that the normalized number of samples from each bin is close
        # to the weight for that bin
        model.train(
            batch_handler,
            input_resolution={'spatial': '8km', 'temporal': '30min'},
            n_epoch=n_epoch,
            weight_gen_advers=0.0,
            train_gen=True,
            train_disc=False,
            checkpoint_int=2,
            out_dir=os.path.join(td, 'test_{epoch}'),
        )
        assert np.allclose(
            batch_handler.old_spatial_weights,
            batch_handler.norm_spatial_record,
            atol=deviation,
        )

        out_dir = os.path.join(td, 'dc_gan')
        model.save(out_dir)
        loaded = model.load(out_dir)

        assert isinstance(model.loss_fun, MmdMseLoss)
        assert isinstance(loaded.loss_fun, MmdMseLoss)
        assert model.meta['class'] == 'Sup3rGanSpatialDC'
        assert loaded.meta['class'] == 'Sup3rGanSpatialDC'


def test_train_st_dc(n_epoch=2, log=False):
    """Test data-centric spatiotemporal model training. Check that the temporal
    weights give the correct number of observations from each temporal bin"""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGanDC(
        fp_gen,
        fp_disc,
        learning_rate=1e-4,
        learning_rate_disc=3e-4,
        loss='MmdMseLoss',
    )

    handler = DataHandlerDCforH5(
        FP_WTK,
        FEATURES,
        target=TARGET_COORD,
        shape=(20, 20),
        time_slice=slice(None, None, 1),
    )
    batch_size = 4
    n_batches = 2
    total_count = batch_size * n_batches
    deviation = np.sqrt(1 / (total_count - 1))
    batch_handler = BatchHandlerDC(
        [handler],
        batch_size=batch_size,
        sample_shape=(12, 12, 16),
        s_enhance=3,
        t_enhance=4,
        n_batches=n_batches,
    )

    with tempfile.TemporaryDirectory() as td:
        # test that the normalized number of samples from each bin is close
        # to the weight for that bin
        model.train(
            batch_handler,
            input_resolution={'spatial': '12km', 'temporal': '60min'},
            n_epoch=n_epoch,
            weight_gen_advers=0.0,
            train_gen=True,
            train_disc=False,
            checkpoint_int=2,
            out_dir=os.path.join(td, 'test_{epoch}'),
        )
        assert np.allclose(
            batch_handler.old_temporal_weights,
            batch_handler.norm_temporal_record,
            atol=deviation,
        )

        out_dir = os.path.join(td, 'dc_gan')
        model.save(out_dir)
        loaded = model.load(out_dir)

        assert isinstance(model.loss_fun, MmdMseLoss)
        assert isinstance(loaded.loss_fun, MmdMseLoss)
        assert model.meta['class'] == 'Sup3rGanDC'
        assert loaded.meta['class'] == 'Sup3rGanDC'
