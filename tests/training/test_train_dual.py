"""Test the training of GANs with dual data handler"""

import os
import tempfile

import numpy as np
import pytest

from sup3r.models import Sup3rGan
from sup3r.preprocessing import (
    DataHandler,
    DualBatchHandler,
    DualRasterizer,
)
from sup3r.preprocessing.samplers import DualSampler
from sup3r.utilities.pytest.helpers import BatchHandlerTesterFactory

TARGET_COORD = (39.01, -105.15)
FEATURES = ['u_100m', 'v_100m']


DualBatchHandlerTester = BatchHandlerTesterFactory(
    DualBatchHandler, DualSampler
)


@pytest.mark.parametrize(
    [
        'fp_gen',
        'fp_disc',
        's_enhance',
        't_enhance',
        'sample_shape',
        'mode',
    ],
    [
        (pytest.ST_FP_GEN, pytest.ST_FP_DISC, 3, 4, (12, 12, 16), 'lazy'),
        (pytest.ST_FP_GEN, pytest.ST_FP_DISC, 3, 4, (12, 12, 16), 'eager'),
        (pytest.S_FP_GEN, pytest.S_FP_DISC, 2, 1, (20, 20, 1), 'lazy'),
        (pytest.S_FP_GEN, pytest.S_FP_DISC, 2, 1, (20, 20, 1), 'eager'),
    ],
)
def test_train_h5_nc(
    fp_gen, fp_disc, s_enhance, t_enhance, sample_shape, mode, n_epoch=2
):
    """Test model training with a dual data handler / batch handler with h5 and
    era as hr / lr datasets. Tests both spatiotemporal and spatial models."""

    lr = 1e-5
    kwargs = {
        'features': FEATURES,
        'target': TARGET_COORD,
        'shape': (20, 20),
    }
    hr_handler = DataHandler(
        pytest.FP_WTK,
        **kwargs,
        time_slice=slice(None, None, 1),
    )
    lr_handler = DataHandler(
        pytest.FP_ERA,
        features=FEATURES,
        time_slice=slice(None, None, 30),
    )

    # time indices conflict with t_enhance
    with pytest.raises(AssertionError):
        dual_rasterizer = DualRasterizer(
            data={'low_res': lr_handler.data, 'high_res': hr_handler.data},
            s_enhance=s_enhance,
            t_enhance=t_enhance,
        )

    lr_handler = DataHandler(
        pytest.FP_ERA,
        features=FEATURES,
        time_slice=slice(None, None, t_enhance),
    )

    dual_rasterizer = DualRasterizer(
        data={'low_res': lr_handler.data, 'high_res': hr_handler.data},
        s_enhance=s_enhance,
        t_enhance=t_enhance,
    )

    batch_handler = DualBatchHandlerTester(
        train_containers=[dual_rasterizer],
        val_containers=[],
        sample_shape=sample_shape,
        batch_size=3,
        s_enhance=s_enhance,
        t_enhance=t_enhance,
        n_batches=3,
        mode=mode,
    )

    Sup3rGan.seed()
    model = Sup3rGan(
        fp_gen, fp_disc, learning_rate=lr, loss='MeanAbsoluteError'
    )

    with tempfile.TemporaryDirectory() as td:
        model_kwargs = {
            'input_resolution': {'spatial': '30km', 'temporal': '60min'},
            'n_epoch': n_epoch,
            'weight_gen_advers': 0.0,
            'train_gen': True,
            'train_disc': False,
            'checkpoint_int': 1,
            'out_dir': os.path.join(td, 'test_{epoch}'),
        }

        model.train(batch_handler, **model_kwargs)

        tlossg = model.history['train_loss_gen'].values
        assert np.sum(np.diff(tlossg)) < 0


@pytest.mark.parametrize(
    [
        'fp_gen',
        'fp_disc',
        's_enhance',
        't_enhance',
        'sample_shape',
        'mode',
    ],
    [
        (pytest.ST_FP_GEN, pytest.ST_FP_DISC, 3, 4, (12, 12, 16), 'lazy'),
        (pytest.ST_FP_GEN, pytest.ST_FP_DISC, 3, 4, (12, 12, 16), 'eager'),
        (pytest.S_FP_GEN, pytest.S_FP_DISC, 2, 1, (20, 20, 1), 'lazy'),
        (pytest.S_FP_GEN, pytest.S_FP_DISC, 2, 1, (20, 20, 1), 'eager'),
    ],
)
def test_train_coarse_h5(
    fp_gen, fp_disc, s_enhance, t_enhance, sample_shape, mode, n_epoch=2
):
    """Test model training with a dual data handler / batch handler using h5
    and coarse h5 for hr / lr datasets. Tests both spatiotemporal and spatial
    models."""

    lr = 1e-5
    kwargs = {
        'features': FEATURES,
        'target': TARGET_COORD,
        'shape': (20, 20),
    }
    hr_handler = DataHandler(
        pytest.FP_WTK,
        **kwargs,
        time_slice=slice(None, None, 1),
    )
    lr_handler = DataHandler(
        pytest.FP_WTK,
        **kwargs,
        hr_spatial_coarsen=s_enhance,
        time_slice=slice(None, None, t_enhance),
    )

    dual_rasterizer = DualRasterizer(
        data={'low_res': lr_handler.data, 'high_res': hr_handler.data},
        s_enhance=s_enhance,
        t_enhance=t_enhance,
    )

    batch_handler = DualBatchHandlerTester(
        train_containers=[dual_rasterizer],
        val_containers=[],
        sample_shape=sample_shape,
        batch_size=3,
        s_enhance=s_enhance,
        t_enhance=t_enhance,
        n_batches=3,
        mode=mode,
    )

    Sup3rGan.seed()
    model = Sup3rGan(
        fp_gen, fp_disc, learning_rate=lr, loss='MeanAbsoluteError'
    )

    with tempfile.TemporaryDirectory() as td:
        model_kwargs = {
            'input_resolution': {'spatial': '30km', 'temporal': '60min'},
            'n_epoch': n_epoch,
            'weight_gen_advers': 0.0,
            'train_gen': True,
            'train_disc': False,
            'checkpoint_int': 1,
            'out_dir': os.path.join(td, 'test_{epoch}'),
        }

        model.train(batch_handler, **model_kwargs)

        tlossg = model.history['train_loss_gen'].values
        assert np.sum(np.diff(tlossg)) < 0
