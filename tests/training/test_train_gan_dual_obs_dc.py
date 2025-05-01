"""Test the training of data centric GAN models"""

import os
import tempfile

import pytest

from sup3r.models import Sup3rGan, Sup3rGanWithObsDC
from sup3r.preprocessing import DataHandler, DualBatchHandlerDC, DualRasterizer

TARGET_COORD = (39.01, -105.15)
FEATURES = ['u_10m', 'v_10m']


@pytest.mark.parametrize(
    ('n_space_bins', 'n_time_bins', 'gen_config'),
    [
        (4, 1, 'gen_config_with_concat_masked'),
        (1, 4, 'gen_config_with_concat_masked'),
        (4, 4, 'gen_config_with_concat_masked'),
    ],
)
def test_train_spatial_dual_obs_dc(
    gen_config,
    request,
    n_space_bins,
    n_time_bins,
    full_shape=(20, 20),
    sample_shape=(8, 8, 1),
    n_epoch=4,
):
    """Test dual data-centric spatial model training with observation
    conditioned models. Check that the spatial weights give the correct number
    of observations from each spatial bin"""

    gen_config = request.getfixturevalue(gen_config)()
    Sup3rGan.seed()
    model = Sup3rGanWithObsDC(
        gen_config,
        pytest.S_FP_DISC,
        learning_rate=1e-4,
        onshore_obs_frac={'spatial': 0.1},
        loss={
            'SpatialDerivativeLoss': {},
            'MeanSquaredError': {},
            'term_weights': [0.2, 0.8],
        },
    )

    hr_handler = DataHandler(
        pytest.FP_WTK,
        FEATURES,
        target=TARGET_COORD,
        shape=full_shape,
        time_slice=slice(None, None, 10),
    )
    lr_handler = DataHandler(
        pytest.FP_WTK,
        FEATURES,
        hr_spatial_coarsen=2,
        target=TARGET_COORD,
        shape=full_shape,
        time_slice=slice(None, None, 10),
    )
    batch_size = 1
    n_batches = 10

    handler = DualRasterizer(
        data={'low_res': lr_handler.data, 'high_res': hr_handler.data},
        s_enhance=2,
        t_enhance=1,
    )
    batcher = DualBatchHandlerDC(
        train_containers=[handler],
        val_containers=[handler],
        n_space_bins=n_space_bins,
        n_time_bins=n_time_bins,
        batch_size=batch_size,
        s_enhance=2,
        n_batches=n_batches,
        sample_shape=sample_shape,
    )
    assert batcher.n_space_bins == n_space_bins
    assert batcher.n_time_bins == n_time_bins

    assert all(
        len(c.spatial_weights) == n_space_bins for c in batcher.containers
    )
    assert all(
        len(c.temporal_weights) == n_time_bins for c in batcher.containers
    )

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
