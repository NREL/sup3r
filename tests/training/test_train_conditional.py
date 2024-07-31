"""Test the basic training of conditional moment estimation models."""

import os
import tempfile

import pytest

from sup3r.models import Sup3rCondMom
from sup3r.preprocessing import (
    BatchHandlerMom1,
    BatchHandlerMom1SF,
    BatchHandlerMom2,
    BatchHandlerMom2Sep,
    BatchHandlerMom2SepSF,
    BatchHandlerMom2SF,
    DataHandler,
)

TARGET_COORD = (39.01, -105.15)
FEATURES = ['u_100m', 'v_100m']

ST_SAMPLE_SHAPE = (12, 12, 16)
S_SAMPLE_SHAPE = (12, 12, 1)


@pytest.mark.parametrize(
    (
        'end_t_padding',
        't_enhance_mode',
        'BatcherClass',
        'fp_gen',
        'sample_shape',
        's_enhance',
        't_enhance',
    ),
    [
        (
            False,
            'constant',
            BatchHandlerMom1,
            pytest.ST_FP_GEN,
            ST_SAMPLE_SHAPE,
            3,
            4,
        ),
        (
            True,
            'constant',
            BatchHandlerMom1,
            pytest.ST_FP_GEN,
            ST_SAMPLE_SHAPE,
            3,
            4,
        ),
        (
            False,
            'constant',
            BatchHandlerMom1SF,
            pytest.ST_FP_GEN,
            ST_SAMPLE_SHAPE,
            3,
            4,
        ),
        (
            False,
            'linear',
            BatchHandlerMom1SF,
            pytest.ST_FP_GEN,
            ST_SAMPLE_SHAPE,
            3,
            4,
        ),
        (
            False,
            'constant',
            BatchHandlerMom2,
            pytest.ST_FP_GEN,
            ST_SAMPLE_SHAPE,
            3,
            4,
        ),
        (
            False,
            'constant',
            BatchHandlerMom2SF,
            pytest.ST_FP_GEN,
            ST_SAMPLE_SHAPE,
            3,
            4,
        ),
        (
            False,
            'constant',
            BatchHandlerMom2Sep,
            pytest.ST_FP_GEN,
            ST_SAMPLE_SHAPE,
            3,
            4,
        ),
        (
            False,
            'constant',
            BatchHandlerMom2SepSF,
            pytest.ST_FP_GEN,
            ST_SAMPLE_SHAPE,
            3,
            4,
        ),
        (
            False,
            'constant',
            BatchHandlerMom1,
            pytest.S_FP_GEN,
            S_SAMPLE_SHAPE,
            2,
            1,
        ),
        (
            True,
            'constant',
            BatchHandlerMom1,
            pytest.S_FP_GEN,
            S_SAMPLE_SHAPE,
            2,
            1,
        ),
        (
            False,
            'constant',
            BatchHandlerMom1SF,
            pytest.S_FP_GEN,
            S_SAMPLE_SHAPE,
            2,
            1,
        ),
        (
            False,
            'linear',
            BatchHandlerMom1SF,
            pytest.S_FP_GEN,
            S_SAMPLE_SHAPE,
            2,
            1,
        ),
        (
            False,
            'constant',
            BatchHandlerMom2,
            pytest.S_FP_GEN,
            S_SAMPLE_SHAPE,
            2,
            1,
        ),
        (
            False,
            'constant',
            BatchHandlerMom2SF,
            pytest.S_FP_GEN,
            S_SAMPLE_SHAPE,
            2,
            1,
        ),
        (
            False,
            'constant',
            BatchHandlerMom2Sep,
            pytest.S_FP_GEN,
            S_SAMPLE_SHAPE,
            2,
            1,
        ),
        (
            False,
            'constant',
            BatchHandlerMom2SepSF,
            pytest.S_FP_GEN,
            S_SAMPLE_SHAPE,
            2,
            1,
        ),
    ],
)
def test_train_conditional(
    end_t_padding,
    t_enhance_mode,
    BatcherClass,
    fp_gen,
    sample_shape,
    s_enhance,
    t_enhance,
    full_shape=(20, 20),
    n_epoch=2,
    batch_size=2,
    n_batches=2,
):
    """Test spatial and spatiotemporal model training for 1st and 2nd
    conditional moments."""

    Sup3rCondMom.seed()
    model = Sup3rCondMom(fp_gen, learning_rate=1e-4)
    model_mom1 = Sup3rCondMom(fp_gen, learning_rate=1e-4)

    handler = DataHandler(
        pytest.FP_WTK,
        FEATURES,
        target=TARGET_COORD,
        shape=full_shape,
        time_slice=slice(500, None, 1),
    )

    val_handler = DataHandler(
        pytest.FP_WTK,
        FEATURES,
        target=TARGET_COORD,
        shape=full_shape,
        time_slice=slice(0, 500, 1),
    )

    batch_handler = BatcherClass(
        train_containers=[handler],
        val_containers=[val_handler],
        batch_size=batch_size,
        s_enhance=s_enhance,
        t_enhance=t_enhance,
        n_batches=n_batches,
        lower_models={1: model_mom1},
        sample_shape=sample_shape,
        end_t_padding=end_t_padding,
        time_enhance_mode=t_enhance_mode,
        mode='eager',
    )

    with tempfile.TemporaryDirectory() as td:
        model.train(
            batch_handler,
            input_resolution={'spatial': '12km', 'temporal': '60min'},
            n_epoch=n_epoch,
            checkpoint_int=2,
            out_dir=os.path.join(td, 'test_{epoch}'),
        )
