"""Test the training of GANs with dual data handler"""

import json
import os
import tempfile

import numpy as np
import pytest
import tensorflow as tf
from rex import init_logger
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from sup3r import CONFIG_DIR, TEST_DATA_DIR
from sup3r.models import Sup3rGan
from sup3r.preprocessing import (
    DataHandlerH5,
    DataHandlerNC,
    DualBatchHandler,
    DualExtracter,
    StatsCollection,
)
from sup3r.utilities.pytest.helpers import execute_pytest

FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
FP_ERA = os.path.join(TEST_DATA_DIR, 'test_era5_co_2012.nc')
TARGET_COORD = (39.01, -105.15)
FEATURES = ['U_100m', 'V_100m']
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

init_logger('sup3r', log_level='DEBUG')


np.random.seed(42)


@pytest.mark.parametrize(
    [
        'gen_config',
        'disc_config',
        's_enhance',
        't_enhance',
        'sample_shape',
        'mode',
    ],
    [
        (
            'spatiotemporal/gen_3x_4x_2f.json',
            'spatiotemporal/disc.json',
            3,
            4,
            (12, 12, 16),
            'lazy',
        ),
        (
            'spatial/gen_2x_2f.json',
            'spatial/disc.json',
            2,
            1,
            (10, 10, 1),
            'lazy',
        ),
        (
            'spatiotemporal/gen_3x_4x_2f.json',
            'spatiotemporal/disc.json',
            3,
            4,
            (12, 12, 16),
            'eager',
        ),
        (
            'spatial/gen_2x_2f.json',
            'spatial/disc.json',
            2,
            1,
            (10, 10, 1),
            'eager',
        ),
    ],
)
def test_train(
    gen_config,
    disc_config,
    s_enhance,
    t_enhance,
    sample_shape,
    mode,
    n_epoch=2,
):
    """Test basic model training with only gen content loss. Tests both
    spatiotemporal and spatial models."""

    lr = 9e-5
    hr_handler = DataHandlerH5(
        file_paths=FP_WTK,
        features=FEATURES,
        target=TARGET_COORD,
        shape=(20, 20),
        time_slice=slice(None, None, 10),
    )
    lr_handler = DataHandlerNC(
        file_paths=FP_ERA,
        features=FEATURES,
        time_slice=slice(None, None, 5),
    )

    with pytest.raises(AssertionError):
        dual_extracter = DualExtracter(
            data=(lr_handler.data, hr_handler.data),
            s_enhance=s_enhance,
            t_enhance=t_enhance,
        )

    lr_handler = DataHandlerNC(
        file_paths=FP_ERA,
        features=FEATURES,
        time_slice=slice(None, None, t_enhance * 10),
    )

    dual_extracter = DualExtracter(
        data=(lr_handler.data, hr_handler.data),
        s_enhance=s_enhance,
        t_enhance=t_enhance,
    )

    fp_gen = os.path.join(CONFIG_DIR, gen_config)
    fp_disc = os.path.join(CONFIG_DIR, disc_config)

    Sup3rGan.seed()
    model = Sup3rGan(
        fp_gen,
        fp_disc,
        learning_rate=lr,
        loss='MeanAbsoluteError',
        default_device='/cpu:0',
    )

    with tempfile.TemporaryDirectory() as td:
        means = os.path.join(td, 'means.json')
        stds = os.path.join(td, 'stds.json')
        _ = StatsCollection(
            [dual_extracter],
            means=means,
            stds=stds,
        )

        batch_handler = DualBatchHandler(
            train_containers=[dual_extracter],
            val_containers=[dual_extracter],
            sample_shape=sample_shape,
            batch_size=2,
            s_enhance=s_enhance,
            t_enhance=t_enhance,
            n_batches=3,
            means=means,
            stds=stds,
            mode=mode,
        )

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

        assert 'config_generator' in model.meta
        assert 'config_discriminator' in model.meta
        assert len(model.history) == n_epoch
        assert all(model.history['train_gen_trained_frac'] == 1)
        assert all(model.history['train_disc_trained_frac'] == 0)
        tlossg = model.history['train_loss_gen'].values
        vlossg = model.history['val_loss_gen'].values
        assert np.sum(np.diff(tlossg)) < 0
        assert np.sum(np.diff(vlossg)) < 0
        assert 'test_0' in os.listdir(td)
        assert 'test_1' in os.listdir(td)
        assert 'model_gen.pkl' in os.listdir(td + '/test_1')
        assert 'model_disc.pkl' in os.listdir(td + '/test_1')

        # test save/load functionality
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)
        loaded = model.load(out_dir)

        with open(os.path.join(out_dir, 'model_params.json')) as f:
            model_params = json.load(f)

        assert np.allclose(model_params['optimizer']['learning_rate'], lr)
        assert np.allclose(model_params['optimizer_disc']['learning_rate'], lr)
        assert 'learning_rate_gen' in model.history
        assert 'learning_rate_disc' in model.history

        assert 'config_generator' in loaded.meta
        assert 'config_discriminator' in loaded.meta
        assert model.meta['class'] == 'Sup3rGan'

        # make an un-trained dummy model
        dummy = Sup3rGan(
            fp_gen, fp_disc, learning_rate=lr, loss='MeanAbsoluteError'
        )

        for batch in batch_handler:
            out_og = model._tf_generate(batch.low_res)
            out_dummy = dummy._tf_generate(batch.low_res)
            out_loaded = loaded._tf_generate(batch.low_res)

            # make sure the loaded model generates the same data as the saved
            # model but different than the dummy

            tf.assert_equal(out_og, out_loaded)
            with pytest.raises(InvalidArgumentError):
                tf.assert_equal(out_og, out_dummy)

            # make sure the trained model has less loss than dummy
            loss_og = model.calc_loss(batch.high_res, out_og)[0]
            loss_dummy = dummy.calc_loss(batch.high_res, out_dummy)[0]
            assert loss_og.numpy() < loss_dummy.numpy()

        # test that a new shape can be passed through the generator
        if model.is_5d:
            test_data = np.ones(
                (3, 10, 10, 4, len(FEATURES)), dtype=np.float32
            )
            y_test = model._tf_generate(test_data)
            assert y_test.shape[3] == test_data.shape[3] * t_enhance

        else:
            test_data = np.ones((3, 10, 10, len(FEATURES)), dtype=np.float32)
            y_test = model._tf_generate(test_data)

        assert y_test.shape[0] == test_data.shape[0]
        assert y_test.shape[1] == test_data.shape[1] * s_enhance
        assert y_test.shape[2] == test_data.shape[2] * s_enhance
        assert y_test.shape[-1] == test_data.shape[-1]

        batch_handler.stop()


if __name__ == '__main__':
    execute_pytest(__file__)
