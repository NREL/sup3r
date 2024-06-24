"""Test the basic training of super resolution GAN"""

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
from sup3r.preprocessing import BatchHandler, DataHandlerH5

FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
TARGET_COORD = (39.01, -105.15)
FEATURES = ['U_100m', 'V_100m']

np.random.seed(42)

init_logger('sup3r', log_level='DEBUG')


def _get_handlers():
    """Initialize training and validation handlers used across tests."""

    train_handler = DataHandlerH5(
        FP_WTK,
        features=FEATURES,
        target=TARGET_COORD,
        shape=(20, 20),
        time_slice=slice(None, 3000, 1),
    )

    val_handler = DataHandlerH5(
        FP_WTK,
        features=FEATURES,
        target=TARGET_COORD,
        shape=(20, 20),
        time_slice=slice(3000, None, 1),
    )

    return train_handler, val_handler


@pytest.mark.parametrize(
    ['gen_config', 'disc_config', 's_enhance', 't_enhance', 'sample_shape'],
    [
        (
            'spatiotemporal/gen_3x_4x_2f.json',
            'spatiotemporal/disc.json',
            3,
            4,
            (12, 12, 16),
        ),
        ('spatial/gen_2x_2f.json', 'spatial/disc.json', 2, 1, (10, 10, 1)),
    ],
)
def test_train(
    gen_config,
    disc_config,
    s_enhance,
    t_enhance,
    sample_shape,
    n_epoch=3,
):
    """Test basic model training with only gen content loss. Tests both
    spatiotemporal and spatial models."""

    fp_gen = os.path.join(CONFIG_DIR, gen_config)
    fp_disc = os.path.join(CONFIG_DIR, disc_config)

    lr = 1e-4
    Sup3rGan.seed()
    model = Sup3rGan(
        fp_gen, fp_disc, learning_rate=lr, loss='MeanAbsoluteError'
    )

    train_handler, val_handler = _get_handlers()

    with tempfile.TemporaryDirectory() as td:
        # stats will be calculated since they are given as None
        batch_handler = BatchHandler(
            train_containers=[train_handler],
            val_containers=[val_handler],
            sample_shape=sample_shape,
            batch_size=3,
            s_enhance=s_enhance,
            t_enhance=t_enhance,
            n_batches=3,
            means=None,
            stds=None,
        )

        assert batch_handler.means is not None
        assert batch_handler.stds is not None

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
        assert np.allclose(
            model_params['optimizer_disc']['learning_rate'], lr
        )
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


def test_train_st_weight_update(n_epoch=2):
    """Test basic spatiotemporal model training with discriminators and
    adversarial loss updating."""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(
        fp_gen, fp_disc, learning_rate=1e-4, learning_rate_disc=3e-4
    )

    train_handler, val_handler = _get_handlers()

    batch_handler = BatchHandler(
        [train_handler],
        [val_handler],
        batch_size=2,
        s_enhance=3,
        t_enhance=4,
        n_batches=2,
        sample_shape=(12, 12, 16),
    )

    adaptive_update_bounds = (0.9, 0.99)
    with tempfile.TemporaryDirectory() as td:
        model.train(
            batch_handler,
            input_resolution={'spatial': '12km', 'temporal': '60min'},
            n_epoch=n_epoch,
            weight_gen_advers=1e-6,
            train_gen=True,
            train_disc=True,
            checkpoint_int=10,
            out_dir=os.path.join(td, 'test_{epoch}'),
            adaptive_update_bounds=adaptive_update_bounds,
            adaptive_update_fraction=0.05,
        )

        # check that weight is changed
        check_lower = any(
            frac < adaptive_update_bounds[0]
            for frac in model.history['train_disc_trained_frac'][:-1]
        )
        check_higher = any(
            frac > adaptive_update_bounds[1]
            for frac in model.history['train_disc_trained_frac'][:-1]
        )
        assert check_lower or check_higher
        for e in range(0, n_epoch - 1):
            weight_old = model.history['weight_gen_advers'][e]
            weight_new = model.history['weight_gen_advers'][e + 1]
            if (
                model.history['train_disc_trained_frac'][e]
                < adaptive_update_bounds[0]
            ):
                assert weight_new > weight_old
            if (
                model.history['train_disc_trained_frac'][e]
                > adaptive_update_bounds[1]
            ):
                assert weight_new < weight_old

        batch_handler.stop()


def test_optimizer_update():
    """Test updating optimizer method."""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(
        fp_gen, fp_disc, learning_rate=1e-4, learning_rate_disc=4e-4
    )

    assert model.optimizer.learning_rate == 1e-4
    assert model.optimizer_disc.learning_rate == 4e-4

    model.update_optimizer(option='generator', learning_rate=2)

    assert model.optimizer.learning_rate == 2
    assert model.optimizer_disc.learning_rate == 4e-4

    model.update_optimizer(option='discriminator', learning_rate=0.4)

    assert model.optimizer.learning_rate == 2
    assert model.optimizer_disc.learning_rate == 0.4

    model.update_optimizer(option='all', learning_rate=0.1)

    assert model.optimizer.learning_rate == 0.1
    assert model.optimizer_disc.learning_rate == 0.1


def test_input_res_check():
    """Make sure error is raised for invalid input resolution"""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(
        fp_gen, fp_disc, learning_rate=1e-4, learning_rate_disc=4e-4
    )

    with pytest.raises(RuntimeError):
        model.set_model_params(
            input_resolution={'spatial': '22km', 'temporal': '9min'}
        )


def test_enhancement_check():
    """Make sure error is raised for invalid enhancement factor inputs"""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(
        fp_gen, fp_disc, learning_rate=1e-4, learning_rate_disc=4e-4
    )

    with pytest.raises(RuntimeError):
        model.set_model_params(
            input_resolution={'spatial': '12km', 'temporal': '60min'},
            s_enhance=7,
            t_enhance=3,
        )
