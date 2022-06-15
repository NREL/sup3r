# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN"""
import numpy as np
import tensorflow as tf

from sup3r.models import Sup3rGan, Sup3rGanMmdMse, Sup3rGanCoarseMse


def test_mse_mmd_loss():
    """Test content loss using mse + mmd for content loss."""

    x = np.zeros((6, 10, 10, 8, 3))
    y = np.zeros((6, 10, 10, 8, 3))
    x[:, 7:9, 7:9, :, :] = 1
    y[:, 2:5, 2:5, :, :] = 1

    # distributions differing by only a small peak should give small mse and
    # larger mmd
    mse = Sup3rGan.calc_loss_gen_content(x, y)
    mmd_plus_mse = Sup3rGanMmdMse.calc_loss_gen_content(x, y)

    assert mmd_plus_mse > 2 * mse

    x = np.random.rand(6, 10, 10, 8, 3)
    x /= np.max(x)
    y = np.random.rand(6, 10, 10, 8, 3)
    y /= np.max(y)

    # scaling the same distribution should give high mse and smaller mmd
    mse = Sup3rGan.calc_loss_gen_content(5 * x, x)
    mmd_plus_mse = Sup3rGanMmdMse.calc_loss_gen_content(5 * x, x)

    assert isinstance(mse, tf.Tensor)
    assert isinstance(mmd_plus_mse, tf.Tensor)
    assert mse.numpy().size == 1
    assert mmd_plus_mse.numpy().size == 1

    assert mmd_plus_mse < 2 * mse


def test_coarse_mse_loss():
    """Test the coarse MSE loss on spatial average data"""
    x = np.random.uniform(0, 1, (6, 10, 10, 8, 3))
    y = np.random.uniform(0, 1, (6, 10, 10, 8, 3))

    mse = Sup3rGan.calc_loss_gen_content(x, y)
    coarse_mse = Sup3rGanCoarseMse.calc_loss_gen_content(x, y)

    assert isinstance(mse, tf.Tensor)
    assert isinstance(coarse_mse, tf.Tensor)
    assert mse.numpy().size == 1
    assert coarse_mse.numpy().size == 1
    assert mse.numpy() > 10 * coarse_mse.numpy()
