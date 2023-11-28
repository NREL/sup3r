# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN"""
import numpy as np
import tensorflow as tf

from sup3r.utilities.loss_metrics import (MmdMseLoss, CoarseMseLoss,
                                          TemporalExtremesLoss)


def test_mmd_loss():
    """Test content loss using mse + mmd for content loss."""

    x = np.zeros((6, 10, 10, 8, 3))
    y = np.zeros((6, 10, 10, 8, 3))
    x[:, 7:9, 7:9, :, :] = 1
    y[:, 2:5, 2:5, :, :] = 1

    # distributions differing by only a small peak should give small mse and
    # larger mmd
    mse_fun = tf.keras.losses.MeanSquaredError()
    mmd_mse_fun = MmdMseLoss()

    mse = mse_fun(x, y)
    mmd_plus_mse = mmd_mse_fun(x, y)

    assert mmd_plus_mse > mse

    x = np.random.rand(6, 10, 10, 8, 3)
    x /= np.max(x)
    y = np.random.rand(6, 10, 10, 8, 3)
    y /= np.max(y)

    # scaling the same distribution should give high mse and smaller mmd
    mse = mse_fun(5 * x, x)
    mmd_plus_mse = mmd_mse_fun(5 * x, x)

    assert mmd_plus_mse < mse


def test_coarse_mse_loss():
    """Test the coarse MSE loss on spatial average data"""
    x = np.random.uniform(0, 1, (6, 10, 10, 8, 3))
    y = np.random.uniform(0, 1, (6, 10, 10, 8, 3))

    mse_fun = tf.keras.losses.MeanSquaredError()
    cmse_fun = CoarseMseLoss()

    mse = mse_fun(x, y)
    coarse_mse = cmse_fun(x, y)

    assert isinstance(mse, tf.Tensor)
    assert isinstance(coarse_mse, tf.Tensor)
    assert mse.numpy().size == 1
    assert coarse_mse.numpy().size == 1
    assert mse.numpy() > 10 * coarse_mse.numpy()


def test_tex_loss():
    """Test custom TemporalExtremesLoss function that looks at min/max values
    in the timeseries."""
    loss_obj = TemporalExtremesLoss()

    x = np.zeros((1, 1, 1, 72, 1))
    y = np.zeros((1, 1, 1, 72, 1))

    # loss should be dominated by special min/max values
    x[..., 24, 0] = 20
    y[..., 25, 0] = 25
    loss = loss_obj(x, y)
    assert loss.numpy() > 1.5

    # loss should be dominated by special min/max values
    x[..., 24, 0] = -20
    y[..., 25, 0] = -25
    loss = loss_obj(x, y)
    assert loss.numpy() > 1.5
