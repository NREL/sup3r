# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN"""
import numpy as np
import tensorflow as tf
import pytest

from sup3r.utilities.loss_metrics import (MmdMseLoss, CoarseMseLoss,
                                          TemporalExtremesLoss, LowResLoss,
                                          MaterialDerivativeLoss)
from sup3r.utilities.utilities import spatial_coarsening, temporal_coarsening


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


def test_lr_loss():
    """Test custom LowResLoss that re-coarsens synthetic and true high-res
    fields and calculates pointwise loss on the low-res fields"""

    # test w/o enhance
    t_meth = 'average'
    loss_obj = LowResLoss(s_enhance=1, t_enhance=1, t_method=t_meth,
                          tf_loss='MeanSquaredError')
    xarr = np.random.uniform(-1, 1, (3, 10, 10, 48, 2))
    yarr = np.random.uniform(-1, 1, (3, 10, 10, 48, 2))
    xtensor = tf.convert_to_tensor(xarr)
    ytensor = tf.convert_to_tensor(yarr)
    loss = loss_obj(xtensor, ytensor)
    assert np.allclose(loss, loss_obj._tf_loss(xtensor, ytensor))

    # test 5D with s_enhance
    s_enhance = 5
    loss_obj = LowResLoss(s_enhance=s_enhance, t_enhance=1, t_method=t_meth,
                          tf_loss='MeanSquaredError')
    xarr_lr = spatial_coarsening(xarr, s_enhance=s_enhance, obs_axis=True)
    yarr_lr = spatial_coarsening(yarr, s_enhance=s_enhance, obs_axis=True)
    loss = loss_obj(xtensor, ytensor)
    assert np.allclose(loss, loss_obj._tf_loss(xarr_lr, yarr_lr))

    # test 5D with s/t enhance
    s_enhance = 5
    t_enhance = 12
    loss_obj = LowResLoss(s_enhance=s_enhance, t_enhance=t_enhance,
                          t_method=t_meth, tf_loss='MeanSquaredError')
    xarr_lr = spatial_coarsening(xarr, s_enhance=s_enhance, obs_axis=True)
    yarr_lr = spatial_coarsening(yarr, s_enhance=s_enhance, obs_axis=True)
    xarr_lr = temporal_coarsening(xarr_lr, t_enhance=t_enhance, method=t_meth)
    yarr_lr = temporal_coarsening(yarr_lr, t_enhance=t_enhance, method=t_meth)
    loss = loss_obj(xtensor, ytensor)
    assert np.allclose(loss, loss_obj._tf_loss(xarr_lr, yarr_lr))

    # test 5D with subsample
    t_meth = 'subsample'
    loss_obj = LowResLoss(s_enhance=s_enhance, t_enhance=t_enhance,
                          t_method=t_meth, tf_loss='MeanSquaredError')
    xarr_lr = spatial_coarsening(xarr, s_enhance=s_enhance, obs_axis=True)
    yarr_lr = spatial_coarsening(yarr, s_enhance=s_enhance, obs_axis=True)
    xarr_lr = temporal_coarsening(xarr_lr, t_enhance=t_enhance, method=t_meth)
    yarr_lr = temporal_coarsening(yarr_lr, t_enhance=t_enhance, method=t_meth)
    loss = loss_obj(xtensor, ytensor)
    assert np.allclose(loss, loss_obj._tf_loss(xarr_lr, yarr_lr))

    # test 4D spatial only
    xarr = np.random.uniform(-1, 1, (3, 10, 10, 2))
    yarr = np.random.uniform(-1, 1, (3, 10, 10, 2))
    xtensor = tf.convert_to_tensor(xarr)
    ytensor = tf.convert_to_tensor(yarr)
    s_enhance = 5
    loss_obj = LowResLoss(s_enhance=s_enhance, t_enhance=1, t_method=t_meth,
                          tf_loss='MeanSquaredError')
    xarr_lr = spatial_coarsening(xarr, s_enhance=s_enhance, obs_axis=True)
    yarr_lr = spatial_coarsening(yarr, s_enhance=s_enhance, obs_axis=True)
    loss = loss_obj(xtensor, ytensor)
    assert np.allclose(loss, loss_obj._tf_loss(xarr_lr, yarr_lr))

    # test 4D spatial only with spatial extremes
    loss_obj = LowResLoss(s_enhance=s_enhance, t_enhance=1, t_method=t_meth,
                          tf_loss='MeanSquaredError',
                          ex_loss='SpatialExtremesOnlyLoss')
    ex_loss = loss_obj(xtensor, ytensor)
    assert ex_loss > loss


def test_md_loss():
    """Test the material derivative calculation in the material derivative
    content loss class."""

    x = np.random.rand(6, 10, 10, 8, 3)
    y = x.copy()

    md_loss = MaterialDerivativeLoss()
    u_div = md_loss._compute_md(x, fidx=0)
    v_div = md_loss._compute_md(x, fidx=1)

    u_div_np = np.gradient(y[..., 0], axis=3)
    u_div_np += y[..., 0] * np.gradient(y[..., 0], axis=1)
    u_div_np += y[..., 1] * np.gradient(y[..., 0], axis=2)

    v_div_np = np.gradient(x[..., 1], axis=3)
    v_div_np += y[..., 0] * np.gradient(y[..., 1], axis=1)
    v_div_np += y[..., 1] * np.gradient(y[..., 1], axis=2)

    with pytest.raises(ValueError):
        md_loss._derivative(x, axis=0)

    with pytest.raises(Exception):
        md_loss(x[..., 0], y[..., 0])

    assert np.allclose(u_div, u_div_np)
    assert np.allclose(v_div, v_div_np)
