"""Test the simple linear interpolation model."""
import numpy as np
from scipy.interpolate import interp1d

from sup3r.models import LinearInterp
from sup3r.utilities.utilities import RANDOM_GENERATOR


def test_linear_spatial():
    """Test the linear interp model on the spatial axis"""
    model = LinearInterp(['feature'], s_enhance=2, t_enhance=1,
                         t_centered=False)
    s_vals = RANDOM_GENERATOR.uniform(0, 100, 3)
    lr = np.transpose(np.array([[s_vals, s_vals]]), axes=(1, 2, 0))
    lr = np.repeat(lr, 6, axis=-1)
    lr = np.expand_dims(lr, (0, 4))
    hr = model.generate(lr)
    assert lr.shape[0] == hr.shape[0]
    assert lr.shape[1] * 2 == hr.shape[1]
    assert lr.shape[2] * 2 == hr.shape[2]
    assert lr.shape[3] == hr.shape[3]
    assert lr.shape[4] == hr.shape[4]

    x = np.linspace(-(1 / 4), 2 + (1 / 4), 6)
    ifun = interp1d(np.arange(3), lr[0, 0, :, 0, 0],
                    fill_value='extrapolate')
    truth = ifun(x)
    assert np.allclose(truth, hr[0, 0, :, 0, 0])


def test_linear_temporal():
    """Test the linear interp model on the temporal axis"""
    model = LinearInterp(['feature'], s_enhance=1, t_enhance=3,
                         t_centered=True)
    t_vals = RANDOM_GENERATOR.uniform(0, 100, 3)
    lr = np.ones((2, 2, 3)) * t_vals
    lr = np.expand_dims(lr, (0, 4))
    hr = model.generate(lr)
    assert lr.shape[0] == hr.shape[0]
    assert lr.shape[1] == hr.shape[1]
    assert lr.shape[2] == hr.shape[2]
    assert lr.shape[3] * 3 == hr.shape[3]
    assert lr.shape[4] == hr.shape[4]

    x = np.linspace(-(1 / 3), 2 + (1 / 3), 9)
    ifun = interp1d(np.arange(3), lr[0, 0, 0, :, 0], fill_value='extrapolate')
    truth = ifun(x)
    assert np.allclose(hr[0, 0, 0, :, 0], truth)
