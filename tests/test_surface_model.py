# -*- coding: utf-8 -*-
"""Test the temperature and relative humidity scaling relationships of the
SurfaceSpatialMetModel"""
import os
import tempfile
import pytest
import numpy as np

from rex import Resource

from sup3r.models import Sup3rGan
from sup3r import CONFIG_DIR, TEST_DATA_DIR
from sup3r.models.surface import SurfaceSpatialMetModel
from sup3r.models.multi_step import MultiStepSurfaceMetGan
from sup3r.utilities.utilities import spatial_coarsening

INPUT_FILE_W = os.path.join(TEST_DATA_DIR, 'test_wtk_surface_vars.h5')
FEATURES = ['temperature_2m', 'relativehumidity_2m', 'pressure_0m']


def get_inputs(s_enhance):
    """Get various inputs for the surface model."""

    with Resource(INPUT_FILE_W) as res:
        ti = res.time_index
        meta = res.meta
        temp = res[FEATURES[0]]
        rh = res[FEATURES[1]]
        pres = res[FEATURES[2]]

    shape = (len(ti), 100, 100)
    temp = np.expand_dims(temp.reshape(shape), -1)
    rh = np.expand_dims(rh.reshape(shape), -1)
    pres = np.expand_dims(pres.reshape(shape), -1)

    true_hi_res = np.concatenate((temp, rh, pres), axis=3)
    true_hi_res = [true_hi_res[slice(i * 24, 24 + i * 24)]
                   for i in range(int(len(ti) // 24))]
    true_hi_res = [np.expand_dims(np.mean(thr, axis=0), 0)
                   for thr in true_hi_res]
    true_hi_res = np.concatenate(true_hi_res, 0)
    low_res = spatial_coarsening(true_hi_res, s_enhance, obs_axis=True)
    topo_hr = meta['elevation'].values.reshape(100, 100)
    topo_lr = spatial_coarsening(np.expand_dims(topo_hr, -1), s_enhance,
                                 obs_axis=False)[..., 0]

    return low_res, true_hi_res, topo_lr, topo_hr


def test_surface_model(s_enhance=5):
    """Test the temperature and relative humidity scaling relationships of the
    SurfaceSpatialMetModel"""

    low_res, true_hi_res, topo_lr, topo_hr = get_inputs(s_enhance)

    model = SurfaceSpatialMetModel.load(features=FEATURES, s_enhance=s_enhance)
    hi_res = model.generate(low_res, exogenous_data=[topo_lr, topo_hr])

    diff = true_hi_res - hi_res

    # high res temperature should have very low bias and MAE < 1C
    assert np.abs(diff[..., 0].mean()) < 1e-6
    assert np.abs(diff[..., 0]).mean() < 5

    # high res relative humidity should have very low bias and MAE < 3%
    assert np.abs(diff[..., 1].mean()) < 1e-6
    assert np.abs(diff[..., 1]).mean() < 2

    # high res pressure should have very low bias and MAE < 200 Pa
    assert np.abs(diff[..., 2].mean()) < 5
    assert np.abs(diff[..., 2]).mean() < 200


def test_train_rh_model(s_enhance=10):
    """Test the train method of the RH linear regression model."""
    _, true_hi_res, _, topo_hr = get_inputs(s_enhance)
    true_hr_temp = np.transpose(true_hi_res[..., 0], axes=(1, 2, 0))
    true_hr_rh = np.transpose(true_hi_res[..., 1], axes=(1, 2, 0))

    model = SurfaceSpatialMetModel(FEATURES, s_enhance=s_enhance)
    w_delta_temp, w_delta_topo = model.train(true_hr_temp, true_hr_rh, topo_hr)

    # pretty generous tolerances because the training dataset is so small
    assert np.allclose(w_delta_temp, SurfaceSpatialMetModel.W_DELTA_TEMP,
                       atol=0.6)
    assert np.allclose(w_delta_topo, SurfaceSpatialMetModel.W_DELTA_TOPO,
                       atol=0.01)


def test_multi_step_surface(s_enhance=2, t_enhance=2):
    """Test the multi step surface met model."""

    config_gen = [
        {"class": "FlexiblePadding",
         "paddings": [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]],
         "mode": "REFLECT"},
        {"class": "Conv3D", "filters": 64, "kernel_size": 3, "strides": 1},
        {"class": "Cropping3D", "cropping": 2},
        {"alpha": 0.2, "class": "LeakyReLU"},
        {"class": "SpatioTemporalExpansion", "temporal_mult": t_enhance,
         "temporal_method": "nearest"},
        {"class": "FlexiblePadding",
         "paddings": [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]],
         "mode": "REFLECT"},
        {"class": "Conv3D", "filters": 3, "kernel_size": 3, "strides": 1},
        {"class": "Cropping3D", "cropping": 2}]

    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    model = Sup3rGan(config_gen, fp_disc)
    _ = model.generate(np.ones((4, 10, 10, 6, len(FEATURES))))

    model.set_norm_stats([0.3, 0.9, 0.1], [0.02, 0.07, 0.03])
    model.set_model_params(training_features=FEATURES,
                           output_features=FEATURES,
                           s_enhance=1,
                           t_enhance=t_enhance)

    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, 'model')
        model.save(fp)

        ms_model = MultiStepSurfaceMetGan.load(FEATURES, s_enhance, fp)

        for model in ms_model.models:
            print(model)
            print(model.s_enhance, model.t_enhance)
            assert isinstance(model.s_enhance, int)
            assert isinstance(model.t_enhance, int)

        x = np.ones((2, 10, 10, len(FEATURES)))
        with pytest.raises(AssertionError):
            ms_model.generate(x)

        low_res, _, topo_lr, topo_hr = get_inputs(s_enhance)

        # reduce data because too big for tests
        low_res = low_res[:, :4, :4]
        topo_lr = topo_lr[:4, :4]
        topo_hr = topo_hr[:8, :8]

        hi_res = ms_model.generate(low_res, exogenous_data=[topo_lr, topo_hr])

        target_shape = (1,
                        low_res.shape[1] * s_enhance,
                        low_res.shape[2] * s_enhance,
                        low_res.shape[0] * t_enhance,
                        len(FEATURES))
        assert hi_res.shape == target_shape