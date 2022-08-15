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

INPUT_FILE_W = os.path.join(TEST_DATA_DIR, 'test_wtk_surface_temp_rh.h5')
FEATURES_W = ['temperature_2m', 'relativehumidity_2m']


def get_inputs(s_enhance):
    """Get various inputs for the surface model."""

    with Resource(INPUT_FILE_W) as res:
        ti = res.time_index
        meta = res.meta
        temp = res[FEATURES_W[0]]
        rh = res[FEATURES_W[1]]

    shape = (len(ti), 100, 100)
    temp = np.expand_dims(temp.reshape(shape), -1)
    rh = np.expand_dims(rh.reshape(shape), -1)

    true_hi_res = np.concatenate((temp, rh), axis=3)
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

    model = SurfaceSpatialMetModel.load(s_enhance=s_enhance)
    hi_res = model.generate(low_res, exogenous_data=[topo_lr, topo_hr])

    diff = true_hi_res - hi_res

    # high res temperature should have very low bias and MAE < 1C
    assert np.abs(diff[..., 0].mean()) < 1e-6
    assert np.abs(diff[..., 0]).mean() < 5

    # high res relative humidity should have very low bias and MAE < 3%
    assert np.abs(diff[..., 1].mean()) < 1e-6
    assert np.abs(diff[..., 1]).mean() < 2


def test_multi_step_surface(s_enhance=2):
    """Test the multi step surface met model."""
    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    model = Sup3rGan(fp_gen, fp_disc)
    _ = model.generate(np.ones((4, 10, 10, 6, len(FEATURES_W))))

    model.set_norm_stats([0.3, 0.9], [0.02, 0.07])
    model.set_feature_names(FEATURES_W, FEATURES_W)

    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, 'model')
        model.save(fp)

        ms_model = MultiStepSurfaceMetGan.load(s_enhance, fp)

        x = np.ones((2, 10, 10, len(FEATURES_W)))
        with pytest.raises(AssertionError):
            ms_model.generate(x)

        low_res, _, topo_lr, topo_hr = get_inputs(s_enhance)

        # reduce data because too big for tests
        low_res = low_res[:, :4, :4]
        topo_lr = topo_lr[:4, :4]
        topo_hr = topo_hr[:8, :8]

        hi_res = ms_model.generate(low_res, exogenous_data=[topo_lr, topo_hr])

        assert hi_res.shape == (1, 24, 24, 28, 2)
