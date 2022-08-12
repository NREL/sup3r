# -*- coding: utf-8 -*-
"""Test the temperature and relative humidity scaling relationships of the
SurfaceSpatialMetModel"""
import os
import numpy as np

from rex import Resource

from sup3r import TEST_DATA_DIR
from sup3r.models.surface import SurfaceSpatialMetModel
from sup3r.utilities.utilities import spatial_coarsening

INPUT_FILE_W = os.path.join(TEST_DATA_DIR, 'test_wtk_surface_temp_rh.h5')
FEATURES_W = ['temperature_2m', 'relativehumidity_2m']


def test_surface_model(s_enhance=5):
    """Test the temperature and relative humidity scaling relationships of the
    SurfaceSpatialMetModel"""

    with Resource(INPUT_FILE_W) as res:
        ti = res.time_index
        meta = res.meta
        temp = res[FEATURES_W[0]]
        rh = res[FEATURES_W[1]]

    shape = (len(ti), 100, 100)
    temp = np.expand_dims(temp.reshape(shape), -1)
    rh = np.expand_dims(rh.reshape(shape), -1)

    true_hi_res = np.concatenate((temp, rh), axis=3)
    low_res = spatial_coarsening(true_hi_res, s_enhance, obs_axis=True)
    topo_hr = meta['elevation'].values.reshape(100, 100)
    topo_lr = spatial_coarsening(np.expand_dims(topo_hr, -1), s_enhance,
                                 obs_axis=False)[..., 0]

    model = SurfaceSpatialMetModel.load()
    hi_res = model.generate(low_res, exogenous_data=[topo_lr, topo_hr])

    diff = true_hi_res - hi_res

    # high res temperature should have very low bias and MAE < 1C
    assert np.abs(diff[..., 0].mean()) < 1e-6
    assert np.abs(diff[..., 0]).mean() < 1

    # high res relative humidity should have very low bias and MAE < 3%
    assert np.abs(diff[..., 1].mean()) < 1e-6
    assert np.abs(diff[..., 1]).mean() < 3
