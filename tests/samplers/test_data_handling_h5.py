# -*- coding: utf-8 -*-
"""pytests for data handling"""
import os

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from sup3r import TEST_DATA_DIR
from sup3r.containers import Sampler
from sup3r.preprocessing import (
    BatchHandler,
    SpatialBatchHandler,
)
from sup3r.preprocessing import DataHandlerH5 as DataHandler
from sup3r.utilities import utilities
from sup3r.utilities.pytest.helpers import DummyData

input_files = [os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5'),
               os.path.join(TEST_DATA_DIR, 'test_wtk_co_2013.h5')]
target = (39.01, -105.15)
shape = (20, 20)
features = ['U_100m', 'V_100m', 'BVF2_200m']
sample_shape = (10, 10, 12)
t_enhance = 2
s_enhance = 5
val_split = 0.2
dh_kwargs = {'target': target, 'shape': shape, 'max_delta': 20,
             'sample_shape': sample_shape,
             'lr_only_features': ('BVF*m', 'topography',),
             'time_slice': slice(None, None, 1),
             'worker_kwargs': {'max_workers': 1}}
bh_kwargs = {'batch_size': 8, 'n_batches': 20,
             's_enhance': s_enhance, 't_enhance': t_enhance,
             'worker_kwargs': {'max_workers': 1}}


@pytest.mark.parametrize('method, t_enhance',
                         [('subsample', 2), ('average', 2), ('total', 2),
                          ('subsample', 3), ('average', 3), ('total', 3),
                          ('subsample', 4), ('average', 4), ('total', 4)])
def test_temporal_coarsening(method, t_enhance):
    """Test temporal coarsening of batches"""

    data_handlers = []
    for input_file in input_files:
        data_handler = DataHandler(input_file, features, val_split=0.05,
                                   **dh_kwargs)
        data_handlers.append(data_handler)
    max_workers = 1
    bh_kwargs_new = bh_kwargs.copy()
    bh_kwargs_new['t_enhance'] = t_enhance
    batch_handler = BatchHandler(data_handlers,
                                 temporal_coarsening_method=method,
                                 **bh_kwargs_new)
    assert batch_handler.load_workers == max_workers
    assert batch_handler.norm_workers == max_workers
    assert batch_handler.stats_workers == max_workers

    for batch in batch_handler:
        assert batch.low_res.shape[0] == batch.high_res.shape[0]
        assert batch.low_res.shape == (batch.low_res.shape[0],
                                       sample_shape[0] // s_enhance,
                                       sample_shape[1] // s_enhance,
                                       sample_shape[2] // t_enhance,
                                       len(features))
        assert batch.high_res.shape == (batch.high_res.shape[0],
                                        sample_shape[0], sample_shape[1],
                                        sample_shape[2], len(features) - 1)


def test_no_val_data():
    """Test that the data handler can work with zero validation data."""
    data_handlers = []
    for input_file in input_files:
        data_handler = DataHandler(input_file, features, val_split=0,
                                   **dh_kwargs)
        data_handlers.append(data_handler)
    batch_handler = BatchHandler(data_handlers, **bh_kwargs)
    n = 0
    for _ in batch_handler.val_data:
        n += 1

    assert n == 0
    assert not batch_handler.val_data.any()


def test_smoothing():
    """Check gaussian filtering on low res"""
    data_handlers = []
    for input_file in input_files:
        data_handler = DataHandler(input_file, features[:-1], val_split=0,
                                   **dh_kwargs)
        data_handlers.append(data_handler)
    batch_handler = BatchHandler(data_handlers, smoothing=0.6, **bh_kwargs)
    for batch in batch_handler:
        high_res = batch.high_res
        low_res = utilities.spatial_coarsening(high_res, s_enhance)
        low_res = utilities.temporal_coarsening(low_res, t_enhance)
        low_res_no_smooth = low_res.copy()
        for i in range(low_res_no_smooth.shape[0]):
            for j in range(low_res_no_smooth.shape[-1]):
                for t in range(low_res_no_smooth.shape[-2]):
                    low_res[i, ..., t, j] = gaussian_filter(
                        low_res_no_smooth[i, ..., t, j], 0.6, mode='nearest')
        assert np.array_equal(batch.low_res, low_res)
        assert not np.array_equal(low_res, low_res_no_smooth)


def test_solar_spatial_h5():
    """Test solar spatial batch handling with NaN drop."""
    input_file_s = os.path.join(TEST_DATA_DIR, 'test_nsrdb_co_2018.h5')
    features_s = ['clearsky_ratio']
    target_s = (39.01, -105.13)
    dh_nan = DataHandler(input_file_s, features_s, target=target_s,
                         shape=(20, 20), sample_shape=(10, 10, 12),
                         mask_nan=False)
    dh = DataHandler(input_file_s, features_s, target=target_s,
                     shape=(20, 20), sample_shape=(10, 10, 12),
                     mask_nan=True)
    assert np.nanmax(dh.data) == 1
    assert np.nanmin(dh.data) == 0
    assert not np.isnan(dh.data).any()
    assert np.isnan(dh_nan.data).any()
    for _ in range(10):
        x = dh.get_next()
        assert x.shape == (10, 10, 12, 1)
        assert not np.isnan(x).any()

    batch_handler = SpatialBatchHandler([dh], **bh_kwargs)
    for batch in batch_handler:
        assert not np.isnan(batch.low_res).any()
        assert not np.isnan(batch.high_res).any()
        assert batch.low_res.shape == (8, 2, 2, 1)
        assert batch.high_res.shape == (8, 10, 10, 1)


def test_lr_only_features():
    """Test using BVF as a low-resolution only feature that should be dropped
    from the high-res observations."""
    dh_kwargs_new = dh_kwargs.copy()
    dh_kwargs_new["sample_shape"] = sample_shape
    dh_kwargs_new["lr_only_features"] = 'BVF2*'
    data_handler = DataHandler(input_files[0], features, **dh_kwargs_new)

    bh_kwargs_new = bh_kwargs.copy()
    bh_kwargs_new['norm'] = False
    batch_handler = BatchHandler(data_handler, **bh_kwargs_new)

    for batch in batch_handler:
        assert batch.low_res.shape[-1] == 3
        assert batch.high_res.shape[-1] == 2

        for iobs, data_ind in enumerate(batch_handler.current_batch_indices):
            truth = data_handler.data[data_ind]
            np.allclose(truth[..., 0:2], batch.high_res[iobs])
            truth = utilities.spatial_coarsening(truth, s_enhance=s_enhance,
                                                 obs_axis=False)
            np.allclose(truth[..., ::t_enhance, :], batch.low_res[iobs])


def test_hr_exo_features():
    """Test using BVF as a high-res exogenous feature. For the single data
    handler, this isnt supposed to do anything because the feature is still
    assumed to be in the low-res."""
    dh_kwargs_new = dh_kwargs.copy()
    dh_kwargs_new["sample_shape"] = sample_shape
    dh_kwargs_new["hr_exo_features"] = 'BVF2*'
    data_handler = DataHandler(input_files[0], features, **dh_kwargs_new)
    assert data_handler.hr_exo_features == ['BVF2_200m']

    bh_kwargs_new = bh_kwargs.copy()
    bh_kwargs_new['norm'] = False
    batch_handler = BatchHandler(data_handler, **bh_kwargs_new)

    for batch in batch_handler:
        assert batch.low_res.shape[-1] == 3
        assert batch.high_res.shape[-1] == 3

        for iobs, data_ind in enumerate(batch_handler.current_batch_indices):
            truth = data_handler.data[data_ind]
            np.allclose(truth, batch.high_res[iobs])
            truth = utilities.spatial_coarsening(truth, s_enhance=s_enhance,
                                                 obs_axis=False)
            np.allclose(truth[..., ::t_enhance, :], batch.low_res[iobs])


@pytest.mark.parametrize(['features', 'lr_only_features', 'hr_exo_features'],
                         [(['V_100m'], ['V_100m'], []),
                          (['U_100m'], ['V_100m'], ['V_100m']),
                          (['U_100m'], [], ['U_100m']),
                          (['U_100m', 'V_100m'], [], ['U_100m']),
                          (['U_100m', 'V_100m'], [], ['V_100m', 'U_100m'])])
def test_feature_errors(features, lr_only_features, hr_exo_features):
    """Each of these feature combinations should raise an error due to no
    features left in hr output or bad ordering"""
    sampler = Sampler(
        DummyData(data_shape=(20, 20, 10), features=features),
        feature_sets={'lr_only_features': lr_only_features,
                      'hr_exo_features': hr_exo_features})

    with pytest.raises(Exception):
        _ = sampler.lr_features
        _ = sampler.hr_out_features
        _ = sampler.hr_exo_features
