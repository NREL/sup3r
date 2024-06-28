# -*- coding: utf-8 -*-
"""pytests for data handling"""
import json
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytest
from rex import Resource
from scipy.ndimage.filters import gaussian_filter

from sup3r import TEST_DATA_DIR
from sup3r.preprocessing.batch_handling import (
    BatchHandler,
    SpatialBatchHandler,
)
from sup3r.preprocessing.data_handling import DataHandlerH5 as DataHandler
from sup3r.utilities import utilities

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
             'lr_only_features': ('BVF*m', 'topography'),
             'temporal_slice': slice(None, None, 1),
             'worker_kwargs': {'max_workers': 1}}
bh_kwargs = {'batch_size': 8, 'n_batches': 20,
             's_enhance': s_enhance, 't_enhance': t_enhance,
             'worker_kwargs': {'max_workers': 1}}


@pytest.mark.parametrize('sample_shape', [(10, 10, 10), (5, 5, 10),
                                          (10, 10, 12), (5, 5, 12)])
def test_spatiotemporal_batch_caching(sample_shape):
    """Test that batch observations are found in source data"""

    cache_patternes = []
    with tempfile.TemporaryDirectory() as td:
        for i in range(len(input_files)):
            tmp = os.path.join(td, f'cache_{i}')
            if os.path.exists(tmp):
                os.system(f'rm {tmp}')
            cache_patternes.append(tmp)

        data_handlers = []
        dh_kwargs_new = dh_kwargs.copy()
        dh_kwargs_new['sample_shape'] = sample_shape
        for input_file, cache_pattern in zip(input_files, cache_patternes):
            data_handler = DataHandler(input_file, features,
                                       cache_pattern=cache_pattern,
                                       **dh_kwargs_new)
            data_handlers.append(data_handler)
        st_batch_handler = BatchHandler(data_handlers, **bh_kwargs)

        for batch in st_batch_handler:
            for i, index in enumerate(st_batch_handler.current_batch_indices):
                spatial_1_slice = index[0]
                spatial_2_slice = index[1]
                t_slice = index[2]

                handler_index = st_batch_handler.current_handler_index
                handler = st_batch_handler.data_handlers[handler_index]

                assert np.array_equal(batch.high_res[i, :, :, :],
                                      handler.data[spatial_1_slice,
                                                   spatial_2_slice,
                                                   t_slice, :-1])


def test_topography():
    """Test that topography is batched and extracted correctly"""

    features = ['U_100m', 'V_100m', 'topography']
    data_handler = DataHandler(input_files[0], features, **dh_kwargs)
    ri = data_handler.raster_index
    with Resource(input_files[0]) as res:
        topo = res.get_meta_arr('elevation')[(ri.flatten(),)]
        topo = topo.reshape((ri.shape[0], ri.shape[1]))
    topo_idx = data_handler.features.index('topography')
    assert np.allclose(topo, data_handler.data[..., 0, topo_idx])
    st_batch_handler = BatchHandler([data_handler], **bh_kwargs)
    assert data_handler.hr_out_features == features[:2]
    assert data_handler.data.shape[-1] == len(features)

    for batch in st_batch_handler:
        assert batch.high_res.shape[-1] == 2
        assert batch.low_res.shape[-1] == len(features)


def test_data_caching():
    """Test data extraction class with data caching/loading"""

    with tempfile.TemporaryDirectory() as td:
        cache_pattern = os.path.join(td, 'cached_features_h5')
        handler = DataHandler(input_files[0], features,
                              cache_pattern=cache_pattern,
                              overwrite_cache=True, val_split=0.05,
                              **dh_kwargs)

        assert handler.data is None
        assert handler.val_data is None
        handler.load_cached_data()
        assert handler.data.shape == (shape[0], shape[1],
                                      handler.data.shape[2], len(features))
        assert handler.data.dtype == np.dtype(np.float32)
        assert handler.val_data.dtype == np.dtype(np.float32)

        # test cache data but keep in memory
        cache_pattern = os.path.join(td, 'new_1_cache')
        handler = DataHandler(input_files[0], features,
                              cache_pattern=cache_pattern,
                              overwrite_cache=True, load_cached=True,
                              val_split=0.05,
                              **dh_kwargs)
        assert handler.data is not None
        assert handler.val_data is not None
        assert handler.data.dtype == np.dtype(np.float32)
        assert handler.val_data.dtype == np.dtype(np.float32)

        # test cache data but keep in memory, with no val split
        cache_pattern = os.path.join(td, 'new_2_cache')

        dh_kwargs_new = dh_kwargs.copy()
        dh_kwargs_new['val_split'] = 0
        handler = DataHandler(input_files[0], features,
                              cache_pattern=cache_pattern,
                              overwrite_cache=False, load_cached=True,
                              **dh_kwargs_new)
        assert handler.data is not None
        assert handler.val_data is None
        assert handler.data.dtype == np.dtype(np.float32)


def test_feature_handler():
    """Make sure compute feature is returing float32"""

    handler = DataHandler(input_files[0], features, **dh_kwargs)
    tmp = handler.run_all_data_init()
    assert tmp.dtype == np.dtype(np.float32)

    vars = {}
    var_names = {'temperature_100m': 'T_bottom',
                 'temperature_200m': 'T_top',
                 'pressure_100m': 'P_bottom',
                 'pressure_200m': 'P_top'}
    for k, v in var_names.items():
        tmp = handler.extract_feature([input_files[0]],
                                      handler.raster_index, k)
        assert tmp.dtype == np.dtype(np.float32)
        vars[v] = tmp

    pt_top = utilities.potential_temperature(vars['T_top'],
                                             vars['P_top'])
    pt_bottom = utilities.potential_temperature(vars['T_bottom'],
                                                vars['P_bottom'])
    assert pt_top.dtype == np.dtype(np.float32)
    assert pt_bottom.dtype == np.dtype(np.float32)

    pt_diff = utilities.potential_temperature_difference(
        vars['T_top'], vars['P_top'], vars['T_bottom'], vars['P_bottom'])
    pt_mid = utilities.potential_temperature_average(
        vars['T_top'], vars['P_top'], vars['T_bottom'], vars['P_bottom'])

    assert pt_diff.dtype == np.dtype(np.float32)
    assert pt_mid.dtype == np.dtype(np.float32)

    bvf_squared = utilities.bvf_squared(
        vars['T_top'], vars['T_bottom'], vars['P_top'], vars['P_bottom'], 100)
    assert bvf_squared.dtype == np.dtype(np.float32)


def test_raster_index_caching():
    """Test raster index caching by saving file and then loading"""

    # saving raster file
    with tempfile.TemporaryDirectory() as td:
        raster_file = os.path.join(td, 'raster.txt')
        handler = DataHandler(input_files[0], features,
                              raster_file=raster_file, **dh_kwargs)
        # loading raster file
        handler = DataHandler(input_files[0], features,
                              raster_file=raster_file)
        assert np.allclose(handler.target, target, atol=1)
        assert handler.data.shape == (shape[0], shape[1],
                                      handler.data.shape[2], len(features))
        assert handler.grid_shape == (shape[0], shape[1])


def test_normalization_input():
    """Test correct normalization input"""

    means = {f: 10 for f in features}
    stds = {f: 20 for f in features}
    data_handlers = []
    for input_file in input_files:
        data_handler = DataHandler(input_file, features, **dh_kwargs)
        data_handlers.append(data_handler)
    batch_handler = BatchHandler(data_handlers, means=means,
                                 stds=stds, **bh_kwargs)
    assert all(batch_handler.means[f] == means[f] for f in features)
    assert all(batch_handler.stds[f] == stds[f] for f in features)


def test_stats_caching():
    """Test caching of stdevs and means"""

    data_handlers = []
    for input_file in input_files:
        data_handler = DataHandler(input_file, features, **dh_kwargs)
        data_handlers.append(data_handler)

    with tempfile.TemporaryDirectory() as td:
        means_file = os.path.join(td, 'means.json')
        stdevs_file = os.path.join(td, 'stds.json')
        batch_handler = BatchHandler(data_handlers, stdevs_file=stdevs_file,
                                     means_file=means_file, **bh_kwargs)
        assert os.path.exists(means_file)
        assert os.path.exists(stdevs_file)

        with open(means_file, 'r') as fh:
            means = json.load(fh)
        with open(stdevs_file, 'r') as fh:
            stds = json.load(fh)

        assert all(batch_handler.means[f] == means[f] for f in features)
        assert all(batch_handler.stds[f] == stds[f] for f in features)

        stacked_data = np.concatenate([d.data for d
                                       in batch_handler.data_handlers], axis=2)

        for i in range(len(features)):
            std = np.std(stacked_data[..., i])
            if std == 0:
                std = 1
            mean = np.mean(stacked_data[..., i])
            assert np.allclose(std, 1, atol=1e-2), str(std)
            assert np.allclose(mean, 0, atol=1e-5), str(mean)


def test_unequal_size_normalization():
    """Test correct normalization for data handlers with different numbers of
    elements"""

    data_handlers = []
    for i, input_file in enumerate(input_files):
        tmp_kwargs = dh_kwargs.copy()
        tmp_kwargs['temporal_slice'] = slice(0, (i + 1) * 100)
        data_handler = DataHandler(input_file, features, **tmp_kwargs)
        data_handlers.append(data_handler)
    batch_handler = SpatialBatchHandler(data_handlers, **bh_kwargs)
    stacked_data = np.concatenate(
        [d.data for d in batch_handler.data_handlers], axis=2)

    for i in range(len(features)):
        std = np.std(stacked_data[..., i])
        if std == 0:
            std = 1
        mean = np.mean(stacked_data[..., i])
        assert np.allclose(std, 1, atol=2e-2), str(std)
        assert np.allclose(mean, 0, atol=1e-5), str(mean)


def test_normalization():
    """Test correct normalization"""

    data_handlers = []
    for input_file in input_files:
        data_handler = DataHandler(input_file, features, **dh_kwargs)
        data_handlers.append(data_handler)
    batch_handler = SpatialBatchHandler(data_handlers, **bh_kwargs)
    stacked_data = np.concatenate(
        [d.data for d in batch_handler.data_handlers], axis=2)

    for i in range(len(features)):
        std = np.std(stacked_data[..., i])
        if std == 0:
            std = 1
        mean = np.mean(stacked_data[..., i])
        assert np.allclose(std, 1, atol=1e-2), str(std)
        assert np.allclose(mean, 0, atol=1e-5), str(mean)


def test_spatiotemporal_normalization():
    """Test correct normalization"""

    data_handlers = []
    for input_file in input_files:
        data_handler = DataHandler(input_file, features, **dh_kwargs)
        data_handlers.append(data_handler)
    batch_handler = BatchHandler(data_handlers, **bh_kwargs)
    stacked_data = np.concatenate([d.data for d
                                   in batch_handler.data_handlers], axis=2)

    for i in range(len(features)):
        std = np.std(stacked_data[..., i])
        if std == 0:
            std = 1
        mean = np.mean(stacked_data[..., i])
        assert np.allclose(std, 1, atol=1e-2), str(std)
        assert np.allclose(mean, 0, atol=1e-5), str(mean)


def test_data_extraction():
    """Test data extraction class"""
    handler = DataHandler(input_files[0], features, val_split=0.05,
                          **dh_kwargs)
    assert handler.data.shape == (shape[0], shape[1], handler.data.shape[2],
                                  len(features))
    assert handler.data.dtype == np.dtype(np.float32)
    assert handler.val_data.dtype == np.dtype(np.float32)


def test_hr_coarsening():
    """Test spatial coarsening of the high res field"""
    handler = DataHandler(input_files[0], features, hr_spatial_coarsen=2,
                          val_split=0.05, **dh_kwargs)
    assert handler.data.shape == (shape[0] // 2, shape[1] // 2,
                                  handler.data.shape[2], len(features))
    assert handler.data.dtype == np.dtype(np.float32)
    assert handler.val_data.dtype == np.dtype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        cache_pattern = os.path.join(td, 'cached_features_h5')
        if os.path.exists(cache_pattern):
            os.system(f'rm {cache_pattern}')
        handler = DataHandler(input_files[0], features, hr_spatial_coarsen=2,
                              cache_pattern=cache_pattern, val_split=0.05,
                              overwrite_cache=True, **dh_kwargs)
        assert handler.data is None
        handler.load_cached_data()
        assert handler.data.shape == (shape[0] // 2, shape[1] // 2,
                                      handler.data.shape[2], len(features))
        assert handler.data.dtype == np.dtype(np.float32)
        assert handler.val_data.dtype == np.dtype(np.float32)


def test_validation_batching():
    """Test batching of validation data through
    ValidationData iterator"""

    data_handlers = []
    for input_file in input_files:
        dh_kwargs_new = dh_kwargs.copy()
        dh_kwargs_new['sample_shape'] = (sample_shape[0], sample_shape[1], 1)
        data_handler = DataHandler(input_file, features, val_split=0.05,
                                   **dh_kwargs_new)
        data_handlers.append(data_handler)
    batch_handler = SpatialBatchHandler([data_handler], **bh_kwargs)

    for batch in batch_handler.val_data:
        assert batch.high_res.dtype == np.dtype(np.float32)
        assert batch.low_res.dtype == np.dtype(np.float32)
        assert batch.low_res.shape[0] == batch.high_res.shape[0]
        assert batch.low_res.shape == (batch.low_res.shape[0],
                                       sample_shape[0] // s_enhance,
                                       sample_shape[1] // s_enhance,
                                       len(features))
        assert batch.high_res.shape == (batch.high_res.shape[0],
                                        sample_shape[0], sample_shape[1],
                                        len(features) - 1)


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


@pytest.mark.parametrize('method', ('subsample', 'average', 'total'))
def test_spatiotemporal_validation_batching(method):
    """Test batching of validation data through
    ValidationData iterator"""

    data_handlers = []
    for input_file in input_files:
        data_handler = DataHandler(input_file, features, **dh_kwargs)
        data_handlers.append(data_handler)
    batch_handler = BatchHandler(data_handlers,
                                 temporal_coarsening_method=method,
                                 **bh_kwargs)

    for batch in batch_handler.val_data:
        assert batch.low_res.shape[0] == batch.high_res.shape[0]
        assert batch.low_res.shape == (batch.low_res.shape[0],
                                       sample_shape[0] // s_enhance,
                                       sample_shape[1] // s_enhance,
                                       sample_shape[2] // t_enhance,
                                       len(features))
        assert batch.high_res.shape == (batch.high_res.shape[0],
                                        sample_shape[0], sample_shape[1],
                                        sample_shape[2], len(features) - 1)


@pytest.mark.parametrize('sample_shape', [(10, 10, 10), (5, 5, 10),
                                          (10, 10, 12), (5, 5, 12)])
def test_spatiotemporal_batch_observations(sample_shape):
    """Test that batch observations are found in source data"""

    data_handlers = []
    dh_kwargs_new = dh_kwargs.copy()
    dh_kwargs_new["sample_shape"] = sample_shape
    for input_file in input_files:
        data_handler = DataHandler(input_file, features, **dh_kwargs_new)
        data_handlers.append(data_handler)
    batch_handler = BatchHandler(data_handlers, **bh_kwargs)

    for batch in batch_handler:
        for i, index in enumerate(batch_handler.current_batch_indices):
            spatial_1_slice = index[0]
            spatial_2_slice = index[1]
            t_slice = index[2]

            handler_index = batch_handler.current_handler_index
            handler = batch_handler.data_handlers[handler_index]

            assert np.array_equal(batch.high_res[i, :, :, :],
                                  handler.data[spatial_1_slice,
                                               spatial_2_slice,
                                               t_slice, :-1])


@pytest.mark.parametrize('sample_shape', [(10, 10, 10), (5, 5, 10),
                                          (10, 10, 12), (5, 5, 12)])
def test_spatiotemporal_batch_indices(sample_shape):
    """Test spatiotemporal batch indices for unique
    spatial indices and contiguous increasing temporal slice"""

    data_handlers = []
    dh_kwargs_new = dh_kwargs.copy()
    dh_kwargs_new["sample_shape"] = sample_shape
    for input_file in input_files:
        data_handler = DataHandler(input_file, features, **dh_kwargs_new)
        data_handlers.append(data_handler)
    batch_handler = BatchHandler(data_handlers, **bh_kwargs)

    all_spatial_tuples = []
    for _ in batch_handler:
        for index in batch_handler.current_batch_indices:
            spatial_1_slice = np.arange(index[0].start, index[0].stop)
            spatial_2_slice = np.arange(index[1].start, index[1].stop)
            t_slice = np.arange(index[2].start, index[2].stop)
            spatial_tuples = []
            for s1 in spatial_1_slice:
                for s2 in spatial_2_slice:
                    spatial_tuples.append((s1, s2))
            assert len(spatial_tuples) == len(list(set(spatial_tuples)))

            all_spatial_tuples.append(np.array(spatial_tuples))

            sorted_temporal_slice = t_slice.copy()
            sorted_temporal_slice.sort()
            assert np.array_equal(sorted_temporal_slice, t_slice)

            assert all(t_slice[1:] - t_slice[:-1] == 1)

    comparisons = []
    for i, s1 in enumerate(all_spatial_tuples):
        for j, s2 in enumerate(all_spatial_tuples):
            if i != j:
                comparisons.append(np.array_equal(s1, s2))
    assert not all(comparisons)


def test_spatiotemporal_batch_handling(plot=False):
    """Test spatiotemporal batch handling class"""

    data_handlers = []
    for input_file in input_files:
        dh_kwargs_new = dh_kwargs.copy()
        dh_kwargs_new["sample_shape"] = sample_shape
        data_handler = DataHandler(input_file, features, **dh_kwargs_new)
        data_handlers.append(data_handler)
    batch_handler = BatchHandler(data_handlers, **bh_kwargs)

    for batch in batch_handler:
        assert batch.low_res.shape[0] == batch.high_res.shape[0]

    for i, batch in enumerate(batch_handler):
        assert batch.low_res.shape == (batch.low_res.shape[0],
                                       sample_shape[0] // s_enhance,
                                       sample_shape[1] // s_enhance,
                                       sample_shape[2] // t_enhance,
                                       len(features))
        assert batch.high_res.shape == (batch.high_res.shape[0],
                                        sample_shape[0], sample_shape[1],
                                        sample_shape[2], len(features) - 1)

        if plot:
            for ifeature in range(batch.high_res.shape[-1]):
                data_fine = batch.high_res[0, 0, :, :, ifeature]
                data_coarse = batch.low_res[0, 0, :, :, ifeature]
                fig = plt.figure(figsize=(10, 5))
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                ax1.imshow(data_fine)
                ax2.imshow(data_coarse)
                plt.savefig(f'./{i}_{ifeature}.png')
                plt.close()


def test_batch_handling(plot=False):
    """Test spatial batch handling class"""

    data_handlers = []
    for input_file in input_files:
        data_handler = DataHandler(input_file, features, **dh_kwargs)
        data_handlers.append(data_handler)
    batch_handler = SpatialBatchHandler(data_handlers, **bh_kwargs)

    for batch in batch_handler:
        assert batch.low_res.shape[0] == batch.high_res.shape[0]

    for i, batch in enumerate(batch_handler):
        assert batch.high_res.dtype == np.float32
        assert batch.low_res.dtype == np.float32
        assert batch.low_res.shape == (batch.low_res.shape[0],
                                       sample_shape[0] // s_enhance,
                                       sample_shape[1] // s_enhance,
                                       len(features))
        assert batch.high_res.shape == (batch.high_res.shape[0],
                                        sample_shape[0], sample_shape[1],
                                        len(features) - 1)

        if plot:
            for ifeature in range(batch.high_res.shape[-1]):
                data_fine = batch.high_res[0, :, :, ifeature]
                data_coarse = batch.low_res[0, :, :, ifeature]
                fig = plt.figure(figsize=(10, 5))
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                ax1.imshow(data_fine)
                ax2.imshow(data_coarse)
                plt.savefig(f'./{i}_{ifeature}.png')
                plt.close()


def test_val_data_storage():
    """Test validation data storage from batch handler method"""

    data_handlers = []
    for input_file in input_files:
        data_handler = DataHandler(input_file, features, val_split=val_split,
                                   **dh_kwargs)
        data_handlers.append(data_handler)
    batch_handler = BatchHandler(data_handlers, **bh_kwargs)

    val_observations = 0
    batch_handler.val_data._i = 0
    for batch in batch_handler.val_data:
        assert batch.low_res.shape[0] == batch.high_res.shape[0]
        assert list(batch.low_res.shape[1:3]) == [s // s_enhance for s
                                                  in sample_shape[:2]]
        val_observations += batch.low_res.shape[0]
    n_observations = 0
    for f in input_files:
        handler = DataHandler(f, features, val_split=val_split, **dh_kwargs)
        data = handler.run_all_data_init()
        n_observations += data.shape[2]
    assert val_observations == int(val_split * n_observations)


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
    handler = DataHandler(input_files[0],
                          features,
                          lr_only_features=lr_only_features,
                          hr_exo_features=hr_exo_features,
                          target=target,
                          shape=(20, 20),
                          sample_shape=(5, 5, 4),
                          temporal_slice=slice(None, None, 1),
                          worker_kwargs=dict(max_workers=1),
                          )
    with pytest.raises(Exception):
        _ = handler.lr_features
        _ = handler.hr_out_features
        _ = handler.hr_exo_features
