# -*- coding: utf-8 -*-
"""pytests for data handling"""

import os
import numpy as np
import pytest
import matplotlib.pyplot as plt
import tempfile
import pickle

from sup3r import TEST_DATA_DIR
from sup3r.preprocessing.data_handling import DataHandlerH5 as DataHandler
from sup3r.preprocessing.batch_handling import (BatchHandler, Batch,
                                                SpatialBatchHandler)
from sup3r.utilities import utilities

input_file = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
input_files = [os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5'),
               os.path.join(TEST_DATA_DIR, 'test_wtk_co_2013.h5')]
target = (39.01, -105.15)
targets = [target]
shape = (20, 20)
features = ['U_100m', 'V_100m', 'BVF2_200m']
batch_size = 8
sample_shape = (10, 10, 12)
s_enhance = 5
max_delta = 20
val_split = 0.2
raster_file = os.path.join(tempfile.gettempdir(), 'tmp_raster_h5.txt')
temporal_slice = slice(None, None, 3)
n_batches = 20
t_enhance = 2


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
        for input_file, cache_pattern in zip(input_files, cache_patternes):
            data_handler = DataHandler(input_file, features, target,
                                       shape=shape, max_delta=max_delta,
                                       val_split=val_split,
                                       sample_shape=sample_shape,
                                       temporal_slice=temporal_slice,
                                       cache_pattern=cache_pattern)
            data_handlers.append(data_handler)
        st_batch_handler = BatchHandler(data_handlers, batch_size=batch_size,
                                        n_batches=n_batches,
                                        s_enhance=s_enhance,
                                        t_enhance=t_enhance)

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


def test_data_caching():
    """Test data extraction class with data caching/loading"""

    with tempfile.TemporaryDirectory() as td:
        cache_pattern = os.path.join(td, 'cached_features_h5')
        handler = DataHandler(input_file, features, target=target,
                              shape=shape, max_delta=20,
                              cache_pattern=cache_pattern,
                              overwrite_cache=True)

        assert handler.data is None
        assert handler.val_data is None
        handler.load_cached_data()
        assert handler.data.shape == (shape[0], shape[1],
                                      handler.data.shape[2], len(features))
        assert handler.data.dtype == np.dtype(np.float32)
        assert handler.val_data.dtype == np.dtype(np.float32)

        # test cache data but keep in memory
        cache_pattern = os.path.join(td, 'new_1_cache')
        handler = DataHandler(input_file, features, target=target,
                              shape=shape, max_delta=20,
                              val_split=0.1,
                              cache_pattern=cache_pattern,
                              overwrite_cache=True, load_cached=True)
        assert handler.data is not None
        assert handler.val_data is not None
        assert handler.data.dtype == np.dtype(np.float32)
        assert handler.val_data.dtype == np.dtype(np.float32)

        # test cache data but keep in memory, with no val split
        cache_pattern = os.path.join(td, 'new_2_cache')
        handler = DataHandler(input_file, features, target=target,
                              shape=shape, max_delta=20,
                              val_split=0.0,
                              cache_pattern=cache_pattern,
                              overwrite_cache=False, load_cached=True)
        assert handler.data is not None
        assert handler.val_data is not None
        assert handler.data.dtype == np.dtype(np.float32)
        assert handler.val_data.dtype == np.dtype(np.float32)


def test_feature_handler():
    """Make sure compute feature is returing float32"""

    handler = DataHandler(input_file, features, target=target, shape=shape,
                          max_delta=max_delta, raster_file=raster_file,
                          temporal_slice=temporal_slice)
    tmp = handler.extract_data()
    assert tmp.dtype == np.dtype(np.float32)

    vars = {}
    var_names = {'temperature_100m': 'T_bottom',
                 'temperature_200m': 'T_top',
                 'pressure_100m': 'P_bottom',
                 'pressure_200m': 'P_top'}
    for k, v in var_names.items():
        tmp = handler.extract_feature([input_file], handler.raster_index, k)
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
        handler = DataHandler(input_file, features, target=target,
                              shape=shape, max_delta=max_delta,
                              raster_file=raster_file)
        # loading raster file
        handler = DataHandler(input_file, features, raster_file=raster_file)
        assert np.allclose(handler.target, target, atol=1)
        assert handler.data.shape == (shape[0], shape[1],
                                      handler.data.shape[2], len(features))
        assert handler.grid_shape == (shape[0], shape[1])


def test_normalization_input():
    """Test correct normalization input"""

    means = np.random.rand(len(features))
    stds = np.random.rand(len(features))
    data_handlers = []
    for input_file in input_files:
        data_handler = DataHandler(input_file, features, target,
                                   shape=shape, max_delta=max_delta,
                                   val_split=val_split,
                                   sample_shape=sample_shape,
                                   temporal_slice=temporal_slice)
        data_handlers.append(data_handler)
    batch_handler = BatchHandler(data_handlers, batch_size=batch_size,
                                 n_batches=n_batches,
                                 s_enhance=s_enhance,
                                 t_enhance=t_enhance, means=means,
                                 stds=stds)
    assert np.array_equal(batch_handler.stds, stds)
    assert np.array_equal(batch_handler.means, means)


def test_stats_caching():
    """Test caching of stdevs and means"""

    data_handlers = []
    for input_file in input_files:
        data_handler = DataHandler(input_file, features, target,
                                   shape=shape, max_delta=max_delta,
                                   val_split=val_split,
                                   sample_shape=sample_shape,
                                   temporal_slice=temporal_slice)
        data_handlers.append(data_handler)

    with tempfile.TemporaryDirectory() as td:
        means_file = os.path.join(td, 'means.pkl')
        stdevs_file = os.path.join(td, 'stdevs.pkl')
        batch_handler = BatchHandler(data_handlers, batch_size=batch_size,
                                     n_batches=n_batches,
                                     s_enhance=s_enhance,
                                     t_enhance=t_enhance,
                                     stdevs_file=stdevs_file,
                                     means_file=means_file,
                                     max_workers=None)
        assert os.path.exists(means_file)
        assert os.path.exists(stdevs_file)

        with open(means_file, 'rb') as fh:
            means = pickle.load(fh)
        with open(stdevs_file, 'rb') as fh:
            stdevs = pickle.load(fh)

        assert np.array_equal(means, batch_handler.means)
        assert np.array_equal(stdevs, batch_handler.stds)

        stacked_data = np.concatenate([d.data for d
                                       in batch_handler.data_handlers], axis=2)

        for i in range(len(features)):
            std = np.std(stacked_data[..., i])
            if std == 0:
                std = 1
            mean = np.mean(stacked_data[..., i])
            assert np.allclose(std, 1, atol=1e-3)
            assert np.allclose(mean, 0, atol=1e-3)


def test_normalization():
    """Test correct normalization"""

    data_handlers = []
    for input_file in input_files:
        data_handler = DataHandler(input_file, features, target,
                                   shape=shape, max_delta=max_delta,
                                   val_split=val_split,
                                   sample_shape=sample_shape,
                                   temporal_slice=temporal_slice)
        data_handlers.append(data_handler)
    batch_handler = SpatialBatchHandler(data_handlers, batch_size=batch_size,
                                        n_batches=n_batches,
                                        s_enhance=s_enhance,
                                        max_workers=None)
    stacked_data = np.concatenate(
        [d.data for d in batch_handler.data_handlers], axis=2)

    for i in range(len(features)):
        std = np.std(stacked_data[..., i])
        if std == 0:
            std = 1
        mean = np.mean(stacked_data[..., i])
        assert np.allclose(std, 1, atol=1e-3)
        assert np.allclose(mean, 0, atol=1e-3)


def test_spatiotemporal_normalization():
    """Test correct normalization"""

    data_handlers = []
    for input_file in input_files:
        data_handler = DataHandler(input_file, features, target,
                                   shape=shape, max_delta=max_delta,
                                   val_split=val_split,
                                   sample_shape=sample_shape,
                                   temporal_slice=temporal_slice)
        data_handlers.append(data_handler)
    batch_handler = BatchHandler(data_handlers, batch_size=batch_size,
                                 n_batches=n_batches, s_enhance=s_enhance,
                                 t_enhance=t_enhance)
    stacked_data = np.concatenate([d.data for d
                                   in batch_handler.data_handlers], axis=2)

    for i in range(len(features)):
        std = np.std(stacked_data[..., i])
        if std == 0:
            std = 1
        mean = np.mean(stacked_data[..., i])
        assert np.allclose(std, 1, atol=1e-3)
        assert np.allclose(mean, 0, atol=1e-3)


def test_data_extraction():
    """Test data extraction class"""
    handler = DataHandler(input_file, features, target=target, shape=shape,
                          max_delta=20)
    assert handler.data.shape == (shape[0], shape[1], handler.data.shape[2],
                                  len(features))
    assert handler.data.dtype == np.dtype(np.float32)
    assert handler.val_data.dtype == np.dtype(np.float32)


def test_hr_coarsening():
    """Test spatial coarsening of the high res field"""
    handler = DataHandler(input_file, features, target=target,
                          shape=shape, max_delta=20, hr_spatial_coarsen=2)
    assert handler.data.shape == (shape[0] // 2, shape[1] // 2,
                                  handler.data.shape[2], len(features))
    assert handler.data.dtype == np.dtype(np.float32)
    assert handler.val_data.dtype == np.dtype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        cache_pattern = os.path.join(td, 'cached_features_h5')
        if os.path.exists(cache_pattern):
            os.system(f'rm {cache_pattern}')
        handler = DataHandler(input_file, features, target=target,
                              shape=shape, max_delta=20,
                              hr_spatial_coarsen=2,
                              cache_pattern=cache_pattern,
                              overwrite_cache=True)
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
        data_handler = DataHandler(input_file, features, target,
                                   shape=shape, max_delta=max_delta,
                                   val_split=val_split,
                                   sample_shape=(sample_shape[0],
                                                 sample_shape[1], 1),
                                   temporal_slice=temporal_slice)
        data_handlers.append(data_handler)
    batch_handler = SpatialBatchHandler(data_handlers,
                                        batch_size=batch_size,
                                        n_batches=n_batches,
                                        s_enhance=s_enhance)

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
        data_handler = DataHandler(input_file, features, target,
                                   shape=shape, max_delta=max_delta,
                                   val_split=val_split,
                                   sample_shape=sample_shape,
                                   temporal_slice=temporal_slice)
        data_handlers.append(data_handler)
    max_workers = 1
    batch_handler = BatchHandler(data_handlers, batch_size=batch_size,
                                 n_batches=n_batches,
                                 s_enhance=s_enhance,
                                 t_enhance=t_enhance,
                                 temporal_coarsening_method=method,
                                 max_workers=max_workers)
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
        data_handler = DataHandler(input_file, features, target,
                                   shape=shape, max_delta=max_delta,
                                   val_split=val_split,
                                   sample_shape=sample_shape,
                                   temporal_slice=temporal_slice)
        data_handlers.append(data_handler)
    batch_handler = BatchHandler(data_handlers, batch_size=batch_size,
                                 n_batches=n_batches,
                                 s_enhance=s_enhance,
                                 t_enhance=t_enhance,
                                 temporal_coarsening_method=method)

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
    for input_file in input_files:
        data_handler = DataHandler(input_file, features, target,
                                   shape=shape, max_delta=max_delta,
                                   val_split=val_split,
                                   sample_shape=sample_shape,
                                   temporal_slice=temporal_slice)
        data_handlers.append(data_handler)
    batch_handler = BatchHandler(data_handlers, batch_size=batch_size,
                                 n_batches=n_batches,
                                 s_enhance=s_enhance,
                                 t_enhance=t_enhance)

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
    for input_file in input_files:
        data_handler = DataHandler(input_file, features, target,
                                   shape=shape, max_delta=max_delta,
                                   val_split=val_split,
                                   sample_shape=sample_shape,
                                   temporal_slice=temporal_slice)
        data_handlers.append(data_handler)
    batch_handler = BatchHandler(data_handlers, batch_size=batch_size,
                                 n_batches=n_batches,
                                 s_enhance=s_enhance,
                                 t_enhance=t_enhance)

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
        data_handler = DataHandler(input_file, features, target,
                                   shape=shape, max_delta=max_delta,
                                   val_split=val_split,
                                   sample_shape=sample_shape,
                                   temporal_slice=temporal_slice)
        data_handlers.append(data_handler)
    batch_handler = BatchHandler(data_handlers, batch_size=batch_size,
                                 n_batches=n_batches,
                                 s_enhance=s_enhance,
                                 t_enhance=t_enhance)

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
        data_handler = DataHandler(input_file, features, target,
                                   shape=shape, max_delta=max_delta,
                                   val_split=val_split,
                                   sample_shape=sample_shape,
                                   temporal_slice=temporal_slice)
        data_handlers.append(data_handler)
    batch_handler = SpatialBatchHandler(data_handlers,
                                        batch_size=batch_size,
                                        n_batches=n_batches,
                                        s_enhance=s_enhance)

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
        data_handler = DataHandler(input_file, features, target,
                                   shape=shape, max_delta=max_delta,
                                   val_split=val_split,
                                   sample_shape=sample_shape,
                                   temporal_slice=temporal_slice)
        data_handlers.append(data_handler)
    batch_handler = SpatialBatchHandler(data_handlers,
                                        batch_size=batch_size,
                                        n_batches=n_batches,
                                        s_enhance=s_enhance)

    val_observations = 0
    batch_handler.val_data._i = 0
    for batch in batch_handler.val_data:
        assert batch.low_res.shape[0] == batch.high_res.shape[0]
        assert list(batch.low_res.shape[1:3]) == [s // s_enhance for s
                                                  in sample_shape[:2]]
        val_observations += batch.low_res.shape[0]
    n_observations = 0
    for f in input_files:
        handler = DataHandler(f, features, target, shape,
                              max_delta, raster_file=raster_file,
                              val_split=val_split,
                              temporal_slice=temporal_slice)
        data = handler.extract_data()
        n_observations += data.shape[2]
    assert val_observations == int(val_split * n_observations)


def test_no_val_data():
    """Test that the data handler can work with zero validation data."""
    data_handlers = []
    for input_file in input_files:
        data_handler = DataHandler(input_file, features, target,
                                   shape=shape, max_delta=max_delta,
                                   val_split=0,
                                   max_workers=1,
                                   sample_shape=sample_shape,
                                   temporal_slice=temporal_slice)
        data_handlers.append(data_handler)
    batch_handler = BatchHandler(data_handlers, batch_size=batch_size,
                                 n_batches=n_batches,
                                 s_enhance=s_enhance,
                                 t_enhance=t_enhance)
    n = 0
    for _ in batch_handler.val_data:
        n += 1

    assert n == 0
    assert not batch_handler.val_data.any()


def test_smoothing():
    """Check gaussian filtering on low res"""
    high_res = np.random.rand(5, 12, 12, 24, 3)
    smooth_batch = Batch.get_coarse_batch(high_res, 3, 4, smoothing=0.6)
    batch = Batch.get_coarse_batch(high_res, 3, 4, smoothing=None)

    assert not np.allclose(batch.low_res, smooth_batch.low_res)
    assert np.allclose(batch.high_res, smooth_batch.high_res)
    assert np.allclose(batch.low_res, smooth_batch.low_res, atol=0.5)
