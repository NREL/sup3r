# -*- coding: utf-8 -*-
"""pytests for data handling"""

import os
import numpy as np
import pytest
import matplotlib.pyplot as plt
import tempfile

from sup3r import TEST_DATA_DIR
from sup3r.data_handling.preprocessing import (DataHandlerH5,
                                               BatchHandler,
                                               SpatialBatchHandler)
from sup3r.utilities import utilities

input_file = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
input_files = [os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5'),
               os.path.join(TEST_DATA_DIR, 'test_wtk_co_2013.h5')]
target = (39.01, -105.15)
targets = target
shape = (20, 20)
features = ['U_100m', 'V_100m', 'BVF_squared_200m']
batch_size = 8
spatial_sample_shape = (10, 10)
spatial_res = 5
max_delta = 20
val_split = 0.2
raster_file = os.path.join(tempfile.gettempdir(), 'tmp_raster_h5.txt')
time_pruning = 3
n_batches = 20
temporal_sample_shape = 12
temporal_res = 2

os.system(f'rm -f {raster_file}')


def test_feature_handler():
    """Make sure compute feature is returing float32"""

    handler = DataHandlerH5(input_file, features, target=target, shape=shape,
                            max_delta=max_delta, raster_file=raster_file)

    tmp = handler.extract_data(
        input_file, handler.raster_index, handler.time_index,
        features, time_pruning)
    assert tmp.dtype == np.dtype(np.float32)

    vars = {}
    var_names = {'temperature_100m': 'T_bottom',
                 'temperature_200m': 'T_top',
                 'pressure_100m': 'P_bottom',
                 'pressure_200m': 'P_top'}
    for k, v in var_names.items():
        tmp = handler.extract_feature(input_file, handler.raster_index, k)
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

    bvf_squared = utilities.BVF_squared(
        vars['T_top'], vars['T_bottom'], vars['P_top'], vars['P_bottom'], 100)
    assert bvf_squared.dtype == np.dtype(np.float32)


def test_raster_index_caching():
    """Test raster index caching by saving file and then loading"""

    # saving raster file
    handler = DataHandlerH5(input_file, features, target=target, shape=shape,
                            max_delta=max_delta, raster_file=raster_file)
    handler.get_raster_index(input_file, target, shape)

    # loading raster file
    handler = DataHandlerH5(input_file, features, target=target, shape=shape,
                            max_delta=max_delta, raster_file=raster_file)

    assert handler.data.shape == (shape[0], shape[1],
                                  handler.data.shape[2], len(features))


def test_normalization():
    """Test correct normalization"""

    batch_handler = SpatialBatchHandler.make(
        input_files, features, targets=targets, shape=shape,
        batch_size=batch_size,
        spatial_res=spatial_res,
        spatial_sample_shape=spatial_sample_shape,
        max_delta=max_delta,
        val_split=val_split,
        time_pruning=time_pruning,
        n_batches=n_batches)

    stacked_data = \
        np.concatenate(
            [d.data for d in batch_handler.data_handlers],
            axis=2)

    for i in range(len(features)):
        std = np.std(stacked_data[:, :, :, i])
        if std == 0:
            std = 1
        mean = np.mean(stacked_data[:, :, :, i])
        assert 0.99999 <= std <= 1.00001
        assert -0.00001 <= mean <= 0.00001


def test_spatiotemporal_normalization():
    """Test correct normalization"""

    spatiotemporal_batch_handler = BatchHandler.make(
        input_files, features, targets, shape,
        batch_size=batch_size,
        spatial_res=spatial_res,
        temporal_res=temporal_res,
        temporal_sample_shape=temporal_sample_shape,
        spatial_sample_shape=spatial_sample_shape,
        max_delta=max_delta,
        val_split=val_split,
        time_pruning=time_pruning,
        n_batches=n_batches)

    stacked_data = \
        np.concatenate(
            [d.data for d in spatiotemporal_batch_handler.data_handlers],
            axis=2)

    for i in range(len(features)):
        std = np.std(stacked_data[:, :, :, i])
        if std == 0:
            std = 1
        mean = np.mean(stacked_data[:, :, :, i])
        assert 0.99999 <= std <= 1.00001
        assert -0.00001 <= mean <= 0.00001


def test_data_extraction():
    """Test data extraction class"""
    handler = DataHandlerH5(input_file, features, target=target,
                            shape=shape, max_delta=20)
    assert handler.data.shape == (shape[0], shape[1],
                                  handler.data.shape[2], len(features))
    assert handler.data.dtype == np.dtype(np.float32)
    assert handler.val_data.dtype == np.dtype(np.float32)


def test_validation_batching():
    """Test batching of validation data through
    ValidationData iterator"""

    batch_handler = SpatialBatchHandler.make(
        input_files, features, targets=targets, shape=shape,
        batch_size=batch_size,
        spatial_res=spatial_res,
        spatial_sample_shape=spatial_sample_shape,
        max_delta=max_delta,
        val_split=val_split,
        time_pruning=time_pruning,
        n_batches=n_batches)

    for batch in batch_handler.val_data:
        assert batch.high_res.dtype == np.dtype(np.float32)
        assert batch.low_res.dtype == np.dtype(np.float32)
        assert batch.low_res.shape[0] == batch.high_res.shape[0]
        assert batch.low_res.shape == \
            (batch.low_res.shape[0], spatial_sample_shape[0] // spatial_res,
             spatial_sample_shape[1] // spatial_res, len(features))
        assert batch.high_res.shape == \
            (batch.high_res.shape[0], spatial_sample_shape[0],
             spatial_sample_shape[1], len(features) - 1)


@pytest.mark.parametrize(
    'method, temporal_res',
    [('subsample', 2), ('average', 2), ('total', 2),
     ('subsample', 3), ('average', 3), ('total', 3),
     ('subsample', 4), ('average', 4), ('total', 4)]
)
def test_temporal_coarsening(method, temporal_res):
    """Test temporal coarsening of batches"""

    spatiotemporal_batch_handler = BatchHandler.make(
        input_files, features, targets, shape,
        spatial_sample_shape=spatial_sample_shape,
        temporal_sample_shape=temporal_sample_shape,
        batch_size=batch_size,
        spatial_res=spatial_res,
        temporal_res=temporal_res,
        max_delta=max_delta,
        val_split=val_split,
        time_pruning=time_pruning,
        n_batches=n_batches,
        temporal_coarsening_method=method)

    for batch in spatiotemporal_batch_handler:
        assert batch.low_res.shape[0] == batch.high_res.shape[0]
        assert batch.low_res.shape == \
            (batch.low_res.shape[0],
             spatial_sample_shape[0] // spatial_res,
             spatial_sample_shape[1] // spatial_res,
             temporal_sample_shape // temporal_res,
             len(features))
        assert batch.high_res.shape == \
            (batch.high_res.shape[0],
             spatial_sample_shape[0],
             spatial_sample_shape[1],
             temporal_sample_shape,
             len(features) - 1)


@pytest.mark.parametrize(
    'method', ('subsample', 'average', 'total')
)
def test_spatiotemporal_validation_batching(method):
    """Test batching of validation data through
    ValidationData iterator"""

    spatiotemporal_batch_handler = BatchHandler.make(
        input_files, features, targets, shape,
        spatial_sample_shape=spatial_sample_shape,
        temporal_sample_shape=temporal_sample_shape,
        batch_size=batch_size,
        spatial_res=spatial_res,
        temporal_res=temporal_res,
        max_delta=max_delta,
        val_split=val_split,
        time_pruning=time_pruning,
        n_batches=n_batches,
        temporal_coarsening_method=method)

    for batch in spatiotemporal_batch_handler.val_data:
        assert batch.low_res.shape[0] == batch.high_res.shape[0]
        assert batch.low_res.shape == \
            (batch.low_res.shape[0],
             spatial_sample_shape[0] // spatial_res,
             spatial_sample_shape[1] // spatial_res,
             temporal_sample_shape // temporal_res,
             len(features))
        assert batch.high_res.shape == \
            (batch.high_res.shape[0],
             spatial_sample_shape[0],
             spatial_sample_shape[1],
             temporal_sample_shape,
             len(features) - 1)


@pytest.mark.parametrize(
    'spatial_sample_shape, temporal_sample_shape',
    [((10, 10), 10), ((5, 5), 10),
     ((10, 10), 12), ((5, 5), 12)]
)
def test_spatiotemporal_batch_observations(spatial_sample_shape,
                                           temporal_sample_shape):
    """Test that batch observations are found in source data"""

    spatiotemporal_batch_handler = BatchHandler.make(
        input_files, features, targets, shape,
        spatial_sample_shape=spatial_sample_shape,
        temporal_sample_shape=temporal_sample_shape,
        batch_size=batch_size,
        spatial_res=spatial_res,
        temporal_res=temporal_res,
        max_delta=max_delta,
        val_split=val_split,
        time_pruning=time_pruning,
        n_batches=n_batches)

    for batch in spatiotemporal_batch_handler:
        for i, index in enumerate(
                spatiotemporal_batch_handler.current_batch_indices):
            spatial_1_slice = index[0]
            spatial_2_slice = index[1]
            temporal_slice = index[2]

            handler_index = spatiotemporal_batch_handler.current_handler_index
            handler = spatiotemporal_batch_handler.data_handlers[handler_index]

            assert np.array_equal(
                batch.high_res[i, :, :, :],
                handler.data[spatial_1_slice,
                             spatial_2_slice,
                             temporal_slice, :-1])


@pytest.mark.parametrize(
    'spatial_sample_shape, temporal_sample_shape',
    [((10, 10), 10), ((5, 5), 10),
     ((10, 10), 12), ((5, 5), 12)]
)
def test_spatiotemporal_batch_indices(spatial_sample_shape,
                                      temporal_sample_shape):
    """Test spatiotemporal batch indices for unique
    spatial indices and contiguous increasing temporal slice"""

    spatiotemporal_batch_handler = BatchHandler.make(
        input_files, features, targets, shape,
        spatial_sample_shape=spatial_sample_shape,
        temporal_sample_shape=temporal_sample_shape,
        batch_size=batch_size,
        spatial_res=spatial_res,
        temporal_res=temporal_res,
        max_delta=max_delta,
        val_split=val_split,
        time_pruning=time_pruning,
        n_batches=n_batches)

    all_spatial_tuples = []
    for _ in spatiotemporal_batch_handler:
        for index in spatiotemporal_batch_handler.current_batch_indices:
            spatial_1_slice = np.arange(index[0].start, index[0].stop)
            spatial_2_slice = np.arange(index[1].start, index[1].stop)
            temporal_slice = np.arange(index[2].start, index[2].stop)
            spatial_tuples = []
            for s1 in spatial_1_slice:
                for s2 in spatial_2_slice:
                    spatial_tuples.append(tuple([s1, s2]))
            assert len(spatial_tuples) == len(list(set(spatial_tuples)))

            all_spatial_tuples.append(np.array(spatial_tuples))

            sorted_temporal_slice = temporal_slice.copy()
            sorted_temporal_slice.sort()
            assert np.array_equal(sorted_temporal_slice, temporal_slice)

            assert all(temporal_slice[1:] - temporal_slice[:-1] == 1)

    comparisons = []
    for i, s1 in enumerate(all_spatial_tuples):
        for j, s2 in enumerate(all_spatial_tuples):
            if i != j:
                comparisons.append(np.array_equal(s1, s2))
    assert not all(comparisons)


def test_spatiotemporal_batch_handling(plot=False):
    """Test spatiotemporal batch handling class"""

    spatiotemporal_batch_handler = BatchHandler.make(
        input_files, features, targets, shape,
        spatial_sample_shape=spatial_sample_shape,
        temporal_sample_shape=temporal_sample_shape,
        batch_size=batch_size,
        spatial_res=spatial_res,
        temporal_res=temporal_res,
        max_delta=max_delta,
        val_split=val_split,
        time_pruning=time_pruning,
        n_batches=n_batches)

    for batch in spatiotemporal_batch_handler:
        assert batch.low_res.shape[0] == batch.high_res.shape[0]

    for i, batch in enumerate(spatiotemporal_batch_handler):
        assert batch.low_res.shape == \
            (batch.low_res.shape[0],
             spatial_sample_shape[0] // spatial_res,
             spatial_sample_shape[1] // spatial_res,
             temporal_sample_shape // temporal_res,
             len(features))
        assert batch.high_res.shape == \
            (batch.high_res.shape[0],
             spatial_sample_shape[0],
             spatial_sample_shape[1],
             temporal_sample_shape,
             len(features) - 1)

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

    batch_handler = SpatialBatchHandler.make(
        input_files, features, targets=targets, shape=shape,
        batch_size=batch_size,
        spatial_res=spatial_res,
        spatial_sample_shape=spatial_sample_shape,
        max_delta=max_delta,
        val_split=val_split,
        time_pruning=time_pruning,
        n_batches=n_batches)

    for batch in batch_handler:
        assert batch.low_res.shape[0] == batch.high_res.shape[0]

    for i, batch in enumerate(batch_handler):
        assert batch.high_res.dtype == np.float32
        assert batch.low_res.dtype == np.float32
        assert batch.low_res.shape == \
            (batch.low_res.shape[0],
             spatial_sample_shape[0] // spatial_res,
             spatial_sample_shape[1] // spatial_res,
             len(features))
        assert batch.high_res.shape == \
            (batch.high_res.shape[0],
             spatial_sample_shape[0],
             spatial_sample_shape[1],
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

    batch_handler = SpatialBatchHandler.make(
        input_files, features, targets=targets, shape=shape,
        batch_size=batch_size,
        spatial_res=spatial_res,
        spatial_sample_shape=spatial_sample_shape,
        max_delta=max_delta,
        val_split=val_split,
        time_pruning=time_pruning,
        n_batches=n_batches)

    val_observations = 0
    batch_handler.val_data._i = 0
    for batch in batch_handler.val_data:
        assert batch.low_res.shape[0] == batch.high_res.shape[0]
        assert list(batch.low_res.shape[1:3]) == \
            [s // spatial_res for s in spatial_sample_shape]
        val_observations += batch.low_res.shape[0]

    n_observations = 0
    for f in input_files:

        handler = DataHandlerH5(f, features, target, shape,
                                max_delta, raster_file=raster_file,
                                val_split=val_split,
                                time_pruning=time_pruning)
        data = handler.extract_data(
            f, handler.raster_index, handler.time_index,
            features, time_pruning)
        n_observations += data.shape[2]

    assert val_observations == int(val_split * n_observations)


@pytest.mark.parametrize(
    'spatial_res', (10, 5, 4, 2)
)
def test_spatial_coarsening(spatial_res, plot=False):
    """Test spatial coarsening"""

    handler = DataHandlerH5(input_file, features, target=target,
                            shape=shape, max_delta=20)

    handler_data = handler.extract_data(
        input_file, handler.raster_index, handler.time_index,
        features, time_pruning)
    handler_data = handler_data.transpose((2, 0, 1, 3))
    coarse_data = utilities.spatial_coarsening(handler_data, spatial_res)
    direct_avg = np.zeros(coarse_data.shape)

    for i in range(direct_avg.shape[1]):
        for j in range(direct_avg.shape[1]):
            direct_avg[:, i, j, :] = \
                np.mean(handler_data[:, spatial_res * i:spatial_res * (i + 1),
                                     spatial_res * j:spatial_res * (j + 1),
                                     :], axis=(1, 2))

    np.testing.assert_equal(coarse_data, direct_avg)

    if plot:
        _, ax = plt.subplots(1, 2)
        ax[0].imshow(handler_data[0, :, :, 0])
        ax[1].imshow(coarse_data[0, :, :, 0])
        plt.show()
