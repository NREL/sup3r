# -*- coding: utf-8 -*-
"""pytests for data handling"""

import os
import numpy as np
import pytest
import matplotlib.pyplot as plt
import tempfile
import xarray as xr

from sup3r import TEST_DATA_DIR
from sup3r.preprocessing.data_handling import DataHandlerNC
from sup3r.preprocessing.batch_handling import (BatchHandler,
                                                SpatialBatchHandler)
from sup3r.utilities import utilities

input_file = os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_00_00_00')
input_files = [
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_00_00_00'),
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_01_00_00'),
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_00_00_00'),
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_01_00_00'),
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_00_00_00'),
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_01_00_00'),
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_00_00_00'),
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_01_00_00')]
target = (19, -125)
targets = target
shape = (8, 8)
features = ['U_100m', 'V_100m', 'BVF_MO_200m']
batch_size = 8
spatial_sample_shape = (8, 8)
s_enhance = 2
max_delta = 20
val_split = 0.2
raster_file = os.path.join(tempfile.gettempdir(), 'tmp_raster_nc.txt')
time_pruning = 1
n_batches = 20
temporal_sample_shape = 6
t_enhance = 2
list_chunk_size = 10


def test_height_interpolation():
    """Make sure height interpolation is working as expected.
    Specifically that it is returning the correct number of time steps"""

    height = 250
    features = [f'U_{height}m']
    handler = DataHandlerNC(input_files, features, target=target,
                            shape=shape, max_delta=20)

    raster_index = handler.raster_index

    tmp = xr.open_mfdataset(
        input_files, concat_dim='Time', combine='nested')
    U_tmp = utilities.unstagger_var(tmp, 'U')

    data = handler.extract_data(
        input_files, raster_index, features,
        time_pruning=time_pruning,
        max_extract_workers=None,
        max_compute_workers=None,
        time_chunk_size=100)
    train_data, val_data = handler.split_data(data)

    print(data.shape)

    assert np.array_equal(
        data, np.concatenate([val_data, train_data], axis=2))

    h_array = utilities.calc_height(tmp)

    arrs = []

    h_array = h_array.reshape((U_tmp.shape[0], U_tmp.shape[1],
                               np.product(U_tmp.shape[2:]))).T

    var_array = U_tmp.reshape((U_tmp.shape[0], U_tmp.shape[1],
                               np.product(U_tmp.shape[2:]))).T

    # Interpolate each column of height and var to specified levels
    for t in range(U_tmp.shape[0]):

        arrs.append(np.array(
            [np.interp([height], h, var) for h, var
             in zip(h_array[:, :, t], var_array[:, :, t])],
            np.float32)[np.newaxis, :, :])

    arrs = np.concatenate(arrs, axis=0)

    arrs = arrs.T.reshape(
        (U_tmp.shape[0], U_tmp.shape[2], U_tmp.shape[3]))

    arrs = arrs[tuple([slice(None)] + raster_index)]
    arrs = np.transpose(arrs, (1, 2, 0))

    assert data.shape[2] == arrs.shape[2]

    assert np.array_equal(data, arrs[:, :, :, np.newaxis])


@pytest.mark.parametrize(
    'spatial_sample_shape, temporal_sample_shape',
    [((4, 4), 6), ((2, 2), 6),
     ((4, 4), 4), ((2, 2), 4)]
)
def test_spatiotemporal_batch_caching(spatial_sample_shape,
                                      temporal_sample_shape):
    """Test that batch observations are found in source data"""

    cache_prefixes = []
    n_cache_files = int(np.ceil(len(input_files) / list_chunk_size))
    with tempfile.TemporaryDirectory() as td:
        for i in range(n_cache_files):
            tmp = os.path.join(td, f'cache_{i}')
            if os.path.exists(tmp):
                os.system(f'rm {tmp}')
            cache_prefixes.append(tmp)

        st_batch_handler = BatchHandler.make(
            input_files, features, targets, shape,
            spatial_sample_shape=spatial_sample_shape,
            temporal_sample_shape=temporal_sample_shape,
            batch_size=batch_size,
            s_enhance=s_enhance,
            t_enhance=t_enhance,
            max_delta=max_delta,
            val_split=val_split,
            time_pruning=time_pruning,
            n_batches=n_batches,
            list_chunk_size=list_chunk_size,
            cache_file_prefixes=cache_prefixes)

        for batch in st_batch_handler:
            for i, index in enumerate(
                    st_batch_handler.current_batch_indices):
                spatial_1_slice = index[0]
                spatial_2_slice = index[1]
                temporal_slice = index[2]

                handler_index = st_batch_handler.current_handler_index
                handler = st_batch_handler.data_handlers[handler_index]

                assert np.array_equal(
                    batch.high_res[i, :, :, :],
                    handler.data[spatial_1_slice,
                                 spatial_2_slice,
                                 temporal_slice, :-1])


def test_data_caching():
    """Test data extraction class"""

    with tempfile.TemporaryDirectory() as td:
        cache_prefix = os.path.join(td, 'cached_features_h5')
        if os.path.exists(cache_prefix):
            os.system(f'rm {cache_prefix}')
        handler = DataHandlerNC(input_file, features, target=target,
                                shape=shape, max_delta=20,
                                cache_file_prefix=cache_prefix)
        assert handler.data is None
        handler.load_cached_data()
        assert handler.data.shape == (shape[0], shape[1],
                                      handler.data.shape[2], len(features))
        assert handler.data.dtype == np.dtype(np.float32)
        assert handler.val_data.dtype == np.dtype(np.float32)


def test_feature_handler():
    """Make sure compute feature is returing float32"""

    handler = DataHandlerNC(input_files, features, target=target, shape=shape,
                            max_delta=max_delta)
    tmp = handler.extract_data(
        input_files, handler.raster_index,
        features, time_pruning)
    assert tmp.dtype == np.dtype(np.float32)

    var_names = {'T_bottom': ['T', 100],
                 'T_top': ['T', 200],
                 'P_bottom': ['P', 100],
                 'P_top': ['P', 200]}
    for _, v in var_names.items():
        tmp = handler.extract_feature(
            input_files, handler.raster_index, f'{v[0]}_{v[1]}m')
        assert tmp.dtype == np.dtype(np.float32)


def test_raster_index_caching():
    """Test raster index caching by saving file and then loading"""

    # saving raster file

    with tempfile.TemporaryDirectory() as td:
        raster_file = os.path.join(td, 'raster.txt')
        handler = DataHandlerNC(input_file, features, target=target,
                                shape=shape, max_delta=max_delta,
                                raster_file=raster_file)
        handler.get_raster_index(input_file, target, shape)

        # loading raster file
        handler = DataHandlerNC(input_file, features, target=target,
                                shape=shape, max_delta=max_delta,
                                raster_file=raster_file)

        assert handler.data.shape == (shape[0], shape[1],
                                      handler.data.shape[2], len(features))


def test_normalization():
    """Test correct normalization"""

    batch_handler = SpatialBatchHandler.make(
        input_files, features, targets=targets, shape=shape,
        batch_size=batch_size,
        s_enhance=s_enhance,
        max_delta=max_delta,
        val_split=val_split,
        time_pruning=time_pruning,
        n_batches=n_batches,
        list_chunk_size=list_chunk_size)

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
        s_enhance=s_enhance,
        t_enhance=t_enhance,
        temporal_sample_shape=temporal_sample_shape,
        spatial_sample_shape=spatial_sample_shape,
        max_delta=max_delta,
        val_split=val_split,
        time_pruning=time_pruning,
        n_batches=n_batches,
        list_chunk_size=list_chunk_size)

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
    handler = DataHandlerNC(input_file, features, target=target,
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
        s_enhance=s_enhance,
        max_delta=max_delta,
        val_split=val_split,
        time_pruning=time_pruning,
        n_batches=n_batches,
        spatial_sample_shape=spatial_sample_shape,
        list_chunk_size=list_chunk_size)

    for batch in batch_handler.val_data:
        assert batch.high_res.dtype == np.dtype(np.float32)
        assert batch.low_res.dtype == np.dtype(np.float32)
        assert batch.low_res.shape[0] == batch.high_res.shape[0]
        assert batch.low_res.shape == \
            (batch.low_res.shape[0], spatial_sample_shape[0] // s_enhance,
             spatial_sample_shape[1] // s_enhance, len(features))
        assert batch.high_res.shape == \
            (batch.high_res.shape[0], spatial_sample_shape[0],
             spatial_sample_shape[1], len(features) - 1)


@pytest.mark.parametrize(
    'method, t_enhance',
    [('subsample', 2), ('average', 2), ('total', 2),
     ('subsample', 3), ('average', 3), ('total', 3)]
)
def test_temporal_coarsening(method, t_enhance):
    """Test temporal coarsening of batches"""

    spatiotemporal_batch_handler = BatchHandler.make(
        input_files, features, targets, shape,
        spatial_sample_shape=spatial_sample_shape,
        temporal_sample_shape=temporal_sample_shape,
        batch_size=batch_size,
        s_enhance=s_enhance,
        t_enhance=t_enhance,
        max_delta=max_delta,
        val_split=val_split,
        time_pruning=time_pruning,
        n_batches=n_batches,
        temporal_coarsening_method=method,
        list_chunk_size=list_chunk_size)

    for batch in spatiotemporal_batch_handler:
        assert batch.low_res.shape[0] == batch.high_res.shape[0]
        assert batch.low_res.shape == \
            (batch.low_res.shape[0],
             spatial_sample_shape[0] // s_enhance,
             spatial_sample_shape[1] // s_enhance,
             temporal_sample_shape // t_enhance,
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
        s_enhance=s_enhance,
        t_enhance=t_enhance,
        max_delta=max_delta,
        val_split=val_split,
        time_pruning=time_pruning,
        n_batches=n_batches,
        temporal_coarsening_method=method,
        list_chunk_size=list_chunk_size)

    for batch in spatiotemporal_batch_handler.val_data:
        assert batch.low_res.shape[0] == batch.high_res.shape[0]
        assert batch.low_res.shape == \
            (batch.low_res.shape[0],
             spatial_sample_shape[0] // s_enhance,
             spatial_sample_shape[1] // s_enhance,
             temporal_sample_shape // t_enhance,
             len(features))
        assert batch.high_res.shape == \
            (batch.high_res.shape[0],
             spatial_sample_shape[0],
             spatial_sample_shape[1],
             temporal_sample_shape,
             len(features) - 1)


@pytest.mark.parametrize(
    'spatial_sample_shape, temporal_sample_shape',
    [((4, 4), 6), ((2, 2), 6),
     ((4, 4), 4), ((2, 2), 4)]
)
def test_spatiotemporal_batch_observations(spatial_sample_shape,
                                           temporal_sample_shape):
    """Test that batch observations are found in source data"""

    spatiotemporal_batch_handler = BatchHandler.make(
        input_files, features, targets, shape,
        spatial_sample_shape=spatial_sample_shape,
        temporal_sample_shape=temporal_sample_shape,
        batch_size=batch_size,
        s_enhance=s_enhance,
        t_enhance=t_enhance,
        max_delta=max_delta,
        val_split=val_split,
        time_pruning=time_pruning,
        n_batches=n_batches,
        list_chunk_size=list_chunk_size)

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
    [((4, 4), 6), ((2, 2), 6),
     ((4, 4), 4), ((2, 2), 4)]
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
        s_enhance=s_enhance,
        t_enhance=t_enhance,
        max_delta=max_delta,
        val_split=val_split,
        time_pruning=time_pruning,
        n_batches=n_batches,
        list_chunk_size=list_chunk_size)

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
        s_enhance=s_enhance,
        t_enhance=t_enhance,
        max_delta=max_delta,
        val_split=val_split,
        time_pruning=time_pruning,
        n_batches=n_batches,
        list_chunk_size=list_chunk_size)

    for batch in spatiotemporal_batch_handler:
        assert batch.low_res.shape[0] == batch.high_res.shape[0]

    for i, batch in enumerate(spatiotemporal_batch_handler):
        assert batch.low_res.shape == \
            (batch.low_res.shape[0],
             spatial_sample_shape[0] // s_enhance,
             spatial_sample_shape[1] // s_enhance,
             temporal_sample_shape // t_enhance,
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
        s_enhance=s_enhance,
        spatial_sample_shape=spatial_sample_shape,
        max_delta=max_delta,
        val_split=val_split,
        time_pruning=time_pruning,
        n_batches=n_batches,
        list_chunk_size=list_chunk_size)

    for batch in batch_handler:
        assert batch.low_res.shape[0] == batch.high_res.shape[0]

    for i, batch in enumerate(batch_handler):
        assert batch.high_res.dtype == np.float32
        assert batch.low_res.dtype == np.float32
        assert batch.low_res.shape == \
            (batch.low_res.shape[0],
             spatial_sample_shape[0] // s_enhance,
             spatial_sample_shape[1] // s_enhance,
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
        s_enhance=s_enhance,
        spatial_sample_shape=spatial_sample_shape,
        max_delta=max_delta,
        val_split=val_split,
        time_pruning=time_pruning,
        n_batches=n_batches,
        list_chunk_size=list_chunk_size)

    val_observations = 0
    batch_handler.val_data._i = 0
    for batch in batch_handler.val_data:
        assert batch.low_res.shape[0] == batch.high_res.shape[0]
        assert list(batch.low_res.shape[1:3]) == \
            [s // s_enhance for s in spatial_sample_shape]
        val_observations += batch.low_res.shape[0]

    n_observations = 0
    for f in input_files:

        handler = DataHandlerNC(f, features, target, shape,
                                max_delta, raster_file=raster_file,
                                val_split=val_split,
                                time_pruning=time_pruning)
        data = handler.extract_data(
            input_file, handler.raster_index,
            features, time_pruning)
        n_observations += data.shape[2]

    assert val_observations == int(val_split * n_observations)


@pytest.mark.parametrize(
    's_enhance', (4, 2)
)
def test_spatial_coarsening(s_enhance, plot=False):
    """Test spatial coarsening"""

    handler = DataHandlerNC(input_file, features, target=target,
                            shape=shape, max_delta=20)

    handler_data = handler.extract_data(
        input_file, handler.raster_index,
        features, time_pruning)
    handler_data = handler_data.transpose((2, 0, 1, 3))
    coarse_data = utilities.spatial_coarsening(handler_data, s_enhance)
    direct_avg = np.zeros(coarse_data.shape)

    for i in range(direct_avg.shape[1]):
        for j in range(direct_avg.shape[1]):
            direct_avg[:, i, j, :] = \
                np.mean(handler_data[:, s_enhance * i:s_enhance * (i + 1),
                                     s_enhance * j:s_enhance * (j + 1),
                                     :], axis=(1, 2))

    np.testing.assert_equal(coarse_data, direct_avg)

    if plot:
        _, ax = plt.subplots(1, 2)
        ax[0].imshow(handler_data[0, :, :, 0])
        ax[1].imshow(coarse_data[0, :, :, 0])
        plt.show()
