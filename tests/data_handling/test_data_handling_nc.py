# -*- coding: utf-8 -*-
"""pytests for data handling"""

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from sup3r import TEST_DATA_DIR
from sup3r.preprocessing.batch_handling import (
    BatchHandler,
    SpatialBatchHandler,
)
from sup3r.preprocessing.data_handling import DataHandlerNC as DataHandler
from sup3r.utilities.interpolation import Interpolator
from sup3r.utilities.pytest import make_fake_era_files, make_fake_nc_files

INPUT_FILE = os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_00_00_00')
features = ['U_100m', 'V_100m', 'BVF_MO_200m']
val_split = 0.2
target = (19.3, -123.5)
shape = (8, 8)
sample_shape = (8, 8, 6)
s_enhance = 2
t_enhance = 2
dh_kwargs = dict(target=target,
                 shape=shape,
                 max_delta=20,
                 lr_only_features=('BVF*m', 'topography'),
                 sample_shape=sample_shape,
                 temporal_slice=slice(None, None, 1),
                 worker_kwargs=dict(max_workers=1),
                 single_ts_files=True)
bh_kwargs = dict(batch_size=8, n_batches=20, s_enhance=s_enhance,
                 t_enhance=t_enhance, worker_kwargs=dict(max_workers=1))


def test_topography():
    """Test that topography is batched and extracted correctly"""

    features = ['U_100m', 'V_100m', 'topography']

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 6)
        data_handler = DataHandler(input_files, features, val_split=0.0,
                                   **dh_kwargs)
        ri = data_handler.raster_index
        with xr.open_mfdataset(input_files, concat_dim='Time',
                               combine='nested') as res:
            topo = np.array(res['HGT'][tuple([slice(None)] + ri)])
        topo = np.transpose(topo, (1, 2, 0))[::-1]
        topo_idx = data_handler.features.index('topography')
        assert np.allclose(topo, data_handler.data[..., :, topo_idx])
        st_batch_handler = BatchHandler([data_handler], **bh_kwargs)
        assert data_handler.hr_out_features == features[:2]
        assert data_handler.data.shape[-1] == len(features)

        for batch in st_batch_handler:
            assert batch.high_res.shape[-1] == 2
            assert batch.low_res.shape[-1] == len(features)


def test_height_interpolation():
    """Make sure height interpolation is working as expected.
    Specifically that it is returning the correct number of time steps"""

    height = 250
    features = [f'U_{height}m']
    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        data_handler = DataHandler(input_files, features, val_split=0.0,
                                   **dh_kwargs)
        raster_index = data_handler.raster_index

        data = data_handler.data

        tmp = xr.open_mfdataset(input_files, concat_dim='Time',
                                combine='nested')

        U_tmp = Interpolator.unstagger_var(tmp, 'U', raster_index)
        h_array = Interpolator.calc_height(tmp, raster_index)
        if data_handler.invert_lat:
            data = data[::-1]

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for t in range(data.shape[2]):

                    val = data[i, j, t, :]

                    # get closest U value
                    for h, _ in enumerate(h_array[t, :, i, j][:-1]):
                        lower_hgt = h_array[t, h, i, j]
                        higher_hgt = h_array[t, h + 1, i, j]
                        if lower_hgt <= height <= higher_hgt:
                            alpha = (height - lower_hgt)
                            alpha /= (higher_hgt - lower_hgt)
                            lower_val = U_tmp[t, h, i, j]
                            higher_val = U_tmp[t, h + 1, i, j]
                            compare_val = lower_val * (1 - alpha)
                            compare_val += higher_val * alpha

                    # get vertical standard deviation of U
                    stdev = np.std(U_tmp[t, :, i, j])

                    assert compare_val - stdev <= val <= compare_val + stdev


def test_single_site_extraction():
    """Make sure single location can be extracted from ERA data without
    error."""

    height = 10
    features = [f'windspeed_{height}m']
    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_era_files(td, INPUT_FILE, 8)
        kwargs = dh_kwargs.copy()
        kwargs['shape'] = [1, 1]
        data_handler = DataHandler(input_files, features, val_split=0.0,
                                   **kwargs)

        data = data_handler.data[0, 0, :, 0]

        data_handler = DataHandler(input_files, features, val_split=0.0,
                                   **dh_kwargs)

        baseline = data_handler.data[-1, 0, :, 0]

        assert np.allclose(baseline, data)


@pytest.mark.parametrize('sample_shape',
                         [(4, 4, 6), (2, 2, 6), (4, 4, 4), (2, 2, 4)])
def test_spatiotemporal_batch_caching(sample_shape):
    """Test that batch observations are found in source data"""

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        cache_pattern = os.path.join(td, 'cache_')
        dh_kwargs_new = dh_kwargs.copy()
        dh_kwargs_new['sample_shape'] = sample_shape
        dh_kwargs_new['lr_only_features'] = ['BVF*']
        data_handler = DataHandler(input_files, features,
                                   cache_pattern=cache_pattern,
                                   **dh_kwargs_new)
        batch_handler = BatchHandler([data_handler], **bh_kwargs)

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


def test_data_caching():
    """Test data extraction class"""

    with tempfile.TemporaryDirectory() as td:
        cache_pattern = os.path.join(td, 'cached_features_h5')
        if os.path.exists(cache_pattern):
            os.system(f'rm {cache_pattern}')
        handler = DataHandler(INPUT_FILE, features,
                              cache_pattern=cache_pattern, **dh_kwargs,
                              val_split=0.1)
        assert handler.data is None
        handler.load_cached_data()
        assert handler.data.shape == (shape[0], shape[1],
                                      handler.data.shape[2], len(features))
        assert handler.data.dtype == np.dtype(np.float32)
        assert handler.val_data.dtype == np.dtype(np.float32)


def test_feature_handler():
    """Make sure compute feature is returing float32"""

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        handler = DataHandler(input_files, features, **dh_kwargs)
        tmp = handler.data
        assert tmp.dtype == np.dtype(np.float32)

        var_names = {'T_bottom': ['T', 100],
                     'T_top': ['T', 200],
                     'P_bottom': ['P', 100],
                     'P_top': ['P', 200]}
        for _, v in var_names.items():
            tmp = handler.extract_feature(
                input_files, handler.raster_index, f'{v[0]}_{v[1]}m')
            assert tmp.dtype == np.dtype(np.float32)


def test_get_full_domain():
    """Test data handling without target, shape, or raster_file input"""

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        handler = DataHandler(input_files, features,
                              worker_kwargs=dict(max_workers=1))
        tmp = xr.open_dataset(input_files[0])
        shape = np.array(tmp.XLAT.values).shape[1:]
        target = (tmp.XLAT.values[0, 0, 0], tmp.XLONG.values[0, 0, 0])
        assert handler.grid_shape == shape
        assert handler.target == target


def test_get_target():
    """Test data handling without target or raster_file input"""
    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        handler = DataHandler(input_files, features, shape=(4, 4),
                              worker_kwargs=dict(max_workers=1))
        tmp = xr.open_dataset(input_files[0])
        target = (tmp.XLAT.values[0, 0, 0], tmp.XLONG.values[0, 0, 0])
        assert handler.grid_shape == (4, 4)
        assert handler.target == target


def test_raster_index_caching():
    """Test raster index caching by saving file and then loading"""

    # saving raster file
    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        raster_file = os.path.join(td, 'raster.npy')
        handler = DataHandler(input_files, features, raster_file=raster_file,
                              **dh_kwargs)
        assert handler.lat_lon[0, 0, 0] > handler.lat_lon[-1, 0, 0]
        assert np.allclose(handler.target, handler.lat_lon[-1, 0, :], atol=1)

        # loading raster file
        handler = DataHandler(input_files, features, raster_file=raster_file,
                              worker_kwargs=dict(max_workers=1))
        assert np.allclose(handler.target, target, atol=1)
        assert handler.data.shape == (shape[0], shape[1],
                                      handler.data.shape[2], len(features))
        assert handler.grid_shape == (shape[0], shape[1])


def test_normalization_input():
    """Test correct normalization input"""

    means = {f: 10 for f in features}
    stds = {f: 20 for f in features}
    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        data_handler = DataHandler(input_files, features, **dh_kwargs)
        batch_handler = BatchHandler([data_handler], means=means, stds=stds,
                                     **bh_kwargs)

    assert all(batch_handler.means[f] == means[f] for f in features)
    assert all(batch_handler.stds[f] == stds[f] for f in features)


def test_normalization():
    """Test correct normalization"""

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        data_handler = DataHandler(input_files, features, **dh_kwargs)
        batch_handler = BatchHandler([data_handler], **bh_kwargs)

        stacked_data = np.concatenate(
            [d.data for d in batch_handler.data_handlers], axis=2)

        for i in range(len(features)):
            std = np.std(stacked_data[:, :, :, i])
            if std == 0:
                std = 1
            mean = np.mean(stacked_data[:, :, :, i])
            assert 0.99999 <= std <= 1.00001
            assert -0.00001 <= mean <= 0.00001


def test_spatiotemporal_normalization():
    """Test correct normalization"""

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        data_handler = DataHandler(input_files, features, **dh_kwargs)
        batch_handler = BatchHandler([data_handler], **bh_kwargs)

        stacked_data = np.concatenate(
            [d.data for d in batch_handler.data_handlers], axis=2)

        for i in range(len(features)):
            std = np.std(stacked_data[:, :, :, i])
            if std == 0:
                std = 1
            mean = np.mean(stacked_data[:, :, :, i])
            assert 0.99999 <= std <= 1.00001
            assert -0.00001 <= mean <= 0.00001


def test_data_extraction():
    """Test data extraction class"""
    handler = DataHandler(INPUT_FILE, features, val_split=0.05, **dh_kwargs)
    assert handler.data.shape == (shape[0], shape[1],
                                  handler.data.shape[2], len(features))
    assert handler.data.dtype == np.dtype(np.float32)
    assert handler.val_data.dtype == np.dtype(np.float32)


def test_validation_batching():
    """Test batching of validation data through
    ValidationData iterator"""

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        dh_kwargs_new = dh_kwargs.copy()
        dh_kwargs_new['sample_shape'] = (sample_shape[0], sample_shape[1], 1)
        data_handler = DataHandler(input_files, features, **dh_kwargs_new)
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


@pytest.mark.parametrize(
    'method, t_enhance',
    [('subsample', 2), ('average', 2), ('total', 2),
     ('subsample', 3), ('average', 3), ('total', 3)]
)
def test_temporal_coarsening(method, t_enhance):
    """Test temporal coarsening of batches"""

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        data_handler = DataHandler(input_files, features, **dh_kwargs)
        bh_kwargs_new = bh_kwargs.copy()
        bh_kwargs_new['t_enhance'] = t_enhance
        batch_handler = BatchHandler([data_handler],
                                     temporal_coarsening_method=method,
                                     **bh_kwargs_new)

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


@pytest.mark.parametrize(
    'method', ('subsample', 'average', 'total')
)
def test_spatiotemporal_validation_batching(method):
    """Test batching of validation data through
    ValidationData iterator"""

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        data_handler = DataHandler(input_files, features, **dh_kwargs)
        batch_handler = BatchHandler([data_handler],
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


@pytest.mark.parametrize('sample_shape',
                         [(4, 4, 6), (2, 2, 6), (4, 4, 4), (2, 2, 4)])
def test_spatiotemporal_batch_observations(sample_shape):
    """Test that batch observations are found in source data"""

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        dh_kwargs_new = dh_kwargs.copy()
        dh_kwargs_new['sample_shape'] = sample_shape
        dh_kwargs_new['lr_only_features'] = 'BVF*'
        data_handler = DataHandler(input_files, features, **dh_kwargs_new)
        batch_handler = BatchHandler([data_handler], **bh_kwargs)

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


@pytest.mark.parametrize('sample_shape',
                         [(4, 4, 6), (2, 2, 6), (4, 4, 4), (2, 2, 4)])
def test_spatiotemporal_batch_indices(sample_shape):
    """Test spatiotemporal batch indices for unique
    spatial indices and contiguous increasing temporal slice"""

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        dh_kwargs_new = dh_kwargs.copy()
        dh_kwargs_new['sample_shape'] = sample_shape
        data_handler = DataHandler(input_files, features, **dh_kwargs_new)
        batch_handler = BatchHandler([data_handler], **bh_kwargs)

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

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        data_handler = DataHandler(input_files, features, **dh_kwargs)
        batch_handler = BatchHandler([data_handler], **bh_kwargs)
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

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        data_handler = DataHandler(input_files, features, **dh_kwargs)
        batch_handler = SpatialBatchHandler([data_handler], **bh_kwargs)

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

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        data_handler = DataHandler(input_files, features, val_split=val_split,
                                   **dh_kwargs)
        batch_handler = BatchHandler([data_handler], **bh_kwargs)

        val_observations = 0
        batch_handler.val_data._i = 0
        for batch in batch_handler.val_data:
            assert batch.low_res.shape[0] == batch.high_res.shape[0]
            assert list(batch.low_res.shape[1:3]) == [s // s_enhance for s
                                                      in sample_shape[:2]]
            val_observations += batch.low_res.shape[0]

        n_observations = 0
        for f in input_files:
            handler = DataHandler(f, features, val_split=val_split,
                                  **dh_kwargs)
            data = handler.run_all_data_init()
            n_observations += data.shape[2]

        assert val_observations == int(val_split * n_observations)
