# -*- coding: utf-8 -*-
"""pytests bias correction calculations"""
import h5py
import os
import pytest
import tempfile
import numpy as np
import xarray as xr

from sup3r import TEST_DATA_DIR, CONFIG_DIR
from sup3r.models import Sup3rGan
from sup3r.qa.qa import Sup3rQa
from sup3r.pipeline.forward_pass import ForwardPass, ForwardPassStrategy
from sup3r.bias.bias_calc import LinearCorrection, MonthlyLinearCorrection
from sup3r.bias.bias_transforms import local_linear_bc, monthly_local_linear_bc


FP_NSRDB = os.path.join(TEST_DATA_DIR, 'test_nsrdb_co_2018.h5')
FP_CC = os.path.join(TEST_DATA_DIR, 'rsds_test.nc')

with xr.open_mfdataset(FP_CC) as fh:
    MIN_LAT = np.min(fh.lat.values)
    MIN_LON = np.min(fh.lon.values) - 360
    TARGET = (MIN_LAT, MIN_LON)
    SHAPE = (len(fh.lat.values), len(fh.lon.values))


def test_linear_bc():
    """Test linear bias correction"""

    calc = LinearCorrection(FP_NSRDB, FP_CC, 'ghi', 'rsds',
                            TARGET, SHAPE, bias_handler='DataHandlerNCforCC')

    # test a known in-bounds gid
    bias_gid = 5
    dist, base_gid = calc.get_base_gid(bias_gid, 1)
    bias_data = calc.get_bias_data(bias_gid)
    base_data, _ = calc.get_base_data(calc.base_fps, calc.base_dset,
                                      base_gid, calc.base_handler,
                                      daily_avg=True)
    bias_coord = calc.bias_meta.loc[bias_gid, ['latitude', 'longitude']]
    base_coord = calc.base_meta.loc[base_gid, ['latitude', 'longitude']]
    true_dist = bias_coord.values - base_coord.values
    true_dist = np.hypot(true_dist[0], true_dist[1])
    assert np.allclose(true_dist, dist)
    assert true_dist < 0.1
    true_scalar = base_data.std() / bias_data.std()
    true_adder = base_data.mean() - bias_data.mean() * true_scalar

    scalar, adder = calc.run(knn=1, threshold=0.6, fill_extend=False,
                             max_workers=1)

    assert len(scalar.shape) == 3
    assert len(adder.shape) == 3
    assert scalar.shape[-1] == 1
    assert adder.shape[-1] == 1

    iloc = np.where(calc.bias_gid_raster == bias_gid)
    assert np.allclose(true_scalar, scalar[iloc])
    assert np.allclose(true_adder, adder[iloc])

    corners = ((0, 0, 0), (-1, 0, 0), (0, -1, 0), (-1, -1, 0))
    for corner in corners:
        assert np.isnan(scalar[corner])
        assert np.isnan(adder[corner])
    nan_mask = np.isnan(scalar)
    assert np.isnan(adder[nan_mask]).all()

    # make sure the NN fill works for out-of-bounds pixels
    scalar, adder = calc.run(knn=1, threshold=0.6, fill_extend=True,
                             max_workers=1)

    iloc = np.where(calc.bias_gid_raster == bias_gid)
    assert np.allclose(true_scalar, scalar[iloc])
    assert np.allclose(true_adder, adder[iloc])

    assert not np.isnan(scalar[nan_mask]).any()
    assert not np.isnan(adder[nan_mask]).any()

    # make sure smoothing affects the out-of-bounds pixels but not the in-bound
    smooth_scalar, smooth_adder = calc.run(knn=1, threshold=0.6,
                                           fill_extend=True, smooth_extend=2,
                                           max_workers=1)
    assert np.allclose(smooth_scalar[~nan_mask], scalar[~nan_mask])
    assert np.allclose(smooth_adder[~nan_mask], adder[~nan_mask])
    assert not np.allclose(smooth_scalar[nan_mask], scalar[nan_mask])
    assert not np.allclose(smooth_adder[nan_mask], adder[nan_mask])

    # parallel test
    par_scalar, par_adder = calc.run(knn=1, threshold=0.6, fill_extend=True,
                                     smooth_extend=2, max_workers=2)
    assert np.allclose(smooth_scalar, par_scalar)
    assert np.allclose(smooth_adder, par_adder)


def test_monthly_linear_bc():
    """Test linear bias correction on a month-by-month basis"""

    calc = MonthlyLinearCorrection(FP_NSRDB, FP_CC, 'ghi', 'rsds',
                                   TARGET, SHAPE,
                                   bias_handler='DataHandlerNCforCC')

    # test a known in-bounds gid
    bias_gid = 5
    dist, base_gid = calc.get_base_gid(bias_gid, 1)
    bias_data = calc.get_bias_data(bias_gid)
    base_data, base_ti = calc.get_base_data(calc.base_fps, calc.base_dset,
                                            base_gid, calc.base_handler,
                                            daily_avg=True)
    bias_coord = calc.bias_meta.loc[bias_gid, ['latitude', 'longitude']]
    base_coord = calc.base_meta.loc[base_gid, ['latitude', 'longitude']]
    true_dist = bias_coord.values - base_coord.values
    true_dist = np.hypot(true_dist[0], true_dist[1])
    assert np.allclose(true_dist, dist)
    assert true_dist < 0.1
    base_data = base_data[:31]  # just take Jan for testing
    bias_data = bias_data[:31]  # just take Jan for testing
    true_scalar = base_data.std() / bias_data.std()
    true_adder = base_data.mean() - bias_data.mean() * true_scalar

    scalar, adder = calc.run(knn=1, threshold=0.6, fill_extend=True,
                             max_workers=1)

    assert len(scalar.shape) == 3
    assert len(adder.shape) == 3
    assert scalar.shape[-1] == 12
    assert adder.shape[-1] == 12

    iloc = np.where(calc.bias_gid_raster == bias_gid)
    iloc += (0, )
    assert np.allclose(true_scalar, scalar[iloc])
    assert np.allclose(true_adder, adder[iloc])

    last_mon = base_ti.month[-1]
    assert np.isnan(scalar[..., last_mon:]).all()
    assert np.isnan(adder[..., last_mon:]).all()


def test_linear_transform():
    """Test the linear bc transform method"""
    calc = LinearCorrection(FP_NSRDB, FP_CC, 'ghi', 'rsds',
                            TARGET, SHAPE, bias_handler='DataHandlerNCforCC')
    with tempfile.TemporaryDirectory() as td:
        fp_out = os.path.join(td, 'bc.h5')
        scalar, adder = calc.run(knn=1, threshold=0.6, fill_extend=False,
                                 max_workers=1, fp_out=fp_out)
        test_data = np.ones_like(scalar)
        with pytest.warns():
            out = local_linear_bc(test_data, 'rsds', fp_out,
                                  lr_padded_slice=None, out_range=None)

        scalar, adder = calc.run(knn=1, threshold=0.6, fill_extend=True,
                                 max_workers=1, fp_out=fp_out)
        test_data = np.ones_like(scalar)
        out = local_linear_bc(test_data, 'rsds', fp_out,
                              lr_padded_slice=None, out_range=None)
        assert np.allclose(out, scalar + adder)

        out_range = (0, 10)
        too_big = out > np.max(out_range)
        too_small = out < np.min(out_range)
        out_mask = too_big | too_small
        assert out_mask.any()

        out = local_linear_bc(test_data, 'rsds', fp_out,
                              lr_padded_slice=None, out_range=out_range)

        assert np.allclose(out[too_big], np.max(out_range))
        assert np.allclose(out[too_small], np.min(out_range))

        lr_slice = (slice(1, 2), slice(2, 3), slice(None))
        sliced_out = local_linear_bc(test_data[lr_slice], 'rsds', fp_out,
                                     lr_padded_slice=lr_slice,
                                     out_range=out_range)
        assert np.allclose(out[lr_slice], sliced_out)


def test_montly_linear_transform():
    """Test the montly linear bc transform method"""
    calc = MonthlyLinearCorrection(FP_NSRDB, FP_CC, 'ghi', 'rsds',
                                   TARGET, SHAPE,
                                   bias_handler='DataHandlerNCforCC')
    _, base_ti = calc.get_base_data(calc.base_fps, calc.base_dset,
                                    5, calc.base_handler,
                                    daily_avg=True)
    with tempfile.TemporaryDirectory() as td:
        fp_out = os.path.join(td, 'bc.h5')
        scalar, adder = calc.run(knn=1, threshold=0.6, fill_extend=True,
                                 max_workers=1, fp_out=fp_out)
        test_data = np.ones((scalar.shape[0], scalar.shape[1], len(base_ti)))
        with pytest.warns():
            out = monthly_local_linear_bc(test_data, 'rsds', fp_out,
                                          lr_padded_slice=None,
                                          time_index=base_ti,
                                          temporal_avg=True,
                                          out_range=None)

        im = base_ti.month - 1
        truth = scalar[..., im].mean(axis=-1) + adder[..., im].mean(axis=-1)
        truth = np.expand_dims(truth, axis=-1)
        assert np.allclose(truth, out)

        out = monthly_local_linear_bc(test_data, 'rsds', fp_out,
                                      lr_padded_slice=None,
                                      time_index=base_ti,
                                      temporal_avg=False,
                                      out_range=None)

        for i, m in enumerate(base_ti.month):
            truth = scalar[..., m - 1] + adder[..., m - 1]
            assert np.allclose(truth, out[..., i])


def test_fwp_integration():
    """Test the integration of the bias correction method into the forward pass
    framework"""
    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    features = ['U_100m', 'V_100m']
    target = (13.67, 125.0)
    shape = (8, 8)
    temporal_slice = slice(None, None, 1)
    fwp_chunk_shape = (4, 4, 150)
    input_files = [os.path.join(TEST_DATA_DIR, 'ua_test.nc'),
                   os.path.join(TEST_DATA_DIR, 'va_test.nc'),
                   os.path.join(TEST_DATA_DIR, 'orog_test.nc'),
                   os.path.join(TEST_DATA_DIR, 'zg_test.nc')]

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 10, 10, 6, len(features))))
    model.meta['training_features'] = features
    model.meta['output_features'] = features
    model.meta['s_enhance'] = 3
    model.meta['t_enhance'] = 4

    with tempfile.TemporaryDirectory() as td:
        bias_fp = os.path.join(td, 'bc.h5')
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        scalar = np.random.uniform(0.5, 1, (8, 8, 1))
        adder = np.random.uniform(0, 1, (8, 8, 1))

        with h5py.File(bias_fp, 'w') as f:
            f.create_dataset('U_100m_scalar', data=scalar)
            f.create_dataset('U_100m_adder', data=adder)
            f.create_dataset('V_100m_scalar', data=scalar)
            f.create_dataset('V_100m_adder', data=adder)

        bias_correct_kwargs = {'U_100m': {'feature_name': 'U_100m',
                                          'bias_fp': bias_fp},
                               'V_100m': {'feature_name': 'V_100m',
                                          'bias_fp': bias_fp}}

        strat = ForwardPassStrategy(
            input_files,
            model_kwargs={'model_dir': out_dir},
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=0, temporal_pad=0,
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            out_pattern=os.path.join(td, 'out_{file_id}.nc'),
            max_workers=1,
            input_handler='DataHandlerNCforCC')
        bc_strat = ForwardPassStrategy(
            input_files,
            model_kwargs={'model_dir': out_dir},
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=0, temporal_pad=0,
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            out_pattern=os.path.join(td, 'out_{file_id}.nc'),
            max_workers=1,
            input_handler='DataHandlerNCforCC',
            bias_correct_method='local_linear_bc',
            bias_correct_kwargs=bias_correct_kwargs)

        for ichunk in range(strat.chunks):

            fwp = ForwardPass(strat, chunk_index=ichunk)
            bc_fwp = ForwardPass(bc_strat, chunk_index=ichunk)

            i_scalar = np.expand_dims(scalar, axis=-1)
            i_adder = np.expand_dims(adder, axis=-1)
            i_scalar = i_scalar[bc_fwp.lr_padded_slice[0],
                                bc_fwp.lr_padded_slice[1]]
            i_adder = i_adder[bc_fwp.lr_padded_slice[0],
                              bc_fwp.lr_padded_slice[1]]
            truth = fwp.input_data * i_scalar + i_adder

            assert np.allclose(bc_fwp.input_data, truth, equal_nan=True)


def test_qa_integration():
    """Test BC integration with QA module"""
    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    features = ['U_100m', 'V_100m']
    input_files = [os.path.join(TEST_DATA_DIR, 'ua_test.nc'),
                   os.path.join(TEST_DATA_DIR, 'va_test.nc'),
                   os.path.join(TEST_DATA_DIR, 'orog_test.nc'),
                   os.path.join(TEST_DATA_DIR, 'zg_test.nc')]

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 10, 10, 6, len(features))))
    model.meta['training_features'] = features
    model.meta['output_features'] = features
    model.meta['s_enhance'] = 3
    model.meta['t_enhance'] = 4

    with tempfile.TemporaryDirectory() as td:
        bias_fp = os.path.join(td, 'bc.h5')
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        out_file_path = os.path.join(td, 'sup3r_out.h5')
        with h5py.File(out_file_path, 'w') as f:
            f.create_dataset('meta', data=np.random.uniform(0, 1, 10))

        scalar = np.random.uniform(0.5, 1, (20, 20, 1))
        adder = np.random.uniform(0, 1, (20, 20, 1))

        with h5py.File(bias_fp, 'w') as f:
            f.create_dataset('U_100m_scalar', data=scalar)
            f.create_dataset('U_100m_adder', data=adder)
            f.create_dataset('V_100m_scalar', data=scalar)
            f.create_dataset('V_100m_adder', data=adder)

        bias_correct_kwargs = {'U_100m': {'feature_name': 'U_100m',
                                          'bias_fp': bias_fp},
                               'V_100m': {'feature_name': 'V_100m',
                                          'bias_fp': bias_fp}}

        qa_kw = {'s_enhance': 3,
                 't_enhance': 4,
                 'temporal_coarsening_method': 'average',
                 'features': features,
                 'input_handler': 'DataHandlerNCforCC',
                 }

        bias_correct_kwargs = {'U_100m': {'feature_name': 'U_100m',
                                          'bias_fp': bias_fp,
                                          'lr_padded_slice': None},
                               'V_100m': {'feature_name': 'V_100m',
                                          'bias_fp': bias_fp,
                                          'lr_padded_slice': None}}

        bc_qa_kw = {'s_enhance': 3,
                    't_enhance': 4,
                    'temporal_coarsening_method': 'average',
                    'features': features,
                    'input_handler': 'DataHandlerNCforCC',
                    'bias_correct_method': 'local_linear_bc',
                    'bias_correct_kwargs': bias_correct_kwargs,
                    }

        for idf, feature in enumerate(features):
            with Sup3rQa(input_files, out_file_path, **qa_kw) as qa:
                data_base = qa.get_source_dset(idf, feature)
                data_truth = data_base * scalar + adder
            with Sup3rQa(input_files, out_file_path, **bc_qa_kw) as qa:
                data_bc = qa.get_source_dset(idf, feature)

            assert np.allclose(data_bc, data_truth, equal_nan=True)
