# -*- coding: utf-8 -*-
"""pytests bias correction calculations"""
import os
import shutil
import tempfile

import h5py
import numpy as np
import pytest
import xarray as xr
from scipy import stats

from sup3r import CONFIG_DIR, TEST_DATA_DIR
from sup3r.bias.bias_calc import (
    LinearCorrection,
    MonthlyLinearCorrection,
    SkillAssessment,
)
from sup3r.bias.bias_transforms import local_linear_bc, monthly_local_linear_bc
from sup3r.models import Sup3rGan
from sup3r.pipeline.forward_pass import ForwardPass, ForwardPassStrategy
from sup3r.preprocessing.data_handling import DataHandlerNCforCC
from sup3r.qa.qa import Sup3rQa

FP_NSRDB = os.path.join(TEST_DATA_DIR, 'test_nsrdb_co_2018.h5')
FP_CC = os.path.join(TEST_DATA_DIR, 'rsds_test.nc')

with xr.open_dataset(FP_CC) as fh:
    MIN_LAT = np.min(fh.lat.values.astype(np.float32))
    MIN_LON = np.min(fh.lon.values.astype(np.float32)) - 360
    TARGET = (float(MIN_LAT), float(MIN_LON))
    SHAPE = (len(fh.lat.values), len(fh.lon.values))


def test_smooth_interior_bc():
    """Test linear bias correction with interior smoothing"""

    calc = LinearCorrection(FP_NSRDB, FP_CC, 'ghi', 'rsds',
                            target=TARGET, shape=SHAPE,
                            distance_upper_bound=0.7,
                            bias_handler='DataHandlerNCforCC')
    out = calc.run(fill_extend=False, max_workers=1)
    og_scalar = out['rsds_scalar']
    og_adder = out['rsds_adder']
    nan_mask = np.isnan(og_scalar)
    assert np.isnan(og_adder[nan_mask]).all()

    calc = LinearCorrection(FP_NSRDB, FP_CC, 'ghi', 'rsds',
                            target=TARGET, shape=SHAPE,
                            distance_upper_bound=0.7,
                            bias_handler='DataHandlerNCforCC')
    out = calc.run(fill_extend=True, smooth_interior=0, max_workers=1)
    scalar = out['rsds_scalar']
    adder = out['rsds_adder']
    # Make sure smooth_interior=0 does not change interior pixels
    assert np.allclose(og_scalar[~nan_mask], scalar[~nan_mask])
    assert np.allclose(og_adder[~nan_mask], adder[~nan_mask])
    assert not np.isnan(adder[nan_mask]).any()
    assert not np.isnan(scalar[nan_mask]).any()

    # make sure smoothing affects the interior pixels but not the exterior
    calc = LinearCorrection(FP_NSRDB, FP_CC, 'ghi', 'rsds',
                            target=TARGET, shape=SHAPE,
                            distance_upper_bound=0.7,
                            bias_handler='DataHandlerNCforCC')
    out = calc.run(fill_extend=True, smooth_interior=1, max_workers=1)
    smooth_scalar = out['rsds_scalar']
    smooth_adder = out['rsds_adder']

    assert not np.allclose(smooth_scalar[~nan_mask], scalar[~nan_mask])
    assert not np.allclose(smooth_adder[~nan_mask], adder[~nan_mask])
    assert np.allclose(smooth_scalar[nan_mask], scalar[nan_mask])
    assert np.allclose(smooth_adder[nan_mask], adder[nan_mask])


def test_linear_bc():
    """Test linear bias correction"""

    calc = LinearCorrection(FP_NSRDB, FP_CC, 'ghi', 'rsds',
                            target=TARGET, shape=SHAPE,
                            distance_upper_bound=0.7,
                            bias_handler='DataHandlerNCforCC')

    # test a known in-bounds gid
    bias_gid = 5
    dist, base_gid = calc.get_base_gid(bias_gid)
    bias_data = calc.get_bias_data(bias_gid)
    base_data, _ = calc.get_base_data(calc.base_fps, calc.base_dset,
                                      base_gid, calc.base_handler,
                                      daily_reduction='avg')
    bias_coord = calc.bias_meta.loc[[bias_gid], ['latitude', 'longitude']]
    base_coord = calc.base_meta.loc[base_gid, ['latitude', 'longitude']]
    true_dist = bias_coord.values - base_coord.values
    true_dist = np.hypot(true_dist[:, 0], true_dist[:, 1])
    assert np.allclose(true_dist, dist)
    assert (true_dist < 0.5).all()  # horiz res of bias data is ~0.7 deg
    true_scalar = base_data.std() / bias_data.std()
    true_adder = base_data.mean() - bias_data.mean() * true_scalar

    out = calc.run(fill_extend=False, max_workers=1)
    scalar = out['rsds_scalar']
    adder = out['rsds_adder']

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
    assert len(calc.bad_bias_gids) > 0

    # make sure the NN fill works for out-of-bounds pixels
    calc = LinearCorrection(FP_NSRDB, FP_CC, 'ghi', 'rsds',
                            target=TARGET, shape=SHAPE,
                            distance_upper_bound=0.7,
                            bias_handler='DataHandlerNCforCC')
    out = calc.run(fill_extend=True, max_workers=1)
    scalar = out['rsds_scalar']
    adder = out['rsds_adder']

    iloc = np.where(calc.bias_gid_raster == bias_gid)
    assert np.allclose(true_scalar, scalar[iloc])
    assert np.allclose(true_adder, adder[iloc])

    assert len(calc.bad_bias_gids) > 0
    assert not np.isnan(scalar[nan_mask]).any()
    assert not np.isnan(adder[nan_mask]).any()

    # make sure smoothing affects the out-of-bounds pixels but not the in-bound
    calc = LinearCorrection(FP_NSRDB, FP_CC, 'ghi', 'rsds',
                            target=TARGET, shape=SHAPE,
                            distance_upper_bound=0.7,
                            bias_handler='DataHandlerNCforCC')
    out = calc.run(fill_extend=True, smooth_extend=2, max_workers=1)
    smooth_scalar = out['rsds_scalar']
    smooth_adder = out['rsds_adder']
    assert np.allclose(smooth_scalar[~nan_mask], scalar[~nan_mask])
    assert np.allclose(smooth_adder[~nan_mask], adder[~nan_mask])
    assert not np.allclose(smooth_scalar[nan_mask], scalar[nan_mask])
    assert not np.allclose(smooth_adder[nan_mask], adder[nan_mask])

    # parallel test
    calc = LinearCorrection(FP_NSRDB, FP_CC, 'ghi', 'rsds',
                            target=TARGET, shape=SHAPE,
                            distance_upper_bound=0.7,
                            bias_handler='DataHandlerNCforCC')
    out = calc.run(fill_extend=True, smooth_extend=2, max_workers=2)
    par_scalar = out['rsds_scalar']
    par_adder = out['rsds_adder']
    assert np.allclose(smooth_scalar, par_scalar)
    assert np.allclose(smooth_adder, par_adder)


def test_monthly_linear_bc():
    """Test linear bias correction on a month-by-month basis"""

    calc = MonthlyLinearCorrection(FP_NSRDB, FP_CC, 'ghi', 'rsds',
                                   target=TARGET, shape=SHAPE,
                                   distance_upper_bound=0.7,
                                   bias_handler='DataHandlerNCforCC')

    # test a known in-bounds gid
    bias_gid = 5
    dist, base_gid = calc.get_base_gid(bias_gid)
    bias_data = calc.get_bias_data(bias_gid)
    base_data, base_ti = calc.get_base_data(calc.base_fps, calc.base_dset,
                                            base_gid, calc.base_handler,
                                            daily_reduction='avg')
    bias_coord = calc.bias_meta.loc[[bias_gid], ['latitude', 'longitude']]
    base_coord = calc.base_meta.loc[base_gid, ['latitude', 'longitude']]
    true_dist = bias_coord.values - base_coord.values
    true_dist = np.hypot(true_dist[:, 0], true_dist[:, 1])
    assert np.allclose(true_dist, dist)
    assert (true_dist < 0.5).all()  # horiz res of bias data is ~0.7 deg
    base_data = base_data[:31]  # just take Jan for testing
    bias_data = bias_data[:31]  # just take Jan for testing
    true_scalar = base_data.std() / bias_data.std()
    true_adder = base_data.mean() - bias_data.mean() * true_scalar

    out = calc.run(fill_extend=True, max_workers=1)
    scalar = out['rsds_scalar']
    adder = out['rsds_adder']

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
                            target=TARGET, shape=SHAPE,
                            distance_upper_bound=0.7,
                            bias_handler='DataHandlerNCforCC')
    lat_lon = calc.bias_dh.lat_lon
    with tempfile.TemporaryDirectory() as td:
        fp_out = os.path.join(td, 'bc.h5')
        out = calc.run(fill_extend=False, max_workers=1, fp_out=fp_out)
        scalar = out['rsds_scalar']
        adder = out['rsds_adder']
        test_data = np.ones_like(scalar)
        with pytest.warns():
            out = local_linear_bc(test_data, lat_lon, 'rsds', fp_out,
                                  lr_padded_slice=None, out_range=None)

        out = calc.run(fill_extend=True, max_workers=1, fp_out=fp_out)
        scalar = out['rsds_scalar']
        adder = out['rsds_adder']
        test_data = np.ones_like(scalar)
        out = local_linear_bc(test_data, lat_lon, 'rsds', fp_out,
                              lr_padded_slice=None, out_range=None)
        assert np.allclose(out, scalar + adder)

        out_range = (0, 10)
        too_big = out > np.max(out_range)
        too_small = out < np.min(out_range)
        out_mask = too_big | too_small
        assert out_mask.any()

        out = local_linear_bc(test_data, lat_lon, 'rsds', fp_out,
                              lr_padded_slice=None, out_range=out_range)

        assert np.allclose(out[too_big], np.max(out_range))
        assert np.allclose(out[too_small], np.min(out_range))

        lr_slice = (slice(1, 2), slice(2, 3), slice(None))
        sliced_out = local_linear_bc(test_data[lr_slice], lat_lon[lr_slice],
                                     'rsds', fp_out, lr_padded_slice=lr_slice,
                                     out_range=out_range)
        assert np.allclose(out[lr_slice], sliced_out)


def test_montly_linear_transform():
    """Test the montly linear bc transform method"""
    calc = MonthlyLinearCorrection(FP_NSRDB, FP_CC, 'ghi', 'rsds',
                                   target=TARGET, shape=SHAPE,
                                   distance_upper_bound=0.7,
                                   bias_handler='DataHandlerNCforCC')
    lat_lon = calc.bias_dh.lat_lon
    _, base_ti = calc.get_base_data(calc.base_fps, calc.base_dset,
                                    5, calc.base_handler,
                                    daily_reduction='avg')
    with tempfile.TemporaryDirectory() as td:
        fp_out = os.path.join(td, 'bc.h5')
        out = calc.run(fill_extend=True, max_workers=1, fp_out=fp_out)
        scalar = out['rsds_scalar']
        adder = out['rsds_adder']
        test_data = np.ones((scalar.shape[0], scalar.shape[1], len(base_ti)))
        with pytest.warns():
            out = monthly_local_linear_bc(test_data, lat_lon, 'rsds', fp_out,
                                          lr_padded_slice=None,
                                          time_index=base_ti,
                                          temporal_avg=True,
                                          out_range=None)

        im = base_ti.month - 1
        truth = scalar[..., im].mean(axis=-1) + adder[..., im].mean(axis=-1)
        truth = np.expand_dims(truth, axis=-1)
        assert np.allclose(truth, out)

        out = monthly_local_linear_bc(test_data, lat_lon, 'rsds', fp_out,
                                      lr_padded_slice=None,
                                      time_index=base_ti,
                                      temporal_avg=False,
                                      out_range=None)

        for i, m in enumerate(base_ti.month):
            truth = scalar[..., m - 1] + adder[..., m - 1]
            assert np.allclose(truth, out[..., i])


def test_clearsky_ratio():
    """Test that bias correction of daily clearsky ratio instead of raw ghi
    works."""
    bias_handler_kwargs = {'nsrdb_source_fp': FP_NSRDB, 'nsrdb_agg': 4,
                           'temporal_slice': [0, 30, 1]}
    calc = LinearCorrection(FP_NSRDB, FP_CC,
                            'clearsky_ratio', 'clearsky_ratio',
                            target=TARGET, shape=SHAPE,
                            distance_upper_bound=0.7,
                            bias_handler_kwargs=bias_handler_kwargs,
                            bias_handler='DataHandlerNCforCC')
    out = calc.run(fill_extend=True, max_workers=1)

    assert not np.isnan(out['clearsky_ratio_scalar']).any()
    assert not np.isnan(out['clearsky_ratio_adder']).any()

    base_cs = out['base_clearsky_ratio_mean']
    bias_cs = out['bias_clearsky_ratio_mean']

    assert (base_cs > 0.3).all()
    assert (base_cs < 1.0).all()

    assert (base_cs > 0.3).all()
    assert (bias_cs < 1.0).all()


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

    lat_lon = DataHandlerNCforCC(input_files, features=[], target=target,
                                 shape=shape,
                                 worker_kwargs={'max_workers': 1}).lat_lon

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 10, 10, 6, len(features))))
    model.meta['lr_features'] = features
    model.meta['hr_out_features'] = features
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
            f.create_dataset('latitude', data=lat_lon[..., 0])
            f.create_dataset('longitude', data=lat_lon[..., 1])

        bias_correct_kwargs = {'U_100m': {'feature_name': 'U_100m',
                                          'bias_fp': bias_fp},
                               'V_100m': {'feature_name': 'V_100m',
                                          'bias_fp': bias_fp}}

        strat = ForwardPassStrategy(
            input_files,
            model_kwargs={'model_dir': out_dir},
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=0, temporal_pad=0,
            input_handler_kwargs=dict(target=target, shape=shape,
                                      temporal_slice=temporal_slice,
                                      worker_kwargs=dict(max_workers=1)),
            out_pattern=os.path.join(td, 'out_{file_id}.nc'),
            worker_kwargs=dict(max_workers=1),
            input_handler='DataHandlerNCforCC')
        bc_strat = ForwardPassStrategy(
            input_files,
            model_kwargs={'model_dir': out_dir},
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=0, temporal_pad=0,
            input_handler_kwargs=dict(target=target, shape=shape,
                                      temporal_slice=temporal_slice,
                                      worker_kwargs=dict(max_workers=1)),
            out_pattern=os.path.join(td, 'out_{file_id}.nc'),
            worker_kwargs=dict(max_workers=1),
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
    features = ['U_100m', 'V_100m']
    input_files = [os.path.join(TEST_DATA_DIR, 'ua_test.nc'),
                   os.path.join(TEST_DATA_DIR, 'va_test.nc'),
                   os.path.join(TEST_DATA_DIR, 'orog_test.nc'),
                   os.path.join(TEST_DATA_DIR, 'zg_test.nc')]

    lat_lon = DataHandlerNCforCC(input_files, features=[]).lat_lon

    with tempfile.TemporaryDirectory() as td:
        bias_fp = os.path.join(td, 'bc.h5')

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
            f.create_dataset('latitude', data=lat_lon[..., 0])
            f.create_dataset('longitude', data=lat_lon[..., 1])

        qa_kw = {'s_enhance': 3,
                 't_enhance': 4,
                 'temporal_coarsening_method': 'average',
                 'features': features,
                 'input_handler': 'DataHandlerNCforCC',
                 'worker_kwargs': {'max_workers': 1},
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
                    'worker_kwargs': {'max_workers': 1},
                    }

        for feature in features:
            with Sup3rQa(input_files, out_file_path, **qa_kw) as qa:
                data_base = qa.get_source_dset(feature, feature)
                data_truth = data_base * scalar + adder
            with Sup3rQa(input_files, out_file_path, **bc_qa_kw) as qa:
                data_bc = qa.get_source_dset(feature, feature)

            assert np.allclose(data_bc, data_truth, equal_nan=True)


def test_skill_assessment():
    """Test the skill assessment of a climate model vs. historical data"""
    calc = SkillAssessment(FP_NSRDB, FP_CC, 'ghi', 'rsds',
                           target=TARGET, shape=SHAPE,
                           distance_upper_bound=0.7,
                           bias_handler='DataHandlerNCforCC')

    # test a known in-bounds gid
    bias_gid = 5
    dist, base_gid = calc.get_base_gid(bias_gid)
    bias_data = calc.get_bias_data(bias_gid)
    base_data, _ = calc.get_base_data(calc.base_fps, calc.base_dset,
                                      base_gid, calc.base_handler,
                                      daily_reduction='avg')
    bias_coord = calc.bias_meta.loc[[bias_gid], ['latitude', 'longitude']]
    base_coord = calc.base_meta.loc[base_gid, ['latitude', 'longitude']]
    true_dist = bias_coord.values - base_coord.values
    true_dist = np.hypot(true_dist[:, 0], true_dist[:, 1])
    assert np.allclose(true_dist, dist)
    assert (true_dist < 0.5).all()  # horiz res of bias data is ~0.7 deg
    iloc = np.where(calc.bias_gid_raster == bias_gid)
    iloc += (0, )

    out = calc.run(fill_extend=True, max_workers=1)

    base_mean = base_data.mean()
    bias_mean = bias_data.mean()
    assert np.allclose(out['base_ghi_mean'][iloc], base_mean)
    assert np.allclose(out['bias_rsds_mean'][iloc], bias_mean)
    assert np.allclose(out['rsds_bias'][iloc], bias_mean - base_mean)

    ks = stats.ks_2samp(base_data - base_mean, bias_data - bias_mean)
    assert np.allclose(out['rsds_ks_stat'][iloc], ks.statistic)
    assert np.allclose(out['rsds_ks_p'][iloc], ks.pvalue)


def test_nc_base_file():
    """Test a base file being a .nc like ERA5"""
    calc = SkillAssessment(FP_CC, FP_CC, 'rsds', 'rsds',
                           target=TARGET, shape=SHAPE,
                           distance_upper_bound=0.7,
                           base_handler='DataHandlerNCforCC',
                           bias_handler='DataHandlerNCforCC')

    # test a known in-bounds gid
    bias_gid = 5
    dist, base_gid = calc.get_base_gid(bias_gid)
    assert dist == 0
    assert (calc.nn_dist == 0).all()

    with pytest.raises(RuntimeError) as exc:
        calc.get_base_data(calc.base_fps, calc.base_dset, base_gid,
                           calc.base_handler, daily_reduction='avg')

    good_err = 'only to be used with `base_handler` as a `sup3r.DataHandler` '
    assert good_err in str(exc.value)

    # make sure this doesnt raise error now that calc.base_dh is provided
    calc.get_base_data(calc.base_fps, calc.base_dset,
                       base_gid, calc.base_handler,
                       daily_reduction='avg',
                       base_dh_inst=calc.base_dh)

    out = calc.run(fill_extend=True, max_workers=1)

    assert np.allclose(out['base_rsds_mean_monthly'],
                       out['bias_rsds_mean_monthly'])
    assert np.allclose(out['base_rsds_mean'], out['bias_rsds_mean'])
    assert np.allclose(out['base_rsds_std'], out['bias_rsds_std'])


def test_match_zero_rate():
    """Test feature to match the rate of zeros in the bias data based on the
    zero rate in the base data. Useful for precip where GCMs have a low-precip
    "drizzle" problem."""
    bias_data = np.random.uniform(0, 1, 1000)
    base_data = np.random.uniform(0, 1, 1000)
    base_data[base_data < 0.1] = 0

    skill = SkillAssessment._run_skill_eval(bias_data, base_data, 'f1', 'f1')
    assert skill['bias_f1_zero_rate'] != skill['base_f1_zero_rate']
    assert (bias_data == 0).mean() != (base_data == 0).mean()

    skill = SkillAssessment._run_skill_eval(bias_data, base_data, 'f1', 'f1',
                                            match_zero_rate=True)
    assert (bias_data == 0).mean() == (base_data == 0).mean()
    assert skill['bias_f1_zero_rate'] == skill['base_f1_zero_rate']
    for p in (1, 5, 25, 50, 75, 95, 99):
        assert np.allclose(skill[f'base_f1_percentile_{p}'],
                           np.percentile(base_data, p))

    with tempfile.TemporaryDirectory() as td:
        fp_nsrdb_temp = os.path.join(td, os.path.basename(FP_NSRDB))
        shutil.copy(FP_NSRDB, fp_nsrdb_temp)
        with h5py.File(fp_nsrdb_temp, 'a') as nsrdb_temp:
            ghi = nsrdb_temp['ghi'][...]
            ghi[:1000, :] = 0
            nsrdb_temp['ghi'][...] = ghi
        calc = SkillAssessment(fp_nsrdb_temp, FP_CC, 'ghi', 'rsds',
                               target=TARGET, shape=SHAPE,
                               distance_upper_bound=0.7,
                               bias_handler='DataHandlerNCforCC',
                               match_zero_rate=True)
        out = calc.run(fill_extend=True, max_workers=1)

    bias_rate = out['bias_rsds_zero_rate']
    base_rate = out['base_ghi_zero_rate']
    assert np.allclose(bias_rate, base_rate, rtol=0.005)
