"""pytests QDM bias correction calculations"""

import os
import shutil

import h5py
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sup3r import CONFIG_DIR, TEST_DATA_DIR
from sup3r.models import Sup3rGan
from sup3r.pipeline.forward_pass import ForwardPass, ForwardPassStrategy
from sup3r.bias import (
    local_qdm_bc,
    QuantileDeltaMappingCorrection,
)
from sup3r.preprocessing.data_handling import DataHandlerNC, DataHandlerNCforCC

FP_NSRDB = os.path.join(TEST_DATA_DIR, 'test_nsrdb_co_2018.h5')
FP_CC = os.path.join(TEST_DATA_DIR, 'rsds_test.nc')
FP_CC_LAT_LON = DataHandlerNC(FP_CC, 'rsds').lat_lon

with xr.open_dataset(FP_CC) as fh:
    MIN_LAT = np.min(fh.lat.values.astype(np.float32))
    MIN_LON = np.min(fh.lon.values.astype(np.float32)) - 360
    TARGET = (float(MIN_LAT), float(MIN_LON))
    SHAPE = (len(fh.lat.values), len(fh.lon.values))


@pytest.fixture(scope='module')
def fp_fut_cc(tmpdir_factory):
    """Sample future CC dataset

    The same CC but with an offset (75.0) and negligible noise.
    """
    fn = tmpdir_factory.mktemp('data').join('test_mf.nc')
    ds = xr.open_dataset(FP_CC)
    # Adding an offset
    ds['rsds'] += 75.0
    # adding a noise
    ds['rsds'] += np.random.randn(*ds['rsds'].shape)
    ds.to_netcdf(fn)
    # DataHandlerNCforCC requires a string
    fn = str(fn)
    return fn


@pytest.fixture(scope='module')
def fp_fut_cc_notrend(tmpdir_factory):
    """Sample future CC dataset identical to historical CC

    This is currently a copy of FP_CC, thus no trend on time.
    """
    fn = tmpdir_factory.mktemp('data').join('test_mf_notrend.nc')
    shutil.copyfile(FP_CC, fn)
    # DataHandlerNCforCC requires a string
    fn = str(fn)
    return fn


def test_window_mask():
    """A basic window mask check

    This is a day and a half in each direction, thus one daily point
    on each direction
    """
    d = np.arange(1, 366)
    idx = QuantileDeltaMappingCorrection.window_mask(d, 60, 3)
    assert np.allclose([59, 60, 61], d[idx])


def test_window_mask_even_window_size():
    """An even number rounds down if there isn't resolution in the input"""
    d = np.arange(1, 366)
    idx = QuantileDeltaMappingCorrection.window_mask(d, 60, 4)
    assert np.allclose([59, 60, 61], d[idx])


def test_window_mask_start_of_year():
    """Early in the year, the window rolls over to the end"""
    d = np.arange(1, 366)
    idx = QuantileDeltaMappingCorrection.window_mask(d, 1, 3)
    assert np.allclose([1, 2, 365], d[idx])


def test_window_mask_end_of_year():
    """Early in the year, the window rolls over to the end"""
    d = np.arange(1, 366)
    idx = QuantileDeltaMappingCorrection.window_mask(d, 365, 3)
    assert np.allclose([1, 364, 365], d[idx])


@pytest.fixture(scope='module')
def dist_params(tmpdir_factory, fp_fut_cc):
    """Distribution parameters for standard datasets

    Use the standard datasets to estimate the distributions and save
    in a temporary place to be re-used
    """
    calc = QuantileDeltaMappingCorrection(FP_NSRDB,
                                          FP_CC,
                                          fp_fut_cc,
                                          'ghi',
                                          'rsds',
                                          target=TARGET,
                                          shape=SHAPE,
                                          distance_upper_bound=0.7,
                                          bias_handler='DataHandlerNCforCC',
                                          )
    fn = tmpdir_factory.mktemp('params').join('standard.h5')
    _ = calc.run(fp_out=fn)

    # DataHandlerNCforCC requires a string
    fn = str(fn)

    return fn


def test_qdm_bc(fp_fut_cc):
    """Test QDM bias correction

    Basic standard run. Using only required arguments. If this fails,
    something fundamental is wrong.
    """

    calc = QuantileDeltaMappingCorrection(FP_NSRDB,
                                          FP_CC,
                                          fp_fut_cc,
                                          'ghi',
                                          'rsds',
                                          target=TARGET,
                                          shape=SHAPE,
                                          bias_handler='DataHandlerNCforCC',
                                          )

    out = calc.run()

    # Guarantee that we have some actual values, otherwise most of the
    # remaining tests would be useless
    for v in out:
        assert np.isfinite(out[v]).any(), 'Something wrong, all CDFs are NaN.'

    # Check possible range
    for v in out:
        assert np.nanmin(out[v]) > 0, f'{v} should be all greater than zero.'
        assert np.nanmax(out[v]) < 1300, f'{v} should be all less than 1300.'

    # Each location can be all finite or all NaN, but not both
    for v in out:
        tmp = np.isfinite(out[v].reshape(-1, out[v].shape[-1]))
        assert np.all(
            np.all(tmp, axis=1) == ~np.all(~tmp, axis=1)
        ), f'For each location of {v} it should be all finite or nonte'


def test_parallel(fp_fut_cc):
    """Compare bias correction run serial vs in parallel

    Both modes should give the exact same results.
    """

    s = QuantileDeltaMappingCorrection(FP_NSRDB, FP_CC, fp_fut_cc,
                                       'ghi', 'rsds',
                                       target=TARGET, shape=SHAPE,
                                       distance_upper_bound=0.7,
                                       bias_handler='DataHandlerNCforCC')
    out_s = s.run(max_workers=1)

    p = QuantileDeltaMappingCorrection(FP_NSRDB, FP_CC, fp_fut_cc,
                                       'ghi', 'rsds',
                                       target=TARGET, shape=SHAPE,
                                       distance_upper_bound=0.7,
                                       bias_handler='DataHandlerNCforCC')
    out_p = p.run(max_workers=2)

    for k in out_s.keys():
        assert k in out_p, f'Missing {k} in parallel run'
        assert np.allclose(
            out_s[k], out_p[k], equal_nan=True
        ), f'Different results for {k}'


def test_fill_nan(fp_fut_cc):
    """No NaN when running with fill_extend"""

    c = QuantileDeltaMappingCorrection(FP_NSRDB, FP_CC, fp_fut_cc,
                                       'ghi', 'rsds',
                                       target=TARGET, shape=SHAPE,
                                       distance_upper_bound=0.7,
                                       bias_handler='DataHandlerNCforCC')

    # Without filling, at least one NaN or this test is useless.
    out = c.run(fill_extend=False)
    # Ignore non `params` parameters, such as window_center
    params = (v for v in out.keys() if v.endswith('params'))
    assert np.all(
        [np.isnan(out[v]).any() for v in params]
    ), 'Assume at least one NaN value for each param'

    out = c.run()
    assert np.all(
        [np.isfinite(v).all() for v in out.values()]
    ), 'All NaN values where supposed to be filled'


def test_save_file(tmp_path, fp_fut_cc):
    """Save valid output

    Confirm it saves the output by creating a valid HDF5 file.
    """

    calc = QuantileDeltaMappingCorrection(FP_NSRDB, FP_CC, fp_fut_cc,
                                          'ghi', 'rsds',
                                          target=TARGET, shape=SHAPE,
                                          distance_upper_bound=0.7,
                                          bias_handler='DataHandlerNCforCC')

    filename = os.path.join(tmp_path, 'test_saving.hdf')
    _ = calc.run(filename)

    # File was created
    os.path.isfile(filename)
    # A valid HDF5, can open and read
    with h5py.File(filename, 'r') as f:
        assert 'latitude' in f.keys()


def test_qdm_transform(dist_params):
    """
    WIP: Confirm it runs, but don't verify anything yet.
    """
    data = np.ones((*FP_CC_LAT_LON.shape[:-1], 2))
    time = pd.DatetimeIndex(
        (np.datetime64('2018-01-01'), np.datetime64('2018-01-02'))
    )
    corrected = local_qdm_bc(
        data, FP_CC_LAT_LON, 'ghi', 'rsds', dist_params, time,
    )

    assert not np.isnan(corrected).all(), "Can't compare if only NaN"
    assert not np.allclose(data, corrected, equal_nan=False)


def test_qdm_transform_notrend(tmp_path, dist_params):
    """The no_trend option is equal to a dataset without trend

    The no_trend flag ignores the trend component, thus it must give the
    same result of a full correction based on data distributions that
    modeled historical is equal to modeled future.

    Note: One possible point of confusion here is that the mf is ignored,
    so it is assumed that mo is the distribution to be representative of the
    target data.
    """
    time = pd.DatetimeIndex(
        (np.datetime64('2018-01-01'), np.datetime64('2018-01-02'))
    )
    # Run the standard pipeline with flag 'no_trend'
    corrected = local_qdm_bc(
        np.ones((*FP_CC_LAT_LON.shape[:-1], 2)),
        FP_CC_LAT_LON,
        'ghi',
        'rsds',
        dist_params,
        time,
        no_trend=True,
    )

    # Creates a new distribution with mo == mf
    notrend_params = os.path.join(tmp_path, 'notrend.hdf')
    shutil.copyfile(dist_params, notrend_params)
    with h5py.File(notrend_params, 'r+') as f:
        f['bias_fut_rsds_params'][:] = f['bias_rsds_params'][:]
        f.flush()

    unbiased = local_qdm_bc(
        np.ones((*FP_CC_LAT_LON.shape[:-1], 2)),
        FP_CC_LAT_LON,
        'ghi',
        'rsds',
        notrend_params,
        time,
    )

    assert not np.isnan(corrected).all(), "Can't compare if only NaN"
    assert np.allclose(corrected, unbiased, equal_nan=True)


def test_handler_qdm_bc(fp_fut_cc, dist_params):
    """qdm_bc() method from DataHandler

    WIP: Confirm it runs, but don't verify much yet.
    """
    Handler = DataHandlerNC(fp_fut_cc, 'rsds')
    original = Handler.data.copy()
    Handler.qdm_bc(dist_params, 'ghi')
    corrected = Handler.data

    assert not np.isnan(corrected).all(), "Can't compare if only NaN"

    idx = ~(np.isnan(original) | np.isnan(corrected))
    # Where it is not NaN, it must have differences.
    assert not np.allclose(original[idx], corrected[idx])


def test_bc_identity(tmp_path, fp_fut_cc, dist_params):
    """No (relative) changes if distributions are identical

    If the three distributions are identical, the QDM shouldn't change
    anything. Note that NaNs in any component, i.e. any dataset, would
    propagate into a NaN transformation.
    """
    ident_params = os.path.join(tmp_path, 'identity.hdf')
    shutil.copyfile(dist_params, ident_params)
    with h5py.File(ident_params, 'r+') as f:
        f['base_ghi_params'][:] = f['bias_fut_rsds_params'][:]
        f['bias_rsds_params'][:] = f['bias_fut_rsds_params'][:]
        f.flush()
    Handler = DataHandlerNC(fp_fut_cc, 'rsds')
    original = Handler.data.copy()
    Handler.qdm_bc(ident_params, 'ghi', relative=True)
    corrected = Handler.data

    assert not np.isnan(corrected).all(), "Can't compare if only NaN"

    idx = ~(np.isnan(original) | np.isnan(corrected))
    assert np.allclose(original[idx], corrected[idx])


def test_bc_identity_absolute(tmp_path, fp_fut_cc, dist_params):
    """No (absolute) changes if distributions are identical

    If the three distributions are identical, the QDM shouldn't change
    anything. Note that NaNs in any component, i.e. any dataset, would
    propagate into a NaN transformation.
    """
    ident_params = os.path.join(tmp_path, 'identity.hdf')
    shutil.copyfile(dist_params, ident_params)
    with h5py.File(ident_params, 'r+') as f:
        f['base_ghi_params'][:] = f['bias_fut_rsds_params'][:]
        f['bias_rsds_params'][:] = f['bias_fut_rsds_params'][:]
        f.flush()
    Handler = DataHandlerNC(fp_fut_cc, 'rsds')
    original = Handler.data.copy()
    Handler.qdm_bc(ident_params, 'ghi', relative=False)
    corrected = Handler.data

    assert not np.isnan(corrected).all(), "Can't compare if only NaN"

    idx = ~(np.isnan(original) | np.isnan(corrected))
    assert np.allclose(original[idx], corrected[idx])


def test_bc_model_constant(tmp_path, fp_fut_cc, dist_params):
    """A constant model but different than reference

    If model is constant, there is no trend. If historical biased
    has an offset with historical observed, that same offset should
    be corrected in the target (future modeled).
    """
    offset_params = os.path.join(tmp_path, 'offset.hdf')
    shutil.copyfile(dist_params, offset_params)
    with h5py.File(offset_params, 'r+') as f:
        f['base_ghi_params'][:] = f['bias_fut_rsds_params'][:] - 10
        f['bias_rsds_params'][:] = f['bias_fut_rsds_params'][:]
        f.flush()
    Handler = DataHandlerNC(fp_fut_cc, 'rsds')
    original = Handler.data.copy()
    Handler.qdm_bc(offset_params, 'ghi', relative=False)
    corrected = Handler.data

    assert not np.isnan(corrected).all(), "Can't compare if only NaN"

    idx = ~(np.isnan(original) | np.isnan(corrected))
    assert np.allclose(corrected[idx] - original[idx], -10)


def test_bc_trend(tmp_path, fp_fut_cc, dist_params):
    """A trend should propagate

    Even if modeled future is equal to observed historical, if there
    is a trend between modeled historical vs future, that same trend
    should be applied to correct
    """
    offset_params = os.path.join(tmp_path, 'offset.hdf')
    shutil.copyfile(dist_params, offset_params)
    with h5py.File(offset_params, 'r+') as f:
        f['base_ghi_params'][:] = f['bias_fut_rsds_params'][:]
        f['bias_rsds_params'][:] = f['bias_fut_rsds_params'][:] - 10
        f.flush()
    Handler = DataHandlerNC(fp_fut_cc, 'rsds')
    original = Handler.data.copy()
    Handler.qdm_bc(offset_params, 'ghi', relative=False)
    corrected = Handler.data

    assert not np.isnan(corrected).all(), "Can't compare if only NaN"

    idx = ~(np.isnan(original) | np.isnan(corrected))
    assert np.allclose(corrected[idx] - original[idx], 10)


def test_bc_trend_same_hist(tmp_path, fp_fut_cc, dist_params):
    """A trend should propagate

    If there was no bias in historical (obs vs mod), there is nothing to
    correct, but trust the forecast.
    """
    offset_params = os.path.join(tmp_path, 'offset.hdf')
    shutil.copyfile(dist_params, offset_params)
    with h5py.File(offset_params, 'r+') as f:
        f['base_ghi_params'][:] = f['bias_fut_rsds_params'][:] - 10
        f['bias_rsds_params'][:] = f['bias_fut_rsds_params'][:] - 10
        f.flush()
    Handler = DataHandlerNC(fp_fut_cc, 'rsds')
    original = Handler.data.copy()
    Handler.qdm_bc(offset_params, 'ghi', relative=False)
    corrected = Handler.data

    assert not np.isnan(corrected).all(), "Can't compare if only NaN"

    idx = ~(np.isnan(original) | np.isnan(corrected))
    assert np.allclose(corrected[idx], original[idx])


def test_fwp_integration(tmp_path):
    """Integration of the bias correction method into the forward pass

    Validate two aspects:
    - We should be able to run a forward pass with unbiased data.
    - The bias trend should be observed in the predicted output.
    """
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
                   os.path.join(TEST_DATA_DIR, 'zg_test.nc'),
                   ]

    n_samples = 101
    quantiles = np.linspace(0, 1, n_samples)
    params = {}
    with xr.open_dataset(os.path.join(TEST_DATA_DIR, 'ua_test.nc')) as ds:
        params['bias_U_100m_params'] = (
            np.ones(12)[:, np.newaxis]
            * ds['ua'].quantile(quantiles).to_numpy()
        )
    params['base_Uref_100m_params'] = params['bias_U_100m_params'] - 2.72
    params['bias_fut_U_100m_params'] = params['bias_U_100m_params']
    with xr.open_dataset(os.path.join(TEST_DATA_DIR, 'va_test.nc')) as ds:
        params['bias_V_100m_params'] = (
            np.ones(12)[:, np.newaxis]
            * ds['va'].quantile(quantiles).to_numpy()
        )
    params['base_Vref_100m_params'] = params['bias_V_100m_params'] + 2.72
    params['bias_fut_V_100m_params'] = params['bias_V_100m_params']

    lat_lon = DataHandlerNCforCC(
        input_files,
        features=[],
        target=target,
        shape=shape,
        worker_kwargs={'max_workers': 1},
    ).lat_lon

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 10, 10, 6, len(features))))
    model.meta['lr_features'] = features
    model.meta['hr_out_features'] = features
    model.meta['s_enhance'] = 3
    model.meta['t_enhance'] = 4

    bias_fp = os.path.join(tmp_path, 'bc.h5')
    out_dir = os.path.join(tmp_path, 'st_gan')
    model.save(out_dir)

    with h5py.File(bias_fp, 'w') as f:
        f.create_dataset('latitude', data=lat_lon[..., 0])
        f.create_dataset('longitude', data=lat_lon[..., 1])

        s = lat_lon.shape[:2]
        for k, v in params.items():
            f.create_dataset(k, data=np.broadcast_to(v, (*s, *v.shape)))
        f.attrs['dist'] = 'empirical'
        f.attrs['sampling'] = 'linear'
        f.attrs['log_base'] = 10
        f.attrs['time_window_center'] = [182.5]

    bias_correct_kwargs = {
        'U_100m': {
            'feature_name': 'U_100m',
            'base_dset': 'Uref_100m',
            'bias_fp': bias_fp,
            'time_index': pd.DatetimeIndex(
                [np.datetime64(t) for t in ds.time.values]
            ),
        },
        'V_100m': {
            'feature_name': 'V_100m',
            'base_dset': 'Vref_100m',
            'bias_fp': bias_fp,
            'time_index': pd.DatetimeIndex(
                [np.datetime64(t) for t in ds.time.values]
            ),
        },
    }

    strat = ForwardPassStrategy(
        input_files,
        model_kwargs={'model_dir': out_dir},
        fwp_chunk_shape=fwp_chunk_shape,
        spatial_pad=0,
        temporal_pad=0,
        input_handler_kwargs=dict(
            target=target,
            shape=shape,
            temporal_slice=temporal_slice,
            worker_kwargs=dict(max_workers=1),
        ),
        out_pattern=os.path.join(tmp_path, 'out_{file_id}.nc'),
        worker_kwargs=dict(max_workers=1),
        input_handler='DataHandlerNCforCC',
    )
    bc_strat = ForwardPassStrategy(
        input_files,
        model_kwargs={'model_dir': out_dir},
        fwp_chunk_shape=fwp_chunk_shape,
        spatial_pad=0,
        temporal_pad=0,
        input_handler_kwargs=dict(
            target=target,
            shape=shape,
            temporal_slice=temporal_slice,
            worker_kwargs=dict(max_workers=1),
        ),
        out_pattern=os.path.join(tmp_path, 'out_{file_id}.nc'),
        worker_kwargs=dict(max_workers=1),
        input_handler='DataHandlerNCforCC',
        bias_correct_method='local_qdm_bc',
        bias_correct_kwargs=bias_correct_kwargs,
    )

    for ichunk in range(strat.chunks):
        fwp = ForwardPass(strat, chunk_index=ichunk)
        bc_fwp = ForwardPass(bc_strat, chunk_index=ichunk)

        delta = bc_fwp.input_data - fwp.input_data
        assert np.allclose(
            delta[..., 0], -2.72, atol=1e-03
        ), 'U reference offset is -1'
        assert np.allclose(
            delta[..., 1], 2.72, atol=1e-03
        ), 'V reference offset is 1'

        delta = bc_fwp.run_chunk() - fwp.run_chunk()
        assert delta[..., 0].mean() < 0, 'Predicted U should trend <0'
        assert delta[..., 1].mean() > 0, 'Predicted V should trend >0'
