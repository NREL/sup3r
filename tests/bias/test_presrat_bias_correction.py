"""Validating PresRat correction procedures


Relevant resources used in the tests:
- FP_CC: Filename of standard biased dataset.
- fut_cc: Future dataset sample based on FP_CC + an offset + small noise
- fp_fut_cc: Filname to `fut_cc`.
- fut_cc_notrend: Future dataset identical to FP_CC, i.e. no trend.
- fp_fut_cc_notrend: Filename to fut_cc_notrend.
- presrat_params: Parameters of reference to test PresRat (using fp_fut_cc).
- presrat_notrend_params: Quantiles of future (mf) are identical to bias
  reference (mh). Thus, there is no trend in the model.
- presrat_identity_params: All distributions (oh & mf) are identical to mh,
    i.e. observations equal to model that doesn't change on time.
- presrat_nochanges_params: Like presrat_identity_params, but also all
    zero_rate are zeros, i.e. no values should be forced to be zero.
- presrat_nozeros_params: Same of presrat_params, but no zero_rate, i.e.
    all zero_rate values are equal to 0 (percent).
"""

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
    apply_zero_precipitation_rate,
    local_presrat_bc,
    PresRat,
)
from sup3r.bias.mixins import ZeroRateMixin
from sup3r.preprocessing.data_handling import DataHandlerNC, DataHandlerNCforCC

FP_NSRDB = os.path.join(TEST_DATA_DIR, 'test_nsrdb_co_2018.h5')
FP_CC = os.path.join(TEST_DATA_DIR, 'rsds_test.nc')
FP_CC_LAT_LON = DataHandlerNC(FP_CC, 'rsds').lat_lon
# A reference zero rate threshold that might not make sense physically but for
# testing purposes only. This might change in the future to force edge cases.
ZR_THRESHOLD = 80

with xr.open_dataset(FP_CC) as fh:
    MIN_LAT = np.min(fh.lat.values.astype(np.float32))
    MIN_LON = np.min(fh.lon.values.astype(np.float32)) - 360
    TARGET = (float(MIN_LAT), float(MIN_LON))
    SHAPE = (len(fh.lat.values), len(fh.lon.values))

VAR_MIN = 0
# Fix this max
VAR_MAX = 1300

@pytest.fixture(scope='module')
def precip():
    # lat = np.linspace(13.66, 31.57, 20)
    # lat = np.linspace(38.245528, 40.350785, 20)
    lat = np.array([38.2455282337738, 38.9472804370071, 39.649032596592, 40.3507847105177])
    # lon = np.linspace(254.53125, 256.640625, 20)
    lon = np.array([254.53125, 255.234375, 255.9375, 256.640625])
    t0 = np.datetime64('2015-01-01T12:00:00')
    time = t0 + np.linspace(0, 364, 365, dtype='timedelta64[D]')
    bnds = (-np.timedelta64(12,'h'), np.timedelta64(12, 'h'))
    time_bnds = time[:,np.newaxis] + bnds
    rng = np.random.default_rng()
    # pr = rng.lognormal(3., 1., (time.size, lat.size, lon.size))
    # pr = rng.uniform(0, 1., (time.size, lat.size, lon.size))
    # Transitioning
    pr = rng.normal(210, 87., (time.size, lat.size, lon.size))
    pr = np.where(pr>0, pr, 0)

    ds = xr.Dataset(
        data_vars={
            "rsds": (["time", "lat", "lon"], pr)
        },
        coords={
            "time": ("time", time),
            "time_bnds": (["time", "bnds"], time_bnds),
            "lat": ("lat", lat),
            "lon": ("lon", lon),
        })

    return ds

# FP_CC
@pytest.fixture(scope='module')
def fp_precip(tmpdir_factory, precip):
    """Precipitation sample filename

    DataHandlerNCforCC requires a string
    """
    fn = tmpdir_factory.mktemp('data').join('precip_mh.nc')
    precip.to_netcdf(fn)
    fn = str(fn)
    return fn

# fut_cc
@pytest.fixture(scope='module')
def precip_fut(tmpdir_factory, precip):
    ds = precip.copy(deep=True)

    time = ds['time'] + np.timedelta64(18263,'D')
    time.attrs = ds['time'].attrs
    ds['time'] = time
    # Adding an offset
    # ds['pr'] += 1
    ds['rsds'] += 75
    # adding a small noise
    # ds['pr'] += 1e-6 * np.random.randn(*ds['pr'].shape)
    ds['rsds'] += 1e-4 * np.random.randn(*ds['rsds'].shape)

    return ds['rsds']

@pytest.fixture(scope='module')
def fp_precip_fut(tmpdir_factory, precip_fut):
    """Future precipitation sample filename

    DataHandlerNCforCC requires a string
    """
    fn = tmpdir_factory.mktemp('data').join('precip_mf.nc')
    precip_fut.to_netcdf(fn)
    fn = str(fn)
    return fn

@pytest.fixture(scope='module')
def fp_fut_cc(tmpdir_factory):
    """Sample future CC dataset

    The same CC but with an offset (75.0) and negligible noise.
    """
    fn = tmpdir_factory.mktemp('data').join('test_mf.nc')
    ds = xr.open_dataset(FP_CC)
    # Adding an offset
    ds['rsds'] += 75.0
    # adding a small noise
    ds['rsds'] += 1e-4 * np.random.randn(*ds['rsds'].shape)
    ds.to_netcdf(fn)
    # DataHandlerNCforCC requires a string
    fn = str(fn)
    return fn


@pytest.fixture(scope='module')
def fut_cc(fp_fut_cc):
    """Gives the dataset itself related to fp_fut_cc

    Giving an object in memory makes everything more efficient by avoiding I/O
    reading files and overhead for multiple uses.

    Note that ``Resources`` modifies the dataset so we cannot just load the
    NetCDF. Here we run a couple of checks to confirm that the output
    dataset is as expected by sup3r.

    To use time as expected by sup3r: time = pd.to_datetime(da.time)
    To use latlon as expected by sup3r:
      latlon = np.stack(xr.broadcast(da["lat"], da["lon"] - 360),
                        axis=-1).astype('float32')
    """
    ds = xr.open_dataset(fp_fut_cc)

    # Operating with numpy arrays impose a fixed dimensions order
    # This compute is required here.
    da = ds['rsds'].compute().transpose('lat', 'lon', 'time')

    # The _get_factors() assume latitude as descending and it will
    # silently return wrong values otherwise.
    da = da.sortby('lat', ascending=False)

    latlon = np.stack(
        xr.broadcast(da['lat'], da['lon'] - 360), axis=-1
    )
    # Confirm that dataset order is consistent
    # Somewhere in pipeline latlon are downgraded to f32
    assert np.allclose(latlon.astype('float32'), FP_CC_LAT_LON)

    # Verify data alignment in comparison with expected for FP_CC
    for ii in range(ds.lat.size):
        for jj in range(ds.lon.size):
            assert np.allclose(
                da.sel(lat=latlon[ii, jj, 0]).sel(
                    lon=latlon[ii, jj, 1] + 360
                ),
                da.data[ii, jj],
            )

    return da


@pytest.fixture(scope='module')
def fp_fut_cc_notrend(tmpdir_factory, fp_precip):
    """Sample future CC dataset identical to historical CC

    This is currently a copy of FP_CC, thus no trend on time.
    """
    fn = tmpdir_factory.mktemp('data').join('test_mf_notrend.nc')
    shutil.copyfile(fp_precip, fn)
    # DataHandlerNCforCC requires a string
    fn = str(fn)
    return fn


@pytest.fixture(scope='module')
def fut_cc_notrend(fp_fut_cc_notrend):
    """Extract the dataset from fp_fut_cc_notrend

    The internal process to read such dataset is way more complex than just
    reading it and there are some transformations. This function must provide
    a dataset compatible with the one expected from the standard processing.
    """
    ds = xr.open_dataset(fp_fut_cc_notrend)

    # Although it is the same file, somewhere in the data reading process
    # the longitude is tranformed to the standard [-180 to 180] and it is
    # expected to be like that everywhere.
    ds['lon'] = ds['lon'] - 360

    # Operating with numpy arrays impose a fixed dimensions order
    # This compute is required here.
    da = ds['rsds'].compute().transpose('lat', 'lon', 'time')

    # The _get_factors() assume latitude as descending and it will
    # silently return wrong values otherwise.
    da = da.sortby('lat', ascending=False)

    latlon = np.stack(
        xr.broadcast(da['lat'], da['lon']), axis=-1
    )
    # Confirm that dataset order is consistent
    # Somewhere in pipeline latlon are downgraded to f32
    assert np.allclose(latlon.astype('float32'), FP_CC_LAT_LON)

    # Verify data alignment in comparison with expected for FP_CC
    for ii in range(ds.lat.size):
        for jj in range(ds.lon.size):
            np.allclose(
                da.sel(lat=latlon[ii, jj, 0]).sel(lon=latlon[ii, jj, 1]),
                da.data[ii, jj],
            )

    return da


@pytest.fixture(scope='module')
def presrat_params(tmpdir_factory, fp_precip, fp_precip_fut):
    """PresRat parameters for standard datasets

    Use the standard datasets to estimate the distributions and save
    in a temporary place to be re-used
    """
    calc = PresRat(
        FP_NSRDB,
        fp_precip,
        fp_precip_fut,
        'ghi',
        'rsds',
        target=TARGET,
        shape=SHAPE,
        distance_upper_bound=0.7,
        bias_handler='DataHandlerNCforCC',
    )
    fn = tmpdir_factory.mktemp('params').join('presrat.h5')
    # Physically non-sense threshold choosed to result in gridpoints with and
    # without zero rate correction for the given testing dataset.
    _ = calc.run(max_workers=1, zero_rate_threshold=ZR_THRESHOLD, fp_out=fn)

    # DataHandlerNCforCC requires a string
    fn = str(fn)

    return fn


@pytest.fixture(scope='module')
def presrat_notrend_params(tmpdir_factory, fp_precip, fp_fut_cc_notrend):
    """No change in time

    The bias_fut distribution is equal to bias (mod_his), so no change in
    time.

    We could save some overhead here copying fp_fut_cc and replacing some
    values there. That was done before but missed some variables resulting
    in errors.
    """
    calc = PresRat(
        FP_NSRDB,
        fp_precip,
        fp_fut_cc_notrend,
        'ghi',
        'rsds',
        target=TARGET,
        shape=SHAPE,
        distance_upper_bound=0.7,
        bias_handler='DataHandlerNCforCC',
    )
    fn = tmpdir_factory.mktemp('params').join('presrat_notrend.h5')
    _ = calc.run(zero_rate_threshold=ZR_THRESHOLD, fp_out=fn)

    # DataHandlerNCforCC requires a string
    fn = str(fn)

    return fn


@pytest.fixture(scope='module')
def presrat_identity_params(tmpdir_factory, presrat_params):
    """Identical distribution"""

    fn = tmpdir_factory.mktemp('params').join('presrat_identity.h5')
    shutil.copyfile(presrat_params, fn)

    with h5py.File(fn, 'r+') as f:
        f['bias_rsds_params'][:] = f['bias_fut_rsds_params'][:]
        f['base_ghi_params'][:] = f['bias_fut_rsds_params'][:]
        f.flush()

    return str(fn)


@pytest.fixture(scope='module')
def presrat_nochanges_params(tmpdir_factory, presrat_params):
    """Identical distribution and no zero rate

    All distributions are identical and zero rate changes are all zero,
    therefore, the PresRat correction should not change anything.

    Note that distributions are based on bias_fut, so it is assumed that the
    test cases will be datasets coherent with that bias_fut distribution,
    otherwise it could lead to differences if out of that scale.
    """
    fn = tmpdir_factory.mktemp('params').join('presrat_nochanges.h5')
    shutil.copyfile(presrat_params, fn)

    with h5py.File(fn, 'r+') as f:
        f['bias_fut_rsds_params'][:] = f['bias_rsds_params'][:]
        f['base_ghi_params'][:] = f['bias_rsds_params'][:]
        f['ghi_zero_rate'][:] *= 0
        f['rsds_mean_mf'][:] = f['rsds_mean_mh'][:]
        f.flush()

    return str(fn)


@pytest.fixture(scope='module')
def presrat_nozeros_params(tmpdir_factory, presrat_params):
    """PresRat parameters without zero rate correction

    The same standard parameters but all zero_rate values are equal to zero,
    which means that zero percent, i.e. none, of the values should be forced
    to be zero.
    """
    fn = tmpdir_factory.mktemp('params').join('presrat_nozeros.h5')
    shutil.copyfile(presrat_params, fn)

    with h5py.File(fn, 'r+') as f:
        f['ghi_zero_rate'][:] *= 0
        f.flush()

    return str(fn)


def test_zero_precipitation_rate():
    """Zero rate estimate with extremme thresholds"""
    f = ZeroRateMixin().zero_precipitation_rate
    arr = np.random.randn(100)

    rate = f(arr, threshold=np.median(arr))
    assert rate == 0.5


def test_zero_precipitation_rate_extremes():
    """Zero rate estimate with extremme thresholds"""
    f = ZeroRateMixin().zero_precipitation_rate
    arr = np.arange(10)

    rate = f(arr, threshold=-1)
    assert rate == 0

    rate = f(arr, threshold=0)
    assert rate == 0

    # Remember, 9 is the last value, i.e. the 10th value
    rate = f(arr, threshold=9)
    assert rate == 0.9

    rate = f(arr, threshold=100)
    assert rate == 1


def test_zero_precipitation_rate_nanonly():
    """Zero rate estimate with only NaNs gives NaN"""
    f = ZeroRateMixin().zero_precipitation_rate
    arr = np.nan * np.zeros(10)

    # All NaN gives NaN rate
    rate = f(arr)
    assert np.isnan(rate)


def test_zero_precipitation_rate_nan():
    """Zero rate estimate with NaNs

    NaN shouldn't be counted to find the rate. Thus an array with NaNs should
    give the same results if the NaN were removed before the calculation.
    """
    f = ZeroRateMixin().zero_precipitation_rate
    arr = np.arange(10)

    r1 = f(arr, threshold=5)
    r2 = f(np.concatenate([5 * [np.nan], arr]), threshold=5)
    assert r1 == r2


@pytest.mark.parametrize('threshold', [0, 50, 1e6])
def test_parallel(fp_precip, fp_precip_fut, threshold):
    """Running in parallel must not alter results

    Check with different thresholds, which will result in different zero rates.
    """
    s = PresRat(
        FP_NSRDB,
        fp_precip,
        fp_precip_fut,
        'ghi',
        'rsds',
        target=TARGET,
        shape=SHAPE,
        bias_handler='DataHandlerNCforCC',
    )

    out_s = s.run(max_workers=1, zero_rate_threshold=threshold)

    p = PresRat(
        FP_NSRDB,
        fp_precip,
        fp_precip_fut,
        'ghi',
        'rsds',
        target=TARGET,
        shape=SHAPE,
        bias_handler='DataHandlerNCforCC',
    )

    out_p = p.run(max_workers=2, zero_rate_threshold=threshold)

    for k in out_s.keys():
        assert k in out_p, f'Missing {k} in parallel run'
        assert np.allclose(
            out_s[k], out_p[k], equal_nan=True
        ), f'Different results for {k}'


def test_presrat_calc(fp_precip, fp_precip_fut):
    """Standard PresRat (pre) calculation

    Estimate the required parameters with a standard setup.

    WIP: Just confirm it runs, but not checking much yet.
    """
    calc = PresRat(
        FP_NSRDB,
        fp_precip,
        fp_precip_fut,
        'ghi',
        'rsds',
        target=TARGET,
        shape=SHAPE,
        bias_handler='DataHandlerNCforCC',
    )

    out = calc.run(zero_rate_threshold=ZR_THRESHOLD)

    expected_vars = ['bias_rsds_params', 'bias_fut_rsds_params',
                     'base_ghi_params', 'ghi_zero_rate', 'rsds_mean_mh',
                     'rsds_mean_mf']
    sref = FP_CC_LAT_LON.shape[:2]
    for v in expected_vars:
        assert v in out, f"Missing {v} in the calculated output"
        assert out[v].shape[:2] == sref, "Doesn't match expected spatial shape"
        # This is only true because fill and extend are applied by default.
        assert np.all(np.isfinite(out[v])), f"Unexpected NaN for {v}"

    zero_rate = out['ghi_zero_rate']
    assert np.all((zero_rate >= 0) & (zero_rate <= 1)), 'Out of range [0, 1]'


@pytest.mark.parametrize('threshold', [0, 50, 1e6])
def test_presrat_zero_rate(fp_precip, fp_precip_fut, threshold):
    """Estimate zero_rate within PresRat.run()

    Use thresholds that gives 0%, 100%, and something between.

    Notes
    -----
    - Rate should be zero if threshold is zero for this dataset.
    """
    calc = PresRat(
        FP_NSRDB,
        fp_precip,
        fp_precip_fut,
        'ghi',
        'rsds',
        target=TARGET,
        shape=SHAPE,
        bias_handler='DataHandlerNCforCC',
    )

    out = calc.run(zero_rate_threshold=threshold)

    assert 'ghi_zero_rate' in out, 'Missing ghi_zero_rate in calc output'
    zero_rate = out['ghi_zero_rate']
    # True for this dataset because fill and extend are applied by default.
    assert np.all(np.isfinite(zero_rate)), 'Unexpected NaN for ghi_zero_rate'
    assert np.all((zero_rate >= 0) & (zero_rate <= 1)), 'Out of range [0, 1]'

    if threshold <= 0:
        assert np.all(zero_rate == 0), 'It should be rate 0 for threshold==0'
    elif threshold >= 1e4:
        assert np.all(zero_rate == 1), 'It should be rate 1 for threshold>=1e4'


def test_apply_zero_precipitation_rate():
    """Reinforce the zero precipitation rate, standard run"""
    data = np.array([[[5, 0.1, 3, 0.2, 1]]])
    out = apply_zero_precipitation_rate(data, np.array([[[0.25]]]))

    assert np.allclose([5.0, 0.0, 3, 0.2, 1.0], out, equal_nan=True)


def test_apply_zero_precipitation_rate_nan():
    """Validate with NaN in the input"""
    data = np.array([[[5, 0.1, np.nan, 0.2, 1]]])
    out = apply_zero_precipitation_rate(data, np.array([[[0.25]]]))

    assert np.allclose([5.0, 0.0, np.nan, 0.2, 1.0], out, equal_nan=True)


def test_apply_zero_precipitation_rate_2D():
    """Validate a 2D input"""
    data = np.array(
        [
            [
                [5, 0.1, np.nan, 0.2, 1],
                [5, 0.1, 3, 0.2, 1],
            ]
        ]
    )
    out = apply_zero_precipitation_rate(data, np.array([[[0.25], [0.41]]]))

    assert np.allclose(
        [[5.0, 0.0, np.nan, 0.2, 1.0], [5.0, 0.0, 3, 0.0, 1.0]],
        out,
        equal_nan=True,
    )


def test_presrat(fp_precip, fp_precip_fut):
    """Test PresRat correction procedure

    Basic standard run. Using only required arguments. If this fails,
    something fundamental is wrong.
    """
    calc = PresRat(
        FP_NSRDB,
        fp_precip,
        fp_precip_fut,
        'ghi',
        'rsds',
        target=TARGET,
        shape=SHAPE,
        bias_handler='DataHandlerNCforCC',
    )

    # A high zero_rate_threshold to gets at least something.
    out = calc.run(zero_rate_threshold=50)

    # Guarantee that we have some actual values, otherwise most of the
    # remaining tests would be useless
    for v in out:
        assert np.isfinite(out[v]).any(), 'Something wrong, all CDFs are NaN.'

    # Check possible range
    for v in out:
        assert np.nanmin(out[v]) >= VAR_MIN, f'{v} should be all greater than zero.'
        assert np.nanmax(out[v]) < VAR_MAX, f'{v} should be all less than 1300.'

    # Each location can be all finite or all NaN, but not both
    for v in (v for v in out if len(out[v].shape) > 2):
        tmp = np.isfinite(out[v].reshape(-1, *out[v].shape[2:]))
        assert np.all(
            np.all(tmp, axis=1) == ~np.all(~tmp, axis=1)
        ), f'For each location of {v} it should be all finite or nonte'


def test_presrat_transform(presrat_params, fut_cc):
    """A standard run with local_presrat_bc

    WIP: Confirm it runs only.
    """
    data = fut_cc.values
    time = pd.to_datetime(fut_cc.time)
    latlon = np.stack(
        xr.broadcast(fut_cc['lat'], fut_cc['lon'] - 360), axis=-1
    ).astype('float32')

    corrected = local_presrat_bc(
        data, time, latlon, 'ghi', 'rsds', presrat_params
    )

    assert np.isfinite(corrected).any(), "Can't compare if only NaN"
    # Confirm that there were changes, but at this point stop there.
    assert not np.allclose(data, corrected, equal_nan=False)


def test_presrat_transform_nochanges(presrat_nochanges_params, fut_cc_notrend):
    """The correction should result in no changes at all

    Note that there are a lot of implicit transformations, so we cannot expect
    to be able to esily compare all gridpoints.

    The corrected output must be the same if:
    - The three CDFs are the same, so no change due to QDM. There is one
      caveat here. The data to be corrected is compared with the mf's CDF, and
      if out of distribution, it would lead to differences;
    - All zero rate set to zero percent, so no value is forced to zero;
    - The value to be corrected is the same used to estimate the means for the
      K factor;
    """
    data = fut_cc_notrend.values
    time = pd.to_datetime(fut_cc_notrend.time)
    latlon = np.stack(
        xr.broadcast(fut_cc_notrend['lat'], fut_cc_notrend['lon']),
        axis=-1,
    ).astype('float32')

    corrected = local_presrat_bc(
        data, time, latlon, 'ghi', 'rsds', presrat_nochanges_params
    )

    assert np.isfinite(corrected).any(), "Can't compare if only NaN"

    # The calculations are set in such a way that `run()` only applies to
    # gripoints where there are historical reference and biased data. This is
    # hidden by an implicit interpolation procedure, which results in values
    # that can't be easily reproduced for validation. One solution is to
    # allow the implicit interpolation but compare only where non-interpolated
    # values are available. Let's call it the 'Magic index'.
    idx = (slice(1, 3), slice(0, 3))
    assert np.allclose(
        data[idx], corrected[idx], equal_nan=False
    ), "This case shouldn't modify the data"


def test_presrat_transform_nozerochanges(presrat_nozeros_params, fut_cc):
    """No adjustment to zero

    Correction procedure results in some changes, but since the zero_rate
    correction is all set to zero percent, there are no values adjusted to
    zero.
    """
    data = fut_cc.values
    time = pd.to_datetime(fut_cc.time)
    latlon = np.stack(
        xr.broadcast(fut_cc['lat'], fut_cc['lon'] - 360), axis=-1
    ).astype('float32')

    corrected = local_presrat_bc(
        data,
        time,
        latlon,
        'ghi',
        'rsds',
        presrat_nozeros_params,
    )

    assert np.isfinite(data).any(), "Can't compare if only NaN"
    assert not np.allclose(
        data, corrected, equal_nan=False
    ), 'Expected changes due to bias correction'
    assert not (
        (data != 0) & (corrected == 0)
    ).any(), 'Unexpected value corrected (zero_rate) to zero (dry day)'


def precip_sample(tmpdir_factory):
    t_start = np.datetime64('2015-01-01')
    t_end = np.datetime64('2015-01-20')
    nt = 20

    lat = np.linspace(13.66, 31.57, 20)
    long = np.linspace(125.0, 148.75, 20)
    t0 = np.datetime64('2015-01-01')
    time = t0 + np.linspace(0, 19, 20, dtype='timedelta64[D]')

    ds = xr.Dataset()


def test_fwp_integration(tmp_path, fp_fut_cc, presrat_params):
    """Integration of the PresRat correction method into the forward pass

    Validate two aspects:
    - We should be able to run a forward pass with unbiased data.
    - The bias trend should be observed in the predicted output.
    """
    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    features = ['rsds']
    target = (39.0, -104.5)
    # shape = (8, 8)
    shape = (2, 2)
    temporal_slice = slice(None, None, 1)
    fwp_chunk_shape = (4, 4, 150)
    input_files = [fp_fut_cc]

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 10, 10, 6, len(features))))
    model.meta['lr_features'] = features
    model.meta['hr_out_features'] = features
    model.meta['s_enhance'] = 3
    model.meta['t_enhance'] = 4

    out_dir = os.path.join(tmp_path, 'st_gan')
    model.save(out_dir)

    bias_correct_kwargs = {
                           'rsds': {'feature_name': 'rsds',
                                      'base_dset': 'ghi',
                                      'bias_fp': presrat_params}}

    strat = ForwardPassStrategy(
        input_files,
        model_kwargs={'model_dir': out_dir},
        fwp_chunk_shape=fwp_chunk_shape,
        spatial_pad=0, temporal_pad=0,
        input_handler_kwargs=dict(target=target, shape=shape,
                                  temporal_slice=temporal_slice,
                                  worker_kwargs=dict(max_workers=1)),
        out_pattern=os.path.join(tmp_path, 'out_{file_id}.nc'),
        worker_kwargs=dict(max_workers=1),
        input_handler='DataHandlerNCforCC',
    )
    bc_strat = ForwardPassStrategy(
        input_files,
        model_kwargs={'model_dir': out_dir},
        fwp_chunk_shape=fwp_chunk_shape,
        spatial_pad=0, temporal_pad=0,
        input_handler_kwargs=dict(target=target, shape=shape,
                                  temporal_slice=temporal_slice,
                                  worker_kwargs=dict(max_workers=1)),
        out_pattern=os.path.join(tmp_path, 'out_{file_id}.nc'),
        worker_kwargs=dict(max_workers=1),
        input_handler='DataHandlerNCforCC',
        bias_correct_method='local_presrat_bc',
        bias_correct_kwargs=bias_correct_kwargs,
    )

    for ichunk in range(strat.chunks):
        fwp = ForwardPass(strat, chunk_index=ichunk)
        bc_fwp = ForwardPass(bc_strat, chunk_index=ichunk)

        delta = bc_fwp.input_data - fwp.input_data

        delta = bc_fwp.run_chunk() - fwp.run_chunk()
