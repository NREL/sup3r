"""Validating PresRat correction procedures


Relevant sources used in the tests:
- fp_fut_cc: Future dataset based on FP_CC + an offset + small noise
- fp_fut_cc_notrend: Future dataset identical to FP_CC
- presrat_params: Parameters of reference to test PresRat
- presrat_notrend_params: Bias historical is identical to bias reference
    (historical). Thus, there is no trend in the model.
- presrat_identity_params: All distributions are identical (oh & mf) to mh,
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

from sup3r import TEST_DATA_DIR
from sup3r.bias import (
    apply_zero_precipitation_rate,
    local_presrat_bc,
    PresRat,
)
from sup3r.bias.mixins import ZeroRateMixin
from sup3r.preprocessing.data_handling import DataHandlerNC

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
    # This compute here is required.
    da = ds['rsds'].compute().transpose('lat', 'lon', 'time')
    # Unfortunatelly, _get_factors() assume latitude as descending
    da = da.sortby('lat', ascending=False)
    # data = da.data
    # time = pd.to_datetime(da.time)
    latlon = np.stack(
        xr.broadcast(da['lat'], da['lon'] - 360), axis=-1
    ).astype('float32')
    # latlon = np.stack(np.meshgrid(da['lon'] - 360, da['lat']), axis=-1)[
    #     :, :, ::-1
    # ].astype('float32')
    for ii in range(4):
        for jj in range(4):
            assert np.allclose(
                da.sel(lat=latlon[ii, jj, 0], method='nearest').sel(
                    lon=latlon[ii, jj, 1] + 360, method='nearest'
                )[0],
                da.data[ii, jj, 0],
            )
    assert np.allclose(latlon, FP_CC_LAT_LON)

    return da.compute()


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


@pytest.fixture(scope='module')
def fut_cc_notrend(fp_fut_cc_notrend):
    ds = xr.open_dataset(fp_fut_cc_notrend)
    da = ds['rsds'].compute().transpose('lat', 'lon', 'time')
    # Unfortunatelly, _get_factors() assume latitude as descending
    da = da.sortby('lat', ascending=False)
    latlon = np.stack(
        xr.broadcast(da['lat'], da['lon'] - 360), axis=-1
    ).astype('float32')
    for ii in range(4):
        for jj in range(4):
            np.allclose(
                da.sel(lat=latlon[ii, jj, 0], method='nearest').sel(
                    lon=latlon[ii, jj, 1] + 360, method='nearest'
                )[0],
                da.data[ii, jj, 0],
            )
    assert np.allclose(latlon, FP_CC_LAT_LON)

    return da.compute()


@pytest.fixture(scope='module')
def presrat_params(tmpdir_factory, fp_fut_cc):
    """PresRat parameters for standard datasets

    Use the standard datasets to estimate the distributions and save
    in a temporary place to be re-used
    """
    calc = PresRat(
        FP_NSRDB,
        FP_CC,
        fp_fut_cc,
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
    _ = calc.run(max_workers=1, zero_rate_threshold=80, fp_out=fn)

    # DataHandlerNCforCC requires a string
    fn = str(fn)

    return fn


@pytest.fixture(scope='module')
def presrat_params(tmpdir_factory, fp_fut_cc):
    """PresRat parameters for standard datasets

    Use the standard datasets to estimate the distributions and save
    in a temporary place to be re-used
    """
    calc = PresRat(
        FP_NSRDB,
        FP_CC,
        fp_fut_cc,
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
    _ = calc.run(zero_rate_threshold=80, fp_out=fn)

    # DataHandlerNCforCC requires a string
    fn = str(fn)

    return fn


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
    arr = np.arange(10)

    # All NaN gives NaN rate
    rate = f(np.nan * arr)
    assert np.isnan(rate)


def test_zero_precipitation_rate_nan():
    """Zero rate estimate with NaNs

    NaN shouldn't be counted to find the rate.
    """
    f = ZeroRateMixin().zero_precipitation_rate
    arr = np.arange(10)

    r1 = f(arr, threshold=5)
    r2 = f(np.concatenate([5 * [np.nan], arr]), threshold=5)
    assert r1 == r2


"""
    breakpoint()

    # Physically non sense threshold.
    out = calc.run(zero_rate_threshold=0)

    assert 'ghi_zero_rate' in out, 'Missing ghi_zero_rate in calc output'
    zero_rate = out['ghi_zero_rate']
    assert np.all(np.isfinite(zero_rate)), "Unexpected NaN for ghi_zero_rate"
    assert np.all(zero_rate==0), "It should be all zero percent"

    # Physically non sense threshold.
    out = calc.run(zero_rate_threshold=1e6)

    assert 'ghi_zero_rate' in out, 'Missing ghi_zero_rate in calc output'
    zero_rate = out['ghi_zero_rate']
    assert np.all(np.isfinite(zero_rate)), "Unexpected NaN for ghi_zero_rate"
    assert np.all(zero_rate==1), "It should be all zero percent"
"""


@pytest.mark.parametrize('threshold', [0, 50, 1e6])
def test_parallel(fp_fut_cc, threshold):
    """Running in parallel must not alter results

    Check with different thresholds that will result in different zero rates.
    """
    s = PresRat(
        FP_NSRDB,
        FP_CC,
        fp_fut_cc,
        'ghi',
        'rsds',
        target=TARGET,
        shape=SHAPE,
        bias_handler='DataHandlerNCforCC',
    )

    out_s = s.run(max_workers=1, zero_rate_threshold=threshold)

    p = PresRat(
        FP_NSRDB,
        FP_CC,
        fp_fut_cc,
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


@pytest.mark.parametrize('threshold', [0, 50, 1e6])
def test_presrat_zero_rate(fp_fut_cc, threshold):
    """Estimate zero_rate within PresRat.run()

    Use thresholds that gives 0%, 100%, and something between.
    """
    calc = PresRat(
        FP_NSRDB,
        FP_CC,
        fp_fut_cc,
        'ghi',
        'rsds',
        target=TARGET,
        shape=SHAPE,
        bias_handler='DataHandlerNCforCC',
    )

    out = calc.run(zero_rate_threshold=threshold)

    assert 'ghi_zero_rate' in out, 'Missing ghi_zero_rate in calc output'
    zero_rate = out['ghi_zero_rate']
    assert np.all(np.isfinite(zero_rate)), 'Unexpected NaN for ghi_zero_rate'
    assert np.all((zero_rate >= 0) & (zero_rate <= 1)), 'Out of range [0, 1]'

    if threshold == 0:
        assert np.all(zero_rate >= 0), 'It should be rate 0 for threshold==0'

    """
    calc = PresRat(
        FP_NSRDB,
        FP_CC,
        fp_fut_cc,
        'ghi',
        'rsds',
        target=TARGET,
        shape=SHAPE,
        bias_handler='DataHandlerNCforCC',
    )

    # Physically non sense threshold.
    out = calc.run(zero_rate_threshold=0)

    assert 'ghi_zero_rate' in out, 'Missing ghi_zero_rate in calc output'
    zero_rate = out['ghi_zero_rate']
    assert np.all(np.isfinite(zero_rate)), 'Unexpected NaN for ghi_zero_rate'
    assert np.all(zero_rate == 0), 'Threshold=0, rate should be 0'


def test_presrat_zero_rate_threshold_1e9(fp_fut_cc):
    """Estimate zero_rate within PresRat.run(), zero threshold

    This should give a zero rate answer, since all values are lower.
    """
    calc = PresRat(
        FP_NSRDB,
        FP_CC,
        fp_fut_cc,
        'ghi',
        'rsds',
        target=TARGET,
        shape=SHAPE,
        bias_handler='DataHandlerNCforCC',
    )

    # Physically non sense threshold.
    out = calc.run(zero_rate_threshold=1e9)

    assert 'ghi_zero_rate' in out, 'Missing ghi_zero_rate in calc output'
    zero_rate = out['ghi_zero_rate']
    assert np.all(np.isfinite(zero_rate)), 'Unexpected NaN for ghi_zero_rate'
    assert np.all(zero_rate == 1), 'Threshold=0, rate should be 0'


def test_apply_zero_precipitation_rate():
    data = np.array([[[5, 0.1, 3, 0.2, 1]]])
    out = apply_zero_precipitation_rate(data, np.array([[[0.25]]]))

    assert np.allclose([5.0, 0.0, 3, 0.2, 1.0], out, equal_nan=True)


def test_apply_zero_precipitation_rate_nan():
    data = np.array([[[5, 0.1, np.nan, 0.2, 1]]])
    out = apply_zero_precipitation_rate(data, np.array([[[0.25]]]))

    assert np.allclose([5.0, 0.0, np.nan, 0.2, 1.0], out, equal_nan=True)


def test_apply_zero_precipitation_rate_2D():
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


def test_presrat(fp_fut_cc):
    """Test PresRat correction procedure

    Basic standard run. Using only required arguments. If this fails,
    something fundamental is wrong.
    """
    calc = PresRat(
        FP_NSRDB,
        FP_CC,
        fp_fut_cc,
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
        assert np.nanmin(out[v]) > 0, f'{v} should be all greater than zero.'
        assert np.nanmax(out[v]) < 1300, f'{v} should be all less than 1300.'

    # Each location can be all finite or all NaN, but not both
    for v in (v for v in out if len(out[v].shape) > 2):
        tmp = np.isfinite(out[v].reshape(-1, *out[v].shape[2:]))
        assert np.all(
            np.all(tmp, axis=1) == ~np.all(~tmp, axis=1)
        ), f'For each location of {v} it should be all finite or nonte'


def test_presrat_transform(presrat_params, fut_cc):
    """
    WIP: Confirm it runs, but don't verify anything yet.
    """
    data = fut_cc.values
    time = pd.to_datetime(fut_cc.time)
    latlon = np.stack(
        xr.broadcast(fut_cc['lat'], fut_cc['lon'] - 360), axis=-1
    ).astype('float32')

    corrected = local_presrat_bc(
        data, time, latlon, 'ghi', 'rsds', presrat_params
    )

    assert not np.isnan(corrected).all(), "Can't compare if only NaN"
    assert not np.allclose(data, corrected, equal_nan=False)


def test_presrat_transform_nochanges(presrat_nochanges_params, fut_cc_notrend):
    """The correction should result in no changes at all

    Note that there are a lot of implicit transformations, so we can't expect
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
    idx = (slice(1,3), slice(0,3))
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
