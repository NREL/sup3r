

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


@pytest.fixture(scope='module')
def dist_params(tmpdir_factory, fp_fut_cc):
    """Distribution parameters for standard datasets

    Use the standard datasets to estimate the distributions and save
    in a temporary place to be re-used
    """
    calc = QuantileDeltaMappingCorrection(
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
    fn = tmpdir_factory.mktemp('params').join('standard.h5')
    _ = calc.run(max_workers=1, fp_out=fn)

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
    r2 = f(np.concatenate([5*[np.nan], arr]), threshold=5)
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

def test_presrat_zero_rate(fp_fut_cc):
    """Estimate zero_rate within PresRat.run()"""
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
    out = calc.run(zero_rate_threshold=50)

    assert 'ghi_zero_rate' in out, 'Missing ghi_zero_rate in calc output'
    zero_rate = out['ghi_zero_rate']
    assert np.all(np.isfinite(zero_rate)), "Unexpected NaN for ghi_zero_rate"
    assert np.all((zero_rate>=0) & (zero_rate<=1)), "Out of range [0, 1]"


def test_presrat_zero_rate_threshold_zero(fp_fut_cc):
    """Estimate zero_rate within PresRat.run(), zero threshold

    This should give a zero rate answer, since all values are higher.
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
    assert np.all(np.isfinite(zero_rate)), "Unexpected NaN for ghi_zero_rate"
    assert np.all(zero_rate==0), "Threshold=0, rate should be 0"


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
    assert np.all(np.isfinite(zero_rate)), "Unexpected NaN for ghi_zero_rate"
    assert np.all(zero_rate==1), "Threshold=0, rate should be 0"
