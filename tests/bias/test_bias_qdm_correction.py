"""pytests QDM bias correction calculations"""


def test_qdm_bc():
    """Test QDM bias correction"""

    calc = QuantileDeltaMappingCorrection(FP_NSRDB, FP_CC, FP_FUT_CC,
                            'ghi', 'rsds',
                            target=TARGET, shape=SHAPE,
                            distance_upper_bound=0.7,
                            bias_handler='DataHandlerNCforCC')

    out = calc.run()

    # Guarantee that we have some actual values, otherwise most of the
    # remaining tests would be useless
    for v in out:
        assert np.isfinite(out[v]).any()


def test_parallel():
    """Compare bias correction run serial vs in parallel"""

    s = QuantileDeltaMappingCorrection(FP_NSRDB, FP_CC, FP_FUT_CC,
                            'ghi', 'rsds',
                            target=TARGET, shape=SHAPE,
                            distance_upper_bound=0.7,
                            bias_handler='DataHandlerNCforCC')
    out_s = s.run(max_workers=1)

    p = QuantileDeltaMappingCorrection(FP_NSRDB, FP_CC, FP_FUT_CC,
                            'ghi', 'rsds',
                            target=TARGET, shape=SHAPE,
                            distance_upper_bound=0.7,
                            bias_handler='DataHandlerNCforCC')
    out_p = p.run(max_workers=2)

    for k in out_s.keys():
        assert k in out_p, f"Missing {k} in parallel run"
        assert np.allclose(out_s[k], out_p[k], equal_nan=True), \
            f"Different results for {k}"