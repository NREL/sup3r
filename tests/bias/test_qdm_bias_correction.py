"""pytests QDM bias correction calculations"""

import os
import shutil
import tempfile

import numpy as np
import xarray as xr

from sup3r import TEST_DATA_DIR
from sup3r.bias.bias_calc import QuantileDeltaMappingCorrection

FP_NSRDB = os.path.join(TEST_DATA_DIR, "test_nsrdb_co_2018.h5")
FP_CC = os.path.join(TEST_DATA_DIR, "rsds_test.nc")

# Not ideal but a good start
tmpdir = tempfile.TemporaryDirectory()
FP_FUT_CC = os.path.join(tmpdir.name, 'test_mf.nc')
shutil.copyfile(FP_CC, FP_FUT_CC)

with xr.open_dataset(FP_CC) as fh:
    MIN_LAT = np.min(fh.lat.values.astype(np.float32))
    MIN_LON = np.min(fh.lon.values.astype(np.float32)) - 360
    TARGET = (float(MIN_LAT), float(MIN_LON))
    SHAPE = (len(fh.lat.values), len(fh.lon.values))


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
