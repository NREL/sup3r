"""pytests QDM bias correction calculations"""

import os
import shutil
import tempfile

import h5py
import numpy as np
import xarray as xr

from sup3r import TEST_DATA_DIR
from sup3r.bias.bias_calc import QuantileDeltaMappingCorrection
from sup3r.bias.bias_transforms import local_qdm_bc
from sup3r.preprocessing.data_handling import DataHandlerNC

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


def test_save_file():
    """Save valid output

    Confirm it saves the output by creating a valid HDF5 file.
    """

    calc = QuantileDeltaMappingCorrection(FP_NSRDB, FP_CC, FP_FUT_CC,
                                          'ghi', 'rsds',
                                          target=TARGET, shape=SHAPE,
                                          distance_upper_bound=0.7,
                                          bias_handler='DataHandlerNCforCC')

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "demo.hdf")
        out = calc.run(filename)

        # File was created
        os.path.isfile(filename)
        # A valid HDF5, can open and read
        with h5py.File(filename, "r") as f:
            assert "latitude" in f.keys()


def test_qdm_transform():
    calc = QuantileDeltaMappingCorrection(FP_NSRDB, FP_CC, FP_FUT_CC,
                                          'ghi', 'rsds',
                                          target=TARGET, shape=SHAPE,
                                          distance_upper_bound=0.7,
                                          bias_handler='DataHandlerNCforCC')
    lat_lon = calc.bias_dh.lat_lon
    with tempfile.TemporaryDirectory() as td:
        fp_out = os.path.join(td, 'bc.h5')
        _ = calc.run(max_workers=1, fp_out=fp_out)
        data = np.ones((*lat_lon.shape[:-1], 2))
        corrected = local_qdm_bc(data, lat_lon, 'ghi', 'rsds', fp_out)


def test_handler_qdm_bc():
    """qdm_bc() method from DataHandler

    WIP: Confirm it runs, but don't verify anything yet.
    """
    calc = QuantileDeltaMappingCorrection(FP_NSRDB, FP_CC, FP_FUT_CC,
                                          'ghi', 'rsds',
                                          target=TARGET, shape=SHAPE,
                                          distance_upper_bound=0.7,
                                          bias_handler='DataHandlerNCforCC')
    with tempfile.TemporaryDirectory() as td:
        fp_out = os.path.join(td, 'bc.h5')
        _ = calc.run(max_workers=1, fp_out=fp_out)

        Handler = DataHandlerNC(FP_FUT_CC, 'rsds')
        Handler.qdm_bc(fp_out, 'ghi')
