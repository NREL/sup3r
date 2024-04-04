"""pytests QDM bias correction calculations"""

import os
import shutil
import tempfile

import h5py
import numpy as np
import pytest
import xarray as xr

from sup3r import TEST_DATA_DIR
from sup3r.bias.bias_calc import QuantileDeltaMappingCorrection
from sup3r.bias.bias_transforms import local_qdm_bc
from sup3r.preprocessing.data_handling import DataHandlerNC

FP_NSRDB = os.path.join(TEST_DATA_DIR, "test_nsrdb_co_2018.h5")
FP_CC = os.path.join(TEST_DATA_DIR, "rsds_test.nc")
FP_CC_LAT_LON = DataHandlerNC(FP_CC, "rsds").lat_lon

with xr.open_dataset(FP_CC) as fh:
    MIN_LAT = np.min(fh.lat.values.astype(np.float32))
    MIN_LON = np.min(fh.lon.values.astype(np.float32)) - 360
    TARGET = (float(MIN_LAT), float(MIN_LON))
    SHAPE = (len(fh.lat.values), len(fh.lon.values))


@pytest.fixture(scope="session")
def fp_fut_cc(tmpdir_factory):
    """Sample future CC dataset

    The same CC but with a small offset and negligible noise.
    """
    fn = tmpdir_factory.mktemp("data").join("test_mf.nc")
    ds = xr.open_dataset(FP_CC)
    # Adding an offset
    ds['rsds'] += 50.0
    # adding a noise
    ds['rsds'] += np.random.random(ds['rsds'].shape)
    ds.to_netcdf(fn)
    # DataHandlerNCforCC requires a string
    fn = str(fn)
    return fn


@pytest.fixture(scope="session")
def fp_fut_cc_notrend(tmpdir_factory):
    """Sample future CC dataset

    This is currently a copy of FP_CC, thus no trend on time.
    """
    fn = tmpdir_factory.mktemp("data").join("test_mf.nc")
    shutil.copyfile(FP_CC, fn)
    # DataHandlerNCforCC requires a string
    fn = str(fn)
    return fn


@pytest.fixture(scope="session")
def dist_params(tmpdir_factory, fp_fut_cc):
    """Distribution parameters for standard datasets

    Use the standard datasets to estimate the distributions and save
    in a temporary place to be re-used
    """
    calc = QuantileDeltaMappingCorrection(
        FP_NSRDB,
        FP_CC,
        fp_fut_cc,
        "ghi",
        "rsds",
        target=TARGET,
        shape=SHAPE,
        distance_upper_bound=0.7,
        bias_handler="DataHandlerNCforCC",
    )
    fn = tmpdir_factory.mktemp("params").join("standard.h5")
    _ = calc.run(max_workers=1, fp_out=fn)

    # DataHandlerNCforCC requires a string
    fn = str(fn)

    return fn


def test_qdm_bc(fp_fut_cc):
    """Test QDM bias correction"""

    calc = QuantileDeltaMappingCorrection(FP_NSRDB, FP_CC, fp_fut_cc,
                                          'ghi', 'rsds',
                                          target=TARGET, shape=SHAPE,
                                          distance_upper_bound=0.7,
                                          bias_handler='DataHandlerNCforCC')

    out = calc.run()

    # Guarantee that we have some actual values, otherwise most of the
    # remaining tests would be useless
    for v in out:
        assert np.isfinite(out[v]).any()


def test_parallel(fp_fut_cc):
    """Compare bias correction run serial vs in parallel"""

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
        assert k in out_p, f"Missing {k} in parallel run"
        assert np.allclose(
            out_s[k], out_p[k], equal_nan=True
        ), f"Different results for {k}"


def test_save_file(tmp_path, fp_fut_cc):
    """Save valid output

    Confirm it saves the output by creating a valid HDF5 file.
    """

    calc = QuantileDeltaMappingCorrection(FP_NSRDB, FP_CC, fp_fut_cc,
                                          'ghi', 'rsds',
                                          target=TARGET, shape=SHAPE,
                                          distance_upper_bound=0.7,
                                          bias_handler='DataHandlerNCforCC')

    filename = os.path.join(tmp_path, "test_saving.hdf")
    _ = calc.run(filename)

    # File was created
    os.path.isfile(filename)
    # A valid HDF5, can open and read
    with h5py.File(filename, "r") as f:
        assert "latitude" in f.keys()


def test_qdm_transform(dist_params):
    """
    WIP: Confirm it runs, but don't verify anything yet.
    """
    data = np.ones((*FP_CC_LAT_LON.shape[:-1], 2))
    corrected = local_qdm_bc(data, FP_CC_LAT_LON, "ghi", "rsds", dist_params)
    assert not np.allclose(data, corrected, equal_nan=False)


def test_handler_qdm_bc(fp_fut_cc, dist_params):
    """qdm_bc() method from DataHandler

    WIP: Confirm it runs, but don't verify anything yet.
    """
    Handler = DataHandlerNC(fp_fut_cc, 'rsds')
    corrected = Handler.qdm_bc(dist_params, 'ghi')
