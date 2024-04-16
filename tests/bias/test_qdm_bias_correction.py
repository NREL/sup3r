"""pytests QDM bias correction calculations"""

import os
import shutil

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


@pytest.fixture(scope="module")
def fp_fut_cc(tmpdir_factory):
    """Sample future CC dataset

    The same CC but with an offset (75.0) and negligible noise.
    """
    fn = tmpdir_factory.mktemp("data").join("test_mf.nc")
    ds = xr.open_dataset(FP_CC)
    # Adding an offset
    ds['rsds'] += 75.0
    # adding a noise
    ds['rsds'] += np.random.randn(*ds['rsds'].shape)
    ds.to_netcdf(fn)
    # DataHandlerNCforCC requires a string
    fn = str(fn)
    return fn


@pytest.fixture(scope="module")
def fp_fut_cc_notrend(tmpdir_factory):
    """Sample future CC dataset identical to historical CC

    This is currently a copy of FP_CC, thus no trend on time.
    """
    fn = tmpdir_factory.mktemp("data").join("test_mf_notrend.nc")
    shutil.copyfile(FP_CC, fn)
    # DataHandlerNCforCC requires a string
    fn = str(fn)
    return fn


@pytest.fixture(scope="module")
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
    """Test QDM bias correction

    Basic standard run. Using only required arguments. If this fails,
    something fundamental is wrong.
    """

    calc = QuantileDeltaMappingCorrection(FP_NSRDB, FP_CC, fp_fut_cc,
                                          'ghi', 'rsds',
                                          target=TARGET, shape=SHAPE,
                                          bias_handler='DataHandlerNCforCC')

    out = calc.run()

    # Guarantee that we have some actual values, otherwise most of the
    # remaining tests would be useless
    for v in out:
        assert np.isfinite(out[v]).any(), "Something wrong, all CDFs are NaN."

    # Check possible range
    for v in out:
        assert np.nanmin(out[v]) > 0, f"{v} should be all greater than zero."
        assert np.nanmax(out[v]) < 1300, f"{v} should be all less than 1300."

    # Each location can be all finite or all NaN, but not both
    for v in out:
        tmp = np.isfinite(out[v].reshape(-1, out[v].shape[-1]))
        assert np.all(
            np.all(tmp, axis=1) == ~np.all(~tmp, axis=1)
        ), f"For each location of {v} it should be all finite or nonte"


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

    assert not np.isnan(corrected).all(), "Can't compare if only NaN"
    assert not np.allclose(data, corrected, equal_nan=False)


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
    ident_params = os.path.join(tmp_path, "identity.hdf")
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
    ident_params = os.path.join(tmp_path, "identity.hdf")
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
    offset_params = os.path.join(tmp_path, "offset.hdf")
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
    offset_params = os.path.join(tmp_path, "offset.hdf")
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
    offset_params = os.path.join(tmp_path, "offset.hdf")
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
