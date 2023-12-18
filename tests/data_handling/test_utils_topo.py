# -*- coding: utf-8 -*-
"""pytests for topography utilities"""
import os
import shutil
import tempfile

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pytest
from rex import Resource
from rex import Outputs

from sup3r import TEST_DATA_DIR
from sup3r.preprocessing.data_handling.exo_extraction import (
    TopoExtractH5,
    TopoExtractNC,
)

FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
TARGET = (39.001, -105.15)
SHAPE = (20, 20)
FP_WRF = os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_00_00_00')
WRF_TARGET = (19.3, -123.5)
WRF_SHAPE = (8, 8)


def get_lat_lon_range_h5(fp):
    """Get the min/max lat/lon from an h5 file"""
    with Resource(fp) as wtk:
        lat_range = (wtk.meta['latitude'].min(), wtk.meta['latitude'].max())
        lon_range = (wtk.meta['longitude'].min(), wtk.meta['longitude'].max())
    return lat_range, lon_range


def get_lat_lon_range_nc(fp):
    """Get the min/max lat/lon from a netcdf file"""
    import xarray as xr
    dset = xr.open_dataset(fp)
    lat_range = (dset['lat'].values.min(), dset['lat'].values.max())
    lon_range = (dset['lon'].values.min(), dset['lon'].values.max())
    return lat_range, lon_range


def make_topo_file(fp, td, N=100, offset=0.1):
    """Make a dummy h5 file with high-res topo for testing"""

    if fp.endswith('.h5'):
        lat_range, lon_range = get_lat_lon_range_h5(fp)
    else:
        lat_range, lon_range = get_lat_lon_range_nc(fp)

    lat = np.linspace(lat_range[0] - offset, lat_range[1] + offset, N)
    lon = np.linspace(lon_range[0] - offset, lon_range[1] + offset, N)
    idy, idx = np.meshgrid(np.arange(len(lon)), np.arange(len(lat)))
    lon, lat = np.meshgrid(lon, lat)
    lon, lat = lon.flatten(), lat.flatten()
    idy, idx = idy.flatten(), idx.flatten()
    scale = 30
    elevation = np.sin(scale * np.deg2rad(idy) + scale * np.deg2rad(idx))
    meta = pd.DataFrame({'latitude': lat, 'longitude': lon,
                         'elevation': elevation})

    fp_temp = os.path.join(td, 'elevation.h5')
    with Outputs(fp_temp, mode='w') as out:
        out.meta = meta

    return fp_temp


@pytest.mark.parametrize('s_enhance', [1, 2])
def test_topo_extraction_h5(s_enhance, plot=False):
    """Test the spatial enhancement of a test grid and then the lookup of the
    elevation data to a reference WTK file (also the same file for the test)"""
    with tempfile.TemporaryDirectory() as td:
        fp_exo_topo = make_topo_file(FP_WTK, td)

        te = TopoExtractH5(FP_WTK, fp_exo_topo, s_enhance=s_enhance,
                           t_enhance=1, t_agg_factor=1,
                           target=TARGET, shape=SHAPE)

        hr_elev = te.data

        lat = te.hr_lat_lon[..., 0].flatten()
        lon = te.hr_lat_lon[..., 1].flatten()
        hr_wtk_meta = np.vstack((lat, lon)).T
        hr_wtk_ind = np.arange(len(lat)).reshape(te.hr_shape[:-1])
        assert te.nn.max() == len(hr_wtk_meta)

        for gid in np.random.choice(len(hr_wtk_meta), 50, replace=False):
            idy, idx = np.where(hr_wtk_ind == gid)
            iloc = np.where(te.nn == gid)[0]
            exo_coords = te.source_lat_lon[iloc]

            # make sure all mapped high-res exo coordinates are closest to gid
            # pylint: disable=consider-using-enumerate
            for i in range(len(exo_coords)):
                dist = hr_wtk_meta - exo_coords[i]
                dist = np.hypot(dist[:, 0], dist[:, 1])
                assert np.argmin(dist) == gid

            # make sure the mean elevation makes sense
            test_out = hr_elev[idy, idx, 0, 0]
            true_out = te.source_data[iloc].mean()
            assert np.allclose(test_out, true_out)

        shutil.rmtree('./exo_cache/', ignore_errors=True)

        if plot:
            a = plt.scatter(te.source_lat_lon[:, 1], te.source_lat_lon[:, 0],
                            c=te.source_data, marker='s', s=5)
            plt.colorbar(a)
            plt.savefig(f'./source_elevation_{s_enhance}.png')
            plt.close()

            a = plt.imshow(hr_elev[:, :, 0, 0])
            plt.colorbar(a)
            plt.savefig(f'./hr_elev_{s_enhance}.png')
            plt.close()


def test_bad_s_enhance(s_enhance=10):
    """Test a large s_enhance factor that results in a bad mapping with
    enhanced grid pixels not having source exo data points"""
    with tempfile.TemporaryDirectory() as td:
        fp_exo_topo = make_topo_file(FP_WTK, td)

        with pytest.warns(UserWarning) as warnings:
            te = TopoExtractH5(FP_WTK, fp_exo_topo, s_enhance=s_enhance,
                               t_enhance=1, t_agg_factor=1,
                               target=TARGET, shape=SHAPE,
                               cache_data=False)
            _ = te.data

    good = ['target pixels did not have unique' in str(w.message)
            for w in warnings.list]
    assert any(good)


def test_topo_extraction_nc():
    """Test the spatial enhancement of a test grid and then the lookup of the
    elevation data to a reference WRF file (also the same file for the test)

    We already test proper topo mapping and aggregation in the h5 test so this
    just makes sure that the data can be extracted from a WRF file.
    """
    te = TopoExtractNC(FP_WRF, FP_WRF, s_enhance=1, t_enhance=1,
                       t_agg_factor=1, target=None, shape=None)
    hr_elev = te.data
    assert np.allclose(te.source_data.flatten(), hr_elev.flatten())
