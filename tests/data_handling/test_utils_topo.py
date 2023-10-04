# -*- coding: utf-8 -*-
"""pytests for topography utilities"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.spatial import KDTree

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


@pytest.mark.parametrize('agg_factor', [1, 4, 8])
def test_topo_extraction_h5(agg_factor, plot=False):
    """Test the spatial enhancement of a test grid and then the lookup of the
    elevation data to a reference WTK file (also the same file for the test)"""
    te = TopoExtractH5(FP_WTK, FP_WTK, s_enhance=2, t_enhance=1,
                       t_agg_factor=1, s_agg_factor=agg_factor,
                       target=TARGET, shape=SHAPE)
    hr_elev = te.data

    tree = KDTree(te.source_lat_lon)

    # bottom left
    _, i = tree.query(TARGET, k=agg_factor)
    elev = te.source_data[i].mean()
    assert np.allclose(elev, hr_elev[-1, 0])

    # top right
    _, i = tree.query((39.35, -105.2), k=agg_factor)
    elev = te.source_data[i].mean()
    assert np.allclose(elev, hr_elev[0, 0])

    for idy in range(10, 20):
        for idx in range(10, 20):
            lat, lon = te.hr_lat_lon[idy, idx, :]
            _, i = tree.query((lat, lon), k=agg_factor)
            elev = te.source_data[i].mean()
            assert np.allclose(elev, hr_elev[idy, idx])

    if plot:
        a = plt.scatter(te.source_lat_lon[:, 1], te.source_lat_lon[:, 0],
                        c=te.source_data, marker='s', s=150)
        plt.colorbar(a)
        plt.savefig('./source_elevation.png')
        plt.close()

        a = plt.imshow(hr_elev)
        plt.colorbar(a)
        plt.savefig('./hr_elev.png')
        plt.close()


@pytest.mark.parametrize('agg_factor', [1, 4, 8])
def test_topo_extraction_nc(agg_factor, plot=False):
    """Test the spatial enhancement of a test grid and then the lookup of the
    elevation data to a reference WRF file (also the same file for the test)"""
    te = TopoExtractNC(FP_WRF, FP_WRF, s_enhance=2, t_enhance=1,
                       s_agg_factor=agg_factor, t_agg_factor=1,
                       target=WRF_TARGET, shape=WRF_SHAPE)
    hr_elev = te.data

    tree = KDTree(te.source_lat_lon)

    # bottom left
    _, i = tree.query(WRF_TARGET, k=agg_factor)
    elev = te.source_data[i].mean()
    assert np.allclose(elev, hr_elev[-1, 0])

    # top right
    _, i = tree.query((19.4, -123.6), k=agg_factor)
    elev = te.source_data[i].mean()
    assert np.allclose(elev, hr_elev[0, 0])

    for idy in range(4, 8):
        for idx in range(4, 8):
            lat, lon = te.hr_lat_lon[idy, idx, :]
            _, i = tree.query((lat, lon), k=agg_factor)
            elev = te.source_data[i].mean()
            assert np.allclose(elev, hr_elev[idy, idx])

    if plot:
        a = plt.scatter(te.source_lat_lon[:, 1], te.source_lat_lon[:, 0],
                        c=te.source_data, marker='s', s=150)
        plt.colorbar(a)
        plt.savefig('./source_elevation.png')
        plt.close()

        a = plt.imshow(hr_elev)
        plt.colorbar(a)
        plt.savefig('./hr_elev.png')
        plt.close()
