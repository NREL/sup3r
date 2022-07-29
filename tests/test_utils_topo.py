# -*- coding: utf-8 -*-
"""pytests for topography utilities"""
import os
import numpy as np
import pytest
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

from sup3r import TEST_DATA_DIR
from sup3r.utilities.topo import TopoExtract


FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
TARGET = (39.001, -105.15)
SHAPE = (20, 20)


@pytest.mark.parametrize('agg_factor', [1, 4, 8])
def test_topo_extraction(agg_factor, plot=False):
    """Test the spatial enhancement of a test grid and then the lookup of the
    elevation data to a reference WTK file (also the same file for the test)"""
    te = TopoExtract(FP_WTK, FP_WTK, s_enhance=2, agg_factor=agg_factor,
                     target=TARGET, shape=SHAPE)
    hr_elev = te.hr_elev

    tree = KDTree(te.source_lat_lon)

    # bottom left
    _, i = tree.query(TARGET, k=agg_factor)
    elev = te.source_elevation[i].mean()
    assert np.allclose(elev, hr_elev[-1, 0])

    # top right
    _, i = tree.query((39.35, -105.2), k=agg_factor)
    elev = te.source_elevation[i].mean()
    assert np.allclose(elev, hr_elev[0, 0])

    for idy in range(10, 20):
        for idx in range(10, 20):
            lat, lon = te.hr_lat_lon[idy, idx, :]
            _, i = tree.query((lat, lon), k=agg_factor)
            elev = te.source_elevation[i].mean()
            assert np.allclose(elev, hr_elev[idy, idx])

    if plot:
        a = plt.scatter(te.source_lat_lon[:, 1], te.source_lat_lon[:, 0],
                        c=te.source_elevation, marker='s', s=150)
        plt.colorbar(a)
        plt.savefig('./source_elevation.png')
        plt.close()

        a = plt.imshow(hr_elev)
        plt.colorbar(a)
        plt.savefig('./hr_elev.png')
        plt.close()
