# -*- coding: utf-8 -*-
"""pytests for data handling"""

import os

import dask.array as da
import numpy as np
import pytest
from rex import init_logger

from sup3r import TEST_DATA_DIR
from sup3r.containers.derivers import Deriver, DeriverNC
from sup3r.containers.extracters import ExtracterH5, ExtracterNC
from sup3r.containers.loaders import LoaderH5, LoaderNC
from sup3r.utilities.interpolation import Interpolator
from sup3r.utilities.utilities import (
    spatial_coarsening,
    transform_rotate_wind,
)

h5_files = [
    os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5'),
    os.path.join(TEST_DATA_DIR, 'test_wtk_co_2013.h5'),
]
nc_files = [os.path.join(TEST_DATA_DIR, 'test_era5_co_2012.nc')]

features = ['windspeed_100m', 'winddirection_100m']

init_logger('sup3r', log_level='DEBUG')


def _height_interp(u, orog, zg):
    hgt_array = zg - orog
    u_100m = Interpolator.interp_to_level(
        np.transpose(u, axes=(3, 0, 1, 2)),
        np.transpose(hgt_array, axes=(3, 0, 1, 2)),
        levels=[100],
    )[..., None]
    return np.transpose(u_100m, axes=(1, 2, 0, 3))


def height_interp(container):
    """Interpolate u to u_100m."""
    return _height_interp(container['u'], container['orog'], container['zg'])


def coarse_transform(container):
    """Corasen high res wrangled data."""
    data = spatial_coarsening(container.data, s_enhance=2, obs_axis=False)
    container._lat_lon = spatial_coarsening(
        container.lat_lon, s_enhance=2, obs_axis=False
    )
    return data


@pytest.mark.parametrize(
    ['input_files', 'Loader', 'Extracter', 'Deriver', 'shape', 'target'],
    [
        (nc_files, LoaderNC, ExtracterNC, DeriverNC, (10, 10), (37.25, -107)),
    ],
)
def test_height_interp_nc(
    input_files, Loader, Extracter, Deriver, shape, target
):
    """Test that variables can be interpolated with height correctly"""

    extract_features = ['U_100m']
    raw_features = ['orog', 'zg', 'u']
    no_transform = Extracter(
        Loader(input_files[0], features=raw_features),
        raw_features,
        target=target,
        shape=shape,
    )
    transform = Deriver(
        Extracter(
            Loader(input_files[0], features=raw_features),
            target=target,
            shape=shape,
        ),
        extract_features,
    )

    out = _height_interp(
        orog=no_transform['orog'],
        zg=no_transform['zg'],
        u=no_transform['u'],
    )
    assert da.map_blocks(lambda x, y: x == y, out, transform.data).all()


@pytest.mark.parametrize(
    ['input_files', 'Loader', 'Extracter', 'shape', 'target'],
    [
        (h5_files, LoaderH5, ExtracterH5, (20, 20), (39.01, -105.15)),
        (nc_files, LoaderNC, ExtracterNC, (10, 10), (37.25, -107)),
    ],
)
def test_uv_transform(input_files, Loader, Extracter, shape, target):
    """Test that ws/wd -> u/v transform is done correctly."""

    derive_features = ['U_100m', 'V_100m']
    raw_features = ['windspeed_100m', 'winddirection_100m']
    extracter = Extracter(
        Loader(input_files[0], features=raw_features),
        target=target,
        shape=shape,
    )
    deriver = Deriver(
        extracter, features=derive_features
    )
    u, v = transform_rotate_wind(
        extracter['windspeed_100m'],
        extracter['winddirection_100m'],
        extracter['lat_lon'],
    )
    assert da.map_blocks(lambda x, y: x == y, u, deriver['U_100m']).all()
    assert da.map_blocks(lambda x, y: x == y, v, deriver['V_100m']).all()
    deriver.close()
    extracter.close()


@pytest.mark.parametrize(
    ['input_files', 'Loader', 'Extracter', 'shape', 'target'],
    [
        (h5_files, LoaderH5, ExtracterH5, (20, 20), (39.01, -105.15)),
        (nc_files, LoaderNC, ExtracterNC, (10, 10), (37.25, -107)),
    ],
)
def test_hr_coarsening(input_files, Loader, Extracter, shape, target):
    """Test spatial coarsening of the high res field"""

    features = ['windspeed_100m', 'winddirection_100m']
    extracter = Extracter(
        Loader(input_files[0], features=features),
        target=target,
        shape=shape,
    )
    deriver = Deriver(extracter, features=features, transform=coarse_transform)
    assert deriver.data.shape == (
        shape[0] // 2,
        shape[1] // 2,
        deriver.data.shape[2],
        len(features),
    )
    assert extracter.lat_lon.shape == (shape[0] // 2, shape[1] // 2, 2)
    assert deriver.data.dtype == np.dtype(np.float32)


def execute_pytest(capture='all', flags='-rapP'):
    """Execute module as pytest with detailed summary report.

    Parameters
    ----------
    capture : str
        Log or stdout/stderr capture option. ex: log (only logger),
        all (includes stdout/stderr)
    flags : str
        Which tests to show logs and results for.
    """

    fname = os.path.basename(__file__)
    pytest.main(['-q', '--show-capture={}'.format(capture), fname, flags])


if __name__ == '__main__':
    execute_pytest()
