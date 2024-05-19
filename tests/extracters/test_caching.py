# -*- coding: utf-8 -*-
"""pytests for data handling"""

import os
import tempfile

import dask.array as da
import numpy as np
import pytest
from rex import init_logger

from sup3r import TEST_DATA_DIR
from sup3r.containers import (
    Cacher,
    DeriverH5,
    DeriverNC,
    ExtracterH5,
    ExtracterNC,
    LoaderH5,
    LoaderNC,
)
from sup3r.utilities.pytest.helpers import execute_pytest

h5_files = [
    os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5'),
    os.path.join(TEST_DATA_DIR, 'test_wtk_co_2013.h5'),
]
nc_files = [os.path.join(TEST_DATA_DIR, 'test_era5_co_2012.nc')]

target = (39.01, -105.15)
shape = (20, 20)
features = ['windspeed_100m', 'winddirection_100m']

init_logger('sup3r', log_level='DEBUG')


def test_raster_index_caching():
    """Test raster index caching by saving file and then loading"""

    # saving raster file
    with tempfile.TemporaryDirectory() as td, LoaderH5(
        h5_files[0], features
    ) as loader:
        raster_file = os.path.join(td, 'raster.txt')
        extracter = ExtracterH5(
            loader, raster_file=raster_file, target=target, shape=shape
        )
        # loading raster file
        extracter = ExtracterH5(loader, raster_file=raster_file)
        assert np.allclose(extracter.target, target, atol=1)
        assert extracter.data.shape == (
            shape[0],
            shape[1],
            extracter.data.shape[2],
            len(features),
        )
        assert extracter.shape[:2] == (shape[0], shape[1])


@pytest.mark.parametrize(
    ['input_files', 'Loader', 'Extracter', 'ext', 'shape', 'target'],
    [
        (h5_files, LoaderH5, ExtracterH5, 'h5', (20, 20), (39.01, -105.15)),
        (nc_files, LoaderNC, ExtracterNC, 'nc', (10, 10), (37.25, -107)),
    ],
)
def test_data_caching(input_files, Loader, Extracter, ext, shape, target):
    """Test data extraction with caching/loading"""

    extract_features = ['windspeed_100m', 'winddirection_100m']
    with tempfile.TemporaryDirectory() as td:
        cache_pattern = os.path.join(td, 'cached_{feature}.' + ext)
        extracter = Extracter(
            Loader(input_files[0], extract_features),
            shape=shape,
            target=target,
        )
        _ = Cacher(extracter, cache_kwargs={'cache_pattern': cache_pattern})

        assert extracter.data.shape == (
            shape[0],
            shape[1],
            extracter.data.shape[2],
            len(extract_features),
        )
        assert extracter.data.dtype == np.dtype(np.float32)

        loader = Loader(
            [cache_pattern.format(feature=f) for f in features], features
        )
        assert da.map_blocks(
            lambda x, y: x == y, loader.data, extracter.data
        ).all()


@pytest.mark.parametrize(
    [
        'input_files',
        'Loader',
        'Extracter',
        'Deriver',
        'extract_features',
        'derive_features',
        'ext',
        'shape',
        'target',
    ],
    [
        (
            h5_files,
            LoaderH5,
            ExtracterH5,
            DeriverH5,
            ['windspeed_100m', 'winddirection_100m'],
            ['u_100m', 'v_100m'],
            'h5',
            (20, 20),
            (39.01, -105.15),
        ),
        (
            nc_files,
            LoaderNC,
            ExtracterNC,
            DeriverNC,
            ['u_100m', 'v_100m'],
            ['windspeed_100m', 'winddirection_100m'],
            'nc',
            (10, 10),
            (37.25, -107),
        ),
    ],
)
def test_derived_data_caching(
    input_files,
    Loader,
    Extracter,
    Deriver,
    extract_features,
    derive_features,
    ext,
    shape,
    target,
):
    """Test feature derivation followed by caching/loading"""

    with tempfile.TemporaryDirectory() as td:
        cache_pattern = os.path.join(td, 'cached_{feature}.' + ext)
        extracter = Extracter(
            Loader(input_files[0], extract_features),
            shape=shape,
            target=target,
        )
        deriver = Deriver(extracter, derive_features)
        _ = Cacher(deriver, cache_kwargs={'cache_pattern': cache_pattern})

        assert deriver.data.shape == (
            shape[0],
            shape[1],
            deriver.data.shape[2],
            len(derive_features),
        )
        assert deriver.data.dtype == np.dtype(np.float32)

        loader = Loader(
            [cache_pattern.format(feature=f) for f in derive_features],
            derive_features,
        )
        assert da.map_blocks(
            lambda x, y: x == y, loader.data, deriver.data
        ).all()


if __name__ == '__main__':
    execute_pytest(__file__)
