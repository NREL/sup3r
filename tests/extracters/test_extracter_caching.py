"""Ensure correct functions of :class:`Cacher` objects"""

import os
import tempfile

import dask.array as da
import numpy as np
import pytest
from rex import init_logger

from sup3r import TEST_DATA_DIR
from sup3r.preprocessing import (
    Cacher,
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
    with tempfile.TemporaryDirectory() as td:
        raster_file = os.path.join(td, 'raster.txt')
        extracter = ExtracterH5(
            h5_files[0], raster_file=raster_file, target=target, shape=shape
        )
        # loading raster file
        extracter = ExtracterH5(h5_files[0], raster_file=raster_file)
    assert np.allclose(extracter.target, target, atol=1)
    assert extracter.shape[:3] == (
        shape[0],
        shape[1],
        extracter.shape[2],
    )


@pytest.mark.parametrize(
    [
        'input_files',
        'Loader',
        'Extracter',
        'ext',
        'shape',
        'target',
        'features',
    ],
    [
        (
            h5_files,
            LoaderH5,
            ExtracterH5,
            'h5',
            (20, 20),
            (39.01, -105.15),
            ['windspeed_100m', 'winddirection_100m'],
        ),
        (
            nc_files,
            LoaderNC,
            ExtracterNC,
            'nc',
            (10, 10),
            (37.25, -107),
            ['u_100m', 'v_100m'],
        ),
    ],
)
def test_data_caching(
    input_files, Loader, Extracter, ext, shape, target, features
):
    """Test data extraction with caching/loading"""

    with tempfile.TemporaryDirectory() as td:
        cache_pattern = os.path.join(td, 'cached_{feature}.' + ext)
        extracter = Extracter(
            input_files[0],
            shape=shape,
            target=target,
        )
        cacher = Cacher(
            extracter, cache_kwargs={'cache_pattern': cache_pattern}
        )

        assert extracter.shape[:3] == (
            shape[0],
            shape[1],
            extracter.shape[2],
        )
        assert extracter.data.dtype == np.dtype(np.float32)
        loader = Loader(cacher.out_files)
        assert da.map_blocks(
            lambda x, y: x == y,
            loader[features],
            extracter[features],
        ).all()


if __name__ == '__main__':
    execute_pytest(__file__)
