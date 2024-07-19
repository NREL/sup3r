"""Ensure correct functions of :class:`Cacher` objects"""

import os
import tempfile

import numpy as np
import pytest

from sup3r.preprocessing import Cacher, Extracter, Loader

target = (39.01, -105.15)
shape = (20, 20)
features = ['windspeed_100m', 'winddirection_100m']


def test_raster_index_caching():
    """Test raster index caching by saving file and then loading"""

    # saving raster file
    with tempfile.TemporaryDirectory() as td:
        raster_file = os.path.join(td, 'raster.txt')
        extracter = Extracter(
            pytest.FP_WTK, raster_file=raster_file, target=target, shape=shape
        )
        # loading raster file
        extracter = Extracter(pytest.FP_WTK, raster_file=raster_file)
    assert np.allclose(extracter.target, target, atol=1)
    assert extracter.shape[:3] == (shape[0], shape[1], extracter.shape[2])


@pytest.mark.parametrize(
    ['input_files', 'ext', 'shape', 'target', 'features'],
    [
        (
            pytest.FP_WTK,
            'h5',
            (20, 20),
            (39.01, -105.15),
            ['windspeed_100m', 'winddirection_100m'],
        ),
        (
            pytest.FP_ERA,
            'nc',
            (10, 10),
            (37.25, -107),
            ['u_100m', 'v_100m'],
        ),
    ],
)
def test_data_caching(input_files, ext, shape, target, features):
    """Test data extraction with caching/loading"""

    with tempfile.TemporaryDirectory() as td:
        cache_pattern = os.path.join(td, 'cached_{feature}.' + ext)
        extracter = Extracter(input_files, shape=shape, target=target)
        cacher = Cacher(
            extracter, cache_kwargs={'cache_pattern': cache_pattern}
        )

        assert extracter.shape[:3] == (shape[0], shape[1], extracter.shape[2])
        assert extracter.data.dtype == np.dtype(np.float32)
        loader = Loader(cacher.out_files)
        assert np.array_equal(
            loader.data[features, ...].compute(),
            extracter.data[features, ...].compute(),
        )

        # make sure full domain can be loaded with extracters
        extracter = Extracter(cacher.out_files)
        assert np.array_equal(
            loader.data[features, ...].compute(),
            extracter.data[features, ...].compute(),
        )
