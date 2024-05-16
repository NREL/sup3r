# -*- coding: utf-8 -*-
"""pytests for data handling"""

import os
import tempfile
from glob import glob

import numpy as np
import pytest
from rex import init_logger

from sup3r import TEST_DATA_DIR
from sup3r.containers.loaders import LoaderH5, LoaderNC
from sup3r.containers.wranglers import WranglerH5, WranglerNC

h5_files = [
    os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5'),
    os.path.join(TEST_DATA_DIR, 'test_wtk_co_2013.h5'),
]
nc_files = [os.path.join(TEST_DATA_DIR, 'test_era5_co_2012.nc')]

target = (39.01, -105.15)
shape = (20, 20)
kwargs = {
    'target': target,
    'shape': shape,
    'max_delta': 20,
    'time_slice': slice(None, None, 1),
}
features = ['windspeed_100m', 'winddirection_100m']

init_logger('sup3r', log_level='DEBUG')


def test_raster_index_caching():
    """Test raster index caching by saving file and then loading"""

    # saving raster file
    with tempfile.TemporaryDirectory() as td, LoaderH5(
        h5_files[0], features
    ) as loader:
        raster_file = os.path.join(td, 'raster.txt')
        wrangler = WranglerH5(
            loader, features, raster_file=raster_file, **kwargs
        )
        # loading raster file
        wrangler = WranglerH5(loader, features, raster_file=raster_file)
        assert np.allclose(wrangler.target, target, atol=1)
        assert wrangler.data.shape == (
            shape[0],
            shape[1],
            wrangler.data.shape[2],
            len(features),
        )
        assert wrangler.shape[:2] == (shape[0], shape[1])


@pytest.mark.parametrize(
    ['input_files', 'Loader', 'Wrangler', 'ext'],
    [
        (h5_files, LoaderH5, WranglerH5, 'h5'),
        (nc_files, LoaderNC, WranglerNC, 'nc'),
    ],
)
def test_data_caching(input_files, Loader, Wrangler, ext):
    """Test data extraction with caching/loading"""

    with tempfile.TemporaryDirectory() as td:
        cache_pattern = os.path.join(td, 'cached_{feature}.' + ext)
        with Loader(input_files[0], features) as loader:
            wrangler = Wrangler(
                loader,
                features,
                cache_kwargs={'cache_pattern': cache_pattern},
                **kwargs,
            )

        assert wrangler.data.shape == (
            shape[0],
            shape[1],
            wrangler.data.shape[2],
            len(features),
        )
        assert wrangler.data.dtype == np.dtype(np.float32)

        loader = Loader(glob(cache_pattern.format(feature='*')), features)

        assert np.array_equal(loader.data, wrangler.data)


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
