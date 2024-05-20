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
    LoaderH5,
    LoaderNC,
    WranglerH5,
    WranglerNC,
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


@pytest.mark.parametrize(
    [
        'input_files',
        'Loader',
        'Wrangler',
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
            WranglerH5,
            ['windspeed_100m', 'winddirection_100m'],
            ['u_100m', 'v_100m'],
            'h5',
            (20, 20),
            (39.01, -105.15),
        ),
        (
            nc_files,
            LoaderNC,
            WranglerNC,
            ['u_100m', 'v_100m'],
            ['windspeed_100m', 'winddirection_100m'],
            'nc',
            (10, 10),
            (37.25, -107),
        ),
    ],
)
def test_wrangler_caching(
    input_files,
    Loader,
    Wrangler,
    extract_features,
    derive_features,
    ext,
    shape,
    target,
):
    """Test feature derivation followed by caching/loading"""

    with tempfile.TemporaryDirectory() as td:
        cache_pattern = os.path.join(td, 'cached_{feature}.' + ext)
        wrangler = Wrangler(
            Loader(input_files[0], extract_features),
            derive_features,
            shape=shape,
            target=target,
            cache_kwargs={'cache_pattern': cache_pattern},
        )

        assert wrangler.data.shape == (
            shape[0],
            shape[1],
            wrangler.data.shape[2],
            len(derive_features),
        )
        assert wrangler.data.dtype == np.dtype(np.float32)

        loader = Loader(
            [cache_pattern.format(feature=f) for f in derive_features],
            derive_features,
        )
        assert da.map_blocks(
            lambda x, y: x == y, loader.data, wrangler.data
        ).all()


if __name__ == '__main__':
    execute_pytest(__file__)
