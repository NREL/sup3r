# -*- coding: utf-8 -*-
"""pytests for data handling"""

import os
import tempfile

import numpy as np
import pytest
from rex import init_logger

from sup3r import TEST_DATA_DIR
from sup3r.preprocessing import (
    Cacher,
    DataHandlerH5,
    DataHandlerNC,
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


@pytest.mark.parametrize(
    [
        'input_files',
        'Loader',
        'Deriver',
        'derive_features',
        'ext',
        'shape',
        'target',
    ],
    [
        (
            h5_files,
            LoaderH5,
            DataHandlerH5,
            ['u_100m', 'v_100m'],
            'h5',
            (20, 20),
            (39.01, -105.15),
        ),
        (
            nc_files,
            LoaderNC,
            DataHandlerNC,
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
    Deriver,
    derive_features,
    ext,
    shape,
    target,
):
    """Test feature derivation followed by caching/loading"""

    with tempfile.TemporaryDirectory() as td:
        cache_pattern = os.path.join(td, 'cached_{feature}.' + ext)
        deriver = Deriver(
            file_paths=input_files[0],
            features=derive_features,
            shape=shape,
            target=target,
        )

        cacher = Cacher(
            deriver.data, cache_kwargs={'cache_pattern': cache_pattern}
        )

        assert deriver.shape[:3] == (shape[0], shape[1], deriver.shape[2])
        assert all(
            deriver[f].shape == (*shape, deriver.shape[2])
            for f in derive_features
        )
        assert deriver.data.dtype == np.dtype(np.float32)

        loader = Loader(cacher.out_files, features=derive_features)
        assert np.array_equal(loader.to_array(), deriver.to_array())


if __name__ == '__main__':
    execute_pytest(__file__)
