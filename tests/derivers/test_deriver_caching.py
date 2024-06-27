"""Test caching by :class:`Deriver` objects"""

import os
import tempfile

import numpy as np
import pytest

from sup3r.preprocessing import (
    Cacher,
    DataHandlerH5,
    DataHandlerNC,
    LoaderH5,
    LoaderNC,
)

target = (39.01, -105.15)
shape = (20, 20)
features = ['windspeed_100m', 'winddirection_100m']


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
            pytest.FPS_WTK,
            LoaderH5,
            DataHandlerH5,
            ['u_100m', 'v_100m'],
            'h5',
            (20, 20),
            (39.01, -105.15),
        ),
        (
            pytest.FP_ERA,
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
        assert np.array_equal(loader.as_array(), deriver.as_array())


@pytest.mark.parametrize(
    [
        'input_files',
        'Deriver',
        'derive_features',
        'ext',
        'shape',
        'target',
    ],
    [
        (
            pytest.FPS_WTK,
            DataHandlerH5,
            ['u_100m', 'v_100m'],
            'h5',
            (20, 20),
            (39.01, -105.15),
        ),
        (
            pytest.FP_ERA,
            DataHandlerNC,
            ['windspeed_100m', 'winddirection_100m'],
            'nc',
            (10, 10),
            (37.25, -107),
        ),
    ],
)
def test_caching_with_dh_loading(
    input_files,
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

        loader = Deriver(cacher.out_files, features=derive_features)
        assert np.array_equal(loader.as_array(), deriver.as_array())
