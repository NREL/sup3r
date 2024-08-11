"""Test caching by :class:`Deriver` objects"""

import os
import tempfile

import numpy as np
import pytest

from sup3r.preprocessing import Cacher, DataHandler, Loader
from sup3r.utilities.pytest.helpers import make_fake_dset

target = (39.01, -105.15)
shape = (20, 20)
features = ['windspeed_100m', 'winddirection_100m']


def test_cacher_attrs():
    """Make sure attributes are preserved in cached data."""
    with tempfile.TemporaryDirectory() as td:
        nc = make_fake_dset(shape=(10, 10, 10), features=['windspeed_100m'])
        nc['windspeed_100m'].attrs = {'attrs': 'test'}

        cache_pattern = os.path.join(td, 'cached_{feature}.nc')
        Cacher(data=nc, cache_kwargs={'cache_pattern': cache_pattern})

        out = Loader(cache_pattern.format(feature='windspeed_100m'))
        assert out.data['windspeed_100m'].attrs == {'attrs': 'test'}


@pytest.mark.parametrize(
    ['input_files', 'derive_features', 'ext', 'shape', 'target'],
    [
        (
            pytest.FP_WTK,
            ['u_100m', 'v_100m', 'sza'],
            'h5',
            (20, 20),
            (39.01, -105.15),
        ),
        (
            pytest.FP_ERA,
            ['windspeed_100m', 'winddirection_100m'],
            'nc',
            (10, 10),
            (37.25, -107),
        ),
    ],
)
def test_derived_data_caching(
    input_files, derive_features, ext, shape, target
):
    """Test feature derivation followed by caching/loading"""

    with tempfile.TemporaryDirectory() as td:
        cache_pattern = os.path.join(td, 'cached_{feature}.' + ext)
        deriver = DataHandler(
            file_paths=input_files,
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

        loader = DataHandler(cacher.out_files, features=derive_features)
        assert np.array_equal(
            loader.as_array().compute(), deriver.as_array().compute()
        )


@pytest.mark.parametrize(
    ['input_files', 'derive_features', 'ext', 'shape', 'target'],
    [
        (
            pytest.FP_WTK,
            ['u_100m', 'v_100m'],
            'h5',
            (20, 20),
            (39.01, -105.15),
        ),
        (
            pytest.FP_ERA,
            ['windspeed_100m', 'winddirection_100m'],
            'nc',
            (10, 10),
            (37.25, -107),
        ),
    ],
)
def test_caching_with_dh_loading(
    input_files, derive_features, ext, shape, target
):
    """Test feature derivation followed by caching/loading"""

    with tempfile.TemporaryDirectory() as td:
        cache_pattern = os.path.join(td, 'cached_{feature}.' + ext)
        deriver = DataHandler(
            file_paths=input_files,
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

        loader = DataHandler(cacher.out_files, features=derive_features)
        assert np.array_equal(
            loader.as_array().compute(), deriver.as_array().compute()
        )
