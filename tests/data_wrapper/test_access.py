"""Tests for correct interactions with :class:`Data` - the xr.Dataset
accessor."""

import dask.array as da
import numpy as np
from rex import init_logger

from sup3r.preprocessing.base import DatasetTuple
from sup3r.preprocessing.common import Dimension
from sup3r.utilities.pytest.helpers import (
    execute_pytest,
    make_fake_dset,
)

init_logger('sup3r', log_level='DEBUG')


def test_correct_access_accessor():
    """Make sure accessor _getitem__ method works correctly."""
    nc = make_fake_dset((20, 20, 100, 3), features=['u', 'v'])
    data = nc.sx

    _ = data['u']
    _ = data[['u', 'v']]
    out = data[[Dimension.LATITUDE, Dimension.LONGITUDE], :]
    assert ['u', 'v'] in data
    assert out.shape == (20, 20, 2)
    assert np.array_equal(out, data.lat_lon)
    assert len(data.time_index) == 100
    out = data.isel(time=slice(0, 10))
    assert out.sx.as_array().shape == (20, 20, 10, 3, 2)
    assert hasattr(out.sx, 'time_index')
    out = data[['u', 'v'], slice(0, 10)]
    assert out.shape == (10, 20, 100, 3, 2)
    out = data[['u', 'v'], slice(0, 10), ..., slice(0, 1)]
    assert out.shape == (10, 20, 100, 1, 2)
    out = data.as_array()[..., 0]
    assert out.shape == (20, 20, 100, 3)
    assert np.array_equal(out, data['u'])
    assert np.array_equal(out, data['u', ...])
    assert np.array_equal(out, data[..., 'u'])
    assert np.array_equal(
        data[['v', 'u']].sx.to_dataarray(), data.as_array()[..., [1, 0]]
    )


def test_correct_access_single_member_data():
    """Make sure Data object works correctly."""
    data = DatasetTuple(make_fake_dset((20, 20, 100, 3), features=['u', 'v']))

    _ = data['u']
    _ = data[['u', 'v']]
    out = data[[Dimension.LATITUDE, Dimension.LONGITUDE], :]
    assert ['u', 'v'] in data
    assert out.shape == (20, 20, 2)
    assert np.array_equal(out, data.lat_lon)
    assert len(data.time_index) == 100
    out = data.isel(time=slice(0, 10))
    assert out.sx.as_array().shape == (20, 20, 10, 3, 2)
    assert hasattr(out.sx, 'time_index')
    out = data[['u', 'v'], slice(0, 10)]
    assert out.shape == (10, 20, 100, 3, 2)
    out = data[['u', 'v'], slice(0, 10), ..., slice(0, 1)]
    assert out.shape == (10, 20, 100, 1, 2)
    out = data.as_array()[..., 0]
    assert out.shape == (20, 20, 100, 3)
    assert np.array_equal(out, data['u'])
    assert np.array_equal(out, data['u', ...])
    assert np.array_equal(out, data[..., 'u'])
    assert np.array_equal(
        data.as_array(['v', 'u']), data.as_array()[..., [1, 0]]
    )


def test_correct_access_multi_member_data():
    """Make sure Data object works correctly."""
    data = DatasetTuple(
        (
            make_fake_dset((20, 20, 100, 3), features=['u', 'v']),
            make_fake_dset((20, 20, 100, 3), features=['u', 'v']),
        )
    )

    _ = data['u']
    _ = data[['u', 'v']]
    out = data[[Dimension.LATITUDE, Dimension.LONGITUDE], :]
    lat_lon = data.lat_lon
    time_index = data.time_index
    assert all(o.shape == (20, 20, 2) for o in out)
    assert all(np.array_equal(o, ll) for o, ll in zip(out, lat_lon))
    assert all(len(ti) == 100 for ti in time_index)
    out = data.isel(time=slice(0, 10))
    assert (o.as_array().shape == (20, 20, 10, 3, 2) for o in out)
    assert all(hasattr(o.sx, 'time_index') for o in out)
    out = data[['u', 'v'], slice(0, 10)]
    assert all(o.shape == (10, 20, 100, 3, 2) for o in out)
    out = data[['u', 'v'], slice(0, 10), ..., slice(0, 1)]
    assert all(o.shape == (10, 20, 100, 1, 2) for o in out)
    out = data[..., 0]
    assert all(o.shape == (20, 20, 100, 3) for o in out)
    assert all(np.array_equal(o, d) for o, d in zip(out, data['u']))
    assert all(np.array_equal(o, d) for o, d in zip(out, data['u', ...]))
    assert all(np.array_equal(o, d) for o, d in zip(out, data[..., 'u']))
    assert all(
        np.array_equal(da.moveaxis(d0.to_dataarray().data, 0, -1), d1)
        for d0, d1 in zip(data[['v', 'u']], data[..., [1, 0]])
    )
    out = data[
        (
            (slice(0, 10), slice(0, 10), slice(0, 5), ['u', 'v']),
            (slice(0, 20), slice(0, 20), slice(0, 10), ['u', 'v']),
        )
    ]
    assert out[0].shape == (10, 10, 5, 3, 2)
    assert out[1].shape == (20, 20, 10, 3, 2)


def test_change_values():
    """Test that we can change values in the Data object."""
    data = make_fake_dset((20, 20, 100, 3), features=['u', 'v'])
    data = DatasetTuple(data)

    rand_u = np.random.uniform(0, 20, data['u'].shape)
    data['u'] = rand_u
    assert np.array_equal(rand_u, data['u'])

    rand_v = np.random.uniform(0, 10, data['v'].shape)
    data['v'] = rand_v
    assert np.array_equal(rand_v, data['v'])

    data[['u', 'v']] = da.stack([rand_u, rand_v], axis=-1)
    assert np.array_equal(
        data[['u', 'v']].sx.to_dataarray(), da.stack([rand_u, rand_v], axis=-1)
    )


if __name__ == '__main__':
    execute_pytest(__file__)
