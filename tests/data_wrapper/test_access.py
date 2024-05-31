"""Tests for correct interactions with :class:`Data` - the xr.Dataset
wrapper."""

import numpy as np
import pytest
from rex import init_logger

from sup3r.preprocessing.abstract import Data, DataGroup
from sup3r.utilities.pytest.helpers import (
    execute_pytest,
    make_fake_dset,
)

init_logger('sup3r', log_level='DEBUG')


def test_correct_access():
    """Make sure Data wrapper _getitem__ method works correctly."""
    nc = make_fake_dset((20, 20, 100, 3), features=['u', 'v'])
    data = Data(nc)

    _ = data['u']
    _ = data[['u', 'v']]
    out = data[['latitude', 'longitude']]
    assert out.shape == (20, 20, 2)
    assert np.array_equal(out, data.lat_lon)
    assert len(data.time_index) == 100
    out = data.isel(time=slice(0, 10))
    assert out.to_array().shape == (20, 20, 10, 3, 2)
    assert isinstance(out, Data)
    assert hasattr(out, 'time_index')
    out = data[['u', 'v'], slice(0, 10)]
    assert out.shape == (10, 20, 100, 3, 2)
    out = data[['u', 'v'], slice(0, 10), ..., slice(0, 1)]
    assert out.shape == (10, 20, 100, 1, 2)
    out = data[..., 0]
    assert out.shape == (20, 20, 100, 3)
    assert np.array_equal(out, data['u'])
    assert np.array_equal(out, data['u', ...])
    assert np.array_equal(out, data[..., 'u'])
    assert np.array_equal(data[['v', 'u']], data[..., [1, 0]])


@pytest.mark.parametrize(
    'data',
    [
        (
            make_fake_dset((20, 20, 100, 3), features=['u', 'v']),
            make_fake_dset((20, 20, 100, 3), features=['u', 'v']),
        ),
        make_fake_dset((20, 20, 100, 3), features=['u', 'v']),
    ],
)
def test_correct_access_for_group(data):
    """Make sure DataGroup wrapper works correctly."""
    data = DataGroup(data)

    _ = data['u']
    _ = data[['u', 'v']]
    out = data[['latitude', 'longitude']]
    if data.n_members == 1:
        out = (out,)

    assert all(o.shape == (20, 20, 2) for o in out)
    assert all(np.array_equal(o, data.lat_lon) for o in out)
    assert len(data.time_index) == 100
    out = data.isel(time=slice(0, 10))
    if data.n_members == 1:
        out = (out,)
    assert (o.to_array().shape == (20, 20, 10, 3, 2) for o in out)
    assert all(isinstance(o, Data) for o in out)
    assert all(hasattr(o, 'time_index') for o in out)
    out = data[['u', 'v'], slice(0, 10)]
    if data.n_members == 1:
        out = (out,)
    assert all(o.shape == (10, 20, 100, 3, 2) for o in out)
    out = data[['u', 'v'], slice(0, 10), ..., slice(0, 1)]
    if data.n_members == 1:
        out = (out,)
    assert all(o.shape == (10, 20, 100, 1, 2) for o in out)
    out = data[..., 0]
    if data.n_members == 1:
        assert out.shape == (20, 20, 100, 3)
    else:
        assert all(o.shape == (20, 20, 100, 3) for o in out)

    assert all(np.array_equal(o, d) for o, d in zip(out, data['u']))
    assert all(np.array_equal(o, d) for o, d in zip(out, data['u', ...]))
    assert all(np.array_equal(o, d) for o, d in zip(out, data[..., 'u']))
    assert all(
        np.array_equal(d0, d1)
        for d0, d1 in zip(data[['v', 'u']], data[..., [1, 0]])
    )


if __name__ == '__main__':
    execute_pytest(__file__)
