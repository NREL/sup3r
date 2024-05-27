"""Tests for correct interactions with :class:`Data` - the xr.Dataset
wrapper."""


import numpy as np
from rex import init_logger

from sup3r.containers.abstract import Data
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


if __name__ == '__main__':
    execute_pytest(__file__)
