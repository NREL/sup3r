"""Tests for correct interactions with :class:`Data` - the xr.Dataset
accessor."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from sup3r.preprocessing import Dimension
from sup3r.preprocessing.accessor import Sup3rX
from sup3r.preprocessing.base import Sup3rDataset
from sup3r.utilities.pytest.helpers import (
    make_fake_dset,
)
from sup3r.utilities.utilities import RANDOM_GENERATOR


def test_suffled_dim_order():
    """Make sure when we get arrays from Sup3rX object they come back in
    standard (lats, lons, time, features) order, regardless of internal
    ordering."""

    shape_2d = (2, 2)
    times = 2
    values = RANDOM_GENERATOR.uniform(0, 1, (*shape_2d, times, 3)).astype(
        np.float32
    )
    lats = RANDOM_GENERATOR.uniform(0, 1, shape_2d).astype(np.float32)
    lons = RANDOM_GENERATOR.uniform(0, 1, shape_2d).astype(np.float32)
    time = np.arange(times)
    dim_order = ('south_north', 'west_east', 'time')

    feats = ['u', 'v', 'temp']
    data_vars = {
        f: (dim_order[::-1], values[..., i].transpose(2, 1, 0))
        for i, f in enumerate(feats)
    }
    nc = xr.Dataset(
        coords={
            'latitude': (dim_order[:-1][::-1], lats),
            'longitude': (dim_order[:-1][::-1], lons),
            'time': time,
        },
        data_vars=data_vars,
    )
    snc = Sup3rX(nc)

    assert np.array_equal(snc[feats, ...], values)


@pytest.mark.parametrize(
    'data',
    (
        Sup3rX(make_fake_dset((20, 20, 100, 3), features=['u', 'v'])),
        Sup3rDataset(
            single_member=make_fake_dset((20, 20, 100, 3), features=['u', 'v'])
        ),
    ),
)
def test_correct_single_member_access(data):
    """Make sure _getitem__ methods work correctly for Sup3rX accessor and
    Sup3rDataset wrapper around single xr.Dataset"""
    nc = make_fake_dset((20, 20, 100, 3), features=['u', 'v'])
    data = nc.sx

    _ = data['u']
    _ = data[['u', 'v']]
    out = data[[Dimension.LATITUDE, Dimension.LONGITUDE], :]
    assert ['u', 'v'] in data
    assert out.shape == (20, 20, 2)
    assert np.array_equal(out.compute(), data.lat_lon.compute())
    assert len(data.time_index) == 100
    out = data.isel(time=slice(0, 10))
    assert out.sx.as_array().shape == (20, 20, 10, 3, 2)
    assert hasattr(out.sx, 'time_index')
    out = data[['u', 'v'], slice(0, 10)]
    assert out.shape == (10, 20, 100, 3, 2)
    out = data[['u', 'v'], [0, 1], [2, 3], ..., slice(0, 10)]
    assert out.shape == (2, 2, 100, 3, 2)
    out = data[['u', 'v'], slice(0, 10), ..., slice(0, 1)]
    assert out.shape == (10, 20, 100, 1, 2)
    out = data.as_array()[..., 0]
    assert out.shape == (20, 20, 100, 3)
    assert np.array_equal(out.compute(), data['u', ...].compute())
    data.compute()
    assert data.loaded


def test_correct_multi_member_access():
    """Make sure Data object works correctly."""
    data = Sup3rDataset(
        (
            Sup3rX(make_fake_dset((20, 20, 100, 3), features=['u', 'v'])),
            Sup3rX(make_fake_dset((20, 20, 100, 3), features=['u', 'v'])),
        )
    )

    _ = data['u']
    _ = data[['u', 'v']]
    out = data[[Dimension.LATITUDE, Dimension.LONGITUDE], :]
    lat_lon = data.lat_lon
    time_index = data.time_index
    assert all(o.shape == (20, 20, 2) for o in out)
    assert all(
        np.array_equal(o.compute(), ll.compute())
        for o, ll in zip(out, lat_lon)
    )
    assert all(len(ti) == 100 for ti in time_index)
    out = data.isel(time=slice(0, 10))
    assert (o.as_array().shape == (20, 20, 10, 3, 2) for o in out)
    assert all(hasattr(o.sx, 'time_index') for o in out)
    out = data[['u', 'v'], slice(0, 10)]
    assert all(o.shape == (10, 20, 100, 3, 2) for o in out)
    out = data[['u', 'v'], slice(0, 10), ..., slice(0, 1)]
    assert all(o.shape == (10, 20, 100, 1, 2) for o in out)
    out = data[
        (
            (['u', 'v'], slice(0, 10), slice(0, 10), slice(0, 5)),
            (['u', 'v'], slice(0, 20), slice(0, 20), slice(0, 10)),
        )
    ]
    assert out[0].shape == (10, 10, 5, 3, 2)
    assert out[1].shape == (20, 20, 10, 3, 2)
    data.compute()
    assert data.loaded


def test_change_values():
    """Test that we can change values in the Data object."""
    data = make_fake_dset((20, 20, 100, 3), features=['u', 'v'])
    data = Sup3rDataset(high_res=data)

    rand_u = RANDOM_GENERATOR.uniform(0, 20, data['u', ...].shape)
    data['u'] = rand_u
    assert np.array_equal(rand_u, data['u', ...].compute())

    rand_v = RANDOM_GENERATOR.uniform(0, 10, data['v', ...].shape)
    data['v'] = rand_v
    assert np.array_equal(rand_v, data['v', ...])

    data[['u', 'v']] = da.stack([rand_u, rand_v], axis=-1)
    assert np.array_equal(
        data[['u', 'v']].as_array().data.compute(),
        da.stack([rand_u, rand_v], axis=-1).compute(),
    )
    data['u', slice(0, 10)] = 0
    assert np.allclose(data['u', ...][slice(0, 10)], [0])
