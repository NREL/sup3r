"""Methods used across container objects."""

import logging
from typing import Tuple
from warnings import warn

import xarray as xr

logger = logging.getLogger(__name__)


DIM_ORDER = (
        'space',
        'south_north',
        'west_east',
        'time',
        'level',
        'variable',
    )


def lowered(features):
    """Return a lower case version of the given str or list of strings. Used to
    standardize storage and lookup of features."""

    feats = (
        features.lower()
        if isinstance(features, str)
        else [f.lower() for f in features]
    )
    if features != feats:
        msg = (
            f'Received some upper case features: {features}. '
            f'Using {feats} instead.'
        )
        logger.warning(msg)
        warn(msg)
    return feats


def ordered_dims(dims: Tuple):
    """Return the order of dims that follows the ordering of self.DIM_ORDER
    for the common dim names. e.g dims = ('time', 'south_north', 'dummy',
    'west_east') will return ('south_north', 'west_east', 'time',
    'dummy')."""
    standard = [dim for dim in DIM_ORDER if dim in dims]
    non_standard = [dim for dim in dims if dim not in standard]
    return tuple(standard + non_standard)


def ordered_array(data: xr.DataArray):
    """Transpose arrays so they have a (space, time, ...) or (space, time,
    ..., feature) ordering.

    Parameters
    ----------
    data : xr.DataArray
        xr.DataArray with `.dims` attribute listing all contained dimensions
    """
    return data.transpose(*ordered_dims(data.dims))


def enforce_standard_dim_order(dset: xr.Dataset):
    """Ensure that data dimensions have a (space, time, ...) or (latitude,
    longitude, time, ...) ordering consistent with the order of `DIM_ORDER`"""

    reordered_vars = {
        var: (
            ordered_dims(dset.data_vars[var].dims),
            ordered_array(dset.data_vars[var]).data,
        )
        for var in dset.data_vars
    }

    return xr.Dataset(coords=dset.coords, data_vars=reordered_vars)


def dims_array_tuple(arr):
    """Return a tuple of (dims, array) with dims equal to the ordered slice
    of DIM_ORDER with the same len as arr.shape. This is used to set xr.Dataset
    entries. e.g. dset[var] = (dims, array)"""
    if len(arr.shape) > 1:
        arr = (DIM_ORDER[1 : len(arr.shape) + 1], arr)
    return arr


def all_dtype(keys, type):
    """Check if all elements are the given type. Used to parse keys
    requested from :class:`Container` and :class:`Data`"""
    keys = keys if isinstance(keys, list) else [keys]
    return all(isinstance(key, type) for key in keys)
