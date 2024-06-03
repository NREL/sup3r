"""Methods used across container objects."""

import logging
import pprint
from abc import ABCMeta
from enum import Enum
from inspect import getfullargspec
from typing import ClassVar, Tuple
from warnings import warn

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class Dimension(str, Enum):
    """Dimension names used for DatasetWrapper."""

    FLATTENED_SPATIAL = 'space'
    SOUTH_NORTH = 'south_north'
    WEST_EAST = 'west_east'
    TIME = 'time'
    PRESSURE_LEVEL = 'level'
    VARIABLE = 'variable'
    LATITUDE = 'latitude'
    LONGITUDE = 'longitude'

    def __str__(self):
        return self.value

    @classmethod
    def order(cls):
        """Return standard dimension order."""
        return (
            cls.FLATTENED_SPATIAL,
            cls.SOUTH_NORTH,
            cls.WEST_EAST,
            cls.TIME,
            cls.PRESSURE_LEVEL,
            cls.VARIABLE,
        )

    @classmethod
    def spatial_2d(cls):
        """Return ordered tuple for 2d spatial coordinates."""
        return (cls.SOUTH_NORTH, cls.WEST_EAST)


class FactoryMeta(ABCMeta, type):
    """Meta class to define __name__ attribute of factory generated classes."""

    def __new__(cls, name, bases, namespace, **kwargs):
        """Define __name__"""
        name = namespace.get('__name__', name)
        return super().__new__(cls, name, bases, namespace, **kwargs)


def _log_args(thing, func, *args, **kwargs):
    """Log annotated attributes and args."""

    ann_dict = {
        name: getattr(thing, name)
        for name, val in thing.__annotations__.items()
        if val is not ClassVar
    }
    arg_spec = getfullargspec(func)
    args = args or []
    defaults = arg_spec.defaults or []
    arg_names = arg_spec.args[1 : len(args) + 1]
    kwargs_names = arg_spec.args[-len(defaults) :]
    args_dict = dict(zip(kwargs_names, defaults))
    args_dict.update(dict(zip(arg_names, args)))
    args_dict.update(kwargs)
    args_dict.update(ann_dict)

    name = (
        thing.__name__
        if hasattr(thing, '__name__')
        else thing.__class__.__name__
    )
    logger.info(
        f'Initialized {name} with:\n' f'{pprint.pformat(args_dict, indent=2)}'
    )


def log_args(func):
    """Decorator to log annotations and args."""

    def wrapper(self, *args, **kwargs):
        _log_args(self, func, *args, **kwargs)
        return func(self, *args, **kwargs)

    return wrapper


def _contains_ellipsis(vals):
    return vals is Ellipsis or (
        isinstance(vals, (tuple, list)) and any(v is Ellipsis for v in vals)
    )


def _is_strings(vals):
    return isinstance(vals, str) or (
        isinstance(vals, (tuple, list)) and all(isinstance(v, str) for v in vals)
    )


def _is_ints(vals):
    return isinstance(vals, int) or (
        isinstance(vals, (list, tuple, np.ndarray))
        and all(isinstance(v, int) for v in vals)
    )


def lowered(features):
    """Return a lower case version of the given str or list of strings. Used to
    standardize storage and lookup of features."""

    feats = (
        features.lower()
        if isinstance(features, str)
        else [f.lower() for f in features]
        if isinstance(features, list)
        and all(isinstance(f, str) for f in features)
        else features
    )
    if _is_strings(features) and features != feats:
        msg = (
            f'Received some upper case features: {features}. '
            f'Using {feats} instead.'
        )
        logger.warning(msg)
        warn(msg)
    return feats


def ordered_dims(dims: Tuple):
    """Return the order of dims that follows the ordering of Dimension.order()
    for the common dim names. e.g dims = ('time', 'south_north', 'dummy',
    'west_east') will return ('south_north', 'west_east', 'time',
    'dummy')."""
    standard = [dim for dim in Dimension.order() if dim in dims]
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
    longitude, time, ...) ordering consistent with the order of
    `Dimension.order()`"""

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
    of Dimension.order() with the same len as arr.shape. This is used to set
    xr.Dataset entries. e.g. dset[var] = (dims, array)"""
    if len(arr.shape) > 1:
        arr = (Dimension.order()[1 : len(arr.shape) + 1], arr)
    return arr


def all_dtype(keys, type):
    """Check if all elements are the given type. Used to parse keys
    requested from :class:`Container` and :class:`Data`"""
    keys = keys if isinstance(keys, list) else [keys]
    return all(isinstance(key, type) for key in keys)
