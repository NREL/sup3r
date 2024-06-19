"""Methods used across container objects."""

import logging
import os
import pprint
from abc import ABCMeta
from enum import Enum
from glob import glob
from inspect import getfullargspec, signature
from pathlib import Path
from typing import ClassVar, Optional, Tuple
from warnings import warn

import numpy as np
import xarray as xr

import sup3r.preprocessing
from sup3r.typing import T_Dataset

logger = logging.getLogger(__name__)


class Dimension(str, Enum):
    """Dimension names used for Sup3rX accessor."""

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


def expand_paths(fps):
    """Expand path(s)

    Parameter
    ---------
    fps : str or pathlib.Path or any Sequence of those
        One or multiple paths to file

    Returns
    -------
    list[str]
        A list of expanded unique and sorted paths as str

    Examples
    --------
    >>> expand_paths("myfile.h5")

    >>> expand_paths(["myfile.h5", "*.hdf"])
    """
    if isinstance(fps, (str, Path)):
        fps = (fps,)

    out = []
    for f in fps:
        out.extend(glob(f))
    return sorted(set(out))


def get_source_type(file_paths):
    """Get data source type

    Parameters
    ----------
    file_paths : list | str
        One or more paths to data files, can include a unix-style pat*tern

    Returns
    -------
    source_type : str
        Either "h5" or "nc"
    """
    if file_paths is None:
        return None

    if isinstance(file_paths, str) and '*' in file_paths:
        temp = glob(file_paths)
        if any(temp):
            file_paths = temp

    if not isinstance(file_paths, list):
        file_paths = [file_paths]

    _, source_type = os.path.splitext(file_paths[0])

    if source_type == '.h5':
        return 'h5'
    return 'nc'


def get_input_handler_class(file_paths, input_handler_name):
    """Get the :class:`DataHandler` or :class:`Extracter` object.

    Parameters
    ----------
    file_paths : list | str
        A list of files to extract raster data from. Each file must have
        the same number of timesteps. Can also pass a string with a
        unix-style file path which will be passed through glob.glob
    input_handler_name : str
        data handler class to use for input data. Provide a string name to
        match a class in data_handling.py. If None the correct handler will
        be guessed based on file type and time series properties. The guessed
        handler will default to an extracter type (simple raster / time
        extraction from raw feature data, as opposed to derivation of new
        features)

    Returns
    -------
    HandlerClass : ExtracterH5 | ExtracterNC | DataHandlerH5 | DataHandlerNC
        DataHandler or Extracter class from sup3r.preprocessing.
    """

    HandlerClass = None

    input_type = get_source_type(file_paths)

    if input_handler_name is None:
        if input_type == 'nc':
            input_handler_name = 'DataHandlerNC'
        elif input_type == 'h5':
            input_handler_name = 'DataHandlerH5'

        logger.info(
            '"input_handler" arg was not provided. Using '
            f'"{input_handler_name}". If this is '
            'incorrect, please provide '
            'input_handler="DataHandlerName".'
        )

    if isinstance(input_handler_name, str):
        HandlerClass = getattr(sup3r.preprocessing, input_handler_name, None)

    if HandlerClass is None:
        msg = (
            'Could not find requested data handler class '
            f'"{input_handler_name}" in sup3r.preprocessing.'
        )
        logger.error(msg)
        raise KeyError(msg)

    return HandlerClass


def get_possible_class_args(Class):
    """Get all available arguments for given class by searching through the
    inheritance hierarchy."""
    class_args = list(signature(Class.__init__).parameters.keys())
    if Class.__bases__ == (object,):
        return class_args
    for base in Class.__bases__:
        class_args += get_possible_class_args(base)
    return set(class_args)


def _get_class_kwargs(Classes, kwargs):
    """Go through class and class parents and get matching kwargs."""
    if not isinstance(Classes, list):
        Classes = [Classes]
    out = []
    for cname in Classes:
        class_args = get_possible_class_args(cname)
        out.append({k: v for k, v in kwargs.items() if k in class_args})
    return out if len(out) > 1 else out[0]


def get_class_kwargs(Classes, kwargs):
    """Go through class and class parents and get matching kwargs."""
    if not isinstance(Classes, list):
        Classes = [Classes]
    out = []
    for cname in Classes:
        class_args = get_possible_class_args(cname)
        out.append({k: v for k, v in kwargs.items() if k in class_args})
    check_kwargs(Classes, kwargs)
    return out if len(out) > 1 else out[0]


def check_kwargs(Classes, kwargs):
    """Make sure all kwargs are valid kwargs for the set of given classes."""
    extras = []
    [
        extras.extend(list(_get_class_kwargs(cname, kwargs).keys()))
        for cname in Classes
    ]
    extras = set(kwargs.keys()) - set(extras)
    msg = f'Received unknown kwargs: {extras}'
    if len(extras) > 0:
        logger.warning(msg)
        warn(msg)


def parse_keys(keys):
    """
    Parse keys for complex __getitem__ and __setitem__

    Parameters
    ----------
    keys : string | tuple
        key or key and slice to extract

    Returns
    -------
    key : string
        key to extract
    key_slice : slice | tuple
        Slice or tuple of slices of key to extract
    """
    if isinstance(keys, tuple):
        key = keys[0]
        key_slice = keys[1:]
    else:
        key = keys
        key_slice = (
            slice(None),
            slice(None),
            slice(None),
        )

    return key, key_slice


class FactoryMeta(ABCMeta, type):
    """Meta class to define __name__ attribute of factory generated classes."""

    def __new__(cls, name, bases, namespace, **kwargs):
        """Define __name__"""
        name = namespace.get('__name__', name)
        return super().__new__(cls, name, bases, namespace, **kwargs)

    def __subclasscheck__(cls, subclass):
        """Check if factory built class shares base classes."""
        if super().__subclasscheck__(subclass):
            return True
        if hasattr(subclass, '_legos'):
            return cls._legos == subclass._legos
        return False


def _get_args_dict(thing, func, *args, **kwargs):
    """Get args dict from given object and object method."""

    ann_dict = {
        name: getattr(thing, name)
        for name, val in thing.__annotations__.items()
        if val is not ClassVar
    }
    arg_spec = getfullargspec(func)
    args = args or []
    names = arg_spec.args if 'self' not in arg_spec.args else arg_spec.args[1:]
    names = ['args', *names] if arg_spec.varargs is not None else names
    vals = [None] * len(names)
    defaults = arg_spec.defaults or []
    vals[-len(defaults) :] = defaults
    vals[: len(args)] = args
    args_dict = dict(zip(names, vals))
    args_dict.update(kwargs)
    args_dict.update(ann_dict)

    return args_dict


def get_full_args_dict(Class, func, *args, **kwargs):
    """Get full args dict for given class by searching through the inheritance
    hierarchy.

    Parameters
    ----------
    Class : class object
        Class object to search through
    func : function
        Function to check against args and kwargs
    *args : list
        Positional args for func
    **kwargs : dict
        Keyword arguments for func

    Returns
    -------
    dict
        Dictionary of argument names and values
    """
    args_dict = _get_args_dict(Class, func, *args, **kwargs)
    if (
        not kwargs
        or not hasattr(Class, '__bases__')
        or Class.__bases__ == (object,)
    ):
        return args_dict
    for base in Class.__bases__:
        base_dict = get_full_args_dict(base, base.__init__, *args, **kwargs)
        args_dict.update(
            {k: v for k, v in base_dict.items() if k not in args_dict}
        )
    return args_dict


def _log_args(thing, func, *args, **kwargs):
    """Log annotated attributes and args."""

    args_dict = get_full_args_dict(thing, func, *args, **kwargs)
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


def parse_features(
    features: Optional[str | list] = None, data: T_Dataset = None
):
    """Parse possible inputs for features (list, str, None, 'all'). If 'all'
    this returns all data_vars in data. If None this returns an empty list.

    Note
    ----
    Returns a string if input is a string and list otherwise. Need to manually
    get [features] if a list is required.

    Parameters
    ----------
    features : list | str | None
        Feature request to parse.
    data : T_Dataset
        Data containing available features
    """
    features = (
        list(data.data_vars)
        if features in ('all', ['all']) and data is not None
        else features
    )
    features = lowered(features) if features is not None else []
    return features


def parse_to_list(features=None, data=None):
    """Parse features and return as a list, even if features is a string."""
    features = (
        np.array(
            list(*features)
            if isinstance(features, tuple)
            else features
            if isinstance(features, list)
            else [features]
        )
        .flatten()
        .tolist()
    )
    return parse_features(features=features, data=data)


def _contains_ellipsis(vals):
    return vals is Ellipsis or (
        isinstance(vals, (tuple, list)) and any(v is Ellipsis for v in vals)
    )


def _is_strings(vals):
    return isinstance(vals, str) or (
        isinstance(vals, (tuple, list))
        and all(isinstance(v, str) for v in vals)
    )


def _get_strings(vals):
    return [v for v in vals if _is_strings(v)]


def _is_ints(vals):
    return isinstance(vals, int) or (
        isinstance(vals, (list, tuple, np.ndarray))
        and all(isinstance(v, int) for v in vals)
    )


def _lowered(features):
    return (
        features.lower()
        if isinstance(features, str)
        else [f.lower() if isinstance(f, str) else f for f in features]
    )


def lowered(features):
    """Return a lower case version of the given str or list of strings. Used to
    standardize storage and lookup of features."""

    feats = _lowered(features)
    if _get_strings(features) != _get_strings(feats):
        msg = (
            f'Received some upper case features: {_get_strings(features)}. '
            f'Using {_get_strings(feats)} instead.'
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


def dims_array_tuple(arr):
    """Return a tuple of (dims, array) with dims equal to the ordered slice
    of Dimension.order() with the same len as arr.shape. This is used to set
    xr.Dataset entries. e.g. dset[var] = (dims, array)"""
    if len(arr.shape) > 1:
        arr = (Dimension.order()[1 : len(arr.shape) + 1], arr)
    return arr
