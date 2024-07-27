"""Methods used across container objects."""

import logging
import os
import pprint
from enum import Enum
from glob import glob
from inspect import Parameter, Signature, getfullargspec, signature
from pathlib import Path
from typing import ClassVar, Optional, Tuple, Union
from warnings import warn

import numpy as np
import pandas as pd
import psutil
import xarray as xr

import sup3r.preprocessing

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
    QUANTILE = 'quantile'
    GLOBAL_TIME = 'global_time'

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
    def flat_2d(cls):
        """Return ordered tuple for 2d flattened data."""
        return (cls.FLATTENED_SPATIAL, cls.TIME)

    @classmethod
    def dims_2d(cls):
        """Return ordered tuple for 2d spatial coordinates."""
        return (cls.SOUTH_NORTH, cls.WEST_EAST)

    @classmethod
    def dims_3d(cls):
        """Return ordered tuple for 3d spatiotemporal coordinates."""
        return (cls.SOUTH_NORTH, cls.WEST_EAST, cls.TIME)

    @classmethod
    def dims_4d(cls):
        """Return ordered tuple for 4d spatiotemporal coordinates."""
        return (cls.SOUTH_NORTH, cls.WEST_EAST, cls.TIME, cls.PRESSURE_LEVEL)

    @classmethod
    def dims_3d_bc(cls):
        """Return ordered tuple for 3d spatiotemporal coordinates."""
        return (cls.SOUTH_NORTH, cls.WEST_EAST, cls.TIME)

    @classmethod
    def dims_4d_bc(cls):
        """Return ordered tuple for 4d spatiotemporal coordinates specifically
        for bias correction factor files."""
        return (cls.SOUTH_NORTH, cls.WEST_EAST, cls.TIME, cls.QUANTILE)


def get_date_range_kwargs(time_index):
    """Get kwargs for pd.date_range from a DatetimeIndex. This is used to
    provide a concise time_index representation which can be passed through
    the cli and avoid logging lengthly time indices."""
    freq = (
        f'{(time_index[-1] - time_index[0]).total_seconds() / 60}min'
        if len(time_index) == 2
        else pd.infer_freq(time_index)
    )
    return {
        'start': time_index[0].strftime('%Y-%m-%d %H:%M:%S'),
        'end': time_index[-1].strftime('%Y-%m-%d %H:%M:%S'),
        'freq': freq,
    }


def _mem_check():
    mem = psutil.virtual_memory()
    return (f'Memory usage is {mem.used / 1e9:.3f} GB out of '
            f'{mem.total / 1e9:.3f} GB')


def _compute_chunks_if_dask(arr):
    return (
        arr.compute_chunk_sizes()
        if hasattr(arr, 'compute_chunk_sizes')
        else arr
    )


def _numpy_if_tensor(arr):
    return arr.numpy() if hasattr(arr, 'numpy') else arr


def _compute_if_dask(arr):
    if isinstance(arr, slice):
        return slice(
            _compute_if_dask(arr.start),
            _compute_if_dask(arr.stop),
            _compute_if_dask(arr.step),
        )
    return arr.compute() if hasattr(arr, 'compute') else arr


def _rechunk_if_dask(arr, chunks='auto'):
    if hasattr(arr, 'rechunk'):
        return arr.rechunk(chunks)
    return arr


def _parse_time_slice(value):
    return (
        value
        if isinstance(value, slice)
        else slice(*value)
        if isinstance(value, list)
        else slice(None)
    )


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

    if source_type in ('.h5', '.hdf'):
        return 'h5'
    return 'nc'


def get_input_handler_class(input_handler_name: Optional[str] = None):
    """Get the :class:`DataHandler` or :class:`Extracter` object.

    Parameters
    ----------
    input_handler_name : str
        Class to use for input data. Provide a string name to match a class in
        `sup3r.preprocessing`. If None this will return :class:`Extracter`,
        which uses `ExtracterNC` or `ExtracterH5` depending on file type.  This
        is a simple handler object which does not derive new features from raw
        data.

    Returns
    -------
    HandlerClass : ExtracterH5 | ExtracterNC | DataHandlerH5 | DataHandlerNC
        DataHandler or Extracter class from sup3r.preprocessing.
    """
    if input_handler_name is None:
        input_handler_name = 'Extracter'

        logger.info(
            '"input_handler_name" arg was not provided. Using '
            f'"{input_handler_name}". If this is incorrect, please provide '
            'input_handler_name="DataHandlerName".'
        )

    HandlerClass = (
        getattr(sup3r.preprocessing, input_handler_name, None)
        if isinstance(input_handler_name, str)
        else None
    )

    if HandlerClass is None:
        msg = (
            'Could not find requested data handler class '
            f'"{input_handler_name}" in sup3r.preprocessing.'
        )
        logger.error(msg)
        raise KeyError(msg)

    return HandlerClass


def get_class_params(Class):
    """Get list of `Parameter` instances for a given class."""
    params = (
        list(Class.__signature__.parameters.values())
        if hasattr(Class, '__signature__')
        else list(signature(Class.__init__).parameters.values())
    )
    params = [p for p in params if p.name not in ('args', 'kwargs')]
    if Class.__bases__ == (object,):
        return params
    bases = Class.__bases__ + getattr(Class, '_legos', ())
    bases = list(bases) if isinstance(bases, tuple) else [bases]
    return _extend_params(bases, params)


def _extend_params(Classes, params):
    for kls in Classes:
        new_params = get_class_params(kls)
        param_names = [p.name for p in params]
        new_params = [
            p
            for p in new_params
            if p.name not in param_names and p.name not in ('args', 'kwargs')
        ]
        params.extend(new_params)
    return params


def get_composite_signature(Classes, exclude=None):
    """Get signature of an object built from the given list of classes, with
    option to exclude some parameters."""
    params = []
    for kls in Classes:
        new_params = get_class_params(kls)
        param_names = [p.name for p in params]
        new_params = [p for p in new_params if p.name not in param_names]
        params.extend(new_params)
    filtered = (
        params
        if exclude is None
        else [p for p in params if p.name not in exclude]
    )
    defaults = [p for p in filtered if p.default != p.empty]
    filtered = [p for p in filtered if p.default == p.empty] + defaults
    filtered = [
        Parameter(p.name, p.kind)
        if p.kind
        not in (Parameter.KEYWORD_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
        else Parameter(p.name, p.KEYWORD_ONLY, default=p.default)
        for p in filtered
    ]
    return Signature(parameters=filtered)


def get_class_kwargs(Class, kwargs):
    """Get kwargs which match Class signature."""
    param_names = [p.name for p in get_class_params(Class)]
    return {k: v for k, v in kwargs.items() if k in param_names}


def _get_args_dict(thing, func, *args, **kwargs):
    """Get args dict from given object and object method."""

    ann_dict = {
        name: getattr(thing, name)
        for name, val in getattr(thing, '__annotations__', {}).items()
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


def _log_args(thing, func, *args, **kwargs):
    """Log annotated attributes and args."""

    args_dict = _get_args_dict(thing, func, *args, **kwargs)
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


def parse_features(features: Optional[Union[str, list]] = None, data=None):
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
            list(features)
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
