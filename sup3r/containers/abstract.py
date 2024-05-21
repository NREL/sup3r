"""Abstract container classes. These are the fundamental objects that all
classes which interact with data (e.g. handlers, wranglers, loaders, samplers,
batchers) are based on."""

import inspect
import logging
import pprint
from abc import ABC, ABCMeta

import dask.array as da
import numpy as np

logger = logging.getLogger(__name__)


class _ContainerMeta(ABCMeta, type):
    def __call__(cls, *args, **kwargs):
        """Check for required attributes"""
        obj = type.__call__(cls, *args, **kwargs)
        obj._init_check()
        return obj


class AbstractContainer(ABC, metaclass=_ContainerMeta):
    """Lowest level object. This contains an xarray.Dataset and some methods
    for selecting data from the dataset. :class:`Container` implementation
    just requires defining `.data` with an xarray.Dataset."""

    def __init__(self):
        self._features = None
        self._shape = None

    def _init_check(self):
        if 'data' not in dir(self):
            msg = f'{self.__class__.__name__} must implement "data"'
            raise NotImplementedError(msg)

    def __new__(cls, *args, **kwargs):
        """Include arg logging in construction."""
        instance = super().__new__(cls)
        cls._log_args(args, kwargs)
        return instance

    @classmethod
    def _log_args(cls, args, kwargs):
        """Log argument names and values."""
        arg_spec = inspect.getfullargspec(cls.__init__)
        args = args or []
        defaults = arg_spec.defaults or []
        arg_names = arg_spec.args[1 : len(args) + 1]
        kwargs_names = arg_spec.args[-len(defaults) :]
        args_dict = dict(zip(kwargs_names, defaults))
        args_dict.update(dict(zip(arg_names, args)))
        args_dict.update(kwargs)
        logger.info(
            f'Initialized {cls.__name__} with:\n'
            f'{pprint.pformat(args_dict, indent=2)}'
        )

    def to_array(self):
        """Return xr.DataArray of contained xr.Dataset."""
        return self._transpose(
            self.data[sorted(self.features)].to_dataarray()
        ).data

    @property
    def features(self):
        """Features in this container."""
        if self._features is None:
            self._features = list(self.data.data_vars)
        return self._features

    @features.setter
    def features(self, val):
        """Set features in this container."""
        self._features = [f.lower() for f in val]

    @property
    def size(self):
        """Get the "size" of the container."""
        return np.prod(self.shape)

    @property
    def dtype(self):
        """Get data type of contained array."""
        return self.to_array().dtype

    def _transpose(self, data):
        """Transpose arrays so they have a (space, time, ...) ordering. These
        arrays do not have a feature channel"""
        if len(data.shape) <= 3 and 'space' in data.dims:
            return data.transpose('space', 'time', ...)
        if len(data.shape) >= 3:
            dim_order = ('south_north', 'west_east', 'time')
            if 'level' in data.dims:
                dim_order = (*dim_order, 'level')
            if 'variable' in data.dims:
                dim_order = (*dim_order, 'variable')
            return data.transpose(*dim_order)
        return None

    @property
    def shape(self):
        """Get shape of underlying xr.DataArray. Feature channel by default is
        first and time is second, so we shift these to (..., time, features).
        We also sometimes have a level dimension for pressure level data."""
        if self._shape is None:
            self._shape = self.to_array().shape
        return self._shape

    @property
    def time_index(self):
        """Base time index for contained data."""
        return self['time']

    @property
    def lat_lon(self):
        """Base lat lon for contained data."""
        return da.stack([self['latitude'], self['longitude']], axis=-1)

    def __contains__(self, feature):
        return feature.lower() in self.data

    def parse_keys(self, keys):
        """
        Parse keys for complex __getitem__ and __setitem__

        Parameters
        ----------
        keys: string | tuple
            key or key and slice to extract

        Returns
        -------
        key: string
            key to extract
        key_slice: slice | tuple
            Slice or tuple of slices of key to extract
        """
        if isinstance(keys, tuple):
            key = keys[0]
            key_slice = keys[1:]
        else:
            key = keys
            dims = 4 if self.data is None else len(self.shape)
            key_slice = tuple([slice(None)] * (dims - 1))

        return key, key_slice

    def _check_string_keys(self, keys):
        if keys.lower() in self.data.data_vars:
            out = self._transpose(self.data[keys.lower()]).data
        elif keys in self.data:
            out = self.data[keys].data
        elif hasattr(self, keys):
            out = getattr(self, keys)
        elif hasattr(self.data, keys):
            out = self.data[keys]
        else:
            msg = f'Could not find {keys} in features or attributes'
            logger.error(msg)
            raise KeyError(msg)
        return out

    def _check_list_keys(self, keys):
        if all(type(s) is str and s in self.features for s in keys):
            out = self._transpose(self.data[keys].to_dataarray()).data
        elif all(type(s) is str for s in keys):
            out = self.data[keys].to_dataarray().data
        elif all(type(s) is slice for s in keys):
            if len(keys) == 2:
                out = self.data.isel(space=keys[0], time=keys[1])
            elif len(keys) == 3:
                out = self.data.isel(
                    south_north=keys[0], west_east=keys[1], time=keys[2]
                )
            else:
                msg = f'Received too many keys: {keys}.'
                logger.error(msg)
                raise KeyError(msg)
        else:
            msg = f'Could not use the provided set of {keys}.'
            logger.error(msg)
            raise KeyError(msg)
        return out

    def __getitem__(self, keys):
        """Method for accessing self.data or attributes. keys can optionally
        include a feature name as the first element of a keys tuple"""
        key, key_slice = self.parse_keys(keys)
        if isinstance(keys, str):
            return self._check_string_keys(keys)
        if isinstance(keys, (tuple, list)):
            return self._check_list_keys(keys)
        return self.to_array()[key, *key_slice]
