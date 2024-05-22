"""Abstract container classes. These are the fundamental objects that all
classes which interact with data (e.g. handlers, wranglers, loaders, samplers,
batchers) are based on."""

import inspect
import logging
import pprint
from abc import ABC

import dask.array as da
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class DataWrapper:
    """xr.Dataset wrapper with some additional attributes."""

    def __init__(self, data: xr.Dataset):
        self.dset = data
        self.dim_names = (
            'south_north',
            'west_east',
            'time',
            'level',
            'variable',
        )

    def get_dim_names(self, data):
        """Get standard dimension ordering for 2d and 3d+ arrays."""
        return tuple(
            [dim for dim in ('space', *self.dim_names) if dim in data.dims]
        )

    def __getitem__(self, keys):
        return self.dset[keys]

    def __contains__(self, feature):
        return feature.lower() in self.dset

    def __getattr__(self, keys):
        if keys in self.__dict__:
            return self.__dict__[keys]
        if keys in dir(self):
            return getattr(self, keys)
        if hasattr(self.dset, keys):
            return getattr(self.dset, keys)
        msg = f'Could not find attribute {keys} in {self.__class__.__name__}'
        logger.error(msg)
        raise KeyError(msg)

    def __setattr__(self, keys, value):
        self.__dict__[keys] = value

    def __setitem__(self, keys, value):
        if hasattr(value, 'dims') and len(value.dims) >= 2:
            self.dset[keys] = (self.get_dim_names(value), value)
        elif hasattr(value, 'shape'):
            self.dset[keys] = (self.dim_names[: len(value.shape)], value)
        else:
            self.dset[keys] = value

    def to_array(self):
        """Return xr.DataArray of contained xr.Dataset."""
        return self._transpose(
            self.dset[sorted(self.features)].to_dataarray()
        ).data

    @property
    def features(self):
        """Features in this container."""
        return sorted(self.dset.data_vars)

    @property
    def size(self):
        """Get the "size" of the container."""
        return np.prod(self.shape)

    @property
    def dtype(self):
        """Get data type of contained array."""
        return self.to_array().dtype

    def _transpose(self, data):
        """Transpose arrays so they have a (space, time, ...) or (space, time,
        ..., feature) ordering."""
        return data.transpose(*self.get_dim_names(data))

    @property
    def shape(self):
        """Get shape of underlying xr.DataArray. Feature channel by default is
        first and time is second, so we shift these to (..., time, features).
        We also sometimes have a level dimension for pressure level data."""
        dim_dict = dict(self.dset.dims)
        dim_vals = [
            dim_dict[k] for k in ('space', *self.dim_names) if k in dim_dict
        ]
        return (*dim_vals, len(self.features))


class AbstractContainer(ABC):
    """Lowest level object. This contains an xarray.Dataset and some methods
    for selecting data from the dataset. :class:`Container` implementation
    just requires defining `.data` with an xarray.Dataset."""

    def __init__(self):
        self._data = None
        self._features = None

    def __new__(cls, *args, **kwargs):
        """Include arg logging in construction."""
        instance = super().__new__(cls)
        cls._log_args(args, kwargs)
        return instance

    @property
    def data(self) -> DataWrapper:
        """Wrapped xr.Dataset."""
        return self._data

    @data.setter
    def data(self, data):
        """Wrap given data in :class:`DataWrapper` to provide additional
        attributes on top of xr.Dataset."""
        self._data = DataWrapper(data)

    @property
    def features(self):
        """Features in this container."""
        if self._features is None:
            self._features = sorted(self.data.features)
        return self._features

    @features.setter
    def features(self, val):
        """Set features in this container."""
        self._features = [f.lower() for f in val]

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

    @property
    def time_index(self):
        """Base time index for contained data."""
        return self['time']

    @time_index.setter
    def time_index(self, value):
        """Update the time_index attribute with given index."""
        self.data['time'] = value

    @property
    def lat_lon(self):
        """Base lat lon for contained data."""
        return da.stack([self['latitude'], self['longitude']], axis=-1)

    @lat_lon.setter
    def lat_lon(self, lat_lon):
        """Update the lat_lon attribute with array values."""
        self.data['latitude'] = (self.data['latitude'].dims, lat_lon[..., 0])
        self.data['longitude'] = (self.data['longitude'].dims, lat_lon[..., 1])

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
        """Check for string key in `.data` or as an attribute."""
        if keys.lower() in self.data.features:
            out = self._transpose(self.data[keys.lower()]).data
        elif keys in self.data:
            out = self.data[keys].data
        else:
            out = getattr(self, keys)
        return out

    def _slice_data(self, keys):
        """Select a region of data with a list or tuple of slices."""
        if len(keys) == 2:
            out = self.data.isel(space=keys[0], time=keys[1])
        elif len(keys) < 5:
            slice_kwargs = dict(
                zip(['south_north', 'west_east', 'time', 'level'], keys)
            )
            out = self.data.isel(**slice_kwargs)
        else:
            msg = f'Received too many keys: {keys}.'
            logger.error(msg)
            raise KeyError(msg)
        return out

    def _check_list_keys(self, keys):
        """Check if key list contains strings which are attributes or in
        `.data` or if the list is a set of slices to select a region of
        data."""
        if all(type(s) is str and s in self.features for s in keys):
            out = self._transpose(self.data[keys].to_dataarray()).data
        elif all(type(s) is str for s in keys):
            out = self.data[keys].to_dataarray().data
        elif all(type(s) is slice for s in keys):
            out = self._slice_data(keys)
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

    def __getattr__(self, keys):
        if keys in self.__dict__:
            return self.__dict__[keys]
        if keys in dir(self):
            return getattr(self, keys)
        return getattr(self.data, keys)
