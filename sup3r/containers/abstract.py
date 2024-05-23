"""Abstract container classes. These are the fundamental objects that all
classes which interact with data (e.g. handlers, wranglers, loaders, samplers,
batchers) are based on."""

import inspect
import logging
import pprint

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

DIM_NAMES = (
    'space',
    'south_north',
    'west_east',
    'time',
    'level',
    'variable',
)


def get_dim_names(data):
    """Get standard dimension ordering for 2d and 3d+ arrays."""
    return tuple([dim for dim in DIM_NAMES if dim in data.dims])


class Data:
    """Lowest level object. This contains an xarray.Dataset and some methods
    for selecting data from the dataset. This is the thing contained by
    :class:`Container` objects."""

    def __init__(self, data: xr.Dataset):
        self.dset = data
        self._features = None

    def _check_string_keys(self, keys):
        """Check for string key in `.data` or as an attribute."""
        if keys.lower() in self.variables:
            out = self._transpose(self.dset[keys.lower()]).data
        elif keys in self.dset:
            out = self.dset[keys].data
        else:
            out = getattr(self, keys)
        return out

    def slice_dset(self, keys=None, features=None):
        """Use given keys to return a sliced version of the underlying
        xr.Dataset()."""
        keys = (slice(None),) if keys is None else keys
        slice_kwargs = dict(zip(get_dim_names(self.dset), keys))
        return self.dset[self.features if features is None else features].isel(
            **slice_kwargs
        )

    def _slice_data(self, keys, features=None):
        """Select a region of data with a list or tuple of slices."""
        if len(keys) < 5:
            out = self._transpose(
                self.slice_dset(keys, features).to_dataarray()
            ).data
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
            out = self._transpose(self.dset[keys].to_dataarray()).data
        elif all(type(s) is str for s in keys):
            out = self.dset[keys].to_dataarray().data
        elif all(type(s) is slice for s in keys):
            out = self._slice_data(keys)
        elif isinstance(keys[-1], list) and all(
            isinstance(s, slice) for s in keys[:-1]
        ):
            out = self._slice_data(keys[:-1], features=keys[-1])
        elif isinstance(keys[0], list) and all(
            isinstance(s, slice) for s in keys[1:]
        ):
            out = self.slice_data(keys[1:], features=keys[0])
        else:
            msg = (
                'Do not know what to do with the provided key set: ' f'{keys}.'
            )
            logger.error(msg)
            raise KeyError(msg)
        return out

    def __getitem__(self, keys):
        """Method for accessing self.dset or attributes. keys can optionally
        include a feature name as the first element of a keys tuple"""
        if isinstance(keys, str):
            return self._check_string_keys(keys)
        if isinstance(keys, (tuple, list)):
            return self._check_list_keys(keys)
        return self.to_array()[keys]

    def __contains__(self, feature):
        return feature.lower() in self.dset

    def __getattr__(self, keys):
        if keys in self.__dict__:
            return self.__dict__[keys]
        if hasattr(self.dset, keys):
            return getattr(self.dset, keys)
        if keys in dir(self):
            return super().__getattribute__(keys)
        raise AttributeError

    def __setattr__(self, keys, value):
        self.__dict__[keys] = value

    def __setitem__(self, keys, value):
        if hasattr(value, 'dims') and len(value.dims) >= 2:
            self.dset[keys] = (get_dim_names(value), value)
        elif hasattr(value, 'shape'):
            self.dset[keys] = (DIM_NAMES[1 : len(value.shape) + 1], value)
        else:
            self.dset[keys] = value

    @property
    def variables(self):
        """'Features' in the dataset. Called variables here to distinguish them
        from the ordered list of training features. These are ordered
        alphabetically and not necessarily used in training."""
        return sorted(self.dset.data_vars)

    @property
    def features(self):
        """Features in this container."""
        if self._features is None:
            self._features = sorted(self.variables)
        return self._features

    @features.setter
    def features(self, val):
        """Set features in this container."""
        self._features = [f.lower() for f in val]

    def _transpose(self, data):
        """Transpose arrays so they have a (space, time, ...) or (space, time,
        ..., feature) ordering."""
        return data.transpose(*get_dim_names(data))

    def to_array(self):
        """Return xr.DataArray of contained xr.Dataset."""
        return self._transpose(self.dset[self.features].to_dataarray()).data

    @property
    def dtype(self):
        """Get data type of contained array."""
        return self.to_array().dtype

    @property
    def shape(self):
        """Get shape of underlying xr.DataArray. Feature channel by default is
        first and time is second, so we shift these to (..., time, features).
        We also sometimes have a level dimension for pressure level data."""
        dim_dict = dict(self.dset.sizes)
        dim_vals = [dim_dict[k] for k in DIM_NAMES if k in dim_dict]
        return (*dim_vals, len(self.variables))

    @property
    def size(self):
        """Get the "size" of the container."""
        return np.prod(self.shape)

    @property
    def time_index(self):
        """Base time index for contained data."""
        return pd.to_datetime(self['time'])

    @time_index.setter
    def time_index(self, value):
        """Update the time_index attribute with given index."""
        self.dset['time'] = value

    @property
    def dims(self):
        """Get ordered dim names for datasets."""
        return get_dim_names(self.dset)

    @property
    def lat_lon(self):
        """Base lat lon for contained data."""
        return da.stack(
            [self.dset['latitude'], self.dset['longitude']], axis=-1
        )

    @lat_lon.setter
    def lat_lon(self, lat_lon):
        """Update the lat_lon attribute with array values."""
        self.dset['latitude'] = (self.dset['latitude'], lat_lon[..., 0])
        self.dset['longitude'] = (self.dset['longitude'], lat_lon[..., 1])


class AbstractContainer:
    """Lowest level object. This contains an xarray.Dataset and some methods
    for selecting data from the dataset. :class:`Container` implementation
    just requires defining `.data` with an xarray.Dataset."""

    def __init__(self):
        self._data = None

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

    @property
    def data(self) -> Data:
        """Wrapped xr.Dataset."""
        return self._data

    @data.setter
    def data(self, data):
        """Wrap given data in :class:`Data` to provide additional
        attributes on top of xr.Dataset."""
        if isinstance(data, xr.Dataset):
            self._data = Data(data)
        else:
            self._data = data

    def __getitem__(self, keys):
        """Method for accessing self.data or attributes. keys can optionally
        include a feature name as the first element of a keys tuple"""
        return self.data[keys]

    def __getattr__(self, keys):
        if keys in self.__dict__:
            return self.__dict__[keys]
        if hasattr(self.data, keys):
            return getattr(self.data, keys)
        if keys in dir(self):
            return super().__getattribute__(keys)
        raise AttributeError
