"""Abstract container classes. These are the fundamental objects that all
classes which interact with data (e.g. handlers, wranglers, loaders, samplers,
batchers) are based on."""

import logging

import dask.array as da
import numpy as np
import xarray as xr

from sup3r.containers.common import lowered

logger = logging.getLogger(__name__)


class Data:
    """Lowest level object. This contains an xarray.Dataset and some methods
    for selecting data from the dataset. This is the thing contained by
    :class:`Container` objects."""

    DIM_ORDER = (
        'space',
        'south_north',
        'west_east',
        'time',
        'level',
        'variable',
    )

    def __init__(self, data: xr.Dataset):
        try:
            self.dset = self.enforce_standard_dim_order(data)
        except Exception as e:
            msg = ('Unable to enforce standard dimension order for the given '
                   'data. Please remove or standardize the problematic '
                   'variables and try again.')
            raise OSError(msg) from e
        self._features = None

    def enforce_standard_dim_order(self, dset: xr.Dataset):
        """Ensure that data dimensions have a (space, time, ...) or (latitude,
        longitude, time, ...) ordering."""

        reordered_vars = {
            var: (
                self.ordered_dims(dset.data_vars[var].dims),
                self.transpose(dset.data_vars[var]).data,
            )
            for var in dset.data_vars
        }

        return xr.Dataset(coords=dset.coords, data_vars=reordered_vars)

    def _check_string_keys(self, keys):
        """Check for string key in `.data` or as an attribute."""
        if keys.lower() in self.variables:
            out = self.dset[keys.lower()].data
        elif keys in self.dset:
            out = self.dset[keys].data
        else:
            out = getattr(self, keys)
        return out

    def slice_dset(self, keys=None, features=None):
        """Use given keys to return a sliced version of the underlying
        xr.Dataset()."""
        keys = (slice(None),) if keys is None else keys
        slice_kwargs = dict(zip(self.dims, keys))
        features = (
            lowered(features) if features is not None else self.features
        )
        return self.dset[features].isel(**slice_kwargs)

    def ordered_dims(self, dims):
        """Return the order of dims that follows the ordering of self.DIM_ORDER
        for the common dim names. e.g dims = ('time', 'south_north', 'dummy',
        'west_east') will return ('south_north', 'west_east', 'time',
        'dummy')."""
        standard = [dim for dim in self.DIM_ORDER if dim in dims]
        non_standard = [dim for dim in dims if dim not in standard]
        return tuple(standard + non_standard)

    @property
    def dims(self):
        """Get ordered dim names for datasets."""
        return self.ordered_dims(self.dset.dims)

    def _dims_with_array(self, arr):
        if len(arr.shape) > 1:
            arr = (self.DIM_ORDER[1 : len(arr.shape) + 1], arr)
        return arr

    def update(self, new_dset):
        """Update the underlying xr.Dataset with given coordinates and / or
        data variables. These are both provided as dictionaries {name:
        dask.array}.

        Parmeters
        ---------
        new_dset : Dict[str, dask.array]
            Can contain any existing or new variable / coordinate as long as
            they all have a consistent shape.
        """
        coords = dict(self.dset.coords)
        data_vars = dict(self.dset.data_vars)
        coords.update(
            {
                k: self._dims_with_array(v)
                for k, v in new_dset.items()
                if k in coords
            }
        )
        data_vars.update(
            {
                k: self._dims_with_array(v)
                for k, v in new_dset.items()
                if k not in coords
            }
        )
        self.dset = self.enforce_standard_dim_order(
            xr.Dataset(coords=coords, data_vars=data_vars)
        )

    def _slice_data(self, keys, features=None):
        """Select a region of data with a list or tuple of slices."""
        if len(keys) < 5:
            out = self.slice_dset(keys, features).to_dataarray().data
        else:
            msg = f'Received too many keys: {keys}.'
            logger.error(msg)
            raise KeyError(msg)
        return out

    def _check_list_keys(self, keys):
        """Check if key list contains strings which are attributes or in
        `.data` or if the list is a set of slices to select a region of
        data."""
        if all(type(s) is str and s in self for s in keys):
            out = self.to_array(keys)
        elif all(type(s) is slice for s in keys):
            out = self.to_array()[keys]
        elif isinstance(keys[-1], list) and all(
            isinstance(s, slice) for s in keys[:-1]
        ):
            out = self.to_array(keys[-1])[keys[:-1]]
        elif isinstance(keys[0], list) and all(
            isinstance(s, slice) for s in keys[1:]
        ):
            out = self.to_array(keys[0])[keys[1:]]
        else:
            try:
                out = self.to_array()[keys]
            except Exception as e:
                msg = (
                    'Do not know what to do with the provided key set: '
                    f'{keys}.'
                )
                logger.error(msg)
                raise KeyError(msg) from e
        return out

    def __getitem__(self, keys):
        """Method for accessing self.dset or attributes. keys can optionally
        include a feature name as the first element of a keys tuple"""
        if isinstance(keys, str):
            return self._check_string_keys(keys)
        if isinstance(keys, (tuple, list)):
            return self._check_list_keys(keys)
        return self.to_array()[keys]

    def __getattr__(self, keys):
        if keys in self.__dict__:
            return self.__dict__[keys]
        if hasattr(self.dset, keys):
            return getattr(self.dset, keys)
        if keys in dir(self):
            return super().__getattribute__(keys)
        msg = f'Could not get attribute {keys} from {self.__class__.__name__}'
        raise AttributeError(msg)

    def __setattr__(self, keys, value):
        self.__dict__[keys] = value

    def __setitem__(self, variable, data):
        variable = variable.lower()
        if hasattr(data, 'dims') and len(data.dims) >= 2:
            self.dset[variable] = (self.orered_dims(data.dims), data)
        elif hasattr(data, 'shape'):
            self.dset[variable] = self._dims_with_array(data)
        else:
            self.dset[variable] = data

    @property
    def variables(self):
        """'All "features" in the dataset in the order that they were loaded.
        Not necessarily the same as the ordered set of training features."""
        return list(self.dset.data_vars)

    @property
    def features(self):
        """Features in this container."""
        if self._features is None:
            self._features = self.variables
        return self._features

    @features.setter
    def features(self, val):
        """Set features in this container."""
        self._features = lowered(val)

    def transpose(self, data):
        """Transpose arrays so they have a (space, time, ...) or (space, time,
        ..., feature) ordering."""
        return data.transpose(*self.ordered_dims(data.dims))

    def to_array(self, features=None):
        """Return xr.DataArray of contained xr.Dataset."""
        features = self.features if features is None else features
        return da.moveaxis(
            self.dset[lowered(features)].to_dataarray().data, 0, -1
        )

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
        dim_vals = [dim_dict[k] for k in self.DIM_ORDER if k in dim_dict]
        return (*dim_vals, len(self.variables))

    @property
    def size(self):
        """Get the "size" of the container."""
        return np.prod(self.shape)

    @property
    def time_index(self):
        """Base time index for contained data."""
        return self.dset.indexes['time']

    @time_index.setter
    def time_index(self, value):
        """Update the time_index attribute with given index."""
        self.dset['time'] = value

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
