"""Abstract container classes. These are the fundamental objects that all
classes which interact with data (e.g. handlers, wranglers, loaders, samplers,
batchers) are based on."""

import logging

import dask.array as da
import numpy as np
import xarray as xr

from sup3r.containers.common import (
    DIM_ORDER,
    all_dtype,
    dims_array_tuple,
    enforce_standard_dim_order,
    lowered,
    ordered_dims,
)

logger = logging.getLogger(__name__)


class Data:
    """Lowest level object. This contains an xarray.Dataset and some methods
    for selecting data from the dataset. This is the thing contained by
    :class:`Container` objects."""

    def __init__(self, data: xr.Dataset):
        try:
            self.dset = enforce_standard_dim_order(data)
        except Exception as e:
            msg = (
                'Unable to enforce standard dimension order for the given '
                'data. Please remove or standardize the problematic '
                'variables and try again.'
            )
            raise OSError(msg) from e
        self._features = None

    def isel(self, *args, **kwargs):
        """Override xr.Dataset.isel to return wrapped object."""
        return Data(self.dset.isel(*args, **kwargs))

    def sel(self, *args, **kwargs):
        """Override xr.Dataset.sel to return wrapped object."""
        if 'features' in kwargs:
            return self.slice_dset(features=kwargs['features'])
        return Data(self.dset.sel(*args, **kwargs))

    @property
    def time_independent(self):
        """Check whether the data is time-independent. This will need to be
        checked during extractions."""
        return 'time' not in self.variables

    def _parse_features(self, features):
        """Parse possible inputs for features (list, str, None, 'all')"""
        out = (
            list(self.dset.data_vars)
            if features == 'all'
            else features
            if features is not None
            else []
        )
        return lowered(out)

    def slice_dset(self, features='all', keys=None):
        """Use given keys to return a sliced version of the underlying
        xr.Dataset()."""
        keys = (slice(None),) if keys is None else keys
        slice_kwargs = dict(zip(self.dims, keys))
        parsed = self._parse_features(features)
        parsed = (
            parsed if len(parsed) > 0 else ['latitude', 'longitude', 'time']
        )
        return Data(self.dset[parsed].isel(**slice_kwargs))

    def to_array(self, features='all'):
        """Return xr.DataArray of contained xr.Dataset."""
        features = self._parse_features(features)
        features = features if isinstance(features, list) else [features]
        shapes = [self.dset[f].data.shape for f in features]
        if all(s == shapes[0] for s in shapes):
            return da.stack([self.dset[f] for f in features], axis=-1)
        return da.moveaxis(self.dset[features].to_dataarray().data, 0, -1)

    @property
    def dims(self):
        """Get ordered dim names for datasets."""
        return ordered_dims(self.dset.dims)

    def __contains__(self, val):
        vals = val if isinstance(val, (tuple, list)) else [val]
        if all_dtype(vals, str):
            return all(v.lower() in self.variables for v in vals)
        return False

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
                k: dims_array_tuple(v)
                for k, v in new_dset.items()
                if k in coords
            }
        )
        data_vars.update(
            {
                k: dims_array_tuple(v)
                for k, v in new_dset.items()
                if k not in coords
            }
        )
        self.dset = enforce_standard_dim_order(
            xr.Dataset(coords=coords, data_vars=data_vars)
        )

    def get_from_list(self, keys):
        """Check if key list contains strings which are attributes or in
        `.data` or if the list is a set of slices to select a region of
        data."""
        if all_dtype(keys, slice):
            out = self.to_array()[keys]
        elif all_dtype(keys[0], str):
            out = self.to_array(keys[0])[*keys[1:], :]
            out = out.squeeze() if isinstance(keys[0], str) else out
        elif all_dtype(keys[-1], str):
            out = self.get_from_list((keys[-1], *keys[:-1]))
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
        include a feature name as the last element of a keys tuple"""
        if keys in self:
            return self.to_array(keys).squeeze()
        if isinstance(keys, str) and hasattr(self, keys):
            return getattr(self, keys)
        if isinstance(keys, (tuple, list)):
            return self.get_from_list(keys)
        return self.to_array()[keys]

    def __getattr__(self, keys):
        if keys in dir(self):
            return self.__getattribute__(keys)
        if hasattr(self.dset, keys):
            return getattr(self.dset, keys)
        msg = f'Could not get attribute {keys} from {self.__class__.__name__}'
        raise AttributeError(msg)

    def __setattr__(self, keys, value):
        self.__dict__[keys] = value

    def __setitem__(self, variable, data):
        if isinstance(variable, (list, tuple)):
            for i, v in enumerate(variable):
                self[v] = data[..., i]
        variable = variable.lower()
        if hasattr(data, 'dims') and len(data.dims) >= 2:
            self.dset[variable] = (self.orered_dims(data.dims), data)
        elif hasattr(data, 'shape'):
            self.dset[variable] = dims_array_tuple(data)
        else:
            self.dset[variable] = data

    @property
    def variables(self):
        """'All "features" in the dataset in the order that they were loaded.
        Not necessarily the same as the ordered set of training features."""
        return (
            list(self.dset.dims)
            + list(self.dset.data_vars)
            + list(self.dset.coords)
        )

    @property
    def features(self):
        """Features in this container."""
        if self._features is None:
            self._features = list(self.dset.data_vars)
        return self._features

    @features.setter
    def features(self, val):
        """Set features in this container."""
        self._features = self._parse_features(val)

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
        dim_vals = [dim_dict[k] for k in DIM_ORDER if k in dim_dict]
        return (*dim_vals, len(self.dset.data_vars))

    @property
    def size(self):
        """Get the "size" of the container."""
        return np.prod(self.shape)

    @property
    def time_index(self):
        """Base time index for contained data."""
        if not self.time_independent:
            return self.dset.indexes['time']
        return None

    @time_index.setter
    def time_index(self, value):
        """Update the time_index attribute with given index."""
        self.dset['time'] = value

    @property
    def lat_lon(self):
        """Base lat lon for contained data."""
        return self[['latitude', 'longitude']]

    @lat_lon.setter
    def lat_lon(self, lat_lon):
        """Update the lat_lon attribute with array values."""
        self.dset['latitude'] = (self.dset['latitude'], lat_lon[..., 0])
        self.dset['longitude'] = (self.dset['longitude'], lat_lon[..., 1])
