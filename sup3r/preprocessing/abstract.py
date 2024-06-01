"""Abstract data object. These are the fundamental objects that are contained
by :class:`Container` objects."""

import logging
from typing import List, Union

import dask.array as da
import numpy as np
import xarray as xr

from sup3r.preprocessing.common import (
    Dimension,
    dims_array_tuple,
    lowered,
    ordered_array,
    ordered_dims,
)
from sup3r.typing import T_Array, T_XArray

logger = logging.getLogger(__name__)


def _contains_ellipsis(vals):
    return (
        vals is Ellipsis
        or (isinstance(vals, list)
        and any(v is Ellipsis for v in vals))
    )


def _is_str_list(vals):
    return isinstance(vals, str) or (
        isinstance(vals, list) and all(isinstance(v, str) for v in vals)
    )


def _is_int_list(vals):
    return isinstance(vals, int) or (
        isinstance(vals, list) and all(isinstance(v, int) for v in vals)
    )


class ArrayTuple(tuple):
    """Wrapper to add some useful methods to tuples of arrays. These are
    frequently returned from the :class:`Data` class, especially when there
    are multiple members of `.dsets`. We want to be able to calculate shapes,
    sizes, means, stds on these tuples."""

    def size(self):
        """Compute the total size across all tuple members."""
        return np.sum(d.size for d in self)

    def mean(self):
        """Compute the mean across all tuple members."""
        return da.mean(da.array([d.mean() for d in self]))

    def std(self):
        """Compute the standard deviation across all tuple members."""
        return da.mean(da.array([d.std() for d in self]))


class XArrayWrapper(xr.Dataset):
    """Lowest level object. This contains an xarray.Dataset and some methods
    for selecting data from the dataset. This is the thing contained by
    :class:`Container` objects."""

    __slots__ = [
        '_features',
    ]

    def __init__(self, data: xr.Dataset = None, coords=None, data_vars=None):
        if data is not None:
            reordered_vars = {
                var: (
                    ordered_dims(data.data_vars[var].dims),
                    ordered_array(data.data_vars[var]).data,
                )
                for var in data.data_vars
            }
            coords = data.coords
            data_vars = reordered_vars

        try:
            super().__init__(coords=coords, data_vars=data_vars)

        except Exception as e:
            msg = (
                'Unable to enforce standard dimension order for the given '
                'data. Please remove or standardize the problematic '
                'variables and try again.'
            )
            raise OSError(msg) from e
        self._features = None

    def sel(self, *args, **kwargs):
        """Override xr.Dataset.sel to return wrapped object."""
        if 'features' in kwargs:
            return self.slice_dset(features=kwargs['features'])
        return super().sel(*args, **kwargs)

    @property
    def time_independent(self):
        """Check whether the data is time-independent. This will need to be
        checked during extractions."""
        return 'time' not in self.variables

    def _parse_features(self, features):
        """Parse possible inputs for features (list, str, None, 'all')"""
        return lowered(
            list(self.data_vars)
            if features == 'all'
            else [features]
            if isinstance(features, str)
            else features
            if features is not None
            else []
        )

    @property
    def dims(self):
        """Return dims with our own enforced ordering."""
        return ordered_dims(super().dims)

    def slice_dset(self, features='all', keys=None):
        """Use given keys to return a sliced version of the underlying
        xr.Dataset()."""
        keys = (slice(None),) if keys is None else keys
        slice_kwargs = dict(zip(self.dims, keys))
        parsed = self._parse_features(features)
        parsed = (
            parsed
            if len(parsed) > 0
            else [Dimension.LATITUDE, Dimension.LONGITUDE, Dimension.TIME]
        )
        sliced = super().__getitem__(parsed).isel(**slice_kwargs)
        return XArrayWrapper(sliced)

    def as_array(self, features='all') -> T_Array:
        """Return dask.array for the contained xr.Dataset."""
        features = self._parse_features(features)
        arrs = [self[f].data for f in features]
        if all(arr.shape == arrs[0].shape for arr in arrs):
            return da.stack(arrs, axis=-1)
        return (
            super()
            .__getitem__(features)
            .to_dataarray()
            .transpose(*self.dims, ...)
            .data
        )

    def _get_from_list(self, keys):
        if _is_str_list(keys):
            return self.as_array(keys).squeeze()
        if _is_str_list(keys[0]):
            return self.as_array(keys[0]).squeeze()[*keys[1:], :]
        if _is_str_list(keys[-1]):
            return self.as_array(keys[-1]).squeeze()[*keys[:-1], :]
        if _is_int_list(keys):
            return self.as_array().squeeze()[..., keys]
        if _is_int_list(keys[-1]):
            return self.as_array().squeeze()[*keys[:-1]][..., keys[-1]]
        return self.as_array()[keys]

    def __getitem__(self, keys):
        """Method for accessing variables or attributes. keys can optionally
        include a feature name as the last element of a keys tuple"""
        keys = lowered(keys)
        if isinstance(keys, (list, tuple)):
            return self._get_from_list(keys)
        if _contains_ellipsis(keys):
            return self.as_array().squeeze()[keys]
        return super().__getitem__(keys)

    def __contains__(self, vals):
        if isinstance(vals, (list, tuple)) and all(
            isinstance(s, str) for s in vals
        ):
            return all(s.lower() in self for s in vals)
        return super().__contains__(vals)

    def init_new(self, new_dset):
        """Return an updated XArrayWrapper with coords and data_vars replaced
        with those provided.  These are both provided as dictionaries {name:
        dask.array}.

        Parmeters
        ---------
        new_dset : Dict[str, dask.array]
            Can contain any existing or new variable / coordinate as long as
            they all have a consistent shape.
        """
        coords = dict(self.coords)
        data_vars = dict(self.data_vars)
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
        return XArrayWrapper(coords=coords, data_vars=data_vars)

    def __setitem__(self, variable, data):
        if isinstance(variable, (list, tuple)):
            for i, v in enumerate(variable):
                self.update({v: dims_array_tuple(data[..., i])})
        else:
            variable = variable.lower()
            if hasattr(data, 'dims') and len(data.dims) >= 2:
                self.update({variable: (ordered_dims(data.dims), data)})
            elif hasattr(data, 'shape'):
                self.update({variable: dims_array_tuple(data)})
            else:
                self.update({variable: data})

    @property
    def features(self):
        """Features in this container."""
        if not self._features:
            self._features = list(self.data_vars)
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
        dim_dict = dict(self.sizes)
        dim_vals = [dim_dict[k] for k in Dimension.order() if k in dim_dict]
        return (*dim_vals, len(self.data_vars))

    @property
    def size(self):
        """Get the "size" of the container."""
        return np.prod(self.shape)

    @property
    def time_index(self):
        """Base time index for contained data."""
        if not self.time_independent:
            return self.indexes['time']
        return None

    @time_index.setter
    def time_index(self, value):
        """Update the time_index attribute with given index."""
        self['time'] = value

    @property
    def lat_lon(self) -> T_Array:
        """Base lat lon for contained data."""
        return self.as_array([Dimension.LATITUDE, Dimension.LONGITUDE])

    @lat_lon.setter
    def lat_lon(self, lat_lon):
        """Update the lat_lon attribute with array values."""
        self[Dimension.LATITUDE] = (self[Dimension.LATITUDE], lat_lon[..., 0])
        self[Dimension.LONGITUDE] = (
            self[Dimension.LONGITUDE],
            lat_lon[..., 1],
        )


def single_member_check(func):
    """Decorator to return first item of list if there is only one data
    member."""

    def wrapper(self, *args, **kwargs):
        out = func(self, *args, **kwargs)
        if self.n_members == 1:
            return out[0]
        return out

    return wrapper


class Data:
    """Interface for interacting with tuples / lists of :class:`XArrayWrapper`
    objects."""

    def __init__(self, data: Union[List[xr.Dataset], List[XArrayWrapper]]):
        if not isinstance(data, (list, tuple)):
            data = (data,)
        self.dsets = tuple(XArrayWrapper(d) for d in data)
        self.n_members = len(self.dsets)

    @single_member_check
    def __getattr__(self, attr):
        if attr in dir(self):
            return self.__getattribute__(attr)
        out = [getattr(d, attr) for d in self.dsets]
        return out

    @single_member_check
    def __getitem__(self, keys):
        """Method for accessing self.dset or attributes. If keys is a list of
        tuples or list this is interpreted as a request for
        `self.dset[i][keys[i]] for i in range(len(keys)).` Otherwise the we
        will get keys from each member of self.dset."""
        if isinstance(keys, (tuple, list)) and all(
            isinstance(k, (tuple, list)) for k in keys
        ):
            out = ArrayTuple([d[key] for d, key in zip(self.dsets, keys)])
        else:
            out = ArrayTuple(d[keys] for d in self.dsets)
        return out

    @single_member_check
    def isel(self, *args, **kwargs) -> T_XArray:
        """Multi index selection method."""
        out = tuple(d.isel(*args, **kwargs) for d in self.dsets)
        return out

    @single_member_check
    def sel(self, *args, **kwargs) -> T_XArray:
        """Multi dimension selection method."""
        out = tuple(d.sel(*args, **kwargs) for d in self.dsets)
        return out

    def __contains__(self, vals):
        """Check for vals in all of the dset members."""
        return any(d.__contains__(vals) for d in self.dsets)

    def __setitem__(self, variable, data):
        """Set dset member values. Check if values is a tuple / list and if
        so interpret this as sending a tuple / list element to each dset
        member. e.g. `vals[0] -> dsets[0]`, `vals[1] -> dsets[1]`, etc"""
        for i, d in enumerate(self.dsets):
            dat = data[i] if isinstance(data, (tuple, list)) else data
            d.__setitem__(variable, dat)

    def __iter__(self):
        yield from self.dsets
