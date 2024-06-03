"""Abstract data object. These are the fundamental objects that are contained
by :class:`Container` objects."""

import logging
from typing import List, Union

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from xarray import Dataset

from sup3r.preprocessing.common import (
    Dimension,
    _contains_ellipsis,
    _is_ints,
    _is_strings,
    dims_array_tuple,
    lowered,
    ordered_array,
    ordered_dims,
)
from sup3r.typing import T_Array

logger = logging.getLogger(__name__)


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


class DatasetWrapper(Dataset):
    """Lowest level object. This contains an xarray.Dataset and some methods
    for selecting data from the dataset. This is the simplest version of the
    `.data` attribute for :class:`Container` objects.

    Notes
    -----
    Data is accessed through the `__getitem__`. A DatasetWrapper is returned
    when a list of features is requested. e.g __getitem__(['u', 'v']).
    When a single feature is requested a DataArray is returned. e.g.
    `__getitem__('u')`
    When numpy style indexing is used a dask array is returned. e.g.
    `__getitem__('u', ...)` `or self['u', :, slice(0, 10)]`
    """

    __slots__ = ['_features']

    def __init__(
        self, data: xr.Dataset = None, coords=None, data_vars=None, attrs=None
    ):
        """
        Parameters
        ----------
        data : xr.Dataset
            An xarray Dataset instance to wrap with our custom interface
        coords : dict
            Dictionary like object with tuples of (dims, array) for each
            coordinate. e.g. {"latitude": (("south_north", "west_east"), lats)}
        data_vars : dict
            Dictionary like object with tuples of (dims, array) for each
            variable. e.g. {"temperature": (("south_north", "west_east",
            "time", "level"), temp)}
        attrs : dict
            Optional dictionary of attributes to include in the meta data. This
            can be accessed through self.attrs
        """
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
            super().__init__(coords=coords, data_vars=data_vars, attrs=attrs)

        except Exception as e:
            msg = (
                'Unable to enforce standard dimension order for the given '
                'data. Please remove or standardize the problematic '
                'variables and try again.'
            )
            raise OSError(msg) from e
        self._features = None

    @property
    def name(self):
        """Name of dataset. Used to label datasets when grouped in
        :class:`Data` objects. e.g. for low / high res pairs or daily / hourly
        data."""
        return self.attrs.get('name', None)

    def sel(self, *args, **kwargs):
        """Override xr.Dataset.sel to return wrapped object."""
        features = kwargs.pop('features', None)
        if features is not None:
            return self[features].sel(**kwargs)
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
            if 'all' in features
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

    def as_array(self, features='all') -> T_Array:
        """Return dask.array for the contained xr.Dataset."""
        features = self._parse_features(features)
        arrs = [super(DatasetWrapper, self).__getitem__(f) for f in features]
        if all(arr.shape == arrs[0].shape for arr in arrs):
            return da.stack(arrs, axis=-1)
        return (
            super()
            .__getitem__(features)
            .to_dataarray()
            .transpose(*self.dims, ...)
            .data
        )

    def _get_from_tuple(self, keys):
        if _is_strings(keys[0]):
            out = self.as_array(keys[0])[*keys[1:], :]
        elif _is_strings(keys[-1]):
            out = self.as_array(keys[-1])[*keys[:-1], :]
        elif _is_ints(keys[-1]) and not _contains_ellipsis(keys):
            out = self.as_array()[*keys[:-1], ..., keys[-1]]
        else:
            out = self.as_array()[keys]
        return out.squeeze(axis=-1) if out.shape[-1] == 1 else out

    def __getitem__(self, keys):
        """Method for accessing variables or attributes. keys can optionally
        include a feature name as the last element of a keys tuple."""
        keys = lowered(keys)
        if isinstance(keys, slice):
            out = self._get_from_tuple((keys,))
        elif isinstance(keys, tuple):
            out = self._get_from_tuple(keys)
        elif _contains_ellipsis(keys):
            out = self.as_array()[keys]
        elif _is_ints(keys):
            out = self.as_array()[..., keys]
        else:
            out = super().__getitem__(keys)
        return out

    def __contains__(self, vals):
        if isinstance(vals, (list, tuple)) and all(
            isinstance(s, str) for s in vals
        ):
            return all(s.lower() in self for s in vals)
        return super().__contains__(vals)

    def init_new(self, new_dset):
        """Return an updated DatasetWrapper with coords and data_vars replaced
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
        return DatasetWrapper(coords=coords, data_vars=data_vars)

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
            return pd.to_datetime(self.indexes['time'])
        return None

    @time_index.setter
    def time_index(self, value):
        """Update the time_index attribute with given index."""
        self.indexes['time'] = value

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

    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)
        return out if len(out) > 1 else out[0]

    return wrapper


class Data:
    """Interface for interacting with tuples / lists of :class:`DatasetWrapper`
    objects. These objects are distinct from :class:`Collection` objects, which
    also contain multiple data members, because these members have some
    relationship with each other (they can be low / high res pairs, they can be
    daily / hourly versions of the same data, etc). Collections contain
    completely independent instances."""

    def __init__(self, data: Union[List[xr.Dataset], List[DatasetWrapper]]):
        if not isinstance(data, (list, tuple)):
            data = (data,)
        self.dsets = tuple(DatasetWrapper(d) for d in data)
        self.n_members = len(self.dsets)

    @property
    def attrs(self):
        """Return meta data attributes of members."""
        return [d.attrs for d in self.dsets]

    @attrs.setter
    def attrs(self, value):
        """Set meta data attributes of all data members."""
        for d in self.dsets:
            for k, v in value.items():
                d.attrs[k] = v

    def __getattr__(self, attr):
        try:
            out = [getattr(d, attr) for d in self.dsets]
        except Exception as e:
            msg = f'{self.__class__.__name__} has no attribute "{attr}"'
            raise AttributeError(msg) from e
        return out if len(out) > 1 else out[0]

    @single_member_check
    def __getitem__(self, keys):
        """Method for accessing self.dset or attributes. If keys is a list of
        tuples or list this is interpreted as a request for
        `self.dset[i][keys[i]] for i in range(len(keys)).` Otherwise we will
        get keys from each member of self.dset."""
        if isinstance(keys, (tuple, list)) and all(
            isinstance(k, (tuple, list)) for k in keys
        ):
            out = ArrayTuple([d[key] for d, key in zip(self.dsets, keys)])
        else:
            out = ArrayTuple(d[keys] for d in self.dsets)
        return out

    @single_member_check
    def isel(self, *args, **kwargs):
        """Multi index selection method."""
        out = tuple(d.isel(*args, **kwargs) for d in self.dsets)
        return out

    @single_member_check
    def sel(self, *args, **kwargs):
        """Multi dimension selection method."""
        out = tuple(d.sel(*args, **kwargs) for d in self.dsets)
        return out

    @property
    def shape(self):
        """We use the shape of the largest data member. These are assumed to be
        ordered as (low-res, high-res) if there are two members."""
        return [d.shape for d in self.dsets][-1]

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
