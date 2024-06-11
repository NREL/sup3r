"""Accessor for xarray."""

import logging

import dask.array as da
import numpy as np
import pandas as pd
import xarray
import xarray as xr

from sup3r.preprocessing.common import (
    Dimension,
    _contains_ellipsis,
    _is_ints,
    _is_strings,
    dims_array_tuple,
    ordered_array,
    ordered_dims,
    parse_features,
    parse_to_list,
)
from sup3r.typing import T_Array

logger = logging.getLogger(__name__)


@xarray.register_dataarray_accessor('sx')
@xarray.register_dataset_accessor('sx')
class Sup3rX:
    """Accessor for xarray - the suggested way to extend xarray functionality.

    References
    ----------
    https://docs.xarray.dev/en/latest/internals/extending-xarray.html

    Note
    ----
    (1) The most important part of this interface is parsing `__getitem__`
    calls of the form `ds.sx[keys]`. `keys` can be a list of features and
    combinations of feature lists with numpy style indexing. e.g. `ds.sx['u',
    slice(0, 10), ...]` or `ds.sx[['u', 'v'], ..., slice(0, 10)]`.
        (i) Using just a feature or list of features (e.g. `ds.sx['u']` or
        `ds.sx[['u','v']]`) will return a :class:`Sup3rX` instance.
        (ii) Combining named feature requests with numpy style indexing will
        return either a dask.array or numpy.array, depending on whether data is
        still on disk or loaded into memory.
        (iii) Using a named feature of list as the first entry (e.g.
        `ds.sx['u', ...]`) will return an array with the feature channel
        squeezed. `ds.sx[..., 'u']`, on the other hand, will keep the feature
        channel so the result will have a trailing dimension of length 1.
    (2) The `__getitem__` and `__getattr__` methods will cast back to
    `type(self)` if `self._ds.__getitem__` or `self._ds.__getattr__` returns an
    instance of `type(self._ds)` (e.g. a `xr.Dataset`). This means we do not
    have to constantly append `.sx` for successive calls to accessor methods.

    Examples
    --------
    >>> ds = xr.Dataset(...)
    >>> ds.sx[features]
    >>> ds.sx.time_index
    >>> ds.sx.lat_lon

    """

    def __init__(self, ds: xr.Dataset | xr.DataArray):
        """Initialize accessor. Order variables to our standard order.

        Parameters
        ----------
        ds : xr.Dataset | xr.DataArray
            xarray Dataset instance to access with the following methods
        """
        self._ds = ds.to_dataset() if isinstance(ds, xr.DataArray) else ds
        self._ds = self.reorder()
        self._features = None

    def compute(self, **kwargs):
        """Load `._ds` into memory. This updates the internal `xr.Dataset` if
        it has not been loaded already."""
        if not self.loaded:
            self._ds = self._ds.compute(**kwargs)

    @property
    def loaded(self):
        """Check if data has been loaded as numpy arrays."""
        return all(
            isinstance(self._ds[f].data, np.ndarray) for f in self.features
        )

    def good_dim_order(self):
        """Check if dims are in the right order for all variables.

        Returns
        -------
        bool
            Whether the dimensions for each variable in self._ds are in our
            standard order (spatial, time, ..., features)
        """
        return all(
            tuple(self._ds[f].dims) == ordered_dims(self._ds[f].dims)
            for f in self._ds.data_vars
        )

    def reorder(self):
        """Reorder dimensions according to our standard.

        Returns
        -------
        _ds : xr.Dataset
            Dataset with all variables in our standard dimension order
            (spatial, time, ..., features)
        """

        if not self.good_dim_order():
            reordered_vars = {
                var: (
                    ordered_dims(self._ds.data_vars[var].dims),
                    ordered_array(self._ds.data_vars[var]).data,
                )
                for var in self._ds.data_vars
            }
            self._ds = xr.Dataset(
                coords=self._ds.coords,
                data_vars=reordered_vars,
                attrs=self._ds.attrs,
            )
        return self._ds

    def update(self, new_dset, attrs=None):
        """Updated the contained dataset with coords and data_vars replaced
        with those provided. These are both provided as dictionaries {name:
        dask.array}.

        Parmeters
        ---------
        new_dset : Dict[str, dask.array]
            Can contain any existing or new variable / coordinate as long as
            they all have a consistent shape.

        Returns
        -------
        _ds : xr.Dataset
            Updated dataset with provided coordinates and data_vars with
            variables in our standard dimension order.
        """
        coords = dict(self._ds.coords)
        data_vars = dict(self._ds.data_vars)
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
        self._ds = xr.Dataset(coords=coords, data_vars=data_vars, attrs=attrs)
        self._ds = self.reorder()
        return self._ds

    def __getattr__(self, attr):
        """Get attribute and cast to type(self) if a xr.Dataset is returned
        first."""
        out = getattr(self._ds, attr)
        if isinstance(out, (xr.Dataset, xr.DataArray)):
            out = type(self)(out)
        return out

    @property
    def name(self):
        """Name of dataset. Used to label datasets when grouped in
        :class:`Data` objects. e.g. for low / high res pairs or daily / hourly
        data."""
        return self._ds.attrs.get('name', None)

    @name.setter
    def name(self, value):
        """Set name of dataset."""
        self._ds.attrs['name'] = value

    def sel(self, *args, **kwargs):
        """Override xr.Dataset.sel to enable feature selection."""
        features = kwargs.pop('features', None)
        if features is not None:
            out = self._ds[features].sel(*args, **kwargs)
        else:
            out = self._ds.sel(*args, **kwargs)
        return type(self)(out)

    def isel(self, *args, **kwargs):
        """Override xr.Dataset.sel to enable feature selection."""
        findices = kwargs.pop('features', None)
        if findices is not None:
            features = [list(self._ds.data_vars)[fidx] for fidx in findices]
            out = self._ds[features].isel(*args, **kwargs)
        else:
            out = self._ds.isel(*args, **kwargs)
        return type(self)(out)

    @property
    def dims(self):
        """Return dims with our own enforced ordering."""
        return ordered_dims(self._ds.dims)

    def as_array(self, features='all') -> T_Array:
        """Return dask.array for the contained xr.Dataset."""
        features = parse_to_list(data=self._ds, features=features)
        arrs = [self._ds[f].data for f in features]
        if all(arr.shape == arrs[0].shape for arr in arrs):
            return (
                da.stack(arrs, axis=-1)
                if not self.loaded
                else np.stack(arrs, axis=-1)
            )
        return self.as_darray(features=features).data

    def as_darray(self, features='all') -> xr.DataArray:
        """Return xr.DataArray for the contained xr.Dataset."""
        features = parse_to_list(data=self._ds, features=features)
        features = features if isinstance(features, list) else [features]
        return self._ds[features].to_dataarray().transpose(*self.dims, ...)

    def mean(self, skipna=True):
        """Get mean directly from dataset object."""
        return self.as_darray().mean(skipna=skipna)

    def std(self, skipna=True):
        """Get std directly from dataset object."""
        return self.as_darray().std(skipna=skipna)

    def _get_from_tuple(self, keys) -> T_Array:
        """
        Parameters
        ----------
        keys : tuple
            Tuple of keys used to get variable data from self._ds. This is
            checked for different patterns (e.g. list of strings as the first
            or last entry is interpreted as requesting the variables for those
            strings)
        """
        if _is_strings(keys[0]):
            out = self.as_array(keys[0])[*keys[1:], :]
            out = out.squeeze(axis=-1) if out.shape[-1] == 1 else out
        elif _is_strings(keys[-1]):
            out = self.as_array(keys[-1])[*keys[:-1], :]
        elif _is_ints(keys[-1]) and not _contains_ellipsis(keys):
            out = self.as_array()[*keys[:-1], ..., keys[-1]]
        else:
            out = self.as_array()[keys]
        return out

    def __getitem__(self, keys) -> T_Array | xr.Dataset:
        """Method for accessing variables or attributes. keys can optionally
        include a feature name as the last element of a keys tuple."""
        keys = parse_features(data=self._ds, features=keys)
        if isinstance(keys, slice):
            out = self._get_from_tuple((keys,))
        elif isinstance(keys, tuple):
            out = self._get_from_tuple(keys)
        elif _contains_ellipsis(keys):
            out = self.as_array()[keys]
        elif _is_ints(keys):
            out = self.as_array()[..., keys]
        else:
            out = self._ds[keys]
        if isinstance(out, (xr.Dataset, xr.DataArray)):
            out = type(self)(out)
        return out

    def __contains__(self, vals):
        """Check if self._ds contains `vals`.

        Parameters
        ----------
        vals : str | list
            Values to check. Can be a list of strings or a single string.

        Examples
        --------
        bool(['u', 'v'] in self)
        bool('u' in self)
        """
        if isinstance(vals, (list, tuple)) and all(
            isinstance(s, str) for s in vals
        ):
            return all(s.lower() in self._ds for s in vals)
        return self._ds.__contains__(vals)

    def __setitem__(self, keys, data):
        """
        Parameters
        ----------
        keys : str | list | tuple
            keys to set. This can be a string like 'temperature' or a list
            like ['u', 'v']. `data` will be iterated over in the latter case.
        data : T_Array | xr.DataArray
            array object used to set variable data. If `variable` is a list
            then this is expected to have a trailing dimension with length
            equal to the length of the list.
        """
        if isinstance(keys, (list, tuple)) and all(
            isinstance(s, str) for s in keys
        ):
            for i, v in enumerate(keys):
                self._ds.update({v: dims_array_tuple(data[..., i])})
        elif isinstance(keys, str):
            keys = keys.lower()
            if hasattr(data, 'dims') and len(data.dims) >= 2:
                self._ds.update({keys: (ordered_dims(data.dims), data)})
            elif hasattr(data, 'shape'):
                self._ds.update({keys: dims_array_tuple(data)})
            else:
                self._ds.update({keys: data})
        elif _is_strings(keys[0]):
            var_array = self[keys[0]].as_array().squeeze()
            var_array[keys[1:]] = data
            self[keys[0]] = var_array
        else:
            msg = f'Cannot set values for keys {keys}'
            raise KeyError(msg)

    @property
    def features(self):
        """Features in this container."""
        return list(self._ds.data_vars)

    @property
    def dtype(self):
        """Get dtype of underlying array."""
        return self.as_array().dtype

    @property
    def shape(self):
        """Get shape of underlying xr.DataArray, using our standard dimension
        order."""
        dim_dict = dict(self._ds.sizes)
        dim_vals = [dim_dict[k] for k in Dimension.order() if k in dim_dict]
        return (*dim_vals, len(self._ds.data_vars))

    @property
    def size(self):
        """Get size of data contained to use in weight calculations."""
        return np.prod(self.shape)

    @property
    def time_index(self):
        """Base time index for contained data."""
        return (
            pd.to_datetime(self._ds.indexes['time'])
            if 'time' in self._ds.indexes
            else None
        )

    @time_index.setter
    def time_index(self, value):
        """Update the time_index attribute with given index."""
        self._ds.indexes['time'] = value

    @property
    def lat_lon(self) -> T_Array:
        """Base lat lon for contained data."""
        return self.as_array([Dimension.LATITUDE, Dimension.LONGITUDE])

    @lat_lon.setter
    def lat_lon(self, lat_lon):
        """Update the lat_lon attribute with array values."""
        self[[Dimension.LATITUDE, Dimension.LONGITUDE]] = lat_lon
