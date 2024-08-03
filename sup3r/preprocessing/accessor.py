"""Accessor for xarray."""

import logging
from typing import Dict, Union
from warnings import warn

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import mode
from typing_extensions import Self

from sup3r.preprocessing.names import Dimension
from sup3r.preprocessing.utilities import (
    _contains_ellipsis,
    _is_strings,
    _lowered,
    _mem_check,
    _parse_ellipsis,
    dims_array_tuple,
    ordered_array,
    ordered_dims,
    parse_to_list,
)
from sup3r.typing import T_Array

logger = logging.getLogger(__name__)


@xr.register_dataset_accessor('sx')
class Sup3rX:
    """Accessor for xarray - the suggested way to extend xarray functionality.

    References
    ----------
    https://docs.xarray.dev/en/latest/internals/extending-xarray.html

    Note
    ----
    (1) This is an `xr.Dataset` style object which all `xr.Dataset` methods,
    plus more. Maybe the most important part of this interface is parsing
    __getitem__` calls of the form `ds.sx[keys]`. `keys` can be a list of
    features and combinations of feature lists with numpy style indexing.
    e.g. `ds.sx['u', slice(0, 10), ...]` or
    `ds.sx[['u', 'v'], ..., slice(0, 10)]`.
        (i) If ds[keys] returns an `xr.Dataset` object then ds.sx[keys] will
        return a Sup3rX object. e.g. `ds.sx[['u','v']]`) will return a
        :class:`Sup3rX` instance but ds.sx['u'] will return an `xr.DataArray`
        (ii) Combining named feature requests with numpy style indexing will
        return either a dask.array or numpy.array, depending on whether data is
        still on disk or loaded into memory, with a standard dimension order.
        e.g. ds.sx[['u','v'], ...] will return an array with shape (lats, lons,
        times, features), (assuming there is no vertical dimension in the
        underlying data).
    (2) The `__getitem__` and `__getattr__` methods will cast back to
    `type(self)` if `self._ds.__getitem__` or `self._ds.__getattr__` returns an
    instance of `type(self._ds)` (e.g. an `xr.Dataset`). This means we do not
    have to constantly append `.sx` for successive calls to accessor methods.

    Examples
    --------
    >>> ds = xr.Dataset(...)
    >>> feature_data = ds.sx[features]
    >>> ti = ds.sx.time_index
    >>> lat_lon_array = ds.sx.lat_lon
    """

    def __init__(self, ds: Union[xr.Dataset, Self]):
        """Initialize accessor.

        Parameters
        ----------
        ds : xr.Dataset | xr.DataArray
            xarray Dataset instance to access with the following methods
        """
        self._ds = ds
        self._features = None
        self.time_slice = None

    def compute(self, **kwargs):
        """Load `._ds` into memory. This updates the internal `xr.Dataset` if
        it has not been loaded already."""
        if not self.loaded:
            logger.debug(f'Loading dataset into memory: {self._ds}')
            logger.debug(f'Pre-loading: {_mem_check()}')

            for f in self._ds.data_vars:
                self._ds[f] = self._ds[f].compute(**kwargs)
                logger.debug(f'Loaded {f} into memory. {_mem_check()}')
            logger.debug(f'Loaded dataset into memory: {self._ds}')
            logger.debug(f'Post-loading: {_mem_check()}')

    @property
    def loaded(self):
        """Check if data has been loaded as numpy arrays."""
        return all(
            isinstance(self._ds[f].data, np.ndarray)
            for f in list(self._ds.data_vars)
        )

    @property
    def flattened(self):
        """Check if the contained data is flattened 2D data or 3D rasterized
        data."""
        return Dimension.FLATTENED_SPATIAL in self.dims

    @property
    def time_independent(self):
        """Check if the contained data is time independent."""
        return Dimension.TIME not in self.dims

    def update_ds(self, new_dset, attrs=None):
        """Update `self._ds` with coords and data_vars replaced with those
        provided. These are both provided as dictionaries {name: dask.array}.

        Parameters
        ----------
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
        return type(self)(self._ds)

    def __getattr__(self, attr):
        """Get attribute and cast to type(self) if a xr.Dataset is returned
        first."""
        out = getattr(self._ds, attr)
        return type(self)(out) if isinstance(out, xr.Dataset) else out

    def __mul__(self, other):
        """Multiply Sup3rX object by other. Used to compute weighted means and
        stdevs."""
        try:
            return type(self)(other * self._ds)
        except Exception as e:
            raise NotImplementedError(
                f'Multiplication not supported for type {type(other)}.'
            ) from e

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        """Raise Sup3rX object to an integer power. Used to compute weighted
        standard deviations."""
        try:
            return type(self)(self._ds**other)
        except Exception as e:
            raise NotImplementedError(
                f'Exponentiation not supported for type {type(other)}.'
            ) from e

    @property
    def name(self):
        """Name of dataset. Used to label datasets when grouped in
        :class:`Data` objects. e.g. for low / high res pairs or daily / hourly
        data."""
        return self._ds.attrs.get('name', None)

    def sample(self, idx):
        """Get sample from self._ds. The idx should be a tuple of slices for
        the dimensions (south_north, west_east, time) and a list of feature
        names."""
        isel_kwargs = dict(zip(Dimension.dims_3d(), idx[:-1]))
        features = (
            self.features if not _is_strings(idx[-1]) else _lowered(idx[-1])
        )
        out = self._ds[features].isel(**isel_kwargs)
        return out.to_array().transpose(*ordered_dims(out.dims), ...).data

    @name.setter
    def name(self, value):
        """Set name of dataset."""
        self._ds.attrs['name'] = value

    def isel(self, *args, **kwargs):
        """Override xr.Dataset.sel to cast back to Sup3rX object."""
        return type(self)(self._ds.isel(*args, **kwargs))

    @property
    def dims(self):
        """Return dims with our own enforced ordering."""
        return ordered_dims(self._ds.dims)

    def _stack_features(self, arrs):
        return (
            da.stack(arrs, axis=-1)
            if not self.loaded
            else np.stack(arrs, axis=-1)
        )

    def as_array(self, features='all', data=None) -> T_Array:
        """Return dask.array for the contained xr.Dataset."""
        data = data if data is not None else self._ds
        features = parse_to_list(data=data, features=features)
        arrs = [
            data[f].transpose(*ordered_dims(data[f].dims), ...).data
            for f in features
        ]
        if all(arr.shape == arrs[0].shape for arr in arrs):
            return self._stack_features(arrs)
        return self.as_darray(features=features, data=data).data

    def as_darray(self, features='all', data=None) -> xr.DataArray:
        """Return xr.DataArray for the contained xr.Dataset."""
        data = data if data is not None else self._ds
        features = parse_to_list(data=data, features=features)
        features = features if isinstance(features, list) else [features]
        out = data[features]
        return out.to_array().transpose(*ordered_dims(out.dims), ...)

    def mean(self, **kwargs):
        """Get mean directly from dataset object."""
        features = kwargs.pop('features', None)
        out = (
            self._ds[features].mean(**kwargs)
            if features is not None
            else self._ds.mean(**kwargs)
        )
        return type(self)(out) if isinstance(out, xr.Dataset) else out

    def std(self, **kwargs):
        """Get std directly from dataset object."""
        features = kwargs.pop('features', None)
        out = (
            self._ds[features].std(**kwargs)
            if features is not None
            else self._ds.std(**kwargs)
        )
        return type(self)(out) if isinstance(out, xr.Dataset) else out

    def normalize(self, means, stds):
        """Normalize dataset using given means and stds. These are provided as
        dictionaries."""
        feats = set(self._ds.data_vars).intersection(means).intersection(stds)
        for f in feats:
            self._ds[f] = (self._ds[f] - means[f]) / stds[f]

    def interpolate_na(self, **kwargs):
        """Use `xr.DataArray.interpolate_na` to fill NaN values with a dask
        compatible method."""
        features = kwargs.pop('features', list(self.data_vars))
        fill_value = kwargs.pop('fill_value', 'extrapolate')
        for feat in features:
            if 'dim' in kwargs:
                if kwargs['dim'] == Dimension.TIME:
                    kwargs['use_coordinate'] = kwargs.get(
                        'use_coordinate', False
                    )
                self._ds[feat] = self._ds[feat].interpolate_na(
                    **kwargs, fill_value=fill_value
                )
            else:
                horiz = (
                    self._ds[feat]
                    .chunk({Dimension.WEST_EAST: -1})
                    .interpolate_na(
                        dim=Dimension.WEST_EAST,
                        **kwargs,
                        fill_value=fill_value,
                    )
                )
                vert = (
                    self._ds[feat]
                    .chunk({Dimension.SOUTH_NORTH: -1})
                    .interpolate_na(
                        dim=Dimension.SOUTH_NORTH,
                        **kwargs,
                        fill_value=fill_value,
                    )
                )
                self._ds[feat] = (
                    self._ds[feat].dims,
                    (horiz.data + vert.data) / 2.0,
                )
        return type(self)(self._ds)

    def _parse_keys(self, keys):
        """Return set of features and slices for all dimensions contained in
        dataset that can be passed to isel and transposed to standard dimension
        order."""
        standard_dims = ordered_dims(self._ds.dims)
        keys = keys if isinstance(keys, tuple) else (keys,)
        features = (
            list(self.coords)
            if not keys[0]
            else _lowered(keys[0])
            if _is_strings(keys[0]) and keys[0] != 'all'
            else self.features
        )
        dim_keys = () if len(keys) == 1 else keys[1:]
        slices = _parse_ellipsis(dim_keys, dim_num=len(standard_dims))
        return features, dict(zip(standard_dims, slices))

    def __getitem__(self, keys) -> Union[T_Array, Self]:
        """Method for accessing variables or attributes. keys can optionally
        include a feature name as the last element of a keys tuple."""
        features, slices = self._parse_keys(keys)
        out = self._ds[features]
        slices = {k: v for k, v in slices.items() if k in out.dims}
        if slices:
            out = out.isel(**slices)
        if isinstance(keys, (slice, tuple)) or _contains_ellipsis(keys):
            if isinstance(out, xr.DataArray):
                return out.transpose(*ordered_dims(out.dims), ...).data
            return self.as_array(data=out, features=features)
        if isinstance(out, xr.Dataset):
            return type(self)(out)
        return out.transpose(*ordered_dims(out.dims), ...)

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
        feature_check = isinstance(vals, (list, tuple)) and all(
            isinstance(s, str) for s in vals
        )
        if feature_check:
            return all(s.lower() in self._ds for s in vals)
        return self._ds.__contains__(vals)

    def _add_dims_to_data_dict(self, vals):
        """Add dimensions to vals entries if needed. This is used to set values
        of `self._ds` which can require dimensions to be explicitly specified
        for the data being set. e.g. self._ds['u_100m'] = (('south_north',
        'west_east', 'time'), data). We add attributes if available in vals,
        as well"""
        new_vals = {}
        for k, v in vals.items():
            if isinstance(v, tuple):
                new_vals[k] = v
            elif isinstance(v, xr.DataArray):
                data = (
                    ordered_array(v).squeeze(dim='variable').data
                    if 'variable' in v.dims
                    else ordered_array(v).data
                )
                new_vals[k] = (
                    ordered_dims(v.dims),
                    data,
                    getattr(v, 'attrs', {}),
                )
            elif isinstance(v, xr.Dataset):
                data = (
                    ordered_array(v[k]).squeeze(dim='variable').data
                    if 'variable' in v[k].dims
                    else ordered_array(v[k]).data
                )
                new_vals[k] = (
                    ordered_dims(v.dims),
                    data,
                    getattr(v[k], 'attrs', {}),
                )
            elif k in self._ds.data_vars:
                new_vals[k] = (self._ds[k].dims, v)
            elif len(v.shape) > 1:
                val = dims_array_tuple(v)
                msg = (
                    f'Setting data for variable "{k}" without explicitly '
                    f'providing dimensions. Using dims = {tuple(val[0])}.'
                )
                logger.warning(msg)
                warn(msg)
                new_vals[k] = val
            else:
                new_vals[k] = v
        return new_vals

    def assign_coords(self, vals: Dict[str, Union[T_Array, tuple]]):
        """Override :meth:`assign_coords` to enable assignment without
        explicitly providing dimensions if coordinate already exists.

        Parameters
        ----------
        vals : dict
            Dictionary of coord names and either arrays or tuples of (dims,
            array). If dims are not provided this will try to use stored dims
            of the coord, if it exists already.
        """
        self._ds = self._ds.assign_coords(self._add_dims_to_data_dict(vals))
        return type(self)(self._ds)

    def assign(self, vals: Dict[str, Union[T_Array, tuple]]):
        """Override xarray assign method to enable update without explicitly
        providing dimensions if variable already exists.

        Parameters
        ----------
        vals : dict
            Dictionary of variable names and either arrays or tuples of (dims,
            array). If dims are not provided this will try to use stored dims
            of the variable, if it exists already.
        """
        self._ds = self._ds.assign(self._add_dims_to_data_dict(vals))
        return type(self)(self._ds)

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
            _ = self.assign({v: data[..., i] for i, v in enumerate(keys)})
        elif isinstance(keys, str) and keys in self.coords:
            _ = self.assign_coords({keys: data})
        elif isinstance(keys, str):
            _ = self.assign({keys.lower(): data})
        elif isinstance(keys[0], str) and keys[0] not in self.coords:
            var_array = self._ds[keys[0].lower()].data
            var_array[keys[1:]] = data
            _ = self.assign({keys[0].lower(): var_array})
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
    def time_step(self):
        """Get time step in seconds."""
        return float(
            mode(
                (self.time_index[1:] - self.time_index[:-1]).total_seconds(),
                keepdims=False,
            ).mode
        )

    @property
    def lat_lon(self) -> T_Array:
        """Base lat lon for contained data."""
        return self.as_array(
            features=[Dimension.LATITUDE, Dimension.LONGITUDE]
        )

    @lat_lon.setter
    def lat_lon(self, lat_lon):
        """Update the lat_lon attribute with array values."""
        self[[Dimension.LATITUDE, Dimension.LONGITUDE]] = lat_lon

    @property
    def target(self):
        """Return the value of the lower left hand coordinate."""
        return np.asarray(self.lat_lon[-1, 0])

    @property
    def grid_shape(self):
        """Return the shape of the spatial dimensions."""
        return self.lat_lon.shape[:-1]

    @property
    def meta(self):
        """Return dataframe of flattened lat / lon values."""
        return pd.DataFrame(
            columns=[Dimension.LATITUDE, Dimension.LONGITUDE],
            data=self.lat_lon.reshape((-1, 2)),
        )

    def unflatten(self, grid_shape):
        """Convert flattened dataset into rasterized dataset with the given
        grid shape."""
        assert self.flattened, 'Dataset is already unflattened'
        ind = pd.MultiIndex.from_product(
            (np.arange(grid_shape[0]), np.arange(grid_shape[1])),
            names=Dimension.dims_2d(),
        )
        self._ds = self._ds.assign({Dimension.FLATTENED_SPATIAL: ind}).unstack(
            Dimension.FLATTENED_SPATIAL
        )
        return type(self)(self._ds)
