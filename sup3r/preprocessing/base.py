"""Base container classes - object that contains data. All objects that
interact with data are containers. e.g. loaders, extracters, data handlers,
samplers, batch queues, batch handlers.
"""

import logging
from typing import Optional, Tuple

import dask.array as da
import numpy as np
import xarray as xr

from sup3r.preprocessing.common import _log_args, lowered

logger = logging.getLogger(__name__)


class ArrayTuple(tuple):
    """Wrapper to add some useful methods to tuples of arrays. These are
    frequently returned from the :class:`Data` class, especially when there
    are multiple members of `.dsets`. We want to be able to calculate shapes,
    sizes, means, stds on these tuples."""

    def size(self):
        """Compute the total size across all tuple members."""
        return np.sum(d.sx.size for d in self)

    def mean(self):
        """Compute the mean across all tuple members."""
        return da.mean(da.array([d.mean() for d in self]))

    def std(self):
        """Compute the standard deviation across all tuple members."""
        return da.mean(da.array([d.std() for d in self]))


class Data:
    """Interface for interacting with tuples / lists of `xarray.Dataset`
    objects. This class is distinct from :class:`Collection`, which also can
    contain multiple data members, because the members contained here have some
    relationship with each other (they can be low / high res pairs, they can be
    daily / hourly versions of the same data, etc). Collections contain
    completely independent instances."""

    def __init__(self, data: Tuple[xr.Dataset] | xr.Dataset):
        self.dsets = data

    def __len__(self):
        return len(self.dsets) if isinstance(self.dsets, tuple) else 1

    def __getattr__(self, attr):
        """Get attribute through accessor if available. Otherwise use standard
        xarray interface."""
        try:
            out = [
                getattr(d.sx, attr)
                if hasattr(d.sx, attr)
                else getattr(d, attr)
                for d in self
            ]
        except Exception as e:
            msg = f'{self.__class__.__name__} has no attribute "{attr}"'
            raise AttributeError(msg) from e
        return out if len(out) > 1 else out[0]

    def __getitem__(self, keys):
        """Method for accessing self.dset or attributes. If keys is a list of
        tuples or list this is interpreted as a request for
        `self.dset[i][keys[i]] for i in range(len(keys)).` Otherwise we will
        get keys from each member of self.dset."""
        if isinstance(keys, (tuple, list)) and all(
            isinstance(k, (tuple, list)) for k in keys
        ):
            out = [d.sx[key] for d, key in zip(self, keys)]
        else:
            out = [d.sx[keys] for d in self]
        return ArrayTuple(out) if len(out) > 1 else out[0]

    @property
    def shape(self):
        """We use the shape of the largest data member. These are assumed to be
        ordered as (low-res, high-res) if there are two members."""
        return [d.sx.shape for d in self][-1]

    def __contains__(self, vals):
        """Check for vals in all of the dset members."""
        return any(d.sx.__contains__(vals) for d in self)

    def __setitem__(self, variable, data):
        """Set dset member values. Check if values is a tuple / list and if
        so interpret this as sending a tuple / list element to each dset
        member. e.g. `vals[0] -> dsets[0]`, `vals[1] -> dsets[1]`, etc"""
        for i, d in enumerate(self):
            dat = data[i] if isinstance(data, (tuple, list)) else data
            d.sx.__setitem__(variable, dat)

    def __iter__(self):
        yield from (self.dsets if len(self) > 1 else (self.dsets,))


class Container:
    """Basic fundamental object used to build preprocessing objects. Contains
    a (or multiple) wrapped xr.Dataset objects (:class:`Data`) and some methods
    for getting data / attributes."""

    def __init__(
        self,
        data: Optional[xr.Dataset | Tuple[xr.Dataset, ...]] = None,
        features: Optional[list] = None,
    ):
        """
        Parameters
        ----------
        data : xr.Dataset | Tuple[xr.Dataset, xr.Dataset]
            Either a single xr.Dataset or a tuple of datasets. Tuple used for
            dual / paired containers like :class:`DualSamplers`.
        """
        self.data = data
        self.features = features
        self.init_member_names()

    def init_member_names(self):
        """Give members unique names if they do not already exist."""
        if self.data is not None:
            for i, d in enumerate(self.data):
                d.attrs.update({'name': d.attrs.get('name', f'member_{i}')})

    @property
    def attrs(self):
        """Attributes for all data members."""
        attrs = {'n_members': len(self.data)}
        for d in self.data:
            attrs.update(d.attrs)
        return attrs

    def __new__(cls, *args, **kwargs):
        """Include arg logging in construction."""
        instance = super().__new__(cls)
        _log_args(cls, cls.__init__, *args, **kwargs)
        return instance

    def __contains__(self, vals):
        return vals in self.data

    @property
    def data(self) -> Data:
        """Wrapped xr.Dataset."""
        return self._data

    @data.setter
    def data(self, data):
        """Wrap given data in :class:`Data` to provide additional
        attributes on top of xr.Dataset."""
        self._data = data
        if not isinstance(self._data, Data) and self._data is not None:
            self._data = Data(self._data)

    @property
    def features(self):
        """Features in this container."""
        if not self._features or 'all' in self._features:
            self._features = self.data.features
        return self._features

    @features.setter
    def features(self, val):
        """Set features in this container."""
        self._features = (
            lowered([val]) if isinstance(val, str) else lowered(val)
        )

    def __getitem__(self, keys):
        """Method for accessing self.data or attributes. keys can optionally
        include a feature name as the first element of a keys tuple"""
        return self.data[keys]

    def __getattr__(self, attr):
        try:
            return getattr(self.data, attr)
        except Exception as e:
            msg = f'{self.__class__.__name__} object has no attribute "{attr}"'
            raise AttributeError(msg) from e
