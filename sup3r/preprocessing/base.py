"""Base container classes - object that contains data. All objects that
interact with data are containers. e.g. loaders, extracters, data handlers,
samplers, batch queues, batch handlers.
"""

import logging
from typing import Optional, Tuple

import dask.array as da
import numpy as np
import xarray as xr

import sup3r.preprocessing.accessor  # noqa: F401
from sup3r.preprocessing.common import _log_args

logger = logging.getLogger(__name__)


class DatasetTuple(tuple):
    """Interface for interacting with tuples / lists of `xarray.Dataset`
    objects. This class is distinct from :class:`Collection`, which also can
    contain multiple data members, because the members contained here have some
    relationship with each other (they can be low / high res pairs, they can be
    daily / hourly versions of the same data, etc). Collections contain
    completely independent instances."""

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
        if isinstance(keys, int):
            return super().__getitem__(keys)
        if isinstance(keys, (tuple, list)) and all(
            isinstance(k, (tuple, list)) for k in keys
        ):
            out = [d.sx[key] for d, key in zip(self, keys)]
        else:
            out = [d.sx[keys] for d in self]
        return type(self)(out) if len(out) > 1 else out[0]

    @property
    def shape(self):
        """We use the shape of the largest data member. These are assumed to be
        ordered as (low-res, high-res) if there are two members."""
        return [d.sx.shape for d in self][-1]

    @property
    def data_vars(self):
        """The data_vars are determined by the set of data_vars from all data
        members."""
        data_vars = []
        [
            data_vars.append(f)
            for f in np.concatenate([d.data_vars for d in self])
            if f not in data_vars
        ]
        return data_vars

    @property
    def size(self):
        """Return number of elements in the largest data member."""
        return np.prod(self.shape)

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

    def mean(self):
        """Compute the mean across all tuple members."""
        return da.mean(da.array([d.mean() for d in self]))

    def std(self):
        """Compute the standard deviation across all tuple members."""
        return da.mean(da.array([d.std() for d in self]))


class Container:
    """Basic fundamental object used to build preprocessing objects. Contains
    a (or multiple) wrapped xr.Dataset objects (:class:`Data`) and some methods
    for getting data / attributes."""

    def __init__(
        self,
        data: Optional[xr.Dataset | Tuple[xr.Dataset, ...]] = None,
    ):
        """
        Parameters
        ----------
        data : xr.Dataset | Tuple[xr.Dataset, xr.Dataset]
            Either a single xr.Dataset or a tuple of datasets. Tuple used for
            dual / paired containers like :class:`DualSamplers`.
        """
        self.data = DatasetTuple(data) if isinstance(data, tuple) else data

    def __new__(cls, *args, **kwargs):
        """Include arg logging in construction."""
        instance = super().__new__(cls)
        _log_args(cls, cls.__init__, *args, **kwargs)
        return instance

    @property
    def shape(self):
        """Get shape of underlying data."""
        return self.data.sx.shape

    def __contains__(self, vals):
        return vals in self.data

    def __getitem__(self, keys):
        """Method for accessing self.data or attributes. keys can optionally
        include a feature name as the first element of a keys tuple"""
        return self.data[keys]

    def __getattr__(self, attr):
        """Try accessing through Sup3rX accessor first. If not available check
        if available through standard inferface."""
        try:
            return getattr(self.data, attr)
        except Exception as e:
            msg = f'{self.__class__.__name__} object has no attribute "{attr}"'
            raise AttributeError(msg) from e
