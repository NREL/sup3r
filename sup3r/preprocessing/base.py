"""Base container classes - object that contains data. All objects that
interact with data are containers. e.g. loaders, extracters, data handlers,
samplers, batch queues, batch handlers.
"""

import logging
from collections import namedtuple
from typing import Dict, Optional, Tuple

import dask.array as da
import numpy as np
import xarray as xr

import sup3r.preprocessing.accessor  # noqa: F401
from sup3r.preprocessing.accessor import Sup3rX
from sup3r.preprocessing.common import _log_args

logger = logging.getLogger(__name__)


class Sup3rDataset:
    """Interface for interacting with one or two `xr.Dataset` instances
    This is either a simple passthrough for a `xr.Dataset` instance or a
    wrapper around two of them so they work well with Dual objects like
    DualSampler, DualExtracter, DualBatchHandler, etc...)

    Note
    ----
    This may seem similar to :class:`Collection`, which also can
    contain multiple data members, but members of :class:`Collection` objects
    are completely independent while here there are at most two members which
    are related as low / high res versions of the same underlying data."""

    def __init__(self, **dsets: Dict[str, xr.Dataset]):
        dsets = {
            k: Sup3rX(v) if isinstance(v, xr.Dataset) else v
            for k, v in dsets.items()
        }
        self._ds = namedtuple('Dataset', list(dsets))(**dsets)

    def __iter__(self):
        yield from self._ds

    @property
    def dtype(self):
        """Get datatype of first member. Assumed to be constant for all
        members."""
        return self._ds[0].dtype

    def __len__(self):
        return len(self._ds)

    def __getattr__(self, attr):
        """Get attribute through accessor if available. Otherwise use standard
        xarray interface."""
        if hasattr(self._ds, attr):
            return getattr(self._ds, attr)
        out = [self._getattr(ds, attr) for ds in self._ds]
        if len(self._ds) == 1:
            out = out[0]
        return out

    def _getattr(self, dset, attr):
        """Get attribute from single data member."""
        return (
            getattr(dset.sx, attr)
            if hasattr(dset.sx, attr)
            else getattr(dset, attr)
        )

    def _getitem(self, dset, item):
        """Get item from single data member."""
        return dset.sx[item] if hasattr(dset, 'sx') else dset[item]

    def get_dual_item(self, keys):
        """Method for getting items from self._ds when it consists of two
        datasets. If keys is a `List[Tuple]` or `List[List]` this is
        interpreted as a request for `self._ds[i][keys[i]] for i in
        range(len(keys)).` Otherwise we will get keys from each member of
        self.dset.

        Note
        ----
        This casts back to `type(self)` before final return if result of get
        item from each member of `self._ds` is a tuple of `Sup3rX` instances
        """
        if isinstance(keys, (tuple, list)) and all(
            isinstance(k, (tuple, list)) for k in keys
        ):
            out = tuple(
                self._getitem(d, key) for d, key in zip(self._ds, keys)
            )
        else:
            out = tuple(self._getitem(d, keys) for d in self._ds)
        return (
            type(self)(**dict(zip(self._ds._fields, out)))
            if all(isinstance(o, Sup3rX) for o in out)
            else out
        )

    def __getitem__(self, keys):
        """If keys is an int this is interpreted as a request for that member
        of self._ds. If self._ds consists of two members we call
        :meth:`get_dual_item`. Otherwise we get the item from the single member
        of self._ds."""
        if isinstance(keys, int):
            return self._ds[keys]
        if len(self._ds) == 1:
            return self._ds[-1][keys]
        return self.get_dual_item(keys)

    @property
    def shape(self):
        """We use the shape of the largest data member. These are assumed to be
        ordered as (low-res, high-res) if there are two members."""
        return self._ds[-1].shape

    @property
    def data_vars(self):
        """The data_vars are determined by the set of data_vars from all data
        members.

        Note
        ----
        We use features to refer to our own selections and data_vars to refer
        to variables contained in datasets independent of our use of them. e.g.
        a dset might contain ['u', 'v', 'potential_temp'] = data_vars, while
        the features we use might just be ['u','v']
        """
        data_vars = list(self._ds[0].data_vars)
        [
            data_vars.append(f)
            for f in list(self._ds[-1].data_vars)
            if f not in data_vars
        ]
        return data_vars

    @property
    def size(self):
        """Return number of elements in the largest data member."""
        return np.prod(self.shape)

    def __contains__(self, vals):
        """Check for vals in all of the dset members."""
        return any(d.sx.__contains__(vals) for d in self._ds)

    def __setitem__(self, variable, data):
        """Set dset member values. Check if values is a tuple / list and if
        so interpret this as sending a tuple / list element to each dset
        member. e.g. `vals[0] -> dsets[0]`, `vals[1] -> dsets[1]`, etc"""
        for i, d in enumerate(self):
            dat = data[i] if isinstance(data, (tuple, list)) else data
            d.sx.__setitem__(variable, dat)

    def mean(self, skipna=True):
        """Compute the mean across all tuple members."""
        return da.nanmean(da.array([d.mean(skipna=skipna) for d in self._ds]))

    def std(self, skipna=True):
        """Compute the standard deviation across all tuple members."""
        return da.nanmean(da.array([d.std(skipna=skipna) for d in self._ds]))


class Container:
    """Basic fundamental object used to build preprocessing objects. Contains
    a xr.Dataset or wrapped tuple of xr.Dataset objects (:class:`Sup3rDataset`)
    """

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
        self.data = data

    @property
    def data(self) -> Sup3rX:
        """Return a wrapped 1-tuple or 2-tuple xr.Dataset."""
        return self._data

    @data.setter
    def data(self, data):
        """Set data value. Cast to Sup3rX accessor if not already"""
        self._data = Sup3rX(data) if isinstance(data, xr.Dataset) else data

    def __new__(cls, *args, **kwargs):
        """Include arg logging in construction."""
        instance = super().__new__(cls)
        _log_args(cls, cls.__init__, *args, **kwargs)
        return instance

    @property
    def shape(self):
        """Get shape of underlying data."""
        return self.data.shape

    def __contains__(self, vals):
        return vals in self.data

    def __getitem__(self, keys):
        """Get item from underlying data."""
        return self.data[keys]

    def __getattr__(self, attr):
        """Check if attribute is available from `.data`"""
        if hasattr(self.data, attr):
            return getattr(self.data, attr)
        msg = f'{self.__class__.__name__} object has no attribute "{attr}"'
        raise AttributeError(msg)
