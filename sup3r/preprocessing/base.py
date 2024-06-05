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
from sup3r.preprocessing.accessor import Sup3rX
from sup3r.preprocessing.common import _log_args

logger = logging.getLogger(__name__)


class DatasetTuple(tuple):
    """Interface for interacting with single or pairs of `xr.Dataset` instances
    through the Sup3rX accessor. This is always a wrapper around a 1-tuple or a
    2-tuple of xr.Dataset instances (2-tuple used for Dual objects - e.g.
    DualSampler, DualExtracter, DualBatchHandler, etc...)

    Note
    ----
    This may seem similar to :class:`Collection`, which also can
    contain multiple data members, but members of :class:`Collection` objects
    are completely independent while here there are at most two members which
    are related as low / high res versions of the same underlying data."""

    def rewrap(self, out):
        """Rewrap out as a :class:`DatasetTuple` if out meets type
        conditions."""
        if isinstance(out, (xr.Dataset, xr.DataArray, Sup3rX)):
            out = type(self)((out,))
        elif isinstance(out, tuple) and all(
            isinstance(o, type(self)) for o in out
        ):
            out = type(self)(out)
        return out

    def __getattr__(self, attr):
        """Get attribute through accessor if available. Otherwise use standard
        xarray interface."""
        if not self.is_dual:
            out = self._getattr(self.low_res, attr)
        else:
            out = tuple(
                self._getattr(self.low_res, attr),
                self._getattr(self.high_res, attr),
            )
        return self.rewrap(out)

    @property
    def is_dual(self):
        """Check if self is a dual object or single data member."""
        return len(self) == 2

    def _getattr(self, dset, attr):
        """Get attribute from single data member."""
        return self.rewrap(
            getattr(dset.sx, attr)
            if hasattr(dset.sx, attr)
            else getattr(dset, attr)
        )

    def _getitem(self, dset, item):
        """Get item from single data member."""
        return self.rewrap(
            dset.sx[item] if hasattr(dset, 'sx') else dset[item]
        )

    def get_dual_item(self, keys):
        """Get item method used when this is a dual object (a.k.a. a wrapped
        2-tuple)"""
        if isinstance(keys, (tuple, list)) and all(
            isinstance(k, (tuple, list)) for k in keys
        ):
            out = tuple(self._getitem(d, key) for d, key in zip(self, keys))
        else:
            out = tuple(self._getitem(d, keys) for d in self)
        return out

    def __getitem__(self, keys):
        """Method for accessing self.dset or attributes. If keys is a list of
        tuples or list this is interpreted as a request for
        `self.dset[i][keys[i]] for i in range(len(keys)).` Otherwise we will
        get keys from each member of self.dset."""
        if isinstance(keys, int):
            return super().__getitem__(keys)
        if not self.is_dual:
            out = self._getitem(self.low_res, keys)
        else:
            out = self.get_dual_item(keys)
        return self.rewrap(out)

    @property
    def shape(self):
        """We use the shape of the largest data member. These are assumed to be
        ordered as (low-res, high-res) if there are two members."""
        return self.high_res.sx.shape

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
        data_vars = list(self.low_res.data_vars)
        [
            data_vars.append(f)
            for f in list(self.high_res.data_vars)
            if f not in data_vars
        ]
        return data_vars

    @property
    def low_res(self):
        """Get low res data member."""
        return self[0]

    @property
    def high_res(self):
        """Get high res data member (2nd tuple member if there are two
        members)."""
        return self[-1]

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
    a xr.Dataset or wrapped tuple of xr.Dataset objects (:class:`DatasetTuple`)
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
    def data(self) -> DatasetTuple:
        """Return a wrapped 1-tuple or 2-tuple xr.Dataset."""
        return self._data

    @data.setter
    def data(self, data):
        """Set data value. Wrap as :class:`DatasetTuple` if not already."""
        self._data = data
        if self._data is not None and not isinstance(self._data, DatasetTuple):
            tmp = (
                (DatasetTuple((d,)) for d in data)
                if isinstance(data, tuple)
                else (data,)
            )
            self._data = DatasetTuple(tmp)

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
