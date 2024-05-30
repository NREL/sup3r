"""Base container classes - object that contains data. All objects that
interact with data are containers. e.g. loaders, extracters, data handlers,
samplers, batch queues, batch handlers."""

import copy
import logging
from typing import Optional

import numpy as np
import xarray as xr

from sup3r.preprocessing.abstract import Data
from sup3r.preprocessing.common import _log_args, lowered

logger = logging.getLogger(__name__)


class Container:
    """Basic fundamental object used to build preprocessing objects. Contains
    a (or multiple) wrapped xr.Dataset objects (:class:`Data`) and some methods
    for getting data / attributes."""

    def __init__(
        self,
        data: Optional[xr.Dataset] = None,
        features: Optional[list] = None,
    ):
        self.data = data
        self.features = features

    def __new__(cls, *args, **kwargs):
        """Include arg logging in construction."""
        instance = super().__new__(cls)
        _log_args(cls, cls.__init__, *args, **kwargs)
        return instance

    @property
    def is_multi_container(self):
        """Return true if this is contains more than one :class:`Data`
        object."""
        return isinstance(self.data, (tuple, list))

    @property
    def size(self):
        """Get size of contained data. Accounts for possibility of containing
        multiple datasets."""
        if not self.is_multi_container:
            return self.data.size
        return np.sum([d.size for d in self.data])

    @property
    def data(self) -> Data:
        """Wrapped xr.Dataset."""
        return self._data

    @data.setter
    def data(self, data):
        """Wrap given data in :class:`Data` to provide additional
        attributes on top of xr.Dataset."""
        self._data = data
        if isinstance(data, xr.Dataset):
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
        self._features = lowered(val)

    def __getitem__(self, keys):
        """Method for accessing self.data or attributes. keys can optionally
        include a feature name as the first element of a keys tuple"""
        if self.is_multi_container:
            return tuple([d[key] for d, key in zip(self.data, keys)])
        return self.data[keys]

    def get_multi_attr(self, attr):
        """Check if all Data objects contained have the same value for
        `attr` and return attribute."""
        msg = (
            f'Requested {attr} attribute from a container with '
            f'{len(self.data)} Data objects but these objects do not all '
            f'have the same value for {attr}.'
        )
        attr = getattr(self.data[0], attr, None)
        check = all(getattr(d, attr, None) == attr for d in self.data)
        if not check:
            logger.error(msg)
            raise ValueError(msg)
        return attr

    def __getattr__(self, attr):
        if attr in dir(self):
            return self.__getattribute__(attr)
        if self.is_multi_container:
            return self.get_multi_attr(attr)
        if hasattr(self.data, attr):
            return getattr(self.data, attr)
        raise AttributeError


class DualContainer(Container):
    """Pair of two Containers, one for low resolution and one for high
    resolution data."""

    def __init__(self, lr_data: Data, hr_data: Data):
        """
        Parameters
        ----------
        lr_data : Data
            :class:`Data` object containing low-resolution data.
        hr_data : Data
            :class:`Data` object containing high-resolution data.
        """
        self.lr_data = lr_data
        self.hr_data = hr_data
        self.data = (self.lr_data, self.hr_data)
        feats = list(copy.deepcopy(self.lr_data.features))
        feats += [fn for fn in self.hr_data.features if fn not in feats]
        self._features = feats

    def __getitem__(self, keys):
        """Method for accessing self.data."""
        lr_key, hr_key = keys
        return (self.lr_data[lr_key], self.hr_data[hr_key])
