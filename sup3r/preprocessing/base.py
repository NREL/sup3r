"""Base container classes - object that contains data. All objects that
interact with data are containers. e.g. loaders, extracters, data handlers,
samplers, batch queues, batch handlers.
"""

import logging
from typing import Optional

import xarray as xr

from sup3r.preprocessing.common import _log_args, lowered
from sup3r.preprocessing.wrapper import Data

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
