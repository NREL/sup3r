"""Base Container classes. These are general objects that contain data. Data
wranglers, data samplers, data loaders, batch handlers, etc are all
containers."""

import copy
import logging
from typing import Self, Tuple

import dask.array
import numpy as np

from sup3r.containers.abstract import AbstractContainer
from sup3r.utilities.utilities import parse_keys

logger = logging.getLogger(__name__)


class Container(AbstractContainer):
    """Low level object with access to data, knowledge of the data shape, and
    what variables / features are contained."""

    def __init__(self, container: Self):
        super().__init__()
        self.container = container
        self._features = self.container.features
        self._data = self.container.data
        self._shape = self.container.shape

    @property
    def data(self) -> dask.array:
        """Returns the contained data."""
        return self._data

    @data.setter
    def data(self, value):
        """Set data values."""
        self._data = value

    @property
    def size(self):
        """'Size' of container."""
        return np.prod(self.shape)

    @property
    def shape(self):
        """Shape of contained data. Usually (lat, lon, time, features)."""
        return self._shape

    @shape.setter
    def shape(self, shape):
        """Set shape value."""
        self._shape = shape

    @property
    def features(self):
        """Features in this container."""
        return self._features

    @features.setter
    def features(self, features):
        """Update features."""
        self._features = features

    def __contains__(self, feature):
        return feature.lower() in [f.lower() for f in self.features]

    def index(self, feature):
        """Get index of feature."""
        return [f.lower() for f in self.features].index(feature.lower())

    def __getitem__(self, keys):
        """Method for accessing self.data or attributes. keys can optionally
        include a feature name as the first element of a keys tuple"""
        key, key_slice = parse_keys(keys)
        if isinstance(key, str):
            if key in self:
                return self.data[*key_slice, self.index(key)]
            if hasattr(self, key):
                return getattr(self, key)
            raise ValueError(f'Could not get item for "{keys}"')
        return self.data[key, *key_slice]

    def __setitem__(self, keys, value):
        """Set values of data or attributes. keys can optionally include a
        feature name as the first element of a keys tuple."""
        key, key_slice = parse_keys(keys)
        if isinstance(key, str):
            if key in self:
                self.data[*key_slice, self.index(key)] = value
            if hasattr(self, key):
                setattr(self, key, value)
            raise ValueError(f'Could not set item for "{keys}"')
        self.data[key, *key_slice] = value


class ContainerPair(Container):
    """Pair of two Containers, one for low resolution and one for high
    resolution data."""

    def __init__(self, lr_container: Container, hr_container: Container):
        self.lr_container = lr_container
        self.hr_container = hr_container

    @property
    def data(self) -> Tuple[dask.array, dask.array]:
        """Raw data."""
        return (self.lr_container.data, self.hr_container.data)

    @property
    def shape(self) -> Tuple[tuple, tuple]:
        """Shape of raw data"""
        return (self.lr_container.shape, self.hr_container.shape)

    def __getitem__(self, keys):
        """Method for accessing self.data."""
        lr_key, hr_key = keys
        return (self.lr_container[lr_key], self.hr_container[hr_key])

    @property
    def features(self):
        """Get a list of data features including features from both the lr and
        hr data handlers"""
        out = list(copy.deepcopy(self.lr_container.features))
        out += [fn for fn in self.hr_container.features if fn not in out]
        return out
