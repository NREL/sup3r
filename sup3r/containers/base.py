"""Base Container classes. These are general objects that contain data. Data
wranglers, data samplers, data loaders, batch handlers, etc are all
containers."""

import copy
import logging
from typing import Tuple

import numpy as np

from sup3r.containers.abstract import AbstractContainer, DataObject

logger = logging.getLogger(__name__)


class Container(AbstractContainer):
    """Low level object with access to data, knowledge of the data shape, and
    what variables / features are contained."""

    def __init__(self, obj: DataObject):
        super().__init__(obj)

    @property
    def data(self):
        """Returns the contained data."""
        return self.obj

    @property
    def size(self):
        """'Size' of container."""
        return np.prod(self.shape)

    @property
    def shape(self):
        """Shape of contained data. Usually (lat, lon, time, features)."""
        return self.obj.shape

    @property
    def features(self):
        """Features in this container."""
        return self.obj.features

    def __getitem__(self, key):
        """Method for accessing self.data."""
        return self.obj[key]


class ContainerPair(Container):
    """Pair of two Containers, one for low resolution and one for high
    resolution data."""

    def __init__(self, lr_container: Container, hr_container: Container):
        self.lr_container = lr_container
        self.hr_container = hr_container

    @property
    def data(self) -> Tuple[Container, Container]:
        """Raw data."""
        return (self.lr_container, self.hr_container)

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
