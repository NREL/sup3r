"""Base Container classes. These are general objects that contain data. Data
wranglers, data samplers, data loaders, batch handlers, etc are all
containers."""

import copy
import logging

import xarray as xr

from sup3r.containers.abstract import AbstractContainer

logger = logging.getLogger(__name__)


class Container(AbstractContainer):
    """Low level object containing an xarray.Dataset and some methods for
    selecting data from the dataset"""

    def __init__(self, data: xr.Dataset):
        super().__init__()
        self.data = data
        self._features = list(data.data_vars)

    @property
    def data(self):
        """Returns the contained data."""
        return self._data

    @data.setter
    def data(self, value):
        """Set data values."""
        self._data = value


class DualContainer(AbstractContainer):
    """Pair of two Containers, one for low resolution and one for high
    resolution data."""

    def __init__(self, lr_container: Container, hr_container: Container):
        self.lr_container = lr_container
        self.hr_container = hr_container
        self.data = (self.lr_container.data, self.hr_container.data)
        self.shape = (lr_container.shape, hr_container.shape)
        feats = list(copy.deepcopy(self.lr_container.features))
        feats += [fn for fn in self.hr_container.features if fn not in feats]
        self._features = feats

    def __getitem__(self, keys):
        """Method for accessing self.data."""
        lr_key, hr_key = keys
        return (self.lr_container[lr_key], self.hr_container[hr_key])
