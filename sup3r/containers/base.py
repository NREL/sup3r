"""Base Container classes. These are general objects that contain data. Data
wranglers, data samplers, data loaders, batch handlers, etc are all
containers."""

import copy
import inspect
import logging
import pprint
from typing import Optional

import numpy as np
import xarray as xr

from sup3r.containers.abstract import Data

logger = logging.getLogger(__name__)


class Container:
    """Basic fundamental object used to build preprocessing objects. Contains
    a (or multiple) wrapped xr.Dataset objects (:class:`Data`) and some methods
    for getting data / attributes."""

    def __init__(self, data: Optional[xr.Dataset] = None):
        self.data = data
        self._features = None

    def __new__(cls, *args, **kwargs):
        """Include arg logging in construction."""
        instance = super().__new__(cls)
        cls._log_args(args, kwargs)
        return instance

    @classmethod
    def _log_args(cls, args, kwargs):
        """Log argument names and values."""
        arg_spec = inspect.getfullargspec(cls.__init__)
        args = args or []
        defaults = arg_spec.defaults or []
        arg_names = arg_spec.args[1 : len(args) + 1]
        kwargs_names = arg_spec.args[-len(defaults) :]
        args_dict = dict(zip(kwargs_names, defaults))
        args_dict.update(dict(zip(arg_names, args)))
        args_dict.update(kwargs)
        logger.info(
            f'Initialized {cls.__name__} with:\n'
            f'{pprint.pformat(args_dict, indent=2)}'
        )

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
        if isinstance(data, xr.Dataset):
            self._data = Data(data)
        else:
            self._data = data

    @property
    def features(self):
        """Features in this container."""
        if self._features is None:
            self._features = self.data.features
        return self._features

    @features.setter
    def features(self, val):
        """Set features in this container."""
        self._features = [f.lower() for f in val]

    def __getitem__(self, keys):
        """Method for accessing self.data or attributes. keys can optionally
        include a feature name as the first element of a keys tuple"""
        if self.is_multi_container:
            return tuple([d[key] for d, key in zip(self.data, keys)])
        return self.data[keys]

    def consistency_check(self, keys):
        """Check if all Data objects contained have the same value for
        `keys`."""
        msg = (f'Requested {keys} attribute from a container with '
               f'{len(self.data)} Data objects but these objects do not all '
               f'have the same value for {keys}.')
        attr = getattr(self.data[0], keys, None)
        check = all(getattr(d, keys, None) == attr for d in self.data)
        if not check:
            logger.error(msg)
            raise ValueError(msg)

    def get_multi_attr(self, keys):
        """Get attribute while containing multiple :class:`Data` objects."""
        if hasattr(self.data[0], keys):
            self.consistency_check(keys)
        return getattr(self.data[0], keys)

    def __getattr__(self, keys):
        if keys in self.__dict__:
            return self.__dict__[keys]
        if self.is_multi_container:
            return self.get_multi_attr(keys)
        if hasattr(self.data, keys):
            return getattr(self.data, keys)
        if keys in dir(self):
            return super().__getattribute__(keys)
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
