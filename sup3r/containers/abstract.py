"""Abstract container classes. These are the fundamental objects that all
classes which interact with data (e.g. handlers, wranglers, loaders, samplers,
batchers) are based on."""

import inspect
import logging
import pprint
from abc import ABC, ABCMeta, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class _ContainerMeta(ABCMeta, type):
    def __call__(cls, *args, **kwargs):
        """Check for required attributes"""
        obj = type.__call__(cls, *args, **kwargs)
        obj._init_check()
        return obj


class AbstractContainer(ABC, metaclass=_ContainerMeta):
    """Lowest level object. This is the thing "contained" by Container
    classes.

    Notes
    -----
    :class:`Container` implementation just requires: `__getitem__` method and
    `.data`, `.shape`, `.features` attributes. Both `.shape` and `.features`
    are needed because :class:`Container` objects interface with
    :class:`Sampler` objects, which need to know the shape available for
    sampling and what features are available if they need to be split into lr /
    hr feature sets."""

    def _init_check(self):
        required = ['data', 'features']
        missing = [req for req in required if req not in dir(self)]
        if len(missing) > 0:
            msg = f'{self.__class__.__name__} must implement {missing}.'
            raise NotImplementedError(msg)

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
        args_dict = dict(zip(arg_names, args))
        default_dict = dict(zip(kwargs_names, defaults))
        args_dict.update(default_dict)
        args_dict.update(kwargs)
        logger.info(
            f'Initialized {cls.__name__} with:\n'
            f'{pprint.pformat(args_dict, indent=2)}'
        )

    @abstractmethod
    def __getitem__(self, keys):
        """Method for accessing contained data"""

    @property
    def shape(self):
        """Get shape of contained data."""
        return self.data.shape

    @property
    def size(self):
        """Get the "size" of the container."""
        return np.prod(self.shape)
