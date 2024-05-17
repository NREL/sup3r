"""Abstract container classes. These are the fundamental objects that all
classes which interact with data (e.g. handlers, wranglers, loaders, samplers,
batchers) are based on."""

import inspect
import logging
import pprint
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class AbstractContainer(ABC):
    """Lowest level object. This is the thing "contained" by Container
    classes.

    Notes
    -----
    class:`Container` implementation just requires: `__getitem__` method and
    `.data`, `.shape` attributes. `.shape` is needed because class:`Container`
    objects interface with class:`Sampler` objects, which need to know the
    shape available for sampling."""

    def __new__(cls, *args, **kwargs):
        """Run check on required attributes and log arguments."""
        instance = super().__new__(cls)
        cls._init_check()
        cls._log_args(args, kwargs)
        return instance

    @classmethod
    def _init_check(cls):
        required = ['data', 'shape']
        missing = [attr for attr in required if not hasattr(cls, attr)]
        if len(missing) > 0:
            msg = f'{cls.__name__} must implement {missing}.'
            raise NotImplementedError(msg)

    @classmethod
    def _log_args(cls, args, kwargs):
        """Log argument names and values."""
        arg_spec = inspect.getfullargspec(cls.__init__)
        args = args or []
        defaults = arg_spec.defaults or []
        arg_names = arg_spec.args[-len(args) - len(defaults):]
        kwargs_names = arg_spec.args[-len(defaults):]
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
    def size(self):
        """Get the "size" of the container."""
        return np.prod(self.shape)
