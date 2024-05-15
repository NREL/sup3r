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
    """Custom meta for ensuring class:`Container` subclasses have the required
    attributes and for logging arg names / values upon initialization"""

    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj._init_check()
        if hasattr(cls, '__init__'):
            obj._log_args(args, kwargs)
        return obj


class AbstractContainer(ABC, metaclass=_ContainerMeta):
    """Lowest level object. This is the thing "contained" by Container
    classes.

    Notes
    -----
    class:`Container` implementation just requires: `__getitem__` method and
    `.data`, `.shape` attributes. `.shape` is needed because class:`Container`
    objects interface with class:`Sampler` objects, which need to know the
    shape available for sampling."""

    def _init_check(self):
        required = ['data', 'shape']
        missing = [attr for attr in required if not hasattr(self, attr)]
        if len(missing) > 0:
            msg = (f'{self.__class__.__name__} must implement {missing}.')
            raise NotImplementedError(msg)

    @classmethod
    def _log_args(cls, args, kwargs):
        """Log argument names and values."""
        arg_spec = inspect.getfullargspec(cls.__init__)
        args = args or []
        arg_names = arg_spec.args[1:]  # exclude self
        args_dict = dict(zip(arg_names[:len(args)], args))
        defaults = arg_spec.defaults or []
        default_dict = dict(zip(arg_names[-len(defaults):], defaults))
        args_dict.update(default_dict)
        args_dict.update(kwargs)
        logger.info(f'Initialized {cls.__name__} with:\n'
                    f'{pprint.pformat(args_dict, indent=2)}')

    @abstractmethod
    def __getitem__(self, key):
        """Method for accessing contained data"""

    @property
    def size(self):
        """Get the "size" of the container."""
        return np.prod(self.shape)
