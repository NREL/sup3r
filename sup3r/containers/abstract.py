"""Abstract container classes. These are the fundamental objects that all
classes which interact with data (e.g. handlers, wranglers, loaders, samplers,
batchers) are based on."""
import inspect
import logging
import pprint
from abc import ABC, ABCMeta, abstractmethod

logger = logging.getLogger(__name__)


class _ContainerMeta(ABCMeta, type):
    """Custom meta for ensuring Container subclasses have the required
    attributes and for logging arg names / values upon initialization"""

    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj._init_check()
        if hasattr(cls, '__init__'):
            obj._log_args(args, kwargs)
        return obj


class AbstractContainer(ABC, metaclass=_ContainerMeta):
    """Lowest level object. This is the thing "contained" by Container
    classes. It just has a `__getitem__` method and `.data`, `.shape`,
    `.features` attributes"""

    def _init_check(self):
        required = ['data', 'features', 'shape']
        missing = [attr for attr in required if not hasattr(self, attr)]
        if len(missing) > 0:
            msg = (f'{self.__class__.__name__} must implement {missing}.')
            raise NotImplementedError(msg)

    @classmethod
    def _log_args(cls, args, kwargs):
        """Log argument names and values."""
        arg_spec = inspect.getfullargspec(cls.__init__)
        args = args or []
        defaults = arg_spec.defaults or []
        arg_vals = [*args, *defaults]
        arg_names = arg_spec.args[1:]  # exclude self
        args_dict = dict(zip(arg_names, arg_vals))
        args_dict.update(kwargs)
        logger.info(f'Initialized {cls.__name__} with:\n'
                    f'{pprint.pformat(args_dict, indent=2)}')

    @abstractmethod
    def __getitem__(self, key):
        """Method for accessing contained data"""
