"""Container subclass with additional methods for loading the contained
data."""

from typing import ClassVar

from sup3r.preprocessing.utilities import composite_info, get_source_type

from .base import BaseLoader
from .h5 import LoaderH5
from .nc import LoaderNC


class Loader:
    """`Loader` class which parses input file type and returns
    appropriate `TypeSpecificLoader`."""

    TypeSpecificClasses: ClassVar = {'nc': LoaderNC, 'h5': LoaderH5}

    def __new__(cls, file_paths, **kwargs):
        """Override parent class to return type specific class based on
        `source_file`"""
        SpecificClass = cls.TypeSpecificClasses[get_source_type(file_paths)]
        return SpecificClass(file_paths, **kwargs)

    _signature_objs = (BaseLoader,)
    __doc__ = BaseLoader.__doc__

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, trace):
        self._res.close()
