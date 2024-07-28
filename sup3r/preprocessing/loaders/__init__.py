"""Container subclass with additional methods for loading the contained
data."""

from typing import ClassVar

from sup3r.preprocessing.base import TypeAgnosticClass

from .base import BaseLoader
from .h5 import LoaderH5
from .nc import LoaderNC


class Loader(TypeAgnosticClass):
    """`Loader` class which parses input file type and returns
    appropriate `TypeSpecificLoader`."""

    TypeSpecificClasses: ClassVar = {'nc': LoaderNC, 'h5': LoaderH5}
