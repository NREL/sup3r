"""Type agnostic classes which parse input file type and returns a type
specific loader."""

from typing import ClassVar

from .base import TypeAgnosticClass
from .data_handlers import DataHandlerH5, DataHandlerNC
from .loaders import LoaderH5, LoaderNC


class Loader(TypeAgnosticClass):
    """`Loader` class which parses input file type and returns
    appropriate `TypeSpecificLoader`."""

    TypeSpecificClasses: ClassVar = {'nc': LoaderNC, 'h5': LoaderH5}


class DataHandler(TypeAgnosticClass):
    """`DataHandler` class which parses input file type and returns
    appropriate `TypeSpecificDataHandler`."""

    TypeSpecificClasses: ClassVar = {'nc': DataHandlerNC, 'h5': DataHandlerH5}
