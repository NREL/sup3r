"""Composite objects built from loaders, extracters, and derivers."""

from typing import ClassVar

from sup3r.preprocessing.base import TypeAgnosticClass

from .base import ExoData, SingleExoDataStep
from .exo import ExoDataHandler
from .factory import (
    DataHandlerH5,
    DataHandlerH5SolarCC,
    DataHandlerH5WindCC,
    DataHandlerNC,
)
from .nc_cc import DataHandlerNCforCC, DataHandlerNCforCCwithPowerLaw


class DataHandler(TypeAgnosticClass):
    """`DataHandler` class which parses input file type and returns
    appropriate `TypeSpecificDataHandler`."""

    TypeSpecificClasses: ClassVar = {'nc': DataHandlerNC, 'h5': DataHandlerH5}
