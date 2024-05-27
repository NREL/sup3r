"""Composite objects that wrangle data. DataHandlers are the typical
example."""

from .h5 import DataHandlerH5SolarCC, DataHandlerH5WindCC
from .nc import DataHandlerNCforCC, DataHandlerNCforCCwithPowerLaw
