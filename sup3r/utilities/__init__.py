"""Sup3r utilities"""

import os
import sys
from enum import Enum

import cftime
import dask
import h5netcdf
import netCDF4
import numpy as np
import pandas as pd
import phygnn
import rex
import sklearn
import tensorflow as tf
import xarray

from .._version import __version__

VERSION_RECORD = {
    'sup3r': __version__,
    'tensorflow': tf.__version__,
    'sklearn': sklearn.__version__,
    'pandas': pd.__version__,
    'numpy': np.__version__,
    'nrel-phygnn': phygnn.__version__,
    'nrel-rex': rex.__version__,
    'python': sys.version,
    'xarray': xarray.__version__,
    'h5netcdf': h5netcdf.__version__,
    'dask': dask.__version__,
    'netCDF4': netCDF4.__version__,
    'cftime': cftime.__version__
}


class ModuleName(str, Enum):
    """A collection of the module names available in sup3r.
    Each module name should match the name of the click command
    that will be used to invoke its respective cli. As of 5/26/2022,
    this means that all commands are lowercase with underscores
    replaced by dashes.
    """

    FORWARD_PASS = 'forward-pass'
    DATA_EXTRACT = 'data-extract'
    DATA_COLLECT = 'data-collect'
    QA = 'qa'
    SOLAR = 'solar'
    STATS = 'stats'
    BIAS_CALC = 'bias-calc'
    VISUAL_QA = 'visual-qa'
    REGRID = 'regrid'

    def __str__(self):
        return self.value

    def __format__(self, format_spec):
        return str.__format__(self.value, format_spec)

    @classmethod
    def all_names(cls):
        """All module names.

        Returns
        -------
        set
            The set of all module name strings.
        """
        # pylint: disable=no-member
        return {v.value for v in cls.__members__.values()}
