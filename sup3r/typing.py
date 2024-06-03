"""Types used across preprocessing library."""

from typing import TypeVar

import dask
import numpy as np
import xarray as xr

T_DatasetWrapper = TypeVar('T_DatasetWrapper')
T_Dataset = TypeVar('T_Dataset', T_DatasetWrapper, xr.Dataset)
T_Array = TypeVar('T_Array', np.ndarray, dask.array.core.Array)
