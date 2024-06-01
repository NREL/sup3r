"""Types used across preprocessing library."""

from typing import List, Tuple, TypeVar

import dask
import numpy as np
import xarray as xr

T_Array = TypeVar('T_Array', np.ndarray, dask.array.core.Array)
T_Container = TypeVar('T_Container')
T_XArray = TypeVar(
    'T_XArray', xr.Dataset, List[xr.Dataset], Tuple[xr.Dataset, ...]
)
T_XArrayWrapper = TypeVar('T_XArrayWrapper')
T_Data = TypeVar('T_Data')
