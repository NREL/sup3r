"""Types used across preprocessing library."""

from typing import TypeVar

import dask
import numpy as np
import xarray as xr

T_Dataset = TypeVar(
    'T_Dataset', xr.Dataset, TypeVar('Sup3rX'), TypeVar('Sup3rDataset')
)
T_Array = TypeVar('T_Array', np.ndarray, dask.array.core.Array)
