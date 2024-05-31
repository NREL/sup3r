"""Types used across preprocessing library."""

from typing import List, TypeVar

import dask
import numpy as np

T_Array = TypeVar('T_Array', np.ndarray, dask.array.core.Array)
T_Container = TypeVar('T_Container')
T_Data = TypeVar('T_Data')
T_DualData = TypeVar('T_DualData')
T_DataGroup = TypeVar('T_DataGroup', T_Data, List[T_Data])
