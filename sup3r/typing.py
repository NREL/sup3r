"""Types used across preprocessing library."""

from typing import TypeVar

import dask
import numpy as np

T_Array = TypeVar('T_Array', np.ndarray, dask.array.core.Array)
