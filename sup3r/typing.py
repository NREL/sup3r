"""Types used across preprocessing library."""

from typing import Union

import dask
import numpy as np

T_Array = Union[np.ndarray, dask.array.core.Array]
