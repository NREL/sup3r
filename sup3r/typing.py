"""Types used across preprocessing library."""

from typing import TypeVar, Union

import dask
import numpy as np

T_Dataset = TypeVar('T_Dataset')
T_Array = Union[np.ndarray, dask.array.core.Array]
