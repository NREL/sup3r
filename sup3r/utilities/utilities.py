"""Miscellaneous utilities shared across multiple modules"""

import json
import logging
import random
import string
import time

import numpy as np
import pandas as pd
from packaging import version
from scipy import ndimage as nd

logger = logging.getLogger(__name__)

RANDOM_GENERATOR = np.random.default_rng(seed=42)


def safe_serialize(obj):
    """json.dumps with non-serializable object handling."""
    def _default(o):
        if isinstance(o, np.float32):
            return np.float64(o)
        return f"<<non-serializable: {type(o).__qualname__}>>"
    return json.dumps(obj, default=_default)


class Timer:
    """Timer class for timing and storing function call times."""

    def __init__(self):
        self.log = {}
        self.elapsed = 0

    def __call__(self, func, log=False):
        """Time function call and store elapsed time in self.log.

        Parameters
        ----------
        func : function
            Function to time
        log : bool
            Whether to write to active logger

        Returns
        -------
        output of func
        """
        def wrapper(*args, **kwargs):
            """Wrapper with decorator pattern.

            Parameters
            ----------
            *args : list
                positional arguments for fun
            **kwargs : dict
                keyword arguments for fun
            """
            t0 = time.time()
            out = func(*args, **kwargs)
            t_elap = time.time() - t0
            self.elapsed = t_elap
            self.log[f'elapsed:{func.__name__}'] = t_elap
            if log:
                logger.debug(f'Call to {func.__name__} finished in '
                             f'{round(t_elap, 5)} seconds')
            return out

        return wrapper


def generate_random_string(length):
    """Generate random string with given length. Used for naming temporary
    files to avoid collisions."""
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(length))


def temporal_coarsening(data, t_enhance=4, method='subsample'):
    """Coarsen data according to t_enhance resolution

    Parameters
    ----------
    data : T_Array
        5D array with dimensions
        (observations, spatial_1, spatial_2, temporal, features)
    t_enhance : int
        factor by which to coarsen temporal dimension
    method : str
        accepted options: [subsample, average, total, min, max]
        Subsample will take every t_enhance-th time step, average will average
        over t_enhance time steps, total will sum over t_enhance time steps

    Returns
    -------
    coarse_data : T_Array
        5D array with same dimensions as data with new coarse resolution
    """

    if t_enhance is not None and len(data.shape) == 5:
        if method == 'subsample':
            coarse_data = data[:, :, :, ::t_enhance, :]

        elif method == 'average':
            coarse_data = np.nansum(
                np.reshape(
                    data,
                    (
                        data.shape[0],
                        data.shape[1],
                        data.shape[2],
                        -1,
                        t_enhance,
                        data.shape[4],
                    ),
                ),
                axis=4,
            )
            coarse_data /= t_enhance

        elif method == 'max':
            coarse_data = np.max(
                np.reshape(
                    data,
                    (
                        data.shape[0],
                        data.shape[1],
                        data.shape[2],
                        -1,
                        t_enhance,
                        data.shape[4],
                    ),
                ),
                axis=4,
            )

        elif method == 'min':
            coarse_data = np.min(
                np.reshape(
                    data,
                    (
                        data.shape[0],
                        data.shape[1],
                        data.shape[2],
                        -1,
                        t_enhance,
                        data.shape[4],
                    ),
                ),
                axis=4,
            )

        elif method == 'total':
            coarse_data = np.nansum(
                np.reshape(
                    data,
                    (
                        data.shape[0],
                        data.shape[1],
                        data.shape[2],
                        -1,
                        t_enhance,
                        data.shape[4],
                    ),
                ),
                axis=4,
            )

        else:
            msg = (
                f'Did not recognize temporal_coarsening method "{method}", '
                'can only accept one of: [subsample, average, total, max, min]'
            )
            logger.error(msg)
            raise KeyError(msg)

    else:
        coarse_data = data

    return coarse_data


def spatial_coarsening(data, s_enhance=2, obs_axis=True):
    """Coarsen data according to s_enhance resolution

    Parameters
    ----------
    data : T_Array
        5D | 4D | 3D | 2D array with dimensions:
        (n_obs, spatial_1, spatial_2, temporal, features) (obs_axis=True)
        (n_obs, spatial_1, spatial_2, features) (obs_axis=True)
        (spatial_1, spatial_2, temporal, features) (obs_axis=False)
        (spatial_1, spatial_2, temporal_or_features) (obs_axis=False)
        (spatial_1, spatial_2) (obs_axis=False)
    s_enhance : int
        factor by which to coarsen spatial dimensions
    obs_axis : bool
        Flag for if axis=0 is the observation axis. If True (default)
        spatial axis=(1, 2) (zero-indexed), if False spatial axis=(0, 1)

    Returns
    -------
    data : T_Array
        2D, 3D | 4D | 5D array with same dimensions as data with new coarse
        resolution
    """

    if len(data.shape) < 2:
        msg = (
            'Data must be 2D, 3D, 4D, or 5D to do spatial coarsening, but '
            f'received: {data.shape}'
        )
        logger.error(msg)
        raise ValueError(msg)

    if obs_axis and len(data.shape) < 3:
        msg = (
            'Data must be 3D, 4D, or 5D to do spatial coarsening with '
            f'obs_axis=True, but received: {data.shape}'
        )
        logger.error(msg)
        raise ValueError(msg)

    if s_enhance is not None and s_enhance > 1:
        bad1 = obs_axis and (
            data.shape[1] % s_enhance != 0 or data.shape[2] % s_enhance != 0
        )
        bad2 = not obs_axis and (
            data.shape[0] % s_enhance != 0 or data.shape[1] % s_enhance != 0
        )
        if bad1 or bad2:
            msg = (
                's_enhance must evenly divide grid size. '
                f'Received s_enhance: {s_enhance} with data shape: '
                f'{data.shape}'
            )
            logger.error(msg)
            raise ValueError(msg)

        if obs_axis and len(data.shape) == 3:
            data = np.reshape(
                data,
                (
                    data.shape[0],
                    data.shape[1] // s_enhance,
                    s_enhance,
                    data.shape[2] // s_enhance,
                    s_enhance,
                ),
            )
            data = data.sum(axis=(2, 4)) / s_enhance**2

        elif obs_axis:
            data = np.reshape(
                data,
                (
                    data.shape[0],
                    data.shape[1] // s_enhance,
                    s_enhance,
                    data.shape[2] // s_enhance,
                    s_enhance,
                    *data.shape[3:],
                ),
            )
            data = data.sum(axis=(2, 4)) / s_enhance**2

        elif not obs_axis and len(data.shape) == 2:
            data = np.reshape(
                data,
                (
                    data.shape[0] // s_enhance,
                    s_enhance,
                    data.shape[1] // s_enhance,
                    s_enhance,
                ),
            )
            data = data.sum(axis=(1, 3)) / s_enhance**2

        elif not obs_axis:
            data = np.reshape(
                data,
                (
                    data.shape[0] // s_enhance,
                    s_enhance,
                    data.shape[1] // s_enhance,
                    s_enhance,
                    *data.shape[2:],
                ),
            )
            data = data.sum(axis=(1, 3)) / s_enhance**2

        else:
            msg = (
                'Data must be 2D, 3D, 4D, or 5D to do spatial coarsening, but '
                f'received: {data.shape}'
            )
            logger.error(msg)
            raise ValueError(msg)

    return data


def nn_fill_array(array):
    """Fill any NaN values in an np.ndarray from the nearest non-nan values.

    Parameters
    ----------
    array : T_Array
        Input array with NaN values

    Returns
    -------
    array : T_Array
        Output array with NaN values filled
    """

    nan_mask = np.isnan(array)
    indices = nd.distance_transform_edt(
        nan_mask, return_distances=False, return_indices=True
    )
    return array[tuple(indices)]


def pd_date_range(*args, **kwargs):
    """A simple wrapper on the pd.date_range() method that handles the closed
    vs. inclusive kwarg change in pd 1.4.0"""
    incl = version.parse(pd.__version__) >= version.parse('1.4.0')

    if incl and 'closed' in kwargs:
        kwargs['inclusive'] = kwargs.pop('closed')
    elif not incl and 'inclusive' in kwargs:
        kwargs['closed'] = kwargs.pop('inclusive')
        if kwargs['closed'] == 'both':
            kwargs['closed'] = None

    return pd.date_range(*args, **kwargs)
