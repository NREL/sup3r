# -*- coding: utf-8 -*-
"""Miscellaneous utilities for computing features, preparing training data,
timing functions, etc"""

import glob
import logging
import os
import random
import re
import string
import time
from fnmatch import fnmatch
from inspect import signature
from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd
import psutil
import xarray as xr
from packaging import version
from scipy import ndimage as nd
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.ndimage import gaussian_filter, zoom

np.random.seed(42)

logger = logging.getLogger(__name__)


def _get_possible_class_args(Class):
    class_args = list(signature(Class.__init__).parameters.keys())
    if Class.__bases__ == (object,):
        return class_args
    for base in Class.__bases__:
        class_args += _get_possible_class_args(base)
    return class_args


def _get_class_kwargs(Class, kwargs):
    class_args = _get_possible_class_args(Class)
    return {k: v for k, v in kwargs.items() if k in class_args}


def parse_keys(keys):
    """
    Parse keys for complex __getitem__ and __setitem__

    Parameters
    ----------
    keys : string | tuple
        key or key and slice to extract

    Returns
    -------
    key : string
        key to extract
    key_slice : slice | tuple
        Slice or tuple of slices of key to extract
    """
    if isinstance(keys, tuple):
        key = keys[0]
        key_slice = keys[1:]
    else:
        key = keys
        key_slice = (slice(None), slice(None), slice(None),)

    return key, key_slice


class Feature:
    """Class to simplify feature computations. Stores feature height, feature
    basename, name of feature in handle
    """

    def __init__(self, feature, handle):
        """Takes a feature (e.g. U_100m) and gets the height (100), basename
        (U) and determines whether the feature is found in the data handle

        Parameters
        ----------
        feature : str
            Raw feature name e.g. U_100m
        handle : WindX | NSRDBX | xarray
            handle for data file
        """
        self.raw_name = feature
        self.height = self.get_height(feature)
        self.pressure = self.get_pressure(feature)
        self.basename = self.get_basename(feature)
        if self.raw_name in handle:
            self.handle_input = self.raw_name
        elif self.basename in handle:
            self.handle_input = self.basename
        else:
            self.handle_input = None

    @staticmethod
    def get_basename(feature):
        """Get basename of feature. e.g. temperature from temperature_100m

        Parameters
        ----------
        feature : str
            Name of feature. e.g. U_100m

        Returns
        -------
        str
            feature basename
        """
        height = Feature.get_height(feature)
        pressure = Feature.get_pressure(feature)
        if height is not None or pressure is not None:
            suffix = feature.split('_')[-1]
            basename = feature.replace(f'_{suffix}', '')
        else:
            basename = feature
        return basename

    @staticmethod
    def get_height(feature):
        """Get height from feature name to use in height interpolation

        Parameters
        ----------
        feature : str
            Name of feature. e.g. U_100m

        Returns
        -------
        float | None
            height to use for interpolation
            in meters
        """
        height = None
        if isinstance(feature, str):
            height = re.search(r'\d+m', feature)
            if height:
                height = height.group(0).strip('m')
                if not height.isdigit():
                    height = None
        return height

    @staticmethod
    def get_pressure(feature):
        """Get pressure from feature name to use in pressure interpolation

        Parameters
        ----------
        feature : str
            Name of feature. e.g. U_100pa

        Returns
        -------
        float | None
            pressure to use for interpolation in pascals
        """
        pressure = None
        if isinstance(feature, str):
            pressure = re.search(r'\d+pa', feature)
            if pressure:
                pressure = pressure.group(0).strip('pa')
                if not pressure.isdigit():
                    pressure = None
        return pressure


class Timer:
    """Timer class for timing and storing function call times."""

    def __init__(self):
        self.log = {}

    def __call__(self, fun, *args, **kwargs):
        """Time function call and store elapsed time in self.log.

        Parameters
        ----------
        fun : function
            Function to time
        *args : list
            positional arguments for fun
        **kwargs : dict
            keyword arguments for fun

        Returns
        -------
        output of fun
        """
        t0 = time.time()
        out = fun(*args, **kwargs)
        t_elap = time.time() - t0
        self.log[f'elapsed:{fun.__name__}'] = t_elap
        return out


def check_mem_usage():
    """Frequently used memory check."""
    mem = psutil.virtual_memory()
    logger.info(
        f'Current memory usage is {mem.used / 1e9:.3f} GB out of '
        f'{mem.total / 1e9:.3f} GB total.'
    )


def expand_paths(fps):
    """Expand path(s)

    Parameter
    ---------
    fps : str or pathlib.Path or any Sequence of those
        One or multiple paths to file

    Returns
    -------
    list[str]
        A list of expanded unique and sorted paths as str

    Examples
    --------
    >>> expand_paths("myfile.h5")

    >>> expand_paths(["myfile.h5", "*.hdf"])
    """
    if isinstance(fps, (str, Path)):
        fps = (fps,)

    out = []
    for f in fps:
        out.extend(glob.glob(f))
    return sorted(set(out))


def generate_random_string(length):
    """Generate random string with given length. Used for naming temporary
    files to avoid collisions."""
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(length))


def windspeed_log_law(z, a, b, c):
    """Windspeed log profile.

    Parameters
    ----------
    z : float
        Height above ground in meters
    a : float
        Proportional to friction velocity
    b : float
        Related to zero-plane displacement in meters (height above the ground
        at which zero mean wind speed is achieved as a result of flow obstacles
        such as trees or buildings)
    c : float
        Proportional to stability term.

    Returns
    -------
    ws : float
        Value of windspeed at a given height.
    """
    return a * np.log(z + b) + c


def get_time_dim_name(filepath):
    """Get the name of the time dimension in the given file. This is
    specifically for netcdf files.

    Parameters
    ----------
    filepath : str
        Path to the file

    Returns
    -------
    time_key : str
        Name of the time dimension in the given file
    """
    with xr.open_dataset(filepath) as handle:
        valid_vars = set(handle.dims)
        time_key = list({'time', 'Time'}.intersection(valid_vars))
    if len(time_key) > 0:
        return time_key[0]
    return 'time'


def correct_path(path):
    """If running on windows we need to replace backslashes with double
    backslashes so paths can be parsed correctly with safe_open_json"""
    return path.replace('\\', '\\\\')


def round_array(arr, digits=3):
    """Method to round elements in an array or list. Used a lot in logging
    losses from the data-centric model

    Parameters
    ----------
    arr : list | ndarray
        List or array to round elements of
    digits : int, optional
        Number of digits to round to, by default 3

    Returns
    -------
    list
        List with rounded elements
    """
    return [round(a, digits) for a in arr]


def get_chunk_slices(arr_size, chunk_size, index_slice=slice(None)):
    """Get array slices of corresponding chunk size

    Parameters
    ----------
    arr_size : int
        Length of array to slice
    chunk_size : int
        Size of slices to split array into
    index_slice : slice
        Slice specifying starting and ending index of slice list

    Returns
    -------
    list
        List of slices corresponding to chunks of array
    """

    indices = np.arange(0, arr_size)
    indices = indices[slice(index_slice.start, index_slice.stop)]
    step = 1 if index_slice.step is None else index_slice.step
    slices = []
    start = indices[0]
    stop = start + step * chunk_size
    stop = np.min([stop, indices[-1] + 1])

    while start < indices[-1] + 1:
        slices.append(slice(start, stop, step))
        start = stop
        stop += step * chunk_size
        stop = np.min([stop, indices[-1] + 1])
    return slices


def get_raster_shape(raster_index):
    """Method to get shape of raster_index"""

    if any(isinstance(r, slice) for r in raster_index):
        shape = (
            raster_index[0].stop - raster_index[0].start,
            raster_index[1].stop - raster_index[1].start,
        )
    else:
        shape = raster_index.shape
    return shape


def get_wrf_date_range(files):
    """Get wrf date range for cleaner log output. This assumes file names have
    the date pattern (YYYY-MM-DD-HH:MM:SS) or (YYYY_MM_DD_HH_MM_SS) at the end
    of the file name.

    Parameters
    ----------
    files : list
        List of wrf file paths

    Returns
    -------
    date_start : str
        start date
    date_end : str
        end date
    """

    date_start = re.search(
        r'(\d{4}(-|_)\d+(-|_)\d+(-|_)\d+(:|_)\d+(:|_)\d+)', files[0]
    )
    date_start = date_start if date_start is None else date_start[0]
    date_end = re.search(
        r'(\d{4}(-|_)\d+(-|_)\d+(-|_)\d+(:|_)\d+(:|_)\d+)', files[-1]
    )
    date_end = date_end if date_end is None else date_end[0]

    date_start = date_start.replace(':', '_')
    date_end = date_end.replace(':', '_')

    return date_start, date_end


def uniform_box_sampler(data_shape, sample_shape):
    """Returns a 2D spatial slice used to extract a sample from a data array.

    Parameters
    ----------
    data_shape : tuple
        (rows, cols) Size of full grid available for sampling
    sample_shape : tuple
        (rows, cols) Size of grid to sample from data

    Returns
    -------
    slices : list
        List of slices corresponding to row and col extent of arr sample
    """

    shape_1 = (
        data_shape[0] if data_shape[0] < sample_shape[0] else sample_shape[0]
    )
    shape_2 = (
        data_shape[1] if data_shape[1] < sample_shape[1] else sample_shape[1]
    )
    shape = (shape_1, shape_2)
    start_row = np.random.randint(0, data_shape[0] - sample_shape[0] + 1)
    start_col = np.random.randint(0, data_shape[1] - sample_shape[1] + 1)
    stop_row = start_row + shape[0]
    stop_col = start_col + shape[1]

    return [slice(start_row, stop_row), slice(start_col, stop_col)]


def weighted_box_sampler(data_shape, sample_shape, weights):
    """Extracts a temporal slice from data with selection weighted based on
    provided weights

    Parameters
    ----------
    data_shape : tuple
        (rows, cols) Size of full spatial grid available for sampling
    sample_shape : tuple
        (rows, cols) Size of grid to sample from data
    weights : ndarray
        Array of weights used to specify selection strategy. e.g. If weights is
        [0.2, 0.4, 0.1, 0.3] then the upper left quadrant of the spatial
        domain will be sampled 20 percent of the time, the upper right quadrant
        will be sampled 40 percent of the time, etc.

    Returns
    -------
    slices : list
        List of spatial slices [spatial_1, spatial_2]
    """
    max_cols = (
        data_shape[1] if data_shape[1] < sample_shape[1] else sample_shape[1]
    )
    max_rows = (
        data_shape[0] if data_shape[0] < sample_shape[0] else sample_shape[0]
    )
    max_cols = data_shape[1] - max_cols + 1
    max_rows = data_shape[0] - max_rows + 1
    indices = np.arange(0, max_rows * max_cols)
    chunks = np.array_split(indices, len(weights))
    weight_list = []
    for i, w in enumerate(weights):
        weight_list += [w] * len(chunks[i])
    weight_list /= np.sum(weight_list)
    msg = (
        'Must have a sample_shape with a number of elements greater than '
        'or equal to the number of spatial weights.'
    )
    assert len(indices) >= len(weight_list), msg
    start = np.random.choice(indices, p=weight_list)
    row = start // max_cols
    col = start % max_cols
    stop_1 = row + np.min([sample_shape[0], data_shape[0]])
    stop_2 = col + np.min([sample_shape[1], data_shape[1]])

    slice_1 = slice(row, stop_1)
    slice_2 = slice(col, stop_2)

    return [slice_1, slice_2]


def weighted_time_sampler(data_shape, sample_shape, weights):
    """Returns a temporal slice with selection weighted based on
    provided weights used to extract temporal chunk from data

    Parameters
    ----------
    data_shape : tuple
        (rows, cols, n_steps) Size of full spatialtemporal data grid available
        for sampling
    shape : tuple
        (time_steps) Size of time slice to sample from data
    weights : list
        List of weights used to specify selection strategy. e.g. If weights
        is [0.2, 0.8] then the start of the temporal slice will be selected
        from the first half of the temporal extent with 0.8 probability and
        0.2 probability for the second half.

    Returns
    -------
    slice : slice
        time slice with size shape
    """

    shape = data_shape[2] if data_shape[2] < sample_shape else sample_shape
    t_indices = (
        np.arange(0, data_shape[2])
        if sample_shape == 1
        else np.arange(0, data_shape[2] - sample_shape + 1)
    )
    t_chunks = np.array_split(t_indices, len(weights))

    weight_list = []
    for i, w in enumerate(weights):
        weight_list += [w] * len(t_chunks[i])
    weight_list /= np.sum(weight_list)

    start = np.random.choice(t_indices, p=weight_list)
    stop = start + shape

    return slice(start, stop)


def uniform_time_sampler(data_shape, sample_shape, crop_slice=slice(None)):
    """Returns temporal slice used to extract temporal chunk from data.

    Parameters
    ----------
    data_shape : tuple
        (rows, cols, n_steps) Size of full spatialtemporal data grid available
        for sampling
    sample_shape : int
        (time_steps) Size of time slice to sample from data grid
    crop_slice : slice
        Optional slice used to restrict the sampling window.

    Returns
    -------
    slice : slice
        time slice with size shape
    """
    shape = data_shape[2] if data_shape[2] < sample_shape else sample_shape
    indices = np.arange(data_shape[2] + 1)[crop_slice]
    start = np.random.randint(indices[0], indices[-1] - sample_shape + 1)
    stop = start + shape
    return slice(start, stop)


def daily_time_sampler(data, shape, time_index):
    """Finds a random temporal slice from data starting at midnight

    Parameters
    ----------
    data : np.ndarray
        Data array with dimensions
        (spatial_1, spatial_2, temporal, features)
    shape : int
        (time_steps) Size of time slice to sample from data, must be an integer
        less than or equal to 24.
    time_index : pd.Datetimeindex
        Time index that matches the data axis=2

    Returns
    -------
    slice : slice
        time slice with size shape of data starting at the beginning of the day
    """

    msg = (
        f'data {data.shape} and time index ({len(time_index)}) '
        'shapes do not match, cannot sample daily data.'
    )
    assert data.shape[2] == len(time_index), msg

    ti_short = time_index[: -(shape - 1)]
    midnight_ilocs = np.where(
        (ti_short.hour == 0) & (ti_short.minute == 0) & (ti_short.second == 0)
    )[0]

    if not any(midnight_ilocs):
        msg = (
            'Cannot sample time index of shape {} with requested daily '
            'sample shape {}'.format(len(time_index), shape)
        )
        logger.error(msg)
        raise RuntimeError(msg)

    start = np.random.randint(0, len(midnight_ilocs))
    start = midnight_ilocs[start]
    stop = start + shape

    return slice(start, stop)


def nsrdb_sub_daily_sampler(data, shape, time_index, csr_ind=0):
    """Finds a random sample during daylight hours of a day. Nightime is
    assumed to be marked as NaN in feature axis == csr_ind in the data input.

    Parameters
    ----------
    data : np.ndarray
        Data array with dimensions, where [..., csr_ind] is assumed to be
        clearsky ratio with NaN at night.
        (spatial_1, spatial_2, temporal, features)
    shape : int
        (time_steps) Size of time slice to sample from data, must be an integer
        less than or equal to 24.
    time_index : pd.Datetimeindex
        Time index that matches the data axis=2
    csr_ind : int
        Index of the feature axis where clearsky ratio is located and NaN's can
        be found at night.

    Returns
    -------
    tslice : slice
        time slice with size shape of data starting at the beginning of the day
    """

    tslice = daily_time_sampler(data, 24, time_index)
    night_mask = np.isnan(data[:, :, tslice, csr_ind]).any(axis=(0, 1))

    if shape == 24:
        return tslice

    if night_mask.all():
        msg = (
            f'No daylight data found for tslice {tslice} '
            f'{time_index[tslice]}'
        )
        logger.warning(msg)
        warn(msg)
        return tslice

    day_ilocs = np.where(~night_mask)[0]
    padding = shape - len(day_ilocs)
    half_pad = int(np.round(padding / 2))
    new_start = tslice.start + day_ilocs[0] - half_pad
    new_end = new_start + shape
    return slice(new_start, new_end)


def nsrdb_reduce_daily_data(data, shape, csr_ind=0):
    """Takes a 5D array and reduces the axis=3 temporal dim to daylight hours.

    Parameters
    ----------
    data : np.ndarray
        Data array 5D, where [..., csr_ind] is assumed to be
        clearsky ratio with NaN at night.
        (n_obs, spatial_1, spatial_2, temporal, features)
    shape : int
        (time_steps) Size of time slice to sample from data, must be an integer
        less than or equal to 24.
    csr_ind : int
        Index of the feature axis where clearsky ratio is located and NaN's can
        be found at night.

    Returns
    -------
    data : np.ndarray
        Same as input but with axis=3 reduced to dailylight hours with
        requested shape.
    """

    night_mask = np.isnan(data[0, :, :, :, csr_ind]).any(axis=(0, 1))

    if shape == 24:
        return data

    if night_mask.all():
        msg = f'No daylight data found for data of shape {data.shape}'
        logger.warning(msg)
        warn(msg)
        return data

    day_ilocs = np.where(~night_mask)[0]
    padding = shape - len(day_ilocs)
    half_pad = int(np.round(padding / 2))
    start = day_ilocs[0] - half_pad
    end = start + shape
    tslice = slice(start, end)
    return data[:, :, :, tslice, :]


def transform_rotate_wind(ws, wd, lat_lon):
    """Transform windspeed/direction to u and v and align u and v with grid

    Parameters
    ----------
    ws : np.ndarray
        3D array of high res windspeed data
        (spatial_1, spatial_2, temporal)
    wd : np.ndarray
        3D array of high res winddirection data. Angle is in degrees and
        measured relative to the south_north direction.
        (spatial_1, spatial_2, temporal)
    lat_lon : np.ndarray
        3D array of lat lon
        (spatial_1, spatial_2, 2)
        Last dimension has lat / lon in that order

    Returns
    -------
    u : np.ndarray
        3D array of high res U data
        (spatial_1, spatial_2, temporal)
    v : np.ndarray
        3D array of high res V data
        (spatial_1, spatial_2, temporal)
    """
    # get the dy/dx to the nearest vertical neighbor
    invert_lat = False
    if lat_lon[-1, 0, 0] > lat_lon[0, 0, 0]:
        invert_lat = True
        lat_lon = lat_lon[::-1]
        ws = ws[::-1]
        wd = wd[::-1]
    dy = lat_lon[:, :, 0] - np.roll(lat_lon[:, :, 0], 1, axis=0)
    dx = lat_lon[:, :, 1] - np.roll(lat_lon[:, :, 1], 1, axis=0)
    dy = (dy + 90) % 180 - 90
    dx = (dx + 180) % 360 - 180

    # calculate the angle from the vertical
    theta = (np.pi / 2) - np.arctan2(dy, dx)

    if len(theta) > 1:
        theta[0] = theta[1]  # fix the roll row
    wd = np.radians(wd)

    u_rot = np.cos(theta)[:, :, np.newaxis] * ws * np.sin(wd)
    u_rot += np.sin(theta)[:, :, np.newaxis] * ws * np.cos(wd)

    v_rot = -np.sin(theta)[:, :, np.newaxis] * ws * np.sin(wd)
    v_rot += np.cos(theta)[:, :, np.newaxis] * ws * np.cos(wd)

    if invert_lat:
        u_rot = u_rot[::-1]
        v_rot = v_rot[::-1]
    return u_rot, v_rot


def invert_uv(u, v, lat_lon):
    """Transform u and v back to windspeed and winddirection

    Parameters
    ----------
    u : np.ndarray
        3D array of high res U data
        (spatial_1, spatial_2, temporal)
    v : np.ndarray
        3D array of high res V data
        (spatial_1, spatial_2, temporal)
    lat_lon : np.ndarray
        3D array of lat lon
        (spatial_1, spatial_2, 2)
        Last dimension has lat / lon in that order

    Returns
    -------
    ws : np.ndarray
        3D array of high res windspeed data
        (spatial_1, spatial_2, temporal)
    wd : np.ndarray
        3D array of high res winddirection data. Angle is in degrees and
        measured relative to the south_north direction.
        (spatial_1, spatial_2, temporal)
    """
    invert_lat = False
    if lat_lon[-1, 0, 0] > lat_lon[0, 0, 0]:
        invert_lat = True
        lat_lon = lat_lon[::-1]
        u = u[::-1]
        v = v[::-1]
    dy = lat_lon[:, :, 0] - np.roll(lat_lon[:, :, 0], 1, axis=0)
    dx = lat_lon[:, :, 1] - np.roll(lat_lon[:, :, 1], 1, axis=0)
    dy = (dy + 90) % 180 - 90
    dx = (dx + 180) % 360 - 180

    # calculate the angle from the vertical
    theta = (np.pi / 2) - np.arctan2(dy, dx)
    if len(theta) > 1:
        theta[0] = theta[1]  # fix the roll row

    u_rot = np.cos(theta)[:, :, np.newaxis] * u
    u_rot -= np.sin(theta)[:, :, np.newaxis] * v

    v_rot = np.sin(theta)[:, :, np.newaxis] * u
    v_rot += np.cos(theta)[:, :, np.newaxis] * v

    ws = np.hypot(u_rot, v_rot)
    wd = (np.degrees(np.arctan2(u_rot, v_rot)) + 360) % 360

    if invert_lat:
        ws = ws[::-1]
        wd = wd[::-1]
    return ws, wd


def temporal_coarsening(data, t_enhance=4, method='subsample'):
    """Coarsen data according to t_enhance resolution

    Parameters
    ----------
    data : np.ndarray
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
    coarse_data : np.ndarray
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


def temporal_simple_enhancing(data, t_enhance=4, mode='constant'):
    """Upsample data according to t_enhance resolution

    Parameters
    ----------
    data : np.ndarray
        5D array with dimensions
        (observations, spatial_1, spatial_2, temporal, features)
    t_enhance : int
        factor by which to enhance temporal dimension
    mode : str
        interpolation method for enhancement.

    Returns
    -------
    enhanced_data : np.ndarray
        5D array with same dimensions as data with new enhanced resolution
    """

    if t_enhance in [None, 1]:
        enhanced_data = data
    elif t_enhance not in [None, 1] and len(data.shape) == 5:
        if mode == 'constant':
            enhancement = [1, 1, 1, t_enhance, 1]
            enhanced_data = zoom(
                data, enhancement, order=0, mode='nearest', grid_mode=True
            )
        elif mode == 'linear':
            index_t_hr = np.array(list(range(data.shape[3] * t_enhance)))
            index_t_lr = index_t_hr[::t_enhance]
            enhanced_data = interp1d(
                index_t_lr, data, axis=3, fill_value='extrapolate'
            )(index_t_hr)
            enhanced_data = np.array(enhanced_data, dtype=np.float32)
    elif len(data.shape) != 5:
        msg = (
            'Data must be 5D to do temporal enhancing, but '
            f'received: {data.shape}'
        )
        logger.error(msg)
        raise ValueError(msg)

    return enhanced_data


def daily_temporal_coarsening(data, temporal_axis=3):
    """Temporal coarsening for daily average climate change data.

    This method takes the sum of the data in the temporal dimension and divides
    by 24 (for 24 hours per day). Even if there are only 8-12 daylight obs in
    the temporal axis, we want to divide by 24 to give the equivalent of a
    daily average.

    Parameters
    ----------
    data : np.ndarray
        Array of data with a temporal axis as determined by the temporal_axis
        input. Example 4D or 5D input shapes:
        (spatial_1, spatial_2, temporal, features)
        (observations, spatial_1, spatial_2, temporal, features)
    temporal_axis : int
        Axis index of the temporal axis to be averaged. Default is axis=3 for
        the 5D tensor that is fed to the ST-GAN.

    Returns
    -------
    coarse_data : np.ndarray
        Array with same dimensions as data with new coarse resolution,
        temporal dimension is size 1
    """
    coarse_data = np.nansum(data, axis=temporal_axis) / 24
    return np.expand_dims(coarse_data, axis=temporal_axis)


def smooth_data(low_res, training_features, smoothing_ignore, smoothing=None):
    """Smooth data using a gaussian filter

    Parameters
    ----------
    low_res : np.ndarray
        4D | 5D array
        (batch_size, spatial_1, spatial_2, features)
        (batch_size, spatial_1, spatial_2, temporal, features)
    training_features : list | None
        Ordered list of training features input to the generative model
    smoothing_ignore : list | None
        List of features to ignore for the smoothing filter. None will
        smooth all features if smoothing kwarg is not None
    smoothing : float | None
        Standard deviation to use for gaussian filtering of the coarse
        data. This can be tuned by matching the kinetic energy of a low
        resolution simulation with the kinetic energy of a coarsened and
        smoothed high resolution simulation. If None no smoothing is
        performed.

    Returns
    -------
    low_res : np.ndarray
        4D | 5D array
        (batch_size, spatial_1, spatial_2, features)
        (batch_size, spatial_1, spatial_2, temporal, features)
    """

    if smoothing is not None:
        feat_iter = [
            j
            for j in range(low_res.shape[-1])
            if training_features[j] not in smoothing_ignore
        ]
        for i in range(low_res.shape[0]):
            for j in feat_iter:
                if len(low_res.shape) == 5:
                    for t in range(low_res.shape[-2]):
                        low_res[i, ..., t, j] = gaussian_filter(
                            low_res[i, ..., t, j], smoothing, mode='nearest'
                        )
                else:
                    low_res[i, ..., j] = gaussian_filter(
                        low_res[i, ..., j], smoothing, mode='nearest'
                    )
    return low_res


def spatial_coarsening(data, s_enhance=2, obs_axis=True):
    """Coarsen data according to s_enhance resolution

    Parameters
    ----------
    data : np.ndarray
        5D | 4D | 3D array with dimensions:
        (n_obs, spatial_1, spatial_2, temporal, features) (obs_axis=True)
        (n_obs, spatial_1, spatial_2, features) (obs_axis=True)
        (spatial_1, spatial_2, temporal, features) (obs_axis=False)
        (spatial_1, spatial_2, temporal_or_features) (obs_axis=False)
    s_enhance : int
        factor by which to coarsen spatial dimensions
    obs_axis : bool
        Flag for if axis=0 is the observation axis. If True (default)
        spatial axis=(1, 2) (zero-indexed), if False spatial axis=(0, 1)

    Returns
    -------
    data : np.ndarray
        3D | 4D | 5D array with same dimensions as data with new coarse
        resolution
    """

    if len(data.shape) < 3:
        msg = (
            'Data must be 3D, 4D, or 5D to do spatial coarsening, but '
            f'received: {data.shape}'
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

        if obs_axis and len(data.shape) == 5:
            data = np.reshape(
                data,
                (
                    data.shape[0],
                    data.shape[1] // s_enhance,
                    s_enhance,
                    data.shape[2] // s_enhance,
                    s_enhance,
                    data.shape[3],
                    data.shape[4],
                ),
            )
            data = data.sum(axis=(2, 4)) / s_enhance**2

        elif obs_axis and len(data.shape) == 4:
            data = np.reshape(
                data,
                (
                    data.shape[0],
                    data.shape[1] // s_enhance,
                    s_enhance,
                    data.shape[2] // s_enhance,
                    s_enhance,
                    data.shape[3],
                ),
            )
            data = data.sum(axis=(2, 4)) / s_enhance**2

        elif not obs_axis and len(data.shape) == 4:
            data = np.reshape(
                data,
                (
                    data.shape[0] // s_enhance,
                    s_enhance,
                    data.shape[1] // s_enhance,
                    s_enhance,
                    data.shape[2],
                    data.shape[3],
                ),
            )
            data = data.sum(axis=(1, 3)) / s_enhance**2

        elif not obs_axis and len(data.shape) == 3:
            data = np.reshape(
                data,
                (
                    data.shape[0] // s_enhance,
                    s_enhance,
                    data.shape[1] // s_enhance,
                    s_enhance,
                    data.shape[2],
                ),
            )
            data = data.sum(axis=(1, 3)) / s_enhance**2

        else:
            msg = (
                'Data must be 3D, 4D, or 5D to do spatial coarsening, but '
                f'received: {data.shape}'
            )
            logger.error(msg)
            raise ValueError(msg)

    return data


def spatial_simple_enhancing(data, s_enhance=2, obs_axis=True):
    """Simple enhancing according to s_enhance resolution

    Parameters
    ----------
    data : np.ndarray
        5D | 4D | 3D array with dimensions:
        (n_obs, spatial_1, spatial_2, temporal, features) (obs_axis=True)
        (n_obs, spatial_1, spatial_2, features) (obs_axis=True)
        (spatial_1, spatial_2, temporal, features) (obs_axis=False)
        (spatial_1, spatial_2, temporal_or_features) (obs_axis=False)
    s_enhance : int
        factor by which to enhance spatial dimensions
    obs_axis : bool
        Flag for if axis=0 is the observation axis. If True (default)
        spatial axis=(1, 2) (zero-indexed), if False spatial axis=(0, 1)

    Returns
    -------
    enhanced_data : np.ndarray
        3D | 4D | 5D array with same dimensions as data with new enhanced
        resolution
    """

    if len(data.shape) < 3:
        msg = (
            'Data must be 3D, 4D, or 5D to do spatial enhancing, but '
            f'received: {data.shape}'
        )
        logger.error(msg)
        raise ValueError(msg)

    if s_enhance is not None and s_enhance > 1:
        if obs_axis and len(data.shape) == 5:
            enhancement = [1, s_enhance, s_enhance, 1, 1]
            enhanced_data = zoom(
                data, enhancement, order=0, mode='nearest', grid_mode=True
            )

        elif obs_axis and len(data.shape) == 4:
            enhancement = [1, s_enhance, s_enhance, 1]
            enhanced_data = zoom(
                data, enhancement, order=0, mode='nearest', grid_mode=True
            )

        elif not obs_axis and len(data.shape) == 4:
            enhancement = [s_enhance, s_enhance, 1, 1]
            enhanced_data = zoom(
                data, enhancement, order=0, mode='nearest', grid_mode=True
            )

        elif not obs_axis and len(data.shape) == 3:
            enhancement = [s_enhance, s_enhance, 1]
            enhanced_data = zoom(
                data, enhancement, order=0, mode='nearest', grid_mode=True
            )
        else:
            msg = (
                'Data must be 3D, 4D, or 5D to do spatial enhancing, but '
                f'received: {data.shape}'
            )
            logger.error(msg)
            raise ValueError(msg)

    else:
        enhanced_data = data

    return enhanced_data


def lat_lon_coarsening(lat_lon, s_enhance=2):
    """Coarsen lat_lon according to s_enhance resolution

    Parameters
    ----------
    lat_lon : np.ndarray
        2D array with dimensions
        (spatial_1, spatial_2)
    s_enhance : int
        factor by which to coarsen spatial dimensions

    Returns
    -------
    coarse_lat_lon : np.ndarray
        2D array with same dimensions as lat_lon with new coarse resolution
    """
    coarse_lat_lon = lat_lon.reshape(
        -1, s_enhance, lat_lon.shape[1] // s_enhance, s_enhance, 2
    ).sum((3, 1))
    coarse_lat_lon /= s_enhance * s_enhance
    return coarse_lat_lon


def forward_average(array_in):
    """Average neighboring values in an array.  Used to unstagger WRF variable
    values.

    Parameters
    ----------
    array_in : ndarray
        Input array, or array axis

    Returns
    -------
    ndarray
        Array of average values, length will be 1 less than array_in
    """
    return (array_in[:-1] + array_in[1:]) * 0.5


def nn_fill_array(array):
    """Fill any NaN values in an np.ndarray from the nearest non-nan values.

    Parameters
    ----------
    array : np.ndarray
        Input array with NaN values

    Returns
    -------
    array : np.ndarray
        Output array with NaN values filled
    """

    nan_mask = np.isnan(array)
    indices = nd.distance_transform_edt(
        nan_mask, return_distances=False, return_indices=True
    )
    return array[tuple(indices)]


def ignore_case_path_fetch(fp):
    """Get file path which matches fp while ignoring case

    Parameters
    ----------
    fp : str
        file path

    Returns
    -------
    str
        existing file which matches fp
    """

    dirname = os.path.dirname(fp)
    basename = os.path.basename(fp)
    if os.path.exists(dirname):
        for file in os.listdir(dirname):
            if fnmatch(file.lower(), basename.lower()):
                return os.path.join(dirname, file)
    return None


def get_source_type(file_paths):
    """Get data source type

    Parameters
    ----------
    file_paths : list | str
        One or more paths to data files, can include a unix-style pat*tern

    Returns
    -------
    source_type : str
        Either "h5" or "nc"
    """
    if file_paths is None:
        return None

    if isinstance(file_paths, str) and '*' in file_paths:
        temp = glob.glob(file_paths)
        if any(temp):
            file_paths = temp

    if not isinstance(file_paths, list):
        file_paths = [file_paths]

    _, source_type = os.path.splitext(file_paths[0])

    if source_type == '.h5':
        return 'h5'
    return 'nc'


def get_extracter_class(extracter_name):
    """Get the DataHandler class.

    Parameters
    ----------
    extracter_name : str
        :class:`Extracter` class to use for input data. Provide a string name
        to match a class in `sup3r.container.extracters`.
    """

    ExtracterClass = None

    if isinstance(extracter_name, str):
        import sup3r.containers

        ExtracterClass = getattr(sup3r.containers, extracter_name, None)

    if ExtracterClass is None:
        msg = (
            'Could not find requested :class:`Extracter` class '
            f'"{extracter_name}" in sup3r.containers.'
        )
        logger.error(msg)
        raise KeyError(msg)

    return ExtracterClass


def get_input_handler_class(file_paths, input_handler_name):
    """Get the DataHandler class.

    Parameters
    ----------
    file_paths : list | str
        A list of files to extract raster data from. Each file must have
        the same number of timesteps. Can also pass a string with a
        unix-style file path which will be passed through glob.glob
    input_handler_name : str
        data handler class to use for input data. Provide a string name to
        match a class in data_handling.py. If None the correct handler will
        be guessed based on file type and time series properties.

    Returns
    -------
    HandlerClass : DataHandlerH5 | DataHandlerNC
        DataHandler subclass from sup3r.preprocessing.
    """

    HandlerClass = None

    input_type = get_source_type(file_paths)

    if input_handler_name is None:
        if input_type == 'nc':
            input_handler_name = 'DataHandlerNC'
        elif input_type == 'h5':
            input_handler_name = 'DataHandlerH5'

        logger.info(
            '"input_handler" arg was not provided. Using '
            f'"{input_handler_name}". If this is '
            'incorrect, please provide '
            'input_handler="DataHandlerName".'
        )

    if isinstance(input_handler_name, str):
        import sup3r.preprocessing

        HandlerClass = getattr(sup3r.preprocessing, input_handler_name, None)

    if HandlerClass is None:
        msg = (
            'Could not find requested data handler class '
            f'"{input_handler_name}" in sup3r.preprocessing.'
        )
        logger.error(msg)
        raise KeyError(msg)

    return HandlerClass


def np_to_pd_times(times):
    """Convert `np.bytes_` times to DatetimeIndex

    Parameters
    ----------
    times : ndarray | list
        List of `np.bytes_` objects for time indices

    Returns
    -------
    times : pd.DatetimeIndex
        DatetimeIndex for time indices
    """
    tmp = [t.decode('utf-8') for t in times.flatten()]
    tmp = [' '.join(t.split('_')) for t in tmp]
    return pd.DatetimeIndex(tmp)


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


def st_interp(low, s_enhance, t_enhance, t_centered=False):
    """Spatiotemporal bilinear interpolation for low resolution field on a
    regular grid. Used to provide baseline for comparison with gan output

    Parameters
    ----------
    low : ndarray
        Low resolution field to interpolate.
        (spatial_1, spatial_2, temporal)
    s_enhance : int
        Factor by which to enhance the spatial domain
    t_enhance : int
        Factor by which to enhance the temporal domain
    t_centered : bool
        Flag to switch time axis from time-beginning (Default, e.g.
        interpolate 00:00 01:00 to 00:00 00:30 01:00 01:30) to
        time-centered (e.g. interp 01:00 02:00 to 00:45 01:15 01:45 02:15)

    Returns
    -------
    ndarray
        Spatiotemporally interpolated low resolution output
    """
    assert len(low.shape) == 3, 'Input to st_interp must be 3D array'
    msg = 'Input to st_interp cannot include axes with length 1'
    assert not any(s <= 1 for s in low.shape), msg

    lr_y, lr_x, lr_t = low.shape
    hr_y, hr_x, hr_t = lr_y * s_enhance, lr_x * s_enhance, lr_t * t_enhance

    # assume outer bounds of mesh (0, 10) w/ points on inside of that range
    y = np.arange(0, 10, 10 / lr_y) + 5 / lr_y
    x = np.arange(0, 10, 10 / lr_x) + 5 / lr_x

    # remesh (0, 10) with high res spacing
    new_y = np.arange(0, 10, 10 / hr_y) + 5 / hr_y
    new_x = np.arange(0, 10, 10 / hr_x) + 5 / hr_x

    t = np.arange(0, 10, 10 / lr_t)
    new_t = np.arange(0, 10, 10 / hr_t)
    if t_centered:
        t += 5 / lr_t
        new_t += 5 / hr_t

    # set RegularGridInterpolator to do extrapolation
    interp = RegularGridInterpolator(
        (y, x, t), low, bounds_error=False, fill_value=None
    )

    # perform interp
    X, Y, T = np.meshgrid(new_x, new_y, new_t)
    return interp((Y, X, T))
