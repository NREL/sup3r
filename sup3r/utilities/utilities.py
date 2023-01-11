# -*- coding: utf-8 -*-
"""Utilities module for preparing training data

@author: bbenton
"""

import numpy as np
import logging
import glob
from scipy import ndimage as nd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d
from scipy.ndimage import interpolation
from fnmatch import fnmatch
import os
import re
from warnings import warn
import psutil
import pandas as pd
from packaging import version

np.random.seed(42)

logger = logging.getLogger(__name__)


def correct_path(path):
    """If running on windows we need to replace backslashes with double
    backslashes so paths can be parsed correctly with safe_open_json"""
    return path.replace('\\', '\\\\')


def estimate_max_workers(max_workers, process_mem, n_processes):
    """Estimate max number of workers based on available memory

    Parameters
    ----------
    max_workers : int | None
        Max number of workers available
    process_mem : int
        Total number of bytes for minimum size process
    n_processes : int
        Number of processes

    Returns
    -------
    max_workers : int
        Max number of workers available
    """
    mem = psutil.virtual_memory()
    avail_mem = 0.7 * (mem.total - mem.used)
    cpu_count = os.cpu_count()
    if max_workers is not None:
        max_workers = np.min([max_workers, n_processes])
    elif process_mem > 0:
        max_workers = avail_mem / process_mem
        max_workers = np.min([max_workers, n_processes, cpu_count])
    else:
        max_workers = 1
    max_workers = int(np.max([max_workers, 1]))
    return max_workers


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
    indices = indices[index_slice.start:index_slice.stop]
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
    """method to get shape of raster_index"""

    if any(isinstance(r, slice) for r in raster_index):
        shape = (raster_index[0].stop - raster_index[0].start,
                 raster_index[1].stop - raster_index[1].start)
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

    date_start = re.search(r'(\d{4}(-|_)\d+(-|_)\d+(-|_)\d+(:|_)\d+(:|_)\d+)',
                           files[0])
    date_start = date_start if date_start is None else date_start[0]
    date_end = re.search(r'(\d{4}(-|_)\d+(-|_)\d+(-|_)\d+(:|_)\d+(:|_)\d+)',
                         files[-1])
    date_end = date_end if date_end is None else date_end[0]

    date_start = date_start.replace(':', '_')
    date_end = date_end.replace(':', '_')

    return date_start, date_end


def uniform_box_sampler(data, shape):
    '''Extracts a sample cut from data.

    Parameters
    ----------
    data : np.ndarray
        Data array with dimensions
        (spatial_1, spatial_2, temporal, features)
    shape : tuple
        (rows, cols) Size of grid to sample
        from data

    Returns
    -------
    slices : list
        List of slices corresponding to row and col extent of arr sample
    '''

    shape_1 = data.shape[0] if data.shape[0] < shape[0] else shape[0]
    shape_2 = data.shape[1] if data.shape[1] < shape[1] else shape[1]
    shape = (shape_1, shape_2)
    start_row = np.random.randint(0, data.shape[0] - shape[0] + 1)
    start_col = np.random.randint(0, data.shape[1] - shape[1] + 1)
    stop_row = start_row + shape[0]
    stop_col = start_col + shape[1]

    return [slice(start_row, stop_row), slice(start_col, stop_col)]


def weighted_box_sampler(data, shape, weights):
    """Extracts a temporal slice from data with selection weighted based on
    provided weights

    Parameters
    ----------
    data : np.ndarray
        Data array with dimensions
        (spatial_1, spatial_2, temporal, features)
    shape : tuple
        (spatial_1, spatial_2) Size of box to sample from data
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
    max_cols = (data.shape[1] if data.shape[1] < shape[1]
                else shape[1])
    max_rows = (data.shape[0] if data.shape[0] < shape[0]
                else shape[0])
    max_cols = data.shape[1] - max_cols + 1
    max_rows = data.shape[0] - max_rows + 1
    indices = np.arange(0, max_rows * max_cols)
    chunks = np.array_split(indices, len(weights))
    weight_list = []
    for i, w in enumerate(weights):
        weight_list += [w] * len(chunks[i])
    weight_list /= np.sum(weight_list)
    msg = ('Must have a sample_shape with a number of elements greater than '
           'or equal to the number of spatial weights.')
    assert len(indices) >= len(weight_list), msg
    start = np.random.choice(indices, p=weight_list)
    row = start // max_cols
    col = start % max_cols
    stop_1 = row + np.min([shape[0], data.shape[0]])
    stop_2 = col + np.min([shape[1], data.shape[1]])

    slice_1 = slice(row, stop_1)
    slice_2 = slice(col, stop_2)

    return [slice_1, slice_2]


def weighted_time_sampler(data, shape, weights):
    """Extracts a temporal slice from data with selection weighted based on
    provided weights

    Parameters
    ----------
    data : np.ndarray
        Data array with dimensions
        (spatial_1, spatial_2, temporal, features)
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

    shape = data.shape[2] if data.shape[2] < shape else shape
    t_indices = (np.arange(0, data.shape[2]) if shape == 1
                 else np.arange(0, data.shape[2] - shape + 1))
    t_chunks = np.array_split(t_indices, len(weights))

    weight_list = []
    for i, w in enumerate(weights):
        weight_list += [w] * len(t_chunks[i])
    weight_list /= np.sum(weight_list)

    start = np.random.choice(t_indices, p=weight_list)
    stop = start + shape

    return slice(start, stop)


def uniform_time_sampler(data, shape):
    '''Extracts a temporal slice from data.

    Parameters
    ----------
    data : np.ndarray
        Data array with dimensions
        (spatial_1, spatial_2, temporal, features)
    shape : int
        (time_steps) Size of time slice to sample
        from data

    Returns
    -------
    slice : slice
        time slice with size shape
    '''
    shape = data.shape[2] if data.shape[2] < shape else shape
    start = np.random.randint(0, data.shape[2] - shape + 1)
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

    msg = (f'data {data.shape} and time index ({len(time_index)}) '
           'shapes do not match, cannot sample daily data.')
    assert data.shape[2] == len(time_index), msg

    ti_short = time_index[:-(shape - 1)]
    midnight_ilocs = np.where((ti_short.hour == 0)
                              & (ti_short.minute == 0)
                              & (ti_short.second == 0))[0]

    if not any(midnight_ilocs):
        msg = ('Cannot sample time index of shape {} with requested daily '
               'sample shape {}'.format(len(time_index), shape))
        logger.error(msg)
        raise RuntimeError(msg)

    start = np.random.randint(0, len(midnight_ilocs))
    start = midnight_ilocs[start]
    stop = start + shape

    tslice = slice(start, stop)

    return tslice


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
        msg = (f'No daylight data found for tslice {tslice} '
               f'{time_index[tslice]}')
        logger.warning(msg)
        warn(msg)
        return tslice

    else:
        day_ilocs = np.where(~night_mask)[0]
        padding = shape - len(day_ilocs)
        half_pad = int(np.round(padding / 2))
        new_start = tslice.start + day_ilocs[0] - half_pad
        new_end = new_start + shape
        tslice = slice(new_start, new_end)
        return tslice


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
        msg = (f'No daylight data found for data of shape {data.shape}')
        logger.warning(msg)
        warn(msg)
        return data

    else:
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
    """"Coarsen data according to t_enhance resolution

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
                data.reshape(
                    (data.shape[0], data.shape[1],
                     data.shape[2], -1, t_enhance,
                     data.shape[4])), axis=4)
            coarse_data /= t_enhance

        elif method == 'max':
            coarse_data = np.max(
                data.reshape(
                    (data.shape[0], data.shape[1],
                     data.shape[2], -1, t_enhance,
                     data.shape[4])), axis=4)

        elif method == 'min':
            coarse_data = np.min(
                data.reshape(
                    (data.shape[0], data.shape[1],
                     data.shape[2], -1, t_enhance,
                     data.shape[4])), axis=4)

        elif method == 'total':
            coarse_data = np.nansum(
                data.reshape(
                    (data.shape[0], data.shape[1],
                     data.shape[2], -1, t_enhance,
                     data.shape[4])), axis=4)

        else:
            msg = ('Did not recognize temporal_coarsening method "{}", can '
                   'only accept one of: [subsample, average, total, max, min]'
                   .format(method))
            logger.error(msg)
            raise KeyError(msg)

    else:
        coarse_data = data

    return coarse_data


def temporal_simple_enhancing(data, t_enhance=4):
    """"Upsample data according to t_enhance resolution

    Parameters
    ----------
    data : np.ndarray
        5D array with dimensions
        (observations, spatial_1, spatial_2, temporal, features)
    t_enhance : int
        factor by which to enhance temporal dimension

    Returns
    -------
    enhanced_data : np.ndarray
        5D array with same dimensions as data with new enhanced resolution
    """

    if t_enhance in [None, 1]:
        enhanced_data = data
    elif t_enhance not in [None, 1] and len(data.shape) == 5:
        enhancement = [1, 1, 1, t_enhance, 1]
        enhanced_data = interpolation.zoom(data,
                                           enhancement,
                                           order=0)
    elif len(data.shape) != 5:
        msg = ('Data must be 5D to do temporal enhancing, but '
               f'received: {data.shape}')
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
    coarse_data = np.expand_dims(coarse_data, axis=temporal_axis)
    return coarse_data


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
        feat_iter = [j for j in range(low_res.shape[-1])
                     if training_features[j] not in smoothing_ignore]
        for i in range(low_res.shape[0]):
            for j in feat_iter:
                if len(low_res.shape) == 5:
                    for t in range(low_res.shape[-2]):
                        low_res[i, ..., t, j] = gaussian_filter(
                            low_res[i, ..., t, j], smoothing,
                            mode='nearest')
                else:
                    low_res[i, ..., j] = gaussian_filter(
                        low_res[i, ..., j], smoothing, mode='nearest')
    return low_res


def spatial_coarsening(data, s_enhance=2, obs_axis=True):
    """"Coarsen data according to s_enhance resolution

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
        msg = ('Data must be 3D, 4D, or 5D to do spatial coarsening, but '
               f'received: {data.shape}')
        logger.error(msg)
        raise ValueError(msg)

    if s_enhance is not None and s_enhance > 1:
        bad1 = (obs_axis and (data.shape[1] % s_enhance != 0
                              or data.shape[2] % s_enhance != 0))
        bad2 = (not obs_axis and (data.shape[0] % s_enhance != 0
                                  or data.shape[1] % s_enhance != 0))
        if bad1 or bad2:
            msg = ('s_enhance must evenly divide grid size. '
                   f'Received s_enhance: {s_enhance} with data shape: '
                   f'{data.shape}')
            logger.error(msg)
            raise ValueError(msg)

        if obs_axis and len(data.shape) == 5:
            data = data.reshape(data.shape[0],
                                data.shape[1] // s_enhance, s_enhance,
                                data.shape[2] // s_enhance, s_enhance,
                                data.shape[3],
                                data.shape[4])
            data = data.sum(axis=(2, 4)) / s_enhance**2

        elif obs_axis and len(data.shape) == 4:
            data = data.reshape(data.shape[0],
                                data.shape[1] // s_enhance, s_enhance,
                                data.shape[2] // s_enhance, s_enhance,
                                data.shape[3])
            data = data.sum(axis=(2, 4)) / s_enhance**2

        elif not obs_axis and len(data.shape) == 4:
            data = data.reshape(data.shape[0] // s_enhance, s_enhance,
                                data.shape[1] // s_enhance, s_enhance,
                                data.shape[2],
                                data.shape[3])
            data = data.sum(axis=(1, 3)) / s_enhance**2

        elif not obs_axis and len(data.shape) == 3:
            data = data.reshape(data.shape[0] // s_enhance, s_enhance,
                                data.shape[1] // s_enhance, s_enhance,
                                data.shape[2])
            data = data.sum(axis=(1, 3)) / s_enhance**2

        else:
            msg = ('Data must be 3D, 4D, or 5D to do spatial coarsening, but '
                   f'received: {data.shape}')
            logger.error(msg)
            raise ValueError(msg)

    return data


def spatial_simple_enhancing(data, s_enhance=2, obs_axis=True):
    """"Simple enhancing according to s_enhance resolution

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
        msg = ('Data must be 3D, 4D, or 5D to do spatial enhancing, but '
               f'received: {data.shape}')
        logger.error(msg)
        raise ValueError(msg)

    if s_enhance is not None and s_enhance > 1:

        if obs_axis and len(data.shape) == 5:
            enhancement = [1, s_enhance, s_enhance, 1, 1]
            enhanced_data = interpolation.zoom(data,
                                               enhancement,
                                               order=0)

        elif obs_axis and len(data.shape) == 4:
            enhancement = [1, s_enhance, s_enhance, 1]
            enhanced_data = interpolation.zoom(data,
                                               enhancement,
                                               order=0)

        elif not obs_axis and len(data.shape) == 4:
            enhancement = [s_enhance, s_enhance, 1, 1]
            enhanced_data = interpolation.zoom(data,
                                               enhancement,
                                               order=0)

        elif not obs_axis and len(data.shape) == 3:
            enhancement = [s_enhance, s_enhance, 1]
            enhanced_data = interpolation.zoom(data,
                                               enhancement,
                                               order=0)
        else:
            msg = ('Data must be 3D, 4D, or 5D to do spatial enhancing, but '
                   f'received: {data.shape}')
            logger.error(msg)
            raise ValueError(msg)

    else:

        enhanced_data = data

    return enhanced_data


def lat_lon_coarsening(lat_lon, s_enhance=2):
    """"Coarsen lat_lon according to s_enhance resolution

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
    coarse_lat_lon = lat_lon.reshape(-1, s_enhance,
                                     lat_lon.shape[1] // s_enhance,
                                     s_enhance, 2).sum((3, 1))
    coarse_lat_lon /= (s_enhance * s_enhance)
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


def extract_var(data, var, raster_index, time_slice=slice(None)):
    """Extract WRF variable values. This is meant to extract values from
    fields without staggered dimensions

    Parameters
    ----------
    data : xarray
        netcdf data object
    var : str
        Name of variable to be extracted
    raster_index : list
        List of slices for raster index of spatial domain
    time_slice : slice
        slice of time to extract

    Returns
    -------
    ndarray
        Extracted array of variable values.
    """

    idx = [time_slice, slice(None), raster_index[0], raster_index[1]]

    assert not any('stag' in d for d in data[var].dims)

    return np.array(data[var][tuple(idx)], dtype=np.float32)


def unstagger_var(data, var, raster_index, time_slice=slice(None)):
    """Unstagger WRF variable values. Some variables use a staggered grid with
    values associated with grid cell edges. We want to center these values.

    Parameters
    ----------
    data : xarray
        netcdf data object
    var : str
        Name of variable to be unstaggered
    raster_index : list
        List of slices for raster index of spatial domain
    time_slice : slice
        slice of time to extract

    Returns
    -------
    ndarray
        Unstaggered array of variable values.
    """

    idx = [time_slice, slice(None), raster_index[0], raster_index[1]]
    assert any('stag' in d for d in data[var].dims)

    if 'stag' in data[var].dims[2]:
        idx[2] = slice(idx[2].start, idx[2].stop + 1)
    if 'stag' in data[var].dims[3]:
        idx[3] = slice(idx[3].start, idx[3].stop + 1)

    array_in = np.array(data[var][tuple(idx)], dtype=np.float32)

    for i, d in enumerate(data[var].dims):
        if 'stag' in d:
            array_in = np.apply_along_axis(forward_average, i, array_in)

    return array_in


def calc_height(data, raster_index, time_slice=slice(None)):
    """Calculate height from the ground

    Parameters
    ----------
    data : xarray
        netcdf data object
    raster_index : list
        List of slices for raster index of spatial domain
    time_slice : slice
        slice of time to extract

    Returns
    -------
    height_arr : ndarray
        (temporal, vertical_level, spatial_1, spatial_2)
        4D array of heights above ground. In meters.
    """
    if all(field in data for field in ('PHB', 'PH', 'HGT')):
        # Base-state Geopotential(m^2/s^2)
        if any('stag' in d for d in data['PHB'].dims):
            gp = unstagger_var(data, 'PHB', raster_index, time_slice)
        else:
            gp = extract_var(data, 'PHB', raster_index, time_slice)

        # Perturbation Geopotential (m^2/s^2)
        if any('stag' in d for d in data['PH'].dims):
            gp += unstagger_var(data, 'PH', raster_index, time_slice)
        else:
            gp += extract_var(data, 'PH', raster_index, time_slice)

        # Terrain Height (m)
        hgt = data['HGT'][(time_slice,) + tuple(raster_index)]
        if gp.shape != hgt.shape:
            hgt = np.repeat(np.expand_dims(hgt, axis=1), gp.shape[-3], axis=1)
        hgt = gp / 9.81 - hgt
        del gp

    elif all(field in data for field in ('zg', 'orog')):
        gp = data['zg'][(time_slice, slice(None),) + tuple(raster_index)]
        hgt = data['orog'][tuple(raster_index)]
        hgt = np.repeat(np.expand_dims(hgt, axis=0), gp.shape[1], axis=0)
        hgt = np.repeat(np.expand_dims(hgt, axis=0), gp.shape[0], axis=0)
        hgt = gp - hgt
        del gp

    else:
        msg = ('Need either PHB/PH/HGT or zg/orog in data to perform height '
               'interpolation')
        raise ValueError(msg)
    return np.array(hgt)


def calc_pressure(data, var, raster_index, time_slice=slice(None)):
    """Calculate pressure array

    Parameters
    ----------
    data : xarray
        netcdf data object
    var : str
        Feature to extract from data
    raster_index : list
        List of slices for raster index of spatial domain
    time_slice : slice
        slice of time to extract

    Returns
    -------
    height_arr : ndarray
        (temporal, vertical_level, spatial_1, spatial_2)
        4D array of pressure levels in pascals
    """
    idx = (time_slice, slice(None),) + tuple(raster_index)
    p_array = np.zeros(data[var][idx].shape, dtype=np.float32)
    for i in range(p_array.shape[1]):
        p_array[:, i, ...] = data.plev[i]

    return p_array


def interp_to_level(var_array, lev_array, levels):
    """Interpolate var_array to given level(s) based on h_array. Interpolation
    is linear and done for every 'z' column of [var, h] data.

    Parameters
    ----------
    var_array : ndarray
        Array of variable data, for example u-wind in a 4D array of shape
        (time, vertical, lat, lon)
    lev_array : ndarray
        Array of height or pressure values corresponding to the wrf source data
        in the same shape as var_array. If this is height and the requested
        levels are input is a hub height above surface, lev_array should be the
        geopotential height corresponding to every var_array index relative to
        the surface elevation (subtract the elevation at the surface from the
        geopotential height)
    levels : float | list
        level or levels to interpolate to (e.g. final desired hub heights above
        surface elevation)

    Returns
    -------
    out_array : ndarray
        Array of interpolated values.
    """

    msg = ('Input arrays must be the same shape.'
           f'\nvar_array: {var_array.shape}'
           f'\nh_array: {lev_array.shape}')
    assert var_array.shape == lev_array.shape, msg

    levels = ([levels] if isinstance(levels, (int, float, np.float32))
              else levels)

    if np.isnan(lev_array).all():
        msg = 'All pressure level height data is NaN!'
        logger.error(msg)
        raise RuntimeError(msg)

    nans = np.isnan(lev_array)
    bad_min = min(levels) < lev_array[:, 0, :, :]
    bad_max = max(levels) > lev_array[:, -1, :, :]

    if nans.any():
        msg = ('Approximately {:.2f}% of the vertical level '
               'array is NaN. Data will be interpolated or extrapolated '
               'past these NaN values.'
               .format(100 * nans.sum() / nans.size))
        logger.warning(msg)
        warn(msg)

    if bad_min.any():
        msg = ('Approximately {:.2f}% of the lowest vertical levels '
               '(maximum value of {:.3f}) '
               'were greater than the minimum requested level: {}'
               .format(100 * bad_min.sum() / bad_min.size,
                       lev_array[:, 0, :, :].max(), min(levels)))
        logger.warning(msg)
        warn(msg)

    if bad_max.any():
        msg = ('Approximately {:.2f}% of the highest vertical levels '
               '(minimum value of {:.3f}) '
               'were lower than the maximum requested level: {}'
               .format(100 * bad_max.sum() / bad_max.size,
                       lev_array[:, -1, :, :].min(), max(levels)))
        logger.warning(msg)
        warn(msg)

    array_shape = var_array.shape

    # Flatten h_array and var_array along lat, long axis
    shape = (len(levels), array_shape[-4], np.product(array_shape[-2:]))
    out_array = np.zeros(shape, dtype=np.float32).T

    # if multiple vertical levels have identical heights at the desired
    # interpolation level, interpolation to that value will fail because linear
    # slope will be NaN. This is most common if you have multiple pressure
    # levels at zero height at the surface in the case that the data didnt
    # provide underground data.
    for level in levels:
        mask = (lev_array == level)
        lev_array[mask] += np.random.uniform(-1e-5, 0, size=mask.sum())

    # iterate through time indices
    for idt in range(array_shape[0]):
        shape = (array_shape[-3], np.product(array_shape[-2:]))
        h_tmp = lev_array[idt].reshape(shape).T
        var_tmp = var_array[idt].reshape(shape).T
        not_nan = ~np.isnan(h_tmp) & ~np.isnan(var_tmp)

        # Interp each vertical column of height and var to requested levels
        zip_iter = zip(h_tmp, var_tmp, not_nan)
        out_array[:, idt, :] = np.array(
            [interp1d(h[mask], var[mask], fill_value='extrapolate')(levels)
             for h, var, mask in zip_iter], dtype=np.float32)

    # Reshape out_array
    if isinstance(levels, (float, np.float32, int)):
        shape = (1, array_shape[-4], array_shape[-2], array_shape[-1])
        out_array = out_array.T.reshape(shape)
    else:
        shape = (len(levels), array_shape[-4], array_shape[-2],
                 array_shape[-1])
        out_array = out_array.T.reshape(shape)

    return out_array


def interp_var_to_height(data, var, raster_index, heights,
                         time_slice=slice(None)):
    """Interpolate var_array to given level(s) based on h_array. Interpolation
    is linear and done for every 'z' column of [var, h] data.

    Parameters
    ----------
    data : xarray
        netcdf data object
    var : str
        Name of variable to be interpolated
    raster_index : list
        List of slices for raster index of spatial domain
    heights : float | list
        level or levels to interpolate to (e.g. final desired hub heights)
    time_slice : slice
        slice of time to extract

    Returns
    -------
    out_array : ndarray
        Array of interpolated values.
    """
    if len(data[var].dims) == 5:
        raster_index = [0] + raster_index
    logger.debug(f'Interpolating {var} to heights (meters): {heights}')
    hgt = calc_height(data, raster_index, time_slice)
    if data[var].dims == ('plev',):
        arr = np.array(data[var])
        arr = np.expand_dims(arr, axis=(0, 2, 3))
        arr = np.repeat(arr, hgt.shape[0], axis=0)
        arr = np.repeat(arr, hgt.shape[2], axis=2)
        arr = np.repeat(arr, hgt.shape[3], axis=3)
    elif all('stag' not in d for d in data[var].dims):
        arr = extract_var(data, var, raster_index, time_slice)
    else:
        arr = unstagger_var(data, var, raster_index, time_slice)
    return interp_to_level(arr, hgt, heights)[0]


def interp_var_to_pressure(data, var, raster_index, pressures,
                           time_slice=slice(None)):
    """Interpolate var_array to given level(s) based on h_array. Interpolation
    is linear and done for every 'z' column of [var, h] data.

    Parameters
    ----------
    data : xarray
        netcdf data object
    var : str
        Name of variable to be interpolated
    raster_index : list
        List of slices for raster index of spatial domain
    pressures : float | list
        level or levels to interpolate to (e.g. final desired hub heights)
    time_slice : slice
        slice of time to extract

    Returns
    -------
    out_array : ndarray
        Array of interpolated values.
    """
    logger.debug(f'Interpolating {var} to pressures (Pa): {pressures}')
    if len(data[var].dims) == 5:
        raster_index = [0] + raster_index

    if all('stag' not in d for d in data[var].dims):
        arr = extract_var(data, var, raster_index, time_slice)
    else:
        arr = unstagger_var(data, var, raster_index, time_slice)

    p_levels = calc_pressure(data, var, raster_index, time_slice)

    return interp_to_level(arr[:, ::-1], p_levels[:, ::-1], pressures)[0]


def potential_temperature(T, P):
    """Potential temperature of fluid at pressure P and temperature T

    Parameters
    ----------
    T : ndarray
        Temperature in celsius
    P : ndarray
        Pressure of fluid in Pa

    Returns
    -------
    ndarray
        Potential temperature
    """
    out = (T + np.float32(273.15))
    out *= (np.float32(100000) / P) ** np.float32(0.286)
    return out


def invert_pot_temp(PT, P):
    """Potential temperature of fluid at pressure P and temperature T

    Parameters
    ----------
    PT : ndarray
        Potential temperature in Kelvin
    P : ndarray
        Pressure of fluid in Pa

    Returns
    -------
    ndarray
        Temperature in celsius
    """
    out = PT * (P / np.float32(100000)) ** np.float32(0.286)
    out -= np.float32(273.15)
    return out


def potential_temperature_difference(T_top, P_top, T_bottom, P_bottom):
    """Potential temp difference calculation

    Parameters
    ---------
    T_top : ndarray
        Temperature at higher height. Used in the approximation of potential
        temperature derivative
    T_bottom : ndarray
        Temperature at lower height. Used in the approximation of potential
        temperature derivative
    P_top : ndarray
        Pressure at higher height. Used in the approximation of potential
        temperature derivative
    P_bottom : ndarray
        Pressure at lower height. Used in the approximation of potential
        temperature derivative

    Returns
    -------
    ndarray
        Difference in potential temperature between top and bottom levels
    """
    return (potential_temperature(T_top, P_top)
            - potential_temperature(T_bottom, P_bottom))


def potential_temperature_average(T_top, P_top, T_bottom, P_bottom):
    """Potential temp average calculation

    Parameters
    ---------
    T_top : ndarray
        Temperature at higher height. Used in the approximation of potential
        temperature derivative
    T_bottom : ndarray
        Temperature at lower height. Used in the approximation of potential
        temperature derivative
    P_top : ndarray
        Pressure at higher height. Used in the approximation of potential
        temperature derivative
    P_bottom : ndarray
        Pressure at lower height. Used in the approximation of potential
        temperature derivative

    Returns
    -------
    ndarray
        Average of potential temperature between top and bottom levels
    """

    return ((potential_temperature(T_top, P_top)
            + potential_temperature(T_bottom, P_bottom)) / np.float32(2.0))


def inverse_mo_length(U_star, flux_surf):
    """Inverse Monin - Obukhov Length

    Parameters
    ----------
    U_star : ndarray
        (spatial_1, spatial_2, temporal)
        Frictional wind speed
    flux_surf : ndarray
        (spatial_1, spatial_2, temporal)
        Surface heat flux

    Returns
    -------
    ndarray
        (spatial_1, spatial_2, temporal)
        Inverse Monin - Obukhov Length
    """

    denom = -U_star ** 3 * 300
    numer = (0.41 * 9.81 * flux_surf)
    return numer / denom


def bvf_squared(T_top, T_bottom, P_top, P_bottom, delta_h):
    """
    Squared Brunt Vaisala Frequency

    Parameters
    ----------
    T_top : ndarray
        Temperature at higher height. Used in the approximation of potential
        temperature derivative
    T_bottom : ndarray
        Temperature at lower height. Used in the approximation of potential
        temperature derivative
    P_top : ndarray
        Pressure at higher height. Used in the approximation of potential
        temperature derivative
    P_bottom : ndarray
        Pressure at lower height. Used in the approximation of potential
        temperature derivative
    delta_h : float
        Difference in heights between top and bottom levels

    Results
    -------
    ndarray
        Squared Brunt Vaisala Frequency
    """

    bvf2 = np.float32(9.81 / delta_h)
    bvf2 *= potential_temperature_difference(
        T_top, P_top, T_bottom, P_bottom)
    bvf2 /= potential_temperature_average(
        T_top, P_top, T_bottom, P_bottom)

    return bvf2


def gradient_richardson_number(T_top, T_bottom, P_top, P_bottom, U_top,
                               U_bottom, V_top, V_bottom, delta_h):
    """Formula for the gradient richardson number - related to the bouyant
    production or consumption of turbulence divided by the shear production of
    turbulence. Used to indicate dynamic stability

    Parameters
    ----------
    T_top : ndarray
        Temperature at higher height. Used in the approximation of potential
        temperature derivative
    T_bottom : ndarray
        Temperature at lower height. Used in the approximation of potential
        temperature derivative
    P_top : ndarray
        Pressure at higher height. Used in the approximation of potential
        temperature derivative
    P_bottom : ndarray
        Pressure at lower height. Used in the approximation of potential
        temperature derivative
    U_top : ndarray
        Zonal wind component at higher height
    U_bottom : ndarray
        Zonal wind component at lower height
    V_top : ndarray
        Meridional wind component at higher height
    V_bottom : ndarray
        Meridional wind component at lower height
    delta_h : float
        Difference in heights between top and bottom levels

    Returns
    -------
    ndarray
        Gradient Richardson Number
    """

    ws_grad = (U_top - U_bottom) ** 2
    ws_grad += (V_top - V_bottom) ** 2
    ws_grad /= delta_h ** 2
    ws_grad[ws_grad < 1e-6] = 1e-6
    Ri = bvf_squared(
        T_top, T_bottom, P_top, P_bottom, delta_h) / ws_grad
    del ws_grad
    return Ri


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
    indices = nd.distance_transform_edt(nan_mask, return_distances=False,
                                        return_indices=True)
    array = array[tuple(indices)]
    return array


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
    for file in os.listdir(dirname):
        if fnmatch(file.lower(), basename.lower()):
            return os.path.join(dirname, file)
    return None


def rotor_area(h_bottom, h_top, radius=40):
    """Area of circular section between two heights

    Parameters
    ----------
    h_bottom : float
        Lower height
    h_top : float
        Upper height
    radius : float
        Radius of rotor. Default is 40 meters

    Returns
    -------
    area : float
    """

    x_bottom = np.sqrt(radius**2 - h_bottom**2)
    x_top = np.sqrt(radius**2 - h_top**2)
    area = h_top * x_top - h_bottom * x_bottom
    area += radius**2 * np.arctan2(h_top, x_top)
    area -= radius**2 * np.arctan2(h_bottom, x_bottom)
    return area


def rotor_equiv_ws(data, heights):
    """Calculate rotor equivalent wind speed. Follows implementation in 'How
    wind speed shear and directional veer affect the power production of a
    megawatt-scale operational wind turbine. DOI:10.5194/wes-2019-86'

    Parameters
    ----------
    data : dict
        Dictionary of arrays for windspeeds/winddirections at different hub
        heights.
        Each dictionary entry has (spatial_1, spatial_2, temporal)
    heights : list
        List of heights corresponding to the windspeeds/winddirections.
        rotor is assumed to be at mean(heights).

    Returns
    -------
    rews : ndarray
        Array of rotor equivalent windspeeds.
        (spatial_1, spatial_2, temporal)
    """

    rotor_center = np.mean(heights)
    rel_heights = [h - rotor_center for h in heights]
    areas = [rotor_area(rel_heights[i], rel_heights[i + 1])
             for i in range(len(rel_heights) - 1)]
    total_area = np.sum(areas)
    areas /= total_area
    rews = np.zeros(data[list(data.keys())[0]].shape)
    for i in range(len(heights) - 1):
        ws_0 = data[f'windspeed_{heights[i]}m']
        ws_1 = data[f'windspeed_{heights[i + 1]}m']
        wd_0 = data[f'winddirection_{heights[i]}m']
        wd_1 = data[f'winddirection_{heights[i + 1]}m']
        ws_cos_0 = np.cos(np.radians(wd_0)) * ws_0
        ws_cos_1 = np.cos(np.radians(wd_1)) * ws_1
        rews += areas[i] * (ws_cos_0 + ws_cos_1)**3

    rews = 0.5 * np.cbrt(rews)
    return rews


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
    else:
        return 'nc'


def get_input_handler_class(file_paths, input_handler_name):
    """Get the DataHandler class.

    Parameters
    ----------
    file_paths : list | str
        A list of files to extract raster data from. Each file must have
        the same number of timesteps. Can also pass a string with a
        unix-style file path which will be passed through glob.glob
    input_handler : str
        data handler class to use for input data. Provide a string name to
        match a class in data_handling.py. If None the correct handler will
        be guessed based on file type and time series properties.

    Returns
    -------
    HandlerClass : DataHandlerH5 | DataHandlerNC
        DataHandler subclass from sup3r.preprocessing.data_handling.
    """

    HandlerClass = None

    input_type = get_source_type(file_paths)

    if input_handler_name is None:
        if input_type == 'nc':
            input_handler_name = 'DataHandlerNC'
        elif input_type == 'h5':
            input_handler_name = 'DataHandlerH5'

        logger.info('"input_handler" arg was not provided. Using '
                    f'"{input_handler_name}". If this is '
                    'incorrect, please provide '
                    'input_handler="DataHandlerName".')

    if isinstance(input_handler_name, str):
        import sup3r.preprocessing.data_handling
        HandlerClass = getattr(sup3r.preprocessing.data_handling,
                               input_handler_name, None)

    if HandlerClass is None:
        msg = ('Could not find requested data handler class '
               f'"{input_handler_name}" in sup3r.preprocessing.data_handling.')
        logger.error(msg)
        raise KeyError(msg)

    return HandlerClass


def np_to_pd_times(times):
    """Convert np.bytes_ times to DatetimeIndex

    Parameters
    ----------
    times : ndarray | list
        List of np.bytes_ objects for time indices

    Returns
    -------
    times : pd.DatetimeIndex
        DatetimeIndex for time indices
    """
    tmp = [t.decode('utf-8') for t in times.flatten()]
    tmp = [' '.join(t.split('_')) for t in tmp]
    tmp = pd.DatetimeIndex(tmp)
    return tmp


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
    interp = RegularGridInterpolator((y, x, t), low, bounds_error=False,
                                     fill_value=None)

    # perform interp
    X, Y, T = np.meshgrid(new_x, new_y, new_t)
    out = interp((Y, X, T))

    return out


def vorticity_calc(u, v, scale=1):
    """Returns the vorticity field.

    Parameters
    ----------
    u: ndarray
        Longitudinal velocity component
        (lat, lon, temporal)
    v : ndarray
        Latitudinal velocity component
        (lat, lon, temporal)
    scale : float
        Value to scale vorticity by. Typically the spatial resolution, so that
        spatial derivatives can be compared across different resolutions

    Returns
    -------
    ndarray
        vorticity values
        (lat, lon, temporal)
    """
    dudy = np.diff(u, axis=0, append=np.mean(u))
    dvdx = np.diff(v, axis=1, append=np.mean(v))
    diffs = dudy - dvdx
    diffs /= scale
    return diffs
