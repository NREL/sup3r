# -*- coding: utf-8 -*-
"""Utilities module for preparing
training data

@author: bbenton
"""

import numpy as np
import logging
from scipy import ndimage as nd
from fnmatch import fnmatch
import os
import xarray as xr
import re
import warnings

from rex import Resource

np.random.seed(42)

logger = logging.getLogger(__name__)


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
    while start < indices[-1] + 1:
        slices.append(slice(start, stop, step))
        start = stop
        stop += step * chunk_size
        if stop > indices[-1] + 1:
            stop = indices[-1] + 1

    return slices


def get_file_t_steps(file_paths):
    """Get number of time steps in each file. We assume that each netcdf file
    in a file list passed to the handling classes has the same number of time
    steps.

    Parameters
    ----------
    file_paths : list
        List of netcdf file paths

    Returns
    -------
    int
        Number of time steps in each file
    """

    with xr.open_dataset(file_paths[0]) as handle:
        return len(handle.XTIME.values)


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
    the date pattern (YYYY-|_MM-|_DD-|_HH:|_MM:|_SS) at the end of the file
    name.

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
                           files[0])[0]
    date_end = re.search(r'(\d{4}(-|_)\d+(-|_)\d+(-|_)\d+(:|_)\d+(:|_)\d+)',
                         files[-1])[0]

    date_start = date_start.replace(':', '_')
    date_end = date_end.replace(':', '_')

    return date_start, date_end


def uniform_box_sampler(data, shape):
    '''Extracts a sample cut from data.

    Parameters:
    -----------
    data : np.ndarray
        Data array with dimensions
        (spatial_1, spatial_2, temporal, features)
    shape : tuple
        (rows, cols) Size of grid to sample
        from data

    Returns:
    --------
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


def weighted_time_sampler(data, shape, weights):
    """Extracts a temporal slice from data with selection weighted based on
    provided weights

    Parameters
    ----------
    data : np.ndarray
        Data array with dimensions
        (spatial_1, spatial_2, temporal, features)
    shape : tuple
        (time_steps) Size of time slice to sample
        from data
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
    Parameters:
    -----------
    data : np.ndarray
        Data array with dimensions
        (spatial_1, spatial_2, temporal, features)
    shape : tuple
        (time_steps) Size of time slice to sample
        from data
    Returns:
    --------
    slice : slice
        time slice with size shape
    '''
    shape = data.shape[2] if data.shape[2] < shape else shape
    start = np.random.randint(0, data.shape[2] - shape + 1)
    stop = start + shape
    return slice(start, stop)


def daily_time_sampler(data, shape, time_index):
    """
    Extracts a temporal slice from data starting at midnight of a random day

    Parameters:
    -----------
    data : np.ndarray
        Data array with dimensions
        (spatial_1, spatial_2, temporal, features)
    shape : tuple
        (time_steps) Size of time slice to sample from data, must be an integer
        multiple of 24.
    time_index : pd.Datetimeindex
        Time index that matches the data axis=2

    Returns:
    --------
    slice : slice
        time slice with size shape of data starting at the beginning of the day
    """

    msg = (f'data {data.shape} and time index ({len(time_index)}) '
           'shapes do not match, cannot sample daily data.')
    assert data.shape[2] == len(time_index), msg

    ti_short = time_index[:1 - shape]
    midnight_ilocs = np.where((ti_short.hour == 0)
                              & (ti_short.minute == 0)
                              & (ti_short.second == 0))[0]

    if data.shape[2] <= shape:
        start = 0
        stop = data.shape[2]
    else:
        start = np.random.randint(0, len(midnight_ilocs))
        start = midnight_ilocs[start]
        stop = start + shape
    return slice(start, stop)


def transform_rotate_wind(ws, wd, lat_lon):
    """Transform windspeed/direction to u and v and align u and v with grid

    Parameters
    ----------
    ws : np.ndarray
        3D array of high res windspeed data
    wd : np.ndarray
        3D array of high res winddirection data
    lat_lon : np.ndarray
        3D array of lat lon

    Returns
    -------
    u : np.ndarray
        3D array of high res U data
    v : np.ndarray
        3D array of high res V data
    """
    # get the dy/dx to the nearest vertical neighbor
    dy = lat_lon[:, :, 0] - np.roll(lat_lon[:, :, 0], 1, axis=0)
    dx = lat_lon[:, :, 1] - np.roll(lat_lon[:, :, 1], 1, axis=0)

    # calculate the angle from the vertical
    theta = (np.pi / 2) - np.arctan2(dy, dx)
    del dy, dx
    theta[0] = theta[1]  # fix the roll row
    wd = np.radians(wd - 180.0)

    u_rot = np.sin(theta)[:, :, np.newaxis] * ws * np.sin(wd)
    u_rot += np.cos(theta)[:, :, np.newaxis] * ws * np.cos(wd)

    v_rot = np.cos(theta)[:, :, np.newaxis] * ws * np.sin(wd)
    v_rot -= np.sin(theta)[:, :, np.newaxis] * ws * np.cos(wd)
    del theta, ws, wd
    return u_rot, v_rot


def invert_uv(u, v, lat_lon):
    """Transform u and v back to windspeed and winddirection

    Parameters
    ----------
    u : np.ndarray
        3D array of high res U data
    v : np.ndarray
        3D array of high res V data
    lat_lon : np.ndarray
        3D array of lat lon

    Returns
    -------
    ws : np.ndarray
        3D array of high res windspeed data
    wd : np.ndarray
        3D array of high res winddirection data
    """
    # get the dy/dx to the nearest vertical neighbor
    dy = lat_lon[:, :, 0] - np.roll(lat_lon[:, :, 0], 1, axis=0)
    dx = lat_lon[:, :, 1] - np.roll(lat_lon[:, :, 1], 1, axis=0)

    # calculate the angle from the vertical
    theta = (np.pi / 2) - np.arctan2(dy, dx)
    del dy, dx
    theta[0] = theta[1]  # fix the roll row

    u_rot = -np.sin(theta)[:, :, np.newaxis] * u
    u_rot += np.cos(theta)[:, :, np.newaxis] * v

    v_rot = np.cos(theta)[:, :, np.newaxis] * u
    v_rot += np.sin(theta)[:, :, np.newaxis] * v

    ws = np.sqrt(u_rot**2 + v_rot**2)
    wd = np.degrees(np.arctan2(u_rot, v_rot)) + 180.0

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
        accepted options: [subsample, average, total]
        Subsample will take every t_enhance-th time step, average will average
        over t_enhance time steps, total will sum over t_enhance time steps

    Returns
    -------
    coarse_data : np.ndarray
        4D array with same dimensions as data with new coarse resolution
    """

    if t_enhance is not None and len(data.shape) == 5:
        if method == 'subsample':
            coarse_data = data[:, :, :, ::t_enhance, :]
        if method == 'average':
            coarse_data = np.nansum(
                data.reshape(
                    (data.shape[0], data.shape[1],
                     data.shape[2], -1, t_enhance,
                     data.shape[4])), axis=4)
            coarse_data /= t_enhance
        if method == 'total':
            coarse_data = np.nansum(
                data.reshape(
                    (data.shape[0], data.shape[1],
                     data.shape[2], -1, t_enhance,
                     data.shape[4])), axis=4)

    else:
        coarse_data = data

    return coarse_data


def spatial_coarsening(data, s_enhance=2):
    """"Coarsen data according to s_enhance resolution

    Parameters
    ----------
    data : np.ndarray
        4D | 5D array with dimensions
        (n_observations, spatial_1, spatial_2, temporal (optional), features)
    s_enhance : int
        factor by which to coarsen spatial dimensions

    Returns
    -------
    coarse_data : np.ndarray
        4D | 5D array with same dimensions as data with new coarse resolution
    """

    if s_enhance is not None:
        if (data.shape[1] % s_enhance != 0
                or data.shape[2] % s_enhance != 0):
            msg = 's_enhance must evenly divide grid size. '
            msg += f'Received s_enhance: {s_enhance} '
            msg += f'with grid size: ({data.shape[1]}, '
            msg += f'{data.shape[2]})'
            logger.error(msg)
            raise ValueError(msg)

        if len(data.shape) == 5:
            coarse_data = data.reshape(data.shape[0],
                                       -1,
                                       s_enhance,
                                       data.shape[1] // s_enhance,
                                       s_enhance,
                                       data.shape[3],
                                       data.shape[4]).sum((2, 4)) \
                / (s_enhance * s_enhance)

        elif len(data.shape) == 4:
            coarse_data = data.reshape(data.shape[0], -1,
                                       s_enhance,
                                       data.shape[1] // s_enhance,
                                       s_enhance,
                                       data.shape[3]).sum((2, 4)) \
                / (s_enhance * s_enhance)

        else:
            msg = ('Data must be 4D or 5D to do spatial coarsening, but '
                   f'received: {data.shape}')
            logger.error(msg)
            raise ValueError(msg)

    else:
        coarse_data = data

    return coarse_data


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


def unstagger_var(data, var, raster_index, time_slice=slice(None)):
    """
    Unstagger WRF variable values. Some variables use a staggered grid with
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

    # Import Variable values from nc database instance
    array_in = np.array(data[var], np.float32)

    if all('stag' not in d for d in data[var].dims):
        array_in = array_in[
            tuple([time_slice] + [slice(None)] + raster_index)]

    else:
        for i, d in enumerate(data[var].dims):
            if 'stag' in d:
                if 'south_north' in d:
                    idx = tuple(
                        [time_slice] + [slice(None)]
                        + [slice(raster_index[0].start,
                                 raster_index[0].stop + 1)]
                        + [raster_index[1]])
                    array_in = array_in[idx]
                elif 'west_east' in d:
                    idx = tuple(
                        [time_slice] + [slice(None)]
                        + [raster_index[0]]
                        + [slice(raster_index[1].start,
                                 raster_index[1].stop + 1)])
                    array_in = array_in[idx]
                else:
                    idx = tuple(
                        [time_slice] + [slice(None)]
                        + raster_index)
                    array_in = array_in[idx]
                array_in = np.apply_along_axis(
                    forward_average, i, array_in)
    return array_in


def calc_height(data, raster_index, time_slice=slice(None)):
    """
    Calculate height from the ground Parameters
    ----------
    data : xarray
        netcdf data object
    raster_index : list
        List of slices for raster index of spatial domain
    time_slice : slice
        slice of time to extract

    Returns
    ---------
    height_arr : ndarray
        (temporal, vertical_level, spatial_1, spatial_2)
        4D array of heights above ground. In meters.
    """
    # Base-state Geopotential(m^2/s^2)
    phb = unstagger_var(data, 'PHB', raster_index, time_slice)
    # Perturbation Geopotential (m^2/s^2)
    ph = unstagger_var(data, 'PH', raster_index, time_slice)
    # Terrain Height (m)
    hgt = data['HGT'][tuple([time_slice] + raster_index)]

    if phb.shape != hgt.shape:
        hgt = np.expand_dims(hgt, axis=1)
        hgt = np.repeat(hgt, phb.shape[-3], axis=1)

    hgt = (ph + phb) / 9.81 - hgt
    del ph, phb
    return hgt


def interp3D(var_array, h_array, heights):
    """
    Interpolate var_array to given level(s) based on h_array. Interpolation is
    linear and done for every 'z' column of [var, h] data.

    Parameters
    ----------
    var_array : ndarray
        Array of variable values
    h_array : ndarray
        Array of heigh values corresponding to the wrf source data
    heights : float | list
        level or levels to interpolate to (e.g. final desired hub heights)

    Returns
    -------
    out_array : ndarray
        Array of interpolated values.
    """

    msg = ('Input arrays must be the same shape.'
           f'\nvar_array: {var_array.shape}'
           f'\nh_array: {h_array.shape}')
    assert var_array.shape == h_array.shape, msg

    heights = [heights] if isinstance(
        heights, (int, float, np.float32)) else heights
    h_min = np.nanmin(h_array)
    h_max = np.nanmax(h_array)
    height_check = (h_min < min(heights) and max(heights) < h_max)

    if np.isnan(h_min):
        msg = 'All pressure level height data is NaN!'
        logger.error(msg)
        raise RuntimeError(msg)

    if not height_check:
        msg = (f'Heights {heights} exceed the bounds of the pressure levels: '
               f'({h_min}, {h_max})')
        logger.warning(msg)
        warnings.warn(msg)

    array_shape = var_array.shape

    # Flatten h_array and var_array along lat, long axis
    out_array = np.zeros(
        (len(heights), array_shape[-4], np.product(array_shape[-2:]))).T

    for i in range(array_shape[0]):
        h_tmp = h_array[i].reshape(
            (array_shape[-3], np.product(array_shape[-2:]))).T
        var_tmp = var_array[i].reshape(
            (array_shape[-3], np.product(array_shape[-2:]))).T

    # Interpolate each column of height and var to specified levels
        out_array[:, i, :] = np.array(
            [np.interp(heights, h, var)
             for h, var in zip(h_tmp, var_tmp)])

    # Reshape out_array
    if isinstance(heights, (float, np.float32, int)):
        out_array = out_array.T.reshape(
            (1, array_shape[-4], array_shape[-2], array_shape[-1]))
    else:
        out_array = out_array.T.reshape(
            (len(heights), array_shape[-4],
             array_shape[-2], array_shape[-1]))

    return out_array


def interp_var(data, var, raster_index, heights, time_slice=slice(None)):
    """ Interpolate var_array to given level(s) based on h_array. Interpolation
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

    logger.debug(f'Interpolating {var} to heights: {heights}')

    return interp3D(unstagger_var(data, var, raster_index, time_slice),
                    calc_height(data, raster_index, time_slice),
                    heights)[0]


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


def inverse_mo_length(U_surf, V_surf, W_surf, PT_surf):
    """Inverse Monin - Obukhov Length

    Parameters
    ----------
    U_surf : ndarray
        (spatial_1, spatial_2, temporal)
        Surface U wind component
    V_surf : ndarray
        (spatial_1, spatial_2, temporal)
        Surface V wind component
    W_surf : ndarray
        (spatial_1, spatial_2, temporal)
        Surface W wind component
    PT_surf : ndarray
        (spatial_1, spatial_2, temporal)
        Surface potential temperature

    Returns
    -------
    ndarray
        (spatial_1, spatial_2, temporal)
        Monin - Obukhov Length
    """

    U_eddy = U_surf - np.mean(U_surf, axis=2)[:, :, np.newaxis]
    V_eddy = V_surf - np.mean(V_surf, axis=2)[:, :, np.newaxis]
    W_eddy = W_surf - np.mean(W_surf, axis=2)[:, :, np.newaxis]

    PT_flux = W_eddy * (PT_surf - np.mean(PT_surf, axis=2)[:, :, np.newaxis])

    ws_friction = ((U_eddy * W_eddy) ** 2 + (V_eddy * W_eddy) ** 2) ** 0.25
    denom = -ws_friction ** 3 * PT_surf
    numer = (0.41 * 9.81 * PT_flux)
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


def get_source_type(file_paths):
    """Get data source type
    ----------
    file_paths : list
        path to data file
    Returns
    -------
    source_type : str
        Either h5 or nc
    """
    if file_paths is None:
        return None

    if not isinstance(file_paths, list):
        file_paths = [file_paths]

    _, source_type = os.path.splitext(file_paths[0])
    if source_type == '.h5':
        return 'h5'
    else:
        return 'nc'


def get_time_index(file_paths):
    """Get data file handle based on file type
    ----------
    file_paths : list
        path to data file

    Returns
    -------
    time_index : pd.DateTimeIndex | np.ndarray
        Time index from h5 or nc source file(s)
    """
    if get_source_type(file_paths) == 'h5':
        with Resource(file_paths[0], hsds=False) as handle:
            time_index = handle.time_index
    else:
        with xr.open_mfdataset(file_paths, combine='nested',
                               concat_dim='Time') as handle:
            time_index = handle.XTIME.values
    return time_index
