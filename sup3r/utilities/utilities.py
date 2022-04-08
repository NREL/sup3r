# -*- coding: utf-8 -*-
"""Utilities module for preparing
training data"""

import numpy as np
import logging

np.random.seed(42)

logger = logging.getLogger(__name__)


def uniform_box_sampler(data, shape):
    '''
    Extracts a sample cut from data.

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
    slices : List of slices corresponding to row
    and col extent of arr sample
    '''

    slices = []
    if data.shape[0] <= shape[0]:
        start_row = 0
        stop_row = data.shape[0]
    else:
        start_row = np.random.randint(0, data.shape[0] - shape[0])
        stop_row = start_row + shape[0]

    if data.shape[1] <= shape[1]:
        start_col = 0
        stop_col = data.shape[1]
    else:
        start_col = np.random.randint(0, data.shape[1] - shape[1])
        stop_col = start_col + shape[1]

    slices = [slice(start_row, stop_row), slice(start_col, stop_col)]
    return slices


def uniform_time_sampler(data, shape):
    '''
    Extracts a temporal slice from data.

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

    if data.shape[2] <= shape:
        start = 0
        stop = data.shape[2]
    else:
        start = np.random.randint(0, data.shape[2] - shape)
        stop = start + shape
    return slice(start, stop)


def transform_rotate_wind(ws, wd, lat_lon):
    """Transform windspeed/direction to
    u and v and align u and v with grid

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

    return (np.sin(theta)[:, :, np.newaxis] * ws * np.sin(wd)
            + np.cos(theta)[:, :, np.newaxis] * ws * np.cos(wd),
            np.cos(theta)[:, :, np.newaxis] * ws * np.sin(wd)
            - np.sin(theta)[:, :, np.newaxis] * ws * np.cos(wd))


def transform_wind(ws, wd):
    """Maps windspeed and direction to u v

    Parameters
    ----------
    ws : np.ndarray
        3D array of windspeed (spatial_1, spatial_2, temporal)
    wd : int
        3D array of winddirection (spatial_1, spatial_2, temporal)

    Returns
    -------
    u : np.ndarray
        3D array of zonal wind components

    v : np.ndarray
        3D array of meridional wind components
    """

    return (ws * np.cos(np.radians(wd - 180.0)),
            ws * np.sin(np.radians(wd - 180.0)))


def rotate_u_v(u, v, lat_lon):
    """aligns u v with grid

    Parameters
    ----------
    u : np.ndarray
        3D array of zonal wind components
        (spatial_1, spatial_2, temporal)

    v : np.ndarray
        3D array of meridional wind components
        (spatial_1, spatial_2, temporal)
    lat_lon : np.ndarray
        3D array (spatial_1, spatial_2, 2)
        2 channels are lat and lon in that
        order

    Returns
    -------
    u_rot : np.ndarray
        3D array of zonal wind components

    v_rot : np.ndarray
        3D array of meridional wind components
    """

    # get the dy/dx to the nearest vertical neighbor
    dy = lat_lon[:, :, 0] - np.roll(lat_lon[:, :, 0], 1, axis=0)
    dx = lat_lon[:, :, 1] - np.roll(lat_lon[:, :, 1], 1, axis=0)

    # calculate the angle from the vertical
    theta = (np.pi / 2) - np.arctan2(dy, dx)
    del dy, dx
    theta[0] = theta[1]  # fix the roll row

    return (np.sin(theta)[:, :, np.newaxis] * v
            + np.cos(theta)[:, :, np.newaxis] * u,
            np.cos(theta)[:, :, np.newaxis] * v
            - np.sin(theta)[:, :, np.newaxis] * u)


def temporal_coarsening(data, temporal_res=2, method='subsample'):
    """"Coarsen data according to temporal_res resolution

    Parameters
    ----------
    data : np.ndarray
        5D array with dimensions
        (observations, spatial_1, spatial_2, temporal, features)

    temporal_res : int
        factor by which to coarsen temporal dimension

    method : str
        accepted options: [subsample, average, total]
        Subsample will take every temporal_res-th time step,
        average will average over temporal_res time steps,
        total will sum over temporal_res time steps

    Returns
    -------
    coarse_data : np.ndarray
        4D array with same dimensions as data
        with new coarse resolution
    """

    if temporal_res is not None and len(data.shape) == 5:
        if method == 'subsample':
            coarse_data = data[:, :, :, ::temporal_res, :]
        if method == 'average':
            coarse_data = np.average(
                data.reshape(
                    (data.shape[0], data.shape[1],
                     data.shape[2], -1, temporal_res,
                     data.shape[4])), axis=4)
        if method == 'total':
            coarse_data = np.sum(
                data.reshape(
                    (data.shape[0], data.shape[1],
                     data.shape[2], -1, temporal_res,
                     data.shape[4])), axis=4)

    else:
        coarse_data = data

    return coarse_data


def spatial_coarsening(data, spatial_res=2):
    """"Coarsen data according to spatial_res resolution

    Parameters
    ----------
    data : np.ndarray
        4D | 5D array with dimensions
        (n_observations, spatial_1, spatial_2, temporal (optional), features)

    lat_lon : np.ndarray
        2D array with dimensions
        (spatial_1, spatial_2)

    spatial_res : int
        factor by which to coarsen spatial dimensions

    Returns
    -------
    coarse_data : np.ndarray
        4D | 5D array with same dimensions as data
        with new coarse resolution
    """

    if spatial_res is not None:
        if (data.shape[1] % spatial_res != 0
                or data.shape[2] % spatial_res != 0):
            msg = 'spatial_res must evenly divide grid size. '
            msg += f'Received spatial_res: {spatial_res} '
            msg += f'with grid size: ({data.shape[1]}, '
            msg += f'{data.shape[2]})'
            raise ValueError(msg)

        if len(data.shape) == 5:
            coarse_data = data.reshape(data.shape[0],
                                       -1,
                                       spatial_res,
                                       data.shape[1] // spatial_res,
                                       spatial_res,
                                       data.shape[3],
                                       data.shape[4]).sum((2, 4)) \
                / (spatial_res * spatial_res)

        else:
            coarse_data = data.reshape(data.shape[0], -1,
                                       spatial_res,
                                       data.shape[1] // spatial_res,
                                       spatial_res,
                                       data.shape[3]).sum((2, 4)) \
                / (spatial_res * spatial_res)

    else:
        coarse_data = data

    return coarse_data


def lat_lon_coarsening(lat_lon, spatial_res=2):
    """"Coarsen lat_lon according to spatial_res resolution

    Parameters
    ----------
    lat_lon : np.ndarray
        2D array with dimensions
        (spatial_1, spatial_2)

    spatial_res : int
        factor by which to coarsen spatial dimensions

    Returns
    -------
    coarse_lat_lon : np.ndarray
        2D array with same dimensions as lat_lon
        with new coarse resolution
    """
    coarse_lat_lon = lat_lon.reshape(-1, spatial_res,
                                     lat_lon.shape[1] // spatial_res,
                                     spatial_res, 2).sum((3, 1)) \
        / (spatial_res * spatial_res)
    return coarse_lat_lon


def forward_average(array_in):
    """
    Average neighboring values in an array.
    Used to unstagger WRF variable values.
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


def unstagger_var(data, var):
    """
    Unstagger WRF variable values.
    Some variables use a staggered grid with values
    associated with grid cell edges. We want to center
    these values.

    Parameters
    ----------
    data : xarray
        netcdf data object
    var : str
        Name of variable to be unstaggered

    Returns
    -------
    ndarray
        Unstaggered array of variable values.
    """

    # Import Variable values from nc database instance
    array_in = np.array(data[var], np.float32)

    for i, d in enumerate(data[var].dims):
        if 'stag' in d:
            array_in = np.apply_along_axis(
                forward_average, i, array_in
            )
    return array_in


def calc_height(data):
    """
    Calculate height from the ground
    Parameters
    ----------
    data : xarray
        netcdf data object

    Returns
    ---------
    height_arr : ndarray
        (spatial_1, spatial_2, vertical_level, temporal)
        4D array of heights above ground. In meters.
    """
    # Base-state Geopotential(m^2/s^2)
    phb = unstagger_var(data, 'PHB')
    # Perturbation Geopotential (m^2/s^2)
    ph = unstagger_var(data, 'PH')
    # Terrain Height (m)
    hgt = data['HGT']

    if phb.shape != hgt.shape:
        hgt = np.expand_dims(hgt, axis=1)
        hgt = np.repeat(hgt, phb.shape[-3], axis=1)

    hgt = (ph + phb) / 9.81 - hgt
    del ph, phb
    return hgt


def interp3D(var_array, h_array, heights):
    """
    Interpolate var_array to given level(s) based on h_array.
    Interpolation is linear and done for every 'z' column of [var, h] data.
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

    heights = [heights] if isinstance(heights, (int, float)) else heights
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
        logger.error(msg)
        raise RuntimeError(msg)

    array_shape = var_array.shape

    # Reduce h_array and var_array shape to accomodate interpolation
    if len(array_shape) == 4:
        h_array = h_array[0]
        var_array = var_array[0]

    # Flatten h_array and var_array along lat, long axis
    h_array = h_array.reshape((array_shape[-3],
                               np.product(array_shape[-2:]))).T
    var_array = var_array.reshape((array_shape[-3],
                                   np.product(array_shape[-2:]))).T

    # Interpolate each column of height and var to specified levels
    out_array = np.array(
        [np.interp(heights, h, var) for h, var in zip(h_array, var_array)],
        np.float32)
    # Reshape out_array
    if isinstance(heights, (float, int)):
        out_array = out_array.T.reshape((1, array_shape[-2],
                                         array_shape[-1]))
    else:
        out_array = out_array.T.reshape((len(heights), array_shape[-2],
                                         array_shape[-1]))

    return out_array


def interp_var(data, var, heights):
    """
    Interpolate var_array to given level(s) based on h_array.
    Interpolation is linear and done for every 'z' column of [var, h] data.

    Parameters
    ----------
    data : xarray
        netcdf data object
    var : str
        Name of variable to be interpolated
    heights : float | list
        level or levels to interpolate to (e.g. final desired hub heights)
    Returns
    -------
    out_array : ndarray
        Array of interpolated values.
    """

    logger.debug(f'Interpolating {var} to heights: {heights}')
    return interp3D(
        unstagger_var(data, var),
        calc_height(data),
        heights)


def potential_temperature(T, P):
    """Potential temperature of fluid
    at pressure P and temperature T

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
    P0 = 100000
    return np.array(
        (T + 273.15) * (P0 / P) ** (0.286),
        dtype=np.float32)


def potential_temperature_difference(T_top, P_top,
                                     T_bottom, P_bottom):
    """Potential temp difference calculation

    Parameters
    ---------
    T_top : ndarray
        Temperature at higher height.
        Used in the approximation of
        potential temperature derivative
    T_bottom : ndarray
        Temperature at lower height.
        Used in the approximation of
        potential temperature derivative
    P_top : ndarray
        Pressure at higher height.
        Used in the approximation of
        potential temperature derivative
    P_bottom : ndarray
        Pressure at lower height.
        Used in the approximation of
        potential temperature derivative

    Returns
    -------
    ndarray
        Difference in potential temperature between
        top and bottom levels
    """
    return np.array(
        potential_temperature(T_top, P_top)
        - potential_temperature(T_bottom, P_bottom),
        dtype=np.float32)


def potential_temperature_average(T_top, P_top,
                                  T_bottom, P_bottom):
    """Potential temp average calculation

    Parameters
    ---------
    T_top : ndarray
        Temperature at higher height.
        Used in the approximation of
        potential temperature derivative
    T_bottom : ndarray
        Temperature at lower height.
        Used in the approximation of
        potential temperature derivative
    P_top : ndarray
        Pressure at higher height.
        Used in the approximation of
        potential temperature derivative
    P_bottom : ndarray
        Pressure at lower height.
        Used in the approximation of
        potential temperature derivative

    Returns
    -------
    ndarray
        Average of potential temperature between
        top and bottom levels
    """
    return np.array(
        (potential_temperature(T_top, P_top)
         + potential_temperature(T_bottom, P_bottom)) / 2.0,
        dtype=np.float32)


def virtual_var(var, mixing_ratio):
    """Formula for virtual variable
    e.g. virtual temperature, virtual
    potential temperature

    Parameters
    ----------
    var : ndarray
        Variable array (e.g. temperature
        array or potential temperature array)
    mixing_ratio : ndarray
        Ratio of the mass of water vapor to the
        mass of dry air

    Returns
    -------
    ndarray
        Virtual quantity (e.g. virtual temperature
        or virtual potential temperature, for use in
        Richardson number calculation)
    """
    return var * (1 + 0.61 * mixing_ratio)


def saturation_vapor_pressure(T):
    """Saturation Vapor pressure calculation
    using Tetens equation

    Parameters
    ----------
    T : ndarray
        Temperature in celsius

    Returns
    -------
    ndarray
        Pressure in kPa
    """

    Es = T
    Es[T > 0] = 0.61078 * np.exp(
        17.27 * T[T > 0] / (T[T > 0] + 237.3))
    Es[T <= 0] = 0.61078 * np.exp(
        21.875 * T[T <= 0] / (T[T <= 0] + 265.5))
    return Es


def vapor_pressure(T, RH):
    """
    Parameters
    ----------
    T : ndarray
        Temperature in celsius
    RH : ndarray
        Relative humidity

    Returns
    -------
    ndarray
        Pressure in kPa
    """
    Es = saturation_vapor_pressure(T)
    E = RH * Es / 100
    return E


def mixing_ratio(P, T, RH):
    """Mixing ratio calculation for
    use in richardson number calculation

    Parameters
    ----------
    P : ndarray
        Pressure in kPa
    T : ndarray
        Temperature in celsius
    RH : ndarray
        Relative humidity

    Returns
    -------
    ndarray
        Mixing ratio
    """
    vapor_p = vapor_pressure(T, RH)
    return 0.622 * vapor_p / (P - vapor_p)


def MO_length(ws_friction, PT_mean, PT_surface_flux):
    """Monin - Obukhov Length

    Parameters
    ----------
    ws_friction : ndarray
        (spatial_1, spatial_2, temporal)
        Frictional windspeed
    PT_mean : ndarray
        (spatial_1, spatial_2, temporal)
        Vertical average of potential temperature
    PT_surface_flux : ndarray
        (spatial_1, spatial_2, temporal)
        Potential temperature flux at the surface

    Returns
    -------
    ndarray
        (spatial_1, spatial_2, temporal)
        Monin - Obukhov Length
    """

    numer = -ws_friction ** 3 * PT_mean
    denom = (0.41 * 9.81 * PT_surface_flux)
    return numer / denom


def BVF_squared(T_top, T_bottom,
                P_top, P_bottom,
                delta_h):
    """
    Squared Brunt Vaisala Frequency

    Parameters
    ----------
    T_top : ndarray
        Temperature at higher height.
        Used in the approximation of
        potential temperature derivative
    T_bottom : ndarray
        Temperature at lower height.
        Used in the approximation of
        potential temperature derivative
    P_top : ndarray
        Pressure at higher height.
        Used in the approximation of
        potential temperature derivative
    P_bottom : ndarray
        Pressure at lower height.
        Used in the approximation of
        potential temperature derivative
    delta_h : float
        Difference in heights between
        top and bottom levels

    Results
    -------
    ndarray
        Squared Brunt Vaisala Frequency
    """

    return np.array(
        (9.81 / delta_h
         * potential_temperature_difference(
             T_top, P_top, T_bottom, P_bottom)
         / potential_temperature_average(
             T_top, P_top, T_bottom, P_bottom)),
        dtype=np.float32)


def gradient_richardson_number(T_top, T_bottom, P_top,
                               P_bottom, U_top, U_bottom,
                               V_top, V_bottom, delta_h):

    """Formula for the gradient richardson
    number - related to the bouyant production
    or consumption of turbulence divided by the
    shear production of turbulence. Used to indicate
    dynamic stability

    Parameters
    ----------
    T_top : ndarray
        Temperature at higher height.
        Used in the approximation of
        potential temperature derivative
    T_bottom : ndarray
        Temperature at lower height.
        Used in the approximation of
        potential temperature derivative
    P_top : ndarray
        Pressure at higher height.
        Used in the approximation of
        potential temperature derivative
    P_bottom : ndarray
        Pressure at lower height.
        Used in the approximation of
        potential temperature derivative
    U_top : ndarray
        Zonal wind component at higher
        height
    U_bottom : ndarray
        Zonal wind component at lower
        height
    V_top : ndarray
        Meridional wind component at
        higher height
    V_bottom : ndarray
        Meridional wind component at
        lower height
    delta_h : float
        Difference in heights between
        top and bottom levels

    Returns
    -------
    ndarray
        Gradient Richardson Number

    """

    ws_grad = (U_top - U_bottom) ** 2
    ws_grad += (V_top - V_bottom) ** 2
    ws_grad /= delta_h ** 2
    ws_grad[ws_grad < 1e-6] = 1e-6
    return BVF_squared(
        T_top, T_bottom, P_top, P_bottom, delta_h) / ws_grad
