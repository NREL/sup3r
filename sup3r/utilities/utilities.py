# -*- coding: utf-8 -*-
"""Utilities module for preparing
training data"""

import numpy as np
import logging
from wtk.wrf import WRFHeights

np.random.seed(42)

logger = logging.getLogger(__name__)


def extract_feature(handle, raster_index,
                    feature, source_type,
                    level_index=None):
    """Extract single feature from data source

    Parameters
    ----------
    handle : WindX | xarray
        Data Handle for either WTK data
        or WRF data
    raster_index : ndarray
        Raster index array
    feature : str
        Feature to extract from data
    source_type : str
        Either h5 or nc

    Returns
    -------
    ndarray
        Data array for extracted feature

    """

    if source_type == 'h5':
        fdata = handle[feature, :, raster_index.flatten()]
        fdata = fdata.reshape((len(fdata),
                               raster_index.shape[0],
                               raster_index.shape[1]))

    elif source_type == 'nc':

        if handle[feature].shape > 3:
            if level_index is None:
                level_index = 0
                fdata = \
                    handle[feature][: level_index,
                                    raster_index[0][0]:raster_index[0][1],
                                    raster_index[1][0]:raster_index[1][1]]
        else:
            fdata = np.array(handle[feature][:,
                             raster_index[0][0]:raster_index[0][1],
                             raster_index[1][0]:raster_index[1][1]])
    else:
        raise ValueError(
            'Can only handle wtk or wrf data')

    return np.transpose(fdata, (1, 2, 0))


def get_BVF_squared(handle, raster_index, source_type):
    """Compute BVF squared

    Parameters
    ----------
    Parameters
    ----------
    handle : WindX | xarray
        Data Handle for either WTK data
        or WRF data
    raster_index : ndarray
        Raster index array
    source_type : str
        Either wtk or wrf

    Returns
    -------
    ndarray
        BVF squared array

    """

    if source_type == 'wtk':
        required_inputs = ['temperature_200m',
                           'temperature_100m',
                           'pressure_200m',
                           'pressure_100m']

    T_top = extract_feature(
        handle, raster_index, required_inputs[0], source_type)
    T_bottom = extract_feature(
        handle, raster_index, required_inputs[1], source_type)
    P_top = extract_feature(
        handle, raster_index, required_inputs[2], source_type)
    P_bottom = extract_feature(
        handle, raster_index, required_inputs[3], source_type)

    return BVF_squared(T_top, T_bottom,
                       P_top, P_bottom)


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


def transform_rotate_wind(y, lat_lon, features,
                          renamed_features):
    """Transform windspeed/direction to
    u and v and align u and v with grid

    Parameters
    ----------
    y : np.ndarray
        4D array of high res data
    lat_lon : np.ndarray
        3D array of lat lon
    features : list
        list of extracted features
    renamed_features : list
        list of feature names after
        transformations

    Returns
    -------
    y : np.ndarray
        4D array of high res data with
        (windspeed, direction) -> (u, v)
    """

    for i, f in enumerate(renamed_features):
        if f.split('_')[0] == 'windspeed':
            if len(f.split('_')) > 1:
                height = f.split('_')[1]
                j = renamed_features.index(f'winddirection_{height}')
                renamed_features[i] = f'U_{height}'
                renamed_features[j] = f'V_{height}'
            else:
                j = renamed_features.index('winddirection')
                renamed_features[i] = 'U'
                renamed_features[j] = 'V'

            logger.debug(
                f'Transforming {features[i]}, {features[j]}'
                f' to {renamed_features[i]}, {renamed_features[j]}.')
            y = transform_wind(y, i, j)

        if renamed_features[i].split('_')[0] == 'U':
            if len(renamed_features[i].split('_')) > 1:
                height = renamed_features[i].split('_')[1]
                j = renamed_features.index(f'V_{height}')
            else:
                j = renamed_features.index('V')

            logger.debug(
                f'Aligning {renamed_features[i]},'
                f' {renamed_features[j]} with grid.')
            y = rotate_u_v(y, i, j, lat_lon)

    return y


def transform_wind(y, i, j):
    """Maps windspeed and direction to u v

    Parameters
    ----------
    y : np.ndarray
        4D array (spatial_1, spatial_2, temporal, features)
    i : int
        index of windspeed feature on the feature axis
    j : int
        index of winddirection feature on the feature axis

    Returns
    -------
    u : np.ndarray
        3D array of zonal wind components

    v : np.ndarray
        3D array of meridional wind components
    """

    # convert from windspeed and direction to u v
    windspeed = y[:, :, :, i]
    direction = y[:, :, :, j]

    y[:, :, :, i] = windspeed * np.cos(np.radians(direction - 180.0))
    y[:, :, :, j] = windspeed * np.sin(np.radians(direction - 180.0))

    return y


def rotate_u_v(y, i, j, lat_lon):
    """aligns u v with grid

    Parameters
    ----------
    y : np.ndarray
        4D array (spatial_1, spatial_2, temporal, features)
    i : int
        index of u feature along feature axis
    j : int
        index of v feature along feature axis
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

    u = y[:, :, :, i]
    v = y[:, :, :, j]
    lats = lat_lon[:, :, 0]
    lons = lat_lon[:, :, 1]

    # get the dy/dx to the nearest vertical neighbor
    dy = lats - np.roll(lats, 1, axis=0)
    dx = lons - np.roll(lons, 1, axis=0)

    # calculate the angle from the vertical
    theta = (np.pi / 2) - np.arctan2(dy, dx)
    theta[0] = theta[1]  # fix the roll row

    sin2 = np.sin(theta)
    cos2 = np.cos(theta)

    y[:, :, :, i] = np.einsum('ij,ijk->ijk', sin2, v) \
        + np.einsum('ij,ijk->ijk', cos2, u)
    y[:, :, :, j] = np.einsum('ij,ijk->ijk', cos2, v) \
        - np.einsum('ij,ijk->ijk', sin2, u)

    return y


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
        coarse_data = data.copy()

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
        coarse_data = data.copy()

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
    Unstagger WRF variable values

    Parameters
    ----------
    data :
        netcdf data object
    var : str
        Name of variable to be unstaggered

    Returns
    -------
    ndarray
        Unstaggered array of variable values.
    """

    # Import Variable values from nc database instance
    array_in = np.array(data[var])

    # Determine axis to unstagger
    if "U" in var:
        axis = -1
    elif "V" in var:
        axis = -2
    else:  # PH, PHB, and W are staggered in the "z" direction
        axis = -3

    return np.apply_along_axis(forward_average, axis, array_in)


def calc_height(data):
    """
    Calculate height from the ground
    Parameters
    ----------
    data :
        netcdf data object

    Returns
    ---------
    height_arr : ndarray
        4D array of heights above ground
    """
    # Base-state Geopotential(m^2/s^2)
    phb = unstagger_var(data, 'PHB')
    # Perturbation Geopotential (m^2/s^2)
    ph = unstagger_var('PH')
    # Terrain Height (m)
    hgt = data['HGT']

    return (ph + phb) / 9.81 - hgt


def interp_var(data, var, heights):
    """
    Interpolate var_array to given level(s) based on h_array.
    Interpolation is linear and done for every 'z' column of [var, h] data.

    Parameters
    ----------
    data :
        netcdf data object
    var : str
        Name of variable to be unstaggered
    heights : float | list
        level or levels to interpolate to (e.g. final desired hub heights)
    Returns
    -------
    out_array : ndarray
        Array of interpolated values.
    """
    h_array = calc_height(data)
    var_array = unstagger_var(data, var)
    return WRFHeights.interp3D(var_array, h_array, heights)


def potential_temperature(T, P):
    """Potential temperature of fluid
    at pressure P and temperature T

    Parameters
    ----------
    T : ndarray
        Temperature in celsius
    P : ndarray
        Pressure of fluid

    Returns
    -------
    ndarray
        Potential temperature
    """
    return (T - 273.15) * (P / 1000) ** (-0.286)


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
    PT_top = potential_temperature(T_top, P_top)
    PT_bottom = potential_temperature(T_bottom, P_bottom)
    return PT_top - PT_bottom


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

    PT_top = potential_temperature(T_top, P_top)
    PT_bottom = potential_temperature(T_bottom, P_bottom)
    PT_mid = (PT_top + PT_bottom) / 2.0
    return 9.81 / PT_mid * (PT_top - PT_bottom) / delta_h


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

    U_diff = (U_top - U_bottom)
    V_diff = (V_top - V_bottom)
    numer = BVF_squared(T_top, T_bottom,
                        P_top, P_bottom, delta_h)
    denom = (U_diff ** 2 + V_diff ** 2) / delta_h ** 2
    denom[denom < 1e-6] = 1e-6
    return numer / denom
