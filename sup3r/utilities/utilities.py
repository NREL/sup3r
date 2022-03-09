# -*- coding: utf-8 -*-
"""Utilities module for preparing
trraining data"""

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
    if data.shape[0] == shape[0]:
        start_row = 0
    else:
        start_row = np.random.randint(0, data.shape[0] - shape[0])
    stop_row = start_row + shape[0]

    if data.shape[1] == shape[1]:
        start_col = 0
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

    if data.shape[2] == shape:
        start = 0
    else:
        start = np.random.randint(0, data.shape[2] - shape)
    stop = start + shape
    return slice(start, stop)


def transform_rotate_wind(y, lat_lon, features):
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

    Returns
    -------
    y : np.ndarray
        4D array of high res data with
        (windspeed, direction) -> (u, v)
    """

    renamed_features = features.copy()
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

            y = transform_wind(y, i, j)

        if renamed_features[i].split('_')[0] == 'U':
            if len(renamed_features[i].split('_')) > 1:
                height = renamed_features[i].split('_')[1]
                j = renamed_features.index(f'V_{height}')
            else:
                j = renamed_features.index('V')

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

    logger.debug('Transforming speed and direction to U and V')

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

    logger.debug('Aligning U and V with grid coordinate system')

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
