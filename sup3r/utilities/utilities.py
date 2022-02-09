# -*- coding: utf-8 -*-
"""Utilities module for preparing
trraining data"""

import numpy as np
import logging


logger = logging.getLogger(__name__)


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


def temporal_coarsening(data, temporal_res=2):
    """"Coarsen data according to temporal_res resolution

    Parameters
    ----------
    data : np.ndarray
        4D array with dimensions
        (spatial_1, spatial_2, temporal, features)

    temporal_res : int
        factor by which to coarsen temporal dimension

    Returns
    -------
    coarse_data : np.ndarray
        4D array with same dimensions as data
        with new coarse resolution
    """

    if temporal_res is not None:
        coarse_data = data[:, :, ::temporal_res, :]
    else:
        coarse_data = data.copy()

    return coarse_data


def spatial_coarsening(data, spatial_res=2):
    """"Coarsen data according to spatial_res resolution

    Parameters
    ----------
    data : np.ndarray
        4D array with dimensions
        (spatial_1, spatial_2, temporal, features)

    lat_lon : np.ndarray
        2D array with dimensions
        (spatial_1, spatial_2)

    spatial_res : int
        factor by which to coarsen spatial dimensions

    Returns
    -------
    coarse_data : np.ndarray
        4D array with same dimensions as data
        with new coarse resolution
    """

    if spatial_res is not None:
        if data.shape[1] % spatial_res != 0:
            msg = 'spatial_res must evenly divide grid size. '
            msg += f'Received spatial_res: {spatial_res} '
            msg += f'with grid size: ({data.shape[0]}, '
            msg += f'{data.shape[1]})'
            raise ValueError(msg)

        coarse_data = data.reshape(-1, spatial_res,
                                   data.shape[1] // spatial_res,
                                   spatial_res,
                                   data.shape[2],
                                   data.shape[3]).sum((1, 3)) \
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
