# -*- coding: utf-8 -*-
"""Utilities module for preparing
trraining data"""

import numpy as np


def transform_wind(y, i, j):
    """Maps windspeed and direction to u v

    Parameters
    ----------
    y : np.ndarray
        4D array (spatial_1, spatial_2, temporal, features)
    i : int
        index of windspeed feature
    j : int
        index of winddirection feature

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
        index of u feature
    j : int
        index of v feature
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


def get_coarse_data(data, lat_lon,
                    spatial_res=2,
                    temporal_res=None):
    """"Coarsen data according to spatial_res resolution
    and temporal_res temporal sample frequency

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

    temporal_res : (int, int)
        factor by which to coarsen temporal dimension

    Returns
    -------
    coarse_data : np.ndarray
        4D array with same dimensions as data
        with new coarse resolution

    coarse_lat_lon : np.ndarray
        3D array (spatial_1, spatial_2, 2) with
        lat and lon as the 2 channels in that order
        with same resolution as coarse_data
    """

    if temporal_res is not None:
        tmp = data[:, :, ::temporal_res, :]
    else:
        tmp = data

    if spatial_res is not None:
        if data.shape[1] % spatial_res != 0:
            msg = 'spatial_res must evenly divide grid size. '
            msg += f'Received spatial_res: {spatial_res} '
            msg += f'with grid size: ({data.shape[0]}, '
            msg += f'{data.shape[1]})'
            raise ValueError(msg)

        coarse_data = tmp.reshape(-1, spatial_res,
                                  data.shape[1] // spatial_res,
                                  spatial_res,
                                  tmp.shape[2],
                                  tmp.shape[3]).sum((1, 3)) \
            / (spatial_res * spatial_res)

        coarse_lat_lon = lat_lon.reshape(-1, spatial_res,
                                         data.shape[1] // spatial_res,
                                         spatial_res, 2).sum((3, 1)) \
            / (spatial_res * spatial_res)
    else:
        coarse_data = data.copy()
        coarse_lat_lon = lat_lon.copy()

    return coarse_data, coarse_lat_lon
