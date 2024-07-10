"""Miscellaneous utilities shared across the derivers module"""

import logging
import re

import numpy as np

np.random.seed(42)

logger = logging.getLogger(__name__)


def parse_feature(feature):
    """Parse feature name to get the "basename" (i.e. U for u_100m), the height
    (100 for u_100m), and pressure if available (1000 for u_1000pa)."""

    class FeatureStruct:
        """Feature structure storing `basename`, `height`, and `pressure`."""

        def __init__(self):
            height = re.findall(r'_\d+m', feature)
            pressure = re.findall(r'_\d+pa', feature)
            self.basename = (
                feature.replace(height[0], '')
                if height
                else feature.replace(pressure[0], '')
                if pressure
                else feature.split('_(.*)')[0]
                if '_(.*)' in feature
                else feature
            )
            self.height = int(height[0][1:-1]) if height else None
            self.pressure = int(pressure[0][1:-2]) if pressure else None

        def map_wildcard(self, pattern):
            """Return given pattern with wildcard replaced with height if
            available, pressure if available, or just return the basename."""
            if '(.*)' not in pattern:
                return pattern
            return (
                f"{pattern.split('_(.*)')[0]}_{self.height}m"
                if self.height
                else f"{pattern.split('_(.*)')[0]}_{self.pressure}pa"
                if self.pressure
                else f"{pattern.split('_(.*)')[0]}"
            )

    return FeatureStruct()


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


def transform_rotate_wind(ws, wd, lat_lon):
    """Transform windspeed/direction to u and v and align u and v with grid

    Parameters
    ----------
    ws : T_Array
        3D array of high res windspeed data
        (spatial_1, spatial_2, temporal)
    wd : T_Array
        3D array of high res winddirection data. Angle is in degrees and
        measured relative to the south_north direction.
        (spatial_1, spatial_2, temporal)
    lat_lon : T_Array
        3D array of lat lon
        (spatial_1, spatial_2, 2)
        Last dimension has lat / lon in that order

    Returns
    -------
    u : T_Array
        3D array of high res U data
        (spatial_1, spatial_2, temporal)
    v : T_Array
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
    u : T_Array
        3D array of high res U data
        (spatial_1, spatial_2, temporal)
    v : T_Array
        3D array of high res V data
        (spatial_1, spatial_2, temporal)
    lat_lon : T_Array
        3D array of lat lon
        (spatial_1, spatial_2, 2)
        Last dimension has lat / lon in that order

    Returns
    -------
    ws : T_Array
        3D array of high res windspeed data
        (spatial_1, spatial_2, temporal)
    wd : T_Array
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
