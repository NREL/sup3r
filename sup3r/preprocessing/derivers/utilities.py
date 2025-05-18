"""Miscellaneous utilities shared across the derivers module"""

import logging
import re

import dask.array as da
import numpy as np
import pandas as pd
from rex.utilities.solar_position import SolarPosition

from sup3r.utilities.utilities import get_feature_basename

logger = logging.getLogger(__name__)


class SolarZenith:
    """
    Class to compute solar zenith angle. Use SPA from rex and wrap some of
    those methods in ``dask.array.map_blocks`` so this can be computed in
    parallel across chunks.
    """

    @staticmethod
    def _get_zenith(n, zulu, lat_lon):
        """
        Compute solar zenith angle from days, hours, and location

        Parameters
        ----------
        n : da.core.Array
            Days since Greenwich Noon
        zulu : da.core.Array
            Decimal hour in UTC (Zulu Hour)
        lat_lon : da.core.Array
            (latitude, longitude, 2) for site(s) of interest

        Returns
        -------
        zenith : ndarray
            Solar zenith angle in degrees
        """
        lat, lon = lat_lon[..., 0], lat_lon[..., 1]
        lat = lat.flatten()[..., None]
        lon = lon.flatten()[..., None]
        ra, dec = SolarPosition._calc_sun_pos(n)
        zen = da.map_blocks(SolarPosition._calc_hour_angle, n, zulu, ra, lon)
        zen = da.map_blocks(SolarPosition._calc_elevation, dec, zen, lat)
        zen = da.map_blocks(SolarPosition._atm_correction, zen)
        zen = np.degrees(np.pi / 2 - zen)
        return zen

    @staticmethod
    def get_zenith(time_index, lat_lon, ll_chunks=(10, 10, 1)):
        """
        Compute solar zenith angle from time_index and location

        Parameters
        ----------
        time_index : ndarray | pandas.DatetimeIndex | str
            Datetime stamps of interest
        lat_lon : ndarray, da.core.Array
            (latitude, longitude, 2) for site(s) of interest
        ll_chunks : tuple
            Chunks for lat_lon array. To run this on a large domain, even with
            delayed computations through dask, we need to use small chunks for
            the lat lon array.

        Returns
        -------
        zenith : da.core.Array
            Solar zenith angle in degrees
        """
        if not isinstance(time_index, pd.DatetimeIndex):
            if isinstance(time_index, str):
                time_index = [time_index]

            time_index = pd.to_datetime(time_index)

        out_shape = (*lat_lon.shape[:-1], len(time_index))
        lat_lon = da.asarray(lat_lon, chunks=ll_chunks)
        n, zulu = SolarPosition._parse_time(time_index)
        n = da.asarray(n).astype(np.float32)
        zulu = da.asarray(zulu).astype(np.float32)
        zen = SolarZenith._get_zenith(n, zulu, lat_lon)
        return zen.reshape(out_shape)


def parse_feature(feature):
    """Parse feature name to get the "basename" (i.e. U for u_100m), the height
    (100 for u_100m), and pressure if available (1000 for u_1000pa)."""

    class FeatureStruct:
        """Feature structure storing `basename`, `height`, and `pressure`."""

        def __init__(self):
            height = re.findall(r'_\d+m', feature)
            press = re.findall(r'_\d+pa', feature)
            self.basename = get_feature_basename(feature)
            self.height = (
                int(round(float(height[0][1:-1]))) if height else None
            )
            self.pressure = (
                int(round(float(press[0][1:-2]))) if press else None
            )

        def map_wildcard(self, pattern):
            """Return given pattern with wildcard replaced with height if
            available, pressure if available, or just return the basename."""
            if '(.*)' not in pattern:
                return pattern
            return (
                f"{pattern.split('_(.*)')[0]}_{self.height}m"
                if self.height is not None
                else f"{pattern.split('_(.*)')[0]}_{self.pressure}pa"
                if self.pressure is not None
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
    ws : Union[np.ndarray, da.core.Array]
        3D array of high res windspeed data
        (spatial_1, spatial_2, temporal)
    wd : Union[np.ndarray, da.core.Array]
        3D array of high res winddirection data. Angle is in degrees and
        measured clockwise from the north direction. This is direction wind is
        coming from.
        (spatial_1, spatial_2, temporal)
    lat_lon : Union[np.ndarray, da.core.Array]
        3D array of lat lon
        (spatial_1, spatial_2, 2)
        Last dimension has lat / lon in that order

    Returns
    -------
    u : Union[np.ndarray, da.core.Array]
        3D array of high res U data
        (spatial_1, spatial_2, temporal)
    v : Union[np.ndarray, da.core.Array]
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
    u : Union[np.ndarray, da.core.Array]
        3D array of high res U data
        (spatial_1, spatial_2, temporal)
    v : Union[np.ndarray, da.core.Array]
        3D array of high res V data
        (spatial_1, spatial_2, temporal)
    lat_lon : Union[np.ndarray, da.core.Array]
        3D array of lat lon
        (spatial_1, spatial_2, 2)
        Last dimension has lat / lon in that order

    Returns
    -------
    ws : Union[np.ndarray, da.core.Array]
        3D array of high res windspeed data
        (spatial_1, spatial_2, temporal)
    wd : Union[np.ndarray, da.core.Array]
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
