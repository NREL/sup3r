"""Mappings from coord / dim / feature names to standard names and Dimension
class for standardizing dimension orders and names."""

from enum import Enum


class Dimension(str, Enum):
    """Dimension names used for Sup3rX accessor."""

    FLATTENED_SPATIAL = 'space'
    SOUTH_NORTH = 'south_north'
    WEST_EAST = 'west_east'
    TIME = 'time'
    PRESSURE_LEVEL = 'level'
    VARIABLE = 'variable'
    LATITUDE = 'latitude'
    LONGITUDE = 'longitude'
    QUANTILE = 'quantile'
    GLOBAL_TIME = 'global_time'

    def __str__(self):
        return self.value

    @classmethod
    def order(cls):
        """Return standard dimension order."""
        return (
            cls.FLATTENED_SPATIAL,
            cls.SOUTH_NORTH,
            cls.WEST_EAST,
            cls.TIME,
            cls.PRESSURE_LEVEL,
            cls.VARIABLE,
        )

    @classmethod
    def flat_2d(cls):
        """Return ordered tuple for 2d flattened data."""
        return (cls.FLATTENED_SPATIAL, cls.TIME)

    @classmethod
    def dims_2d(cls):
        """Return ordered tuple for 2d spatial coordinates."""
        return (cls.SOUTH_NORTH, cls.WEST_EAST)

    @classmethod
    def dims_3d(cls):
        """Return ordered tuple for 3d spatiotemporal coordinates."""
        return (cls.SOUTH_NORTH, cls.WEST_EAST, cls.TIME)

    @classmethod
    def dims_4d(cls):
        """Return ordered tuple for 4d spatiotemporal coordinates."""
        return (cls.SOUTH_NORTH, cls.WEST_EAST, cls.TIME, cls.PRESSURE_LEVEL)

    @classmethod
    def dims_3d_bc(cls):
        """Return ordered tuple for 3d spatiotemporal coordinates."""
        return (cls.SOUTH_NORTH, cls.WEST_EAST, cls.TIME)

    @classmethod
    def dims_4d_bc(cls):
        """Return ordered tuple for 4d spatiotemporal coordinates specifically
        for bias correction factor files."""
        return (cls.SOUTH_NORTH, cls.WEST_EAST, cls.TIME, cls.QUANTILE)


FEATURE_NAMES = {
    'elevation': 'topography',
    'orog': 'topography',
    'hgt': 'topography',
}

DIM_NAMES = {
    'lat': Dimension.SOUTH_NORTH,
    'lon': Dimension.WEST_EAST,
    'xlat': Dimension.SOUTH_NORTH,
    'xlong': Dimension.WEST_EAST,
    'latitude': Dimension.SOUTH_NORTH,
    'longitude': Dimension.WEST_EAST,
    'plev': Dimension.PRESSURE_LEVEL,
    'isobaricInhPa': Dimension.PRESSURE_LEVEL,
    'xtime': Dimension.TIME,
}

COORD_NAMES = {
    'lat': Dimension.LATITUDE,
    'lon': Dimension.LONGITUDE,
    'xlat': Dimension.LATITUDE,
    'xlong': Dimension.LONGITUDE,
    'plev': Dimension.PRESSURE_LEVEL,
    'isobaricInhPa': Dimension.PRESSURE_LEVEL,
}
