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

COORD_NAMES = {
    'lat': Dimension.LATITUDE,
    'lon': Dimension.LONGITUDE,
    'xlat': Dimension.LATITUDE,
    'xlong': Dimension.LONGITUDE,
    'plev': Dimension.PRESSURE_LEVEL,
    'isobaricInhPa': Dimension.PRESSURE_LEVEL,
    'pressure_level': Dimension.PRESSURE_LEVEL,
    'xtime': Dimension.TIME,
    'valid_time': Dimension.TIME
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
    'pressure_level': Dimension.PRESSURE_LEVEL,
    'xtime': Dimension.TIME,
    'valid_time': Dimension.TIME
}


# ERA5 variable names

# variables available on a single level (e.g. surface)
SFC_VARS = [
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    '100m_u_component_of_wind',
    '100m_v_component_of_wind',
    'surface_pressure',
    '2m_temperature',
    'geopotential',
    'total_precipitation',
    'convective_available_potential_energy',
    '2m_dewpoint_temperature',
    'convective_inhibition',
    'surface_latent_heat_flux',
    'instantaneous_moisture_flux',
    'mean_total_precipitation_rate',
    'mean_sea_level_pressure',
    'friction_velocity',
    'lake_cover',
    'high_vegetation_cover',
    'land_sea_mask',
    'k_index',
    'forecast_surface_roughness',
    'northward_turbulent_surface_stress',
    'eastward_turbulent_surface_stress',
    'sea_surface_temperature',
]

# variables available on multiple pressure levels
LEVEL_VARS = [
    'u_component_of_wind',
    'v_component_of_wind',
    'geopotential',
    'temperature',
    'relative_humidity',
    'specific_humidity',
    'divergence',
    'vertical_velocity',
    'pressure',
    'potential_vorticity',
]

ERA_NAME_MAP = {
    'u10': 'u_10m',
    'v10': 'v_10m',
    'u100': 'u_100m',
    'v100': 'v_100m',
    't': 'temperature',
    't2m': 'temperature_2m',
    'sp': 'pressure_0m',
    'r': 'relativehumidity',
    'relative_humidity': 'relativehumidity',
    'q': 'specifichumidity',
    'd': 'divergence',
}
