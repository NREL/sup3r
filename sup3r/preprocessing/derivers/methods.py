"""Derivation methods for deriving features from raw data."""

import copy
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np

from sup3r.preprocessing.accessor import Sup3rX
from sup3r.preprocessing.base import Sup3rDataset
from sup3r.preprocessing.names import Dimension

from .utilities import SolarZenith, invert_uv, transform_rotate_wind

logger = logging.getLogger(__name__)


class DerivedFeature(ABC):
    """Abstract class for special features which need to be derived from raw
    features

    Note
    ----
    `inputs` list will be used to search already derived / loaded data so this
    should include all features required for a successful `.compute` call.
    """

    inputs: Tuple[str, ...] = ()

    @classmethod
    @abstractmethod
    def compute(cls, data: Union[Sup3rX, Sup3rDataset], **kwargs):
        """Compute method for derived feature. This can use any of the features
        contained in the xr.Dataset data and the attributes (e.g.
        `.lat_lon`, `.time_index` accessed through Sup3rX accessor).

        Parameters
        ----------
        data : Union[Sup3rX, Sup3rDataset]
            Initialized and standardized through a :class:`Loader` with a
            specific spatiotemporal extent rasterized for the features
            contained using a :class:`Rasterizer`.
        kwargs : dict
            Optional keyword arguments used in derivation. height is a typical
            example. Could also be pressure.
        """


class SurfaceRH(DerivedFeature):
    """Surface Relative humidity feature for computing rh from dewpoint
    temperature and ambient temperature. This is in a 0 - 100 scale to match
    the ERA5 pressure level relative humidity scale.

    https://earthscience.stackexchange.com/questions/24156/era5-single-level-calculate-relative-humidity

    https://journals.ametsoc.org/view/journals/bams/86/2/bams-86-2-225.xml?tab_body=pdf
    """

    inputs = ('d2m', 'temperature_2m')

    @classmethod
    def compute(cls, data):
        """Compute surface relative humidity."""
        water_vapor_pressure = 6.1078 * np.exp(
            17.1 * data['d2m'] / (235 + data['d2m'])
        )
        saturation_water_vapor_pressure = 6.1078 * np.exp(
            17.1 * data['temperature_2m'] / (235 + data['temperature_2m'])
        )
        return 100 * water_vapor_pressure / saturation_water_vapor_pressure


class ClearSkyRatio(DerivedFeature):
    """Clear Sky Ratio feature class. Inputs here are typically found in H5
    data like the NSRDB"""

    inputs = ('ghi', 'clearsky_ghi')

    @classmethod
    def compute(cls, data):
        """Compute the clearsky ratio

        Returns
        -------
        cs_ratio : ndarray
            Clearsky ratio, e.g. the all-sky ghi / the clearsky ghi. NaN where
            nighttime.
        """
        # need to use a nightime threshold of 1 W/m2 because cs_ghi is stored
        # in integer format and weird binning patterns happen in the clearsky
        # ratio and cloud mask between 0 and 1 W/m2 and sunrise/sunset
        night_mask = data['clearsky_ghi'] <= 1

        # set any timestep with any nighttime equal to NaN to avoid weird
        # sunrise/sunset artifacts.
        night_mask = np.asarray(night_mask.any(axis=(0, 1)))

        cs_ratio = data['ghi'] / data['clearsky_ghi']
        cs_ratio[..., night_mask] = np.nan
        return cs_ratio.astype(np.float32)


class ClearSkyRatioCC(DerivedFeature):
    """Clear Sky Ratio feature class for computing from climate change netcdf
    data
    """

    inputs = ('rsds', 'clearsky_ghi')

    @classmethod
    def compute(cls, data):
        """Compute the daily average climate change clearsky ratio

        Parameters
        ----------
        data : Union[Sup3rX, Sup3rDataset]
            xarray dataset used for this compuation, must include clearsky_ghi
            and rsds (rsds==ghi for cc datasets)

        Returns
        -------
        cs_ratio : ndarray
            Clearsky ratio, e.g. the all-sky ghi / the clearsky ghi. This is
            assumed to be daily average data for climate change source data.
        """
        cs_ratio = data['rsds'] / data['clearsky_ghi']
        cs_ratio = np.minimum(cs_ratio, 1)
        return np.maximum(cs_ratio, 0)


class CloudMask(DerivedFeature):
    """Cloud Mask feature class. Inputs here are typically found in H5 data
    like the NSRDB."""

    inputs = ('ghi', 'clearky_ghi')

    @classmethod
    def compute(cls, data):
        """
        Returns
        -------
        cloud_mask : ndarray
            Cloud mask, e.g. 1 where cloudy, 0 where clear. NaN where
            nighttime. Data is float32 so it can be normalized without any
            integer weirdness.
        """
        # need to use a nightime threshold of 1 W/m2 because cs_ghi is stored
        # in integer format and weird binning patterns happen in the clearsky
        # ratio and cloud mask between 0 and 1 W/m2 and sunrise/sunset
        night_mask = data['clearsky_ghi'] <= 1

        # set any timestep with any nighttime equal to NaN to avoid weird
        # sunrise/sunset artifacts.
        night_mask = np.asarray(night_mask.any(axis=(0, 1)))

        cloud_mask = data['ghi'] < data['clearsky_ghi']
        cloud_mask = cloud_mask.astype(np.float32)
        cloud_mask[night_mask] = np.nan
        return cloud_mask.astype(np.float32)


class PressureWRF(DerivedFeature):
    """Pressure feature class for WRF data. Needed since P is perturbation
    pressure.
    """

    inputs = ('p_(.*)', 'pb_(.*)')

    @classmethod
    def compute(cls, data, height):
        """Method to compute pressure from NETCDF data"""
        return data[f'p_{height}m'] + data[f'pb_{height}m']


class Windspeed(DerivedFeature):
    """Windspeed feature from rasterized data"""

    inputs = ('u_(.*)', 'v_(.*)')

    @classmethod
    def compute(cls, data, height):
        """Compute windspeed"""

        ws, _ = invert_uv(
            data[f'u_{height}m'],
            data[f'v_{height}m'],
            data.lat_lon,
        )
        return ws


class Winddirection(DerivedFeature):
    """Winddirection feature from rasterized data"""

    inputs = ('u_(.*)', 'v_(.*)')

    @classmethod
    def compute(cls, data, height):
        """Compute winddirection"""
        _, wd = invert_uv(
            data[f'u_{height}m'],
            data[f'v_{height}m'],
            data.lat_lon,
        )
        return wd


class UWindPowerLaw(DerivedFeature):
    """U wind component feature class with needed inputs method and compute
    method. Uses power law extrapolation to get values above surface

    https://csl.noaa.gov/projects/lamar/windshearformula.html
    https://www.tandfonline.com/doi/epdf/10.1080/00022470.1977.10470503
    """

    ALPHA = 0.2
    NEAR_SFC_HEIGHT = 10

    inputs = ('uas',)

    @classmethod
    def compute(cls, data, height):
        """Method to compute U wind component from data

        Parameters
        ----------
        data : Union[Sup3rX, Sup3rDataset]
            Initialized and standardized through a :class:`Loader` with a
            specific spatiotemporal extent rasterized for the features
            contained using a :class:`Rasterizer`.
        height : str | int
            Height at which to compute the derived feature

        Returns
        -------
        ndarray
            Derived feature array

        """
        return data['uas'] * (float(height) / cls.NEAR_SFC_HEIGHT) ** cls.ALPHA


class VWindPowerLaw(DerivedFeature):
    """V wind component feature class with needed inputs method and compute
    method. Uses power law extrapolation to get values above surface

    https://csl.noaa.gov/projects/lamar/windshearformula.html
    https://www.tandfonline.com/doi/epdf/10.1080/00022470.1977.10470503
    """

    ALPHA = 0.2
    NEAR_SFC_HEIGHT = 10

    inputs = ('vas',)

    @classmethod
    def compute(cls, data, height):
        """Method to compute V wind component from data"""

        return data['vas'] * (float(height) / cls.NEAR_SFC_HEIGHT) ** cls.ALPHA


class UWind(DerivedFeature):
    """U wind component feature class with needed inputs method and compute
    method
    """

    inputs = ('windspeed_(.*)', 'winddirection_(.*)')

    @classmethod
    def compute(cls, data, height):
        """Method to compute U wind component from data"""
        u, _ = transform_rotate_wind(
            data[f'windspeed_{height}m'],
            data[f'winddirection_{height}m'],
            data.lat_lon,
        )
        return u


class VWind(DerivedFeature):
    """V wind component feature class with needed inputs method and compute
    method
    """

    inputs = ('windspeed_(.*)', 'winddirection_(.*)')

    @classmethod
    def compute(cls, data, height):
        """Method to compute V wind component from data"""

        _, v = transform_rotate_wind(
            data[f'windspeed_{height}m'],
            data[f'winddirection_{height}m'],
            data.lat_lon,
        )
        return v


class USolar(DerivedFeature):
    """U wind component feature class with needed inputs method and compute
    method for NSRDB data (which has just a single windspeed hub height)
    """

    inputs = ('wind_speed', 'wind_direction')

    @classmethod
    def compute(cls, data):
        """Method to compute U wind component from data"""
        u, _ = transform_rotate_wind(
            data['wind_speed'],
            data['wind_direction'],
            data.lat_lon,
        )
        return u


class VSolar(DerivedFeature):
    """V wind component feature class with needed inputs method and compute
    method for NSRDB data (which has just a single windspeed hub height)
    """

    inputs = ('wind_speed', 'wind_direction')

    @classmethod
    def compute(cls, data):
        """Method to compute U wind component from data"""
        _, v = transform_rotate_wind(
            data['wind_speed'],
            data['wind_direction'],
            data.lat_lon,
        )
        return v


class TempNCforCC(DerivedFeature):
    """Air temperature variable from climate change nc files"""

    inputs = ('ta_(.*)',)

    @classmethod
    def compute(cls, data, height):
        """Method to compute ta in Celsius from ta source in Kelvin"""
        out = data[f'ta_{height}m']
        units = out.attrs.get('units', 'K')
        if units == 'K':
            out -= 273.15
            out.attrs['units'] = 'C'
        return out


class Tas(DerivedFeature):
    """Air temperature near surface variable from climate change nc files"""

    inputs = ('tas',)

    @classmethod
    def compute(cls, data):
        """Method to compute tas in Celsius from tas source in Kelvin"""
        out = data[cls.inputs[0]]
        units = out.attrs.get('units', 'K')
        if units == 'K':
            out -= 273.15
            out.attrs['units'] = 'C'
        return out


class TasMin(Tas):
    """Daily min air temperature near surface variable from climate change nc
    files
    """

    inputs = ('tasmin',)


class TasMax(Tas):
    """Daily max air temperature near surface variable from climate change nc
    files
    """

    inputs = ('tasmax',)


class Sza(DerivedFeature):
    """Solar zenith angle derived feature."""

    @classmethod
    def compute(cls, data):
        """Compute method for sza."""
        sza = SolarZenith.get_zenith(data.time_index, data.lat_lon)
        return sza.astype(np.float32)


class Latitude(DerivedFeature):
    """latitude feature with time dimension included."""

    @classmethod
    def compute(cls, data):
        """Compute method for latitude."""
        lat = data[Dimension.LATITUDE]
        lat = lat.expand_dims(Dimension.TIME, axis=-1)
        lat = np.repeat(lat, len(data.time_index), axis=-1)
        return lat.astype(np.float32)


class Longitude(DerivedFeature):
    """longitude feature with time dimension included."""

    @classmethod
    def compute(cls, data):
        """Compute method for longitude."""
        lon = data[Dimension.LONGITUDE]
        lon = lon.expand_dims(Dimension.TIME, axis=-1)
        lon = np.repeat(lon, len(data.time_index), axis=-1)
        return lon.astype(np.float32)


RegistryBase = {
    'u_(.*)': UWind,
    'v_(.*)': VWind,
    'relativehumidity_2m': SurfaceRH,
    'windspeed_(.*)': Windspeed,
    'winddirection_(.*)': Winddirection,
    'cloud_mask': CloudMask,
    'clearsky_ratio': ClearSkyRatio,
    'sza': Sza,
    'latitude_feature': Latitude,
    'longitude_feature': Longitude,
}

RegistryH5WindCC = {
    **RegistryBase,
    'temperature_max_(.*)m': 'temperature_(.*)m',
    'temperature_min_(.*)m': 'temperature_(.*)m',
    'relativehumidity_max_(.*)m': 'relativehumidity_(.*)m',
    'relativehumidity_min_(.*)m': 'relativehumidity_(.*)m',
}

RegistryH5SolarCC = {
    **RegistryH5WindCC,
    'windspeed': 'wind_speed',
    'winddirection': 'wind_direction',
    'U': USolar,
    'V': VSolar,
}

RegistryNCforCC = copy.deepcopy(RegistryBase)
RegistryNCforCC.update(
    {
        'u_(.*)': 'ua_(.*)',
        'v_(.*)': 'va_(.*)',
        'relativehumidity_2m': 'hurs',
        'relativehumidity_min_2m': 'hursmin',
        'relativehumidity_max_2m': 'hursmax',
        'clearsky_ratio': ClearSkyRatioCC,
        'temperature_(.*)': TempNCforCC,
        'temperature_2m': Tas,
        'temperature_max_2m': TasMax,
        'temperature_min_2m': TasMin,
        'pressure_(.*)': 'level_(.*)',
    }
)


RegistryNCforCCwithPowerLaw = {
    **RegistryNCforCC,
    'u_(.*)': UWindPowerLaw,
    'v_(.*)': VWindPowerLaw,
}
