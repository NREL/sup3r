"""Sup3r derived features.

@author: bbenton
"""

import logging
from abc import ABC, abstractmethod

import numpy as np

from sup3r.containers.extracters import Extracter
from sup3r.utilities.utilities import (
    invert_uv,
    transform_rotate_wind,
)

np.random.seed(42)

logger = logging.getLogger(__name__)


class DerivedFeature(ABC):
    """Abstract class for special features which need to be derived from raw
    features
    """

    @classmethod
    @abstractmethod
    def compute(cls, container: Extracter, **kwargs):
        """Compute method for derived feature. This can use any of the features
        contained in the :class:`Extracter` data and the attributes (e.g.
        `.lat_lon`, `.time_index`). To access the data contained in the
        extracter just use the feature name. e.g. container['windspeed_100m'].
        This will also work for attributes e.g. container['lat_lon'].

        Parameters
        ----------
        container : Extracter
            Extracter type container. This has been initialized on a
            :class:`Loader` object and extracted a specific spatiotemporal
            extent for the features contained in the loader. These features are
            exposed through a `__getitem__` method such that container[feature]
            will return the feature data for the specified extent.
        **kwargs : dict
            Optional keyword arguments used in derivation. height is a typical
            example. Could also be pressure.
        """


class ClearSkyRatioH5(DerivedFeature):
    """Clear Sky Ratio feature class for computing from H5 data"""

    @classmethod
    def compute(cls, container):
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
        night_mask = container['clearsky_ghi'] <= 1

        # set any timestep with any nighttime equal to NaN to avoid weird
        # sunrise/sunset artifacts.
        night_mask = night_mask.any(axis=(0, 1))
        container['clearsky_ghi'][..., night_mask] = np.nan

        cs_ratio = container['ghi'] / container['clearsky_ghi']
        return cs_ratio.astype(np.float32)


class ClearSkyRatioCC(DerivedFeature):
    """Clear Sky Ratio feature class for computing from climate change netcdf
    data
    """

    @classmethod
    def compute(cls, container):
        """Compute the daily average climate change clearsky ratio

        Parameters
        ----------
        container : Extracter
            data container used for this compuation, must include clearsky_ghi
            and rsds (rsds==ghi for cc datasets)

        Returns
        -------
        cs_ratio : ndarray
            Clearsky ratio, e.g. the all-sky ghi / the clearsky ghi. This is
            assumed to be daily average data for climate change source data.
        """
        cs_ratio = container['rsds'] / container['clearsky_ghi']
        cs_ratio = np.minimum(cs_ratio, 1)
        return np.maximum(cs_ratio, 0)


class CloudMaskH5(DerivedFeature):
    """Cloud Mask feature class for computing from H5 data"""

    @classmethod
    def compute(cls, container):
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
        night_mask = container['clearsky_ghi'] <= 1

        # set any timestep with any nighttime equal to NaN to avoid weird
        # sunrise/sunset artifacts.
        night_mask = night_mask.any(axis=(0, 1))

        cloud_mask = container['ghi'] < container['clearsky_ghi']
        cloud_mask = cloud_mask.astype(np.float32)
        cloud_mask[night_mask] = np.nan
        return cloud_mask.astype(np.float32)


class PressureNC(DerivedFeature):
    """Pressure feature class for NETCDF data. Needed since P is perturbation
    pressure.
    """

    @classmethod
    def compute(cls, container, height):
        """Method to compute pressure from NETCDF data"""
        return container[f'P_{height}m'] + container[f'PB_{height}m']


class WindspeedNC(DerivedFeature):
    """Windspeed feature from netcdf data"""

    @classmethod
    def compute(cls, container, height):
        """Compute windspeed"""

        ws, _ = invert_uv(
            container[f'U_{height}m'],
            container[f'V_{height}m'],
            container['lat_lon'],
        )
        return ws


class WinddirectionNC(DerivedFeature):
    """Winddirection feature from netcdf data"""

    @classmethod
    def compute(cls, container, height):
        """Compute winddirection"""
        _, wd = invert_uv(
            container[f'U_{height}m'],
            container[f'V_{height}m'],
            container['lat_lon'],
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

    @classmethod
    def compute(cls, container, height):
        """Method to compute U wind component from data

        Parameters
        ----------
        container : Extracter
            Dictionary of raw feature arrays to use for derivation
        height : str | int
            Height at which to compute the derived feature

        Returns
        -------
        ndarray
            Derived feature array

        """
        return (
            container['uas']
            * (float(height) / cls.NEAR_SFC_HEIGHT) ** cls.ALPHA
        )


class VWindPowerLaw(DerivedFeature):
    """V wind component feature class with needed inputs method and compute
    method. Uses power law extrapolation to get values above surface

    https://csl.noaa.gov/projects/lamar/windshearformula.html
    https://www.tandfonline.com/doi/epdf/10.1080/00022470.1977.10470503
    """

    ALPHA = 0.2
    NEAR_SFC_HEIGHT = 10

    @classmethod
    def compute(cls, container, height):
        """Method to compute V wind component from data"""

        return (
            container['vas']
            * (float(height) / cls.NEAR_SFC_HEIGHT) ** cls.ALPHA
        )


class UWind(DerivedFeature):
    """U wind component feature class with needed inputs method and compute
    method
    """

    @classmethod
    def compute(cls, container, height):
        """Method to compute U wind component from data"""
        u, _ = transform_rotate_wind(
            container[f'windspeed_{height}m'],
            container[f'winddirection_{height}m'],
            container['lat_lon'],
        )
        return u


class VWind(DerivedFeature):
    """V wind component feature class with needed inputs method and compute
    method
    """

    @classmethod
    def compute(cls, container, height):
        """Method to compute V wind component from data"""

        _, v = transform_rotate_wind(
            container[f'windspeed_{height}m'],
            container[f'winddirection_{height}m'],
            container['lat_lon'],
        )
        return v


class TempNCforCC(DerivedFeature):
    """Air temperature variable from climate change nc files"""

    @classmethod
    def compute(cls, container, height):
        """Method to compute ta in Celsius from ta source in Kelvin"""

        return container[f'ta_{height}m'] - 273.15


class Tas(DerivedFeature):
    """Air temperature near surface variable from climate change nc files"""

    CC_FEATURE_NAME = 'tas'
    """Source CC.nc dataset name for air temperature variable. This can be
    changed in subclasses for other temperature datasets."""

    @classmethod
    def compute(cls, container):
        """Method to compute tas in Celsius from tas source in Kelvin"""
        return container[cls.CC_FEATURE_NAME] - 273.15


class TasMin(Tas):
    """Daily min air temperature near surface variable from climate change nc
    files
    """

    CC_FEATURE_NAME = 'tasmin'


class TasMax(Tas):
    """Daily max air temperature near surface variable from climate change nc
    files
    """

    CC_FEATURE_NAME = 'tasmax'


RegistryBase = {
    'U_(.*)': UWind,
    'V_(.*)': VWind,
}

RegistryNC = {
    **RegistryBase,
    'Windspeed_(.*)': WindspeedNC,
    'Winddirection_(.*)': WinddirectionNC,
}

RegistryH5 = {
    **RegistryBase,
    'cloud_mask': CloudMaskH5,
    'clearsky_ratio': ClearSkyRatioH5,
}

RegistryH5WindCC = {
    **RegistryH5,
    'temperature_max_(.*)m': 'temperature_(.*)m',
    'temperature_min_(.*)m': 'temperature_(.*)m',
    'relativehumidity_max_(.*)m': 'relativehumidity_(.*)m',
    'relativehumidity_min_(.*)m': 'relativehumidity_(.*)m',
}

RegistryH5SolarCC = {
    **RegistryH5WindCC,
    'windspeed': 'wind_speed',
    'winddirection': 'wind_direction',
    'U': UWind,
    'V': VWind,
}

RegistryNCforCC = {
    **RegistryNC,
    'U_(.*)': 'ua_(.*)',
    'V_(.*)': 'va_(.*)',
    'relativehumidity_2m': 'hurs',
    'relativehumidity_min_2m': 'hursmin',
    'relativehumidity_max_2m': 'hursmax',
    'clearsky_ratio': ClearSkyRatioCC,
    'Pressure_(.*)': 'level_(.*)',
    'Temperature_(.*)': TempNCforCC,
    'temperature_2m': Tas,
    'temperature_max_2m': TasMax,
    'temperature_min_2m': TasMin,
}


RegistryNCforCCwithPowerLaw = {
    **RegistryNCforCC,
    'U_(.*)': UWindPowerLaw,
    'V_(.*)': VWindPowerLaw,
}
