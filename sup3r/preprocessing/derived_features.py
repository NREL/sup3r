"""Sup3r derived features.

@author: bbenton
"""

import logging
import re
from abc import ABC, abstractmethod

import numpy as np
import xarray as xr
from rex import Resource

from sup3r.utilities.utilities import (
    bvf_squared,
    inverse_mo_length,
    invert_pot_temp,
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
    def inputs(cls, feature):
        """Required inputs for derived feature"""

    @classmethod
    @abstractmethod
    def compute(cls, data, height):
        """Compute method for derived feature"""


class ClearSkyRatioH5(DerivedFeature):
    """Clear Sky Ratio feature class for computing from H5 data"""

    @classmethod
    def inputs(cls, feature):
        """Get list of raw features used in calculation of the clearsky ratio

        Parameters
        ----------
        feature : str
            Clearsky ratio feature name, needs to be "clearsky_ratio"

        Returns
        -------
        list
            List of required features for clearsky_ratio: clearsky_ghi, ghi
        """
        assert feature == 'clearsky_ratio'
        return ['clearsky_ghi', 'ghi']

    @classmethod
    def compute(cls, data, height=None):
        """Compute the clearsky ratio

        Parameters
        ----------
        data : dict
            dictionary of feature arrays used for this compuation, must include
            clearsky_ghi and ghi
        height : str | int
            Placeholder to match interface with other compute methods

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
        night_mask = night_mask.any(axis=(0, 1))
        data['clearsky_ghi'][..., night_mask] = np.nan

        cs_ratio = data['ghi'] / data['clearsky_ghi']
        return cs_ratio.astype(np.float32)


class ClearSkyRatioCC(DerivedFeature):
    """Clear Sky Ratio feature class for computing from climate change netcdf
    data
    """

    @classmethod
    def inputs(cls, feature):
        """Get list of raw features used in calculation of the clearsky ratio

        Parameters
        ----------
        feature : str
            Clearsky ratio feature name, needs to be "clearsky_ratio"

        Returns
        -------
        list
            List of required features for clearsky_ratio: clearsky_ghi, rsds
            (rsds==ghi for cc datasets)
        """
        assert feature == 'clearsky_ratio'
        return ['clearsky_ghi', 'rsds']

    @classmethod
    def compute(cls, data, height=None):
        """Compute the daily average climate change clearsky ratio

        Parameters
        ----------
        data : dict
            dictionary of feature arrays used for this compuation, must include
            clearsky_ghi and rsds (rsds==ghi for cc datasets)
        height : str | int
            Placeholder to match interface with other compute methods

        Returns
        -------
        cs_ratio : ndarray
            Clearsky ratio, e.g. the all-sky ghi / the clearsky ghi. This is
            assumed to be daily average data for climate change source data.
        """
        cs_ratio = data['rsds'] / data['clearsky_ghi']
        cs_ratio = np.minimum(cs_ratio, 1)
        return np.maximum(cs_ratio, 0)


class CloudMaskH5(DerivedFeature):
    """Cloud Mask feature class for computing from H5 data"""

    @classmethod
    def inputs(cls, feature):
        """Get list of raw features used in calculation of the cloud mask

        Parameters
        ----------
        feature : str
            Cloud mask feature name, needs to be "cloud_mask"

        Returns
        -------
        list
            List of required features for cloud_mask: clearsky_ghi, ghi
        """
        assert feature == 'cloud_mask'
        return ['clearsky_ghi', 'ghi']

    @classmethod
    def compute(cls, data, height=None):
        """Compute the cloud mask

        Parameters
        ----------
        data : dict
            dictionary of feature arrays used for this compuation, must include
            clearsky_ghi and ghi
        height : str | int
            Placeholder to match interface with other compute methods

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
        night_mask = night_mask.any(axis=(0, 1))

        cloud_mask = data['ghi'] < data['clearsky_ghi']
        cloud_mask = cloud_mask.astype(np.float32)
        cloud_mask[night_mask] = np.nan
        return cloud_mask.astype(np.float32)


class PotentialTempNC(DerivedFeature):
    """Potential Temperature feature class for NETCDF data. Needed since T is
    perturbation potential temperature.
    """

    @classmethod
    def inputs(cls, feature):
        """Get list of inputs needed for compute method."""
        height = Feature.get_height(feature)
        return [f'T_{height}m']

    @classmethod
    def compute(cls, data, height):
        """Method to compute Potential Temperature from NETCDF data

        Parameters
        ----------
        data : dict
            Dictionary of raw feature arrays to use for derivation
        height : str | int
            Height at which to compute the derived feature

        Returns
        -------
        ndarray
            Derived feature array

        """
        return data[f'T_{height}m'] + 300


class TempNC(DerivedFeature):
    """Temperature feature class for NETCDF data. Needed since T is potential
    temperature not standard temp.
    """

    @classmethod
    def inputs(cls, feature):
        """Get list of inputs needed for compute method."""
        height = Feature.get_height(feature)
        return [f'PotentialTemp_{height}m', f'Pressure_{height}m']

    @classmethod
    def compute(cls, data, height):
        """Method to compute T from NETCDF data

        Parameters
        ----------
        data : dict
            Dictionary of raw feature arrays to use for derivation
        height : str | int
            Height at which to compute the derived feature

        Returns
        -------
        ndarray
            Derived feature array

        """
        return invert_pot_temp(data[f'PotentialTemp_{height}m'],
                               data[f'Pressure_{height}m'])


class PressureNC(DerivedFeature):
    """Pressure feature class for NETCDF data. Needed since P is perturbation
    pressure.
    """

    @classmethod
    def inputs(cls, feature):
        """Get list of inputs needed for compute method."""
        height = Feature.get_height(feature)
        return [f'P_{height}m', f'PB_{height}m']

    @classmethod
    def compute(cls, data, height):
        """Method to compute pressure from NETCDF data

        Parameters
        ----------
        data : dict
            Dictionary of raw feature arrays to use for derivation
        height : str | int
            Height at which to compute the derived feature

        Returns
        -------
        ndarray
            Derived feature array

        """
        return data[f'P_{height}m'] + data[f'PB_{height}m']


class BVFreqSquaredNC(DerivedFeature):
    """BVF Squared feature class with needed inputs method and compute
    method
    """

    @classmethod
    def inputs(cls, feature):
        """Get list of inputs needed for compute method."""
        height = Feature.get_height(feature)
        return [f'PT_{height}m', f'PT_{int(height) - 100}m']

    @classmethod
    def compute(cls, data, height):
        """Method to compute BVF squared from NETCDF data

        Parameters
        ----------
        data : dict
            Dictionary of raw feature arrays to use for derivation
        height : str | int
            Height at which to compute the derived feature

        Returns
        -------
        ndarray
            Derived feature array

        """
        # T is perturbation potential temperature for wrf and the
        # base potential temperature is 300K
        bvf2 = np.float32(9.81 / 100)
        bvf2 *= (data[f'PT_{height}m'] - data[f'PT_{int(height) - 100}m'])
        bvf2 /= (data[f'PT_{height}m'] + data[f'PT_{int(height) - 100}m'])
        bvf2 /= np.float32(2)
        return bvf2


class InverseMonNC(DerivedFeature):
    """Inverse MO feature class with needed inputs method and compute method"""

    @classmethod
    def inputs(cls, feature):
        """Required inputs for inverse MO from NETCDF data

        Parameters
        ----------
        feature : str
            raw feature name. e.g. RMOL

        Returns
        -------
        list
            List of required features for computing RMOL
        """
        assert feature == 'RMOL'
        return ['UST', 'HFX']

    @classmethod
    def compute(cls, data, height=None):
        """Method to compute Inverse MO from NC data

        Parameters
        ----------
        data : dict
            Dictionary of raw feature arrays to use for derivation
        height : str | int
            Placeholder to match interface with other compute methods

        Returns
        -------
        ndarray
            Derived feature array

        """
        return inverse_mo_length(data['UST'], data['HFX'])


class BVFreqMon(DerivedFeature):
    """BVF MO feature class with needed inputs method and compute method"""

    @classmethod
    def inputs(cls, feature):
        """Required inputs for computing BVF times inverse MO from data

        Parameters
        ----------
        feature : str
            raw feature name. e.g. BVF_MO_100m

        Returns
        -------
        list
            List of required features for computing BVF_MO
        """
        height = Feature.get_height(feature)
        return [f'BVF2_{height}m', 'RMOL']

    @classmethod
    def compute(cls, data, height):
        """Method to compute BVF MO from data

        Parameters
        ----------
        data : dict
            Dictionary of raw feature arrays to use for derivation
        height : str | int
            Height at which to compute the derived feature

        Returns
        -------
        ndarray
            Derived feature array

        """
        bvf_mo = data[f'BVF2_{height}m']
        mask = data['RMOL'] != 0
        bvf_mo[mask] /= data['RMOL'][mask]

        # making this zero when not both bvf and mo are negative
        bvf_mo[data['RMOL'] >= 0] = 0
        bvf_mo[bvf_mo < 0] = 0

        return bvf_mo


class BVFreqSquaredH5(DerivedFeature):
    """BVF Squared feature class with needed inputs method and compute
    method
    """

    @classmethod
    def inputs(cls, feature):
        """Required inputs for computing BVF squared

        Parameters
        ----------
        feature : str
            raw feature name. e.g. BVF2_100m

        Returns
        -------
        list
            List of required features for computing BVF2
        """
        height = Feature.get_height(feature)
        return [
            f'temperature_{height}m', f'temperature_{int(height) - 100}m',
            f'pressure_{height}m', f'pressure_{int(height) - 100}m'
        ]

    @classmethod
    def compute(cls, data, height):
        """Method to compute BVF squared from H5 data

        Parameters
        ----------
        data : dict
            Dictionary of raw feature arrays to use for derivation
        height : str | int
            Height at which to compute the derived feature

        Returns
        -------
        ndarray
            Derived feature array

        """
        return bvf_squared(data[f'temperature_{height}m'],
                           data[f'temperature_{int(height) - 100}m'],
                           data[f'pressure_{height}m'],
                           data[f'pressure_{int(height) - 100}m'], 100)


class WindspeedNC(DerivedFeature):
    """Windspeed feature from netcdf data"""

    @classmethod
    def inputs(cls, feature):
        """Required inputs for computing windspeed from netcdf data

        Parameters
        ----------
        feature : str
            raw feature name. e.g. BVF_MO_100m

        Returns
        -------
        list
            List of required features for computing windspeed
        """
        height = Feature.get_height(feature)
        return [f'U_{height}m', f'V_{height}m', 'lat_lon']

    @classmethod
    def compute(cls, data, height):
        """Compute windspeed

        Parameters
        ----------
        data : dict
            Dictionary of raw feature arrays to use for derivation
        height : str | int
            Height at which to compute the derived feature

        Returns
        -------
        ndarray
            Derived feature array
        """
        ws, _ = invert_uv(data[f'U_{height}m'], data[f'V_{height}m'],
                          data['lat_lon'])
        return ws


class WinddirectionNC(DerivedFeature):
    """Winddirection feature from netcdf data"""

    @classmethod
    def inputs(cls, feature):
        """Required inputs for computing windspeed from netcdf data

        Parameters
        ----------
        feature : str
            raw feature name. e.g. BVF_MO_100m

        Returns
        -------
        list
            List of required features for computing windspeed
        """
        height = Feature.get_height(feature)
        return [f'U_{height}m', f'V_{height}m', 'lat_lon']

    @classmethod
    def compute(cls, data, height):
        """Compute winddirection

        Parameters
        ----------
        data : dict
            Dictionary of raw feature arrays to use for derivation
        height : str | int
            Height at which to compute the derived feature

        Returns
        -------
        ndarray
            Derived feature array
        """
        _, wd = invert_uv(data[f'U_{height}m'], data[f'V_{height}m'],
                          data['lat_lon'])
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
    def inputs(cls, feature):
        """Required inputs for computing U wind component

        Parameters
        ----------
        feature : str
            raw feature name. e.g. U_100m

        Returns
        -------
        list
            List of required features for computing U
        """
        features = ['uas']
        return features

    @classmethod
    def compute(cls, data, height):
        """Method to compute U wind component from data

        Parameters
        ----------
        data : dict
            Dictionary of raw feature arrays to use for derivation
        height : str | int
            Height at which to compute the derived feature

        Returns
        -------
        ndarray
            Derived feature array

        """
        return data['uas'] * (float(height) / cls.NEAR_SFC_HEIGHT)**cls.ALPHA


class VWindPowerLaw(DerivedFeature):
    """V wind component feature class with needed inputs method and compute
    method. Uses power law extrapolation to get values above surface

    https://csl.noaa.gov/projects/lamar/windshearformula.html
    https://www.tandfonline.com/doi/epdf/10.1080/00022470.1977.10470503
    """

    ALPHA = 0.2
    NEAR_SFC_HEIGHT = 10

    @classmethod
    def inputs(cls, feature):
        """Required inputs for computing V wind component

        Parameters
        ----------
        feature : str
            raw feature name. e.g. V_100m

        Returns
        -------
        list
            List of required features for computing V
        """
        features = ['vas']
        return features

    @classmethod
    def compute(cls, data, height):
        """Method to compute V wind component from data

        Parameters
        ----------
        data : dict
            Dictionary of raw feature arrays to use for derivation
        height : str | int
            Height at which to compute the derived feature

        Returns
        -------
        ndarray
            Derived feature array

        """
        return data['vas'] * (float(height) / cls.NEAR_SFC_HEIGHT)**cls.ALPHA


class UWind(DerivedFeature):
    """U wind component feature class with needed inputs method and compute
    method
    """

    @classmethod
    def inputs(cls, feature):
        """Required inputs for computing U wind component

        Parameters
        ----------
        feature : str
            raw feature name. e.g. U_100m

        Returns
        -------
        list
            List of required features for computing U
        """
        height = Feature.get_height(feature)
        features = [
            f'windspeed_{height}m', f'winddirection_{height}m', 'lat_lon'
        ]
        return features

    @classmethod
    def compute(cls, data, height):
        """Method to compute U wind component from data

        Parameters
        ----------
        data : dict
            Dictionary of raw feature arrays to use for derivation
        height : str | int
            Height at which to compute the derived feature

        Returns
        -------
        ndarray
            Derived feature array

        """
        u, _ = transform_rotate_wind(data[f'windspeed_{height}m'],
                                     data[f'winddirection_{height}m'],
                                     data['lat_lon'])
        return u


class VWind(DerivedFeature):
    """V wind component feature class with needed inputs method and compute
    method
    """

    @classmethod
    def inputs(cls, feature):
        """Required inputs for computing V wind component

        Parameters
        ----------
        feature : str
            raw feature name. e.g. V_100m

        Returns
        -------
        list
            List of required features for computing V
        """
        height = Feature.get_height(feature)
        return [
            f'windspeed_{height}m', f'winddirection_{height}m', 'lat_lon'
        ]

    @classmethod
    def compute(cls, data, height):
        """Method to compute V wind component from data

        Parameters
        ----------
        data : dict
            Dictionary of raw feature arrays to use for derivation
        height : str | int
            Height at which to compute the derived feature

        Returns
        -------
        ndarray
            Derived feature array

        """
        _, v = transform_rotate_wind(data[f'windspeed_{height}m'],
                                     data[f'winddirection_{height}m'],
                                     data['lat_lon'])
        return v


class TempNCforCC(DerivedFeature):
    """Air temperature variable from climate change nc files"""

    @classmethod
    def inputs(cls, feature):
        """Required inputs for computing ta

        Parameters
        ----------
        feature : str
            raw feature name. e.g. ta

        Returns
        -------
        list
            List of required features for computing ta
        """
        height = Feature.get_height(feature)
        return [f'ta_{height}m']

    @classmethod
    def compute(cls, data, height):
        """Method to compute ta in Celsius from ta source in Kelvin

        Parameters
        ----------
        data : dict
            Dictionary of raw feature arrays to use for derivation
        height : str | int
            Height at which to compute the derived feature

        Returns
        -------
        ndarray
            Derived feature array
        """
        return data[f'ta_{height}m'] - 273.15


class Tas(DerivedFeature):
    """Air temperature near surface variable from climate change nc files"""

    CC_FEATURE_NAME = 'tas'
    """Source CC.nc dataset name for air temperature variable. This can be
    changed in subclasses for other temperature datasets."""

    @classmethod
    def inputs(cls, feature):
        """Required inputs for computing tas

        Parameters
        ----------
        feature : str
            raw feature name. e.g. tas

        Returns
        -------
        list
            List of required features for computing tas
        """
        return [cls.CC_FEATURE_NAME]

    @classmethod
    def compute(cls, data, height):
        """Method to compute tas in Celsius from tas source in Kelvin

        Parameters
        ----------
        data : dict
            Dictionary of raw feature arrays to use for derivation
        height : str | int
            Height at which to compute the derived feature

        Returns
        -------
        ndarray
            Derived feature array
        """
        return data[cls.CC_FEATURE_NAME] - 273.15


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


class LatLonNC:
    """Lat Lon feature class with compute method"""

    @staticmethod
    def compute(file_paths, raster_index):
        """Get lats and lons

        Parameters
        ----------
        file_paths : list
            path to data file
        raster_index : list
            List of slices for raster

        Returns
        -------
        ndarray
            lat lon array
            (spatial_1, spatial_2, 2)
        """
        fp = file_paths if isinstance(file_paths, str) else file_paths[0]
        handle = xr.open_dataset(fp)
        valid_vars = set(handle.variables)
        lat_key = {'XLAT', 'lat', 'latitude', 'south_north'}.intersection(
            valid_vars)
        lat_key = next(iter(lat_key))
        lon_key = {'XLONG', 'lon', 'longitude', 'west_east'}.intersection(
            valid_vars)
        lon_key = next(iter(lon_key))

        if len(handle.variables[lat_key].dims) == 4:
            idx = (0, raster_index[0], raster_index[1], 0)
        elif len(handle.variables[lat_key].dims) == 3:
            idx = (0, raster_index[0], raster_index[1])
        elif len(handle.variables[lat_key].dims) == 2:
            idx = (raster_index[0], raster_index[1])

        if len(handle.variables[lat_key].dims) == 1:
            lons = handle.variables[lon_key].values
            lats = handle.variables[lat_key].values
            lons, lats = np.meshgrid(lons, lats)
            lat_lon = np.dstack(
                (lats[tuple(raster_index)], lons[tuple(raster_index)]))
        else:
            lats = handle.variables[lat_key].values[idx]
            lons = handle.variables[lon_key].values[idx]
            lat_lon = np.dstack((lats, lons))

        return lat_lon


class TopoH5:
    """Topography feature class with compute method"""

    @staticmethod
    def compute(file_paths, raster_index):
        """Get topography corresponding to raster

        Parameters
        ----------
        file_paths : list
            path to data file
        raster_index : ndarray
            Raster index array

        Returns
        -------
        ndarray
            topo array
            (spatial_1, spatial_2)
        """
        with Resource(file_paths[0], hsds=False) as handle:
            idx = (raster_index.flatten(),)
            topo = handle.get_meta_arr('elevation')[idx]
            topo = topo.reshape((raster_index.shape[0], raster_index.shape[1]))
        return topo


class LatLonH5:
    """Lat Lon feature class with compute method"""

    @staticmethod
    def compute(file_paths, raster_index):
        """Get lats and lons corresponding to raster for use in
        windspeed/direction -> u/v mapping

        Parameters
        ----------
        file_paths : list
            path to data file
        raster_index : ndarray
            Raster index array

        Returns
        -------
        ndarray
            lat lon array
            (spatial_1, spatial_2, 2)
        """
        with Resource(file_paths[0], hsds=False) as handle:
            lat_lon = handle.lat_lon[(raster_index.flatten(),)]
            return lat_lon.reshape(
                (raster_index.shape[0], raster_index.shape[1], 2))


class Feature:
    """Class to simplify feature computations. Stores feature height, feature
    basename, name of feature in handle
    """

    def __init__(self, feature, handle):
        """Takes a feature (e.g. U_100m) and gets the height (100), basename
        (U) and determines whether the feature is found in the data handle

        Parameters
        ----------
        feature : str
            Raw feature name e.g. U_100m
        handle : WindX | NSRDBX | xarray
            handle for data file
        """
        self.raw_name = feature
        self.height = self.get_height(feature)
        self.pressure = self.get_pressure(feature)
        self.basename = self.get_basename(feature)
        if self.raw_name in handle:
            self.handle_input = self.raw_name
        elif self.basename in handle:
            self.handle_input = self.basename
        else:
            self.handle_input = None

    @staticmethod
    def get_basename(feature):
        """Get basename of feature. e.g. temperature from temperature_100m

        Parameters
        ----------
        feature : str
            Name of feature. e.g. U_100m

        Returns
        -------
        str
            feature basename
        """
        height = Feature.get_height(feature)
        pressure = Feature.get_pressure(feature)
        if height is not None or pressure is not None:
            suffix = feature.split('_')[-1]
            basename = feature.replace(f'_{suffix}', '')
        else:
            basename = feature
        return basename

    @staticmethod
    def get_height(feature):
        """Get height from feature name to use in height interpolation

        Parameters
        ----------
        feature : str
            Name of feature. e.g. U_100m

        Returns
        -------
        float | None
            height to use for interpolation
            in meters
        """
        height = None
        if isinstance(feature, str):
            height = re.search(r'\d+m', feature)
            if height:
                height = height.group(0).strip('m')
                if not height.isdigit():
                    height = None
        return height

    @staticmethod
    def get_pressure(feature):
        """Get pressure from feature name to use in pressure interpolation

        Parameters
        ----------
        feature : str
            Name of feature. e.g. U_100pa

        Returns
        -------
        float | None
            pressure to use for interpolation in pascals
        """
        pressure = None
        if isinstance(feature, str):
            pressure = re.search(r'\d+pa', feature)
            if pressure:
                pressure = pressure.group(0).strip('pa')
                if not pressure.isdigit():
                    pressure = None
        return pressure
