# -*- coding: utf-8 -*-
"""
Sup3r feature handling module.

@author: bbenton
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import as_completed
import logging
import numpy as np
import re
import xarray as xr
import psutil

from rex import Resource
from rex.utilities.execution import SpawnProcessPool
from sup3r.utilities.utilities import (invert_pot_temp, invert_uv,
                                       rotor_equiv_ws,
                                       transform_rotate_wind,
                                       bvf_squared,
                                       get_raster_shape,
                                       inverse_mo_length,
                                       vorticity_calc
                                       )


np.random.seed(42)

logger = logging.getLogger(__name__)


class DerivedFeature(ABC):
    """Abstract class for special features which need to be derived from raw
    features"""

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
        cs_ratio = cs_ratio.astype(np.float32)
        return cs_ratio


class ClearSkyRatioCC(DerivedFeature):
    """Clear Sky Ratio feature class for computing from climate change netcdf
    data"""

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
        cs_ratio = np.maximum(cs_ratio, 0)

        return cs_ratio


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
        cloud_mask = cloud_mask.astype(np.float32)
        return cloud_mask


class PotentialTempNC(DerivedFeature):
    """Potential Temperature feature class for NETCDF data. Needed since T is
    perturbation potential temperature."""

    @classmethod
    def inputs(cls, feature):
        height = Feature.get_height(feature)
        features = [f'T_{height}m']
        return features

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
    temperature not standard temp."""

    @classmethod
    def inputs(cls, feature):
        height = Feature.get_height(feature)
        features = [f'PotentialTemp_{height}m',
                    f'Pressure_{height}m']
        return features

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
    pressure."""

    @classmethod
    def inputs(cls, feature):
        height = Feature.get_height(feature)
        features = [f'P_{height}m',
                    f'PB_{height}m']
        return features

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
    method"""

    @classmethod
    def inputs(cls, feature):
        height = Feature.get_height(feature)
        features = [f'PT_{height}m',
                    f'PT_{int(height) - 100}m']

        return features

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
        features = ['UST', 'HFX']
        return features

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
        features = [f'BVF2_{height}m', 'RMOL']
        return features

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
        bvf_mo[mask] = bvf_mo[mask] / data['RMOL'][mask]

        # making this zero when not both bvf and mo are negative
        bvf_mo[data['RMOL'] >= 0] = 0
        bvf_mo[bvf_mo < 0] = 0

        return bvf_mo


class BVFreqSquaredH5(DerivedFeature):
    """BVF Squared feature class with needed inputs method and compute
    method"""

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
        features = [f'temperature_{height}m',
                    f'temperature_{int(height) - 100}m',
                    f'pressure_{height}m',
                    f'pressure_{int(height) - 100}m']

        return features

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
        return bvf_squared(
            data[f'temperature_{height}m'],
            data[f'temperature_{int(height) - 100}m'],
            data[f'pressure_{height}m'],
            data[f'pressure_{int(height) - 100}m'],
            100)


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
        features = [f'U_{height}m', f'V_{height}m', 'lat_lon']
        return features

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
        features = [f'U_{height}m', f'V_{height}m', 'lat_lon']
        return features

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


class Veer(DerivedFeature):
    """Veer at a given height"""

    HEIGHTS = [40, 60, 80, 100, 120]

    @classmethod
    def inputs(cls, feature):
        """Required inputs for computing Veer

        Parameters
        ----------
        feature : str
            raw feature name. e.g. BVF_MO_100m

        Returns
        -------
        list
            List of required features for computing REWS
        """

        rotor_center = Feature.get_height(feature)
        if rotor_center is None:
            heights = cls.HEIGHTS
        else:
            heights = [int(rotor_center) - i * 20 for i in [-2, -1, 0, 1, 2]]
        features = []
        for height in heights:
            features.append(f'winddirection_{height}m')
        return features

    @classmethod
    def compute(cls, data, height):
        """Compute Veer

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

        if height is None:
            heights = cls.HEIGHTS
        else:
            heights = [int(height) - i * 20 for i in [-2, -1, 0, 1, 2]]
        veer = 0
        for i in range(0, len(heights), 2):
            tmp = np.radians(data[f'winddirection_{height[i + 1]}'])
            tmp -= np.radians(data[f'winddirection_{height[i]}'])
            veer += np.abs(tmp)
        veer /= (heights[-1] - heights[0])
        return veer


class Shear(DerivedFeature):
    """Wind shear at a given height"""

    @classmethod
    def inputs(cls, feature):
        """Required inputs for computing Veer

        Parameters
        ----------
        feature : str
            raw feature name. e.g. BVF_MO_100m

        Returns
        -------
        list
            List of required features for computing Veer
        """

        height = Feature.get_height(feature)
        heights = [int(height), int(height) + 20]
        features = []
        for height in heights:
            features.append(f'winddirection_{height}m')
        return features

    @classmethod
    def compute(cls, data, height):
        """Compute REWS

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

        heights = [int(height), int(height) + 20]
        shear = np.cos(np.radians(data[f'winddirection_{int(height) + 20}m']))
        shear -= np.cos(np.radians(data[f'winddirection_{int(height)}m']))
        shear /= (heights[-1] - heights[0])
        return shear


class Rews(DerivedFeature):
    """Rotor equivalent wind speed"""

    HEIGHTS = [40, 60, 80, 100, 120]

    @classmethod
    def inputs(cls, feature):
        """Required inputs for computing REWS

        Parameters
        ----------
        feature : str
            raw feature name. e.g. BVF_MO_100m

        Returns
        -------
        list
            List of required features for computing REWS
        """

        rotor_center = Feature.get_height(feature)
        if rotor_center is None:
            heights = cls.HEIGHTS
        else:
            heights = [int(rotor_center) - i * 20 for i in [-2, -1, 0, 1, 2]]
        features = []
        for height in heights:
            features.append(f'windspeed_{height}m')
            features.append(f'winddirection_{height}m')
        return features

    @classmethod
    def compute(cls, data, height):
        """Compute REWS

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

        if height is None:
            heights = cls.HEIGHTS
        else:
            heights = [int(height) - i * 20 for i in [-2, -1, 0, 1, 2]]
        rews = rotor_equiv_ws(data, heights)
        return rews


class UWind(DerivedFeature):
    """U wind component feature class with needed inputs method and compute
    method"""

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
        features = [f'windspeed_{height}m', f'winddirection_{height}m',
                    'lat_lon']
        return features

    @classmethod
    def compute(cls, data, height):
        """Method to compute U wind component from H5 data

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


class Vorticity(DerivedFeature):
    """Vorticity feature class with needed inputs method and compute
    method"""

    @classmethod
    def inputs(cls, feature):
        """Required inputs for computing vorticity

        Parameters
        ----------
        feature : str
            raw feature name. e.g. vorticity_100m

        Returns
        -------
        list
            List of required features for computing vorticity
        """
        height = Feature.get_height(feature)
        features = [f'U_{height}m', f'V_{height}m']
        return features

    @classmethod
    def compute(cls, data, height):
        """Method to compute vorticity

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
        vort = vorticity_calc(data[f'U_{height}m'],
                              data[f'V_{height}m'])
        return vort


class VWind(DerivedFeature):
    """V wind component feature class with needed inputs method and compute
    method"""

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
        features = [f'windspeed_{height}m', f'winddirection_{height}m',
                    'lat_lon']
        return features

    @classmethod
    def compute(cls, data, height):
        """Method to compute V wind component from H5 data

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
    files"""
    CC_FEATURE_NAME = 'tasmin'


class TasMax(Tas):
    """Daily max air temperature near surface variable from climate change nc
    files"""
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
        lat_key = 'XLAT'
        lon_key = 'XLONG'
        if lat_key not in handle.variables:
            lat_key = 'latitude'
        if lon_key not in handle.variables:
            lon_key = 'longitude'
        if len(handle.variables[lat_key].dims) == 3:
            idx = (0, raster_index[0], raster_index[1])
        elif len(handle.variables[lat_key].dims) == 4:
            idx = (0, raster_index[0], raster_index[1], 0)
        else:
            idx = (raster_index[0], raster_index[1])
        lats = handle.variables[lat_key].values[idx]
        lons = handle.variables[lon_key].values[idx]
        lat_lon = np.dstack((lats, lons))
        return lat_lon


class LatLonNCforCC:
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
        lats = handle.lat.values
        lons = handle.lon.values
        if handle.lat.dims != ('lat',):
            lats = lats[handle.lat.dims.index('lat')]
        if handle.lon.dims != ('lon',):
            lons = lons[handle.lon.dims.index('lon')]

        assert len(lats.shape) == 1, f'Got bad lats shape: {lats.shape}'
        assert len(lons.shape) == 1, f'Got bad lons shape: {lons.shape}'

        lons, lats = np.meshgrid(lons, lats)
        lat_lon = np.dstack((lats[tuple(raster_index)],
                             lons[tuple(raster_index)]))
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
            idx = tuple([raster_index.flatten()])
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
            lat_lon = handle.lat_lon[tuple([raster_index.flatten()])]
            lat_lon = lat_lon.reshape((raster_index.shape[0],
                                       raster_index.shape[1], 2))
        return lat_lon


class Feature:
    """Class to simplify feature computations. Stores feature height, feature
    basename, name of feature in handle"""

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
        pressure = re.search(r'\d+pa', feature)
        if pressure:
            pressure = pressure.group(0).strip('pa')
            if not pressure.isdigit():
                pressure = None
        return pressure


class FeatureHandler:
    """Feature Handler with cache for previously loaded features used in other
    calculations """

    @classmethod
    def valid_handle_features(cls, features, handle_features):
        """Check if features are in handle

        Parameters
        ----------
        features : str | list
            Raw feature names e.g. U_100m
        handle_features : list
            Features available in raw data

        Returns
        -------
        bool
            Whether feature basename is in handle
        """

        if features is None:
            return False

        return all(Feature.get_basename(f) in handle_features
                   or f in handle_features for f in features)

    @classmethod
    def valid_input_features(cls, features, handle_features):
        """Check if features are in handle or have compute methods

        Parameters
        ----------
        features : str | list
            Raw feature names e.g. U_100m
        handle_features : list
            Features available in raw data

        Returns
        -------
        bool
            Whether feature basename is in handle
        """

        if features is None:
            return False

        if all(Feature.get_basename(f) in handle_features
               or f in handle_features
               or cls.lookup(f, 'compute') is not None for f in features):
            return True
        return False

    @classmethod
    def pop_old_data(cls, data, chunk_number, all_features):
        """Remove input feature data if no longer needed for requested features

        Parameters
        ----------
        data : dict
            dictionary of feature arrays with integer keys for chunks and str
            keys for features.  e.g. data[chunk_number][feature] = array.
            (spatial_1, spatial_2, temporal)
        chunk_number : int
            time chunk index to check
        all_features : list
            list of all requested features including those requiring derivation
            from input features

        """
        if data:
            old_keys = [f for f in data[chunk_number]
                        if f not in all_features]
            for k in old_keys:
                data[chunk_number].pop(k)

    @classmethod
    def serial_extract(cls, file_paths, raster_index, time_chunks,
                       input_features, **kwargs):
        """Extract features in series

        Parameters
        ----------
        file_paths : list
            list of file paths
        raster_index : ndarray
            raster index for spatial domain
        time_chunks : list
            List of slices to chunk data feature extraction along time
            dimension
        input_features : list
            list of input feature strings
        kwargs : dict
            kwargs passed to source handler for data extraction. e.g. This
            could be {'parallel': True,
                      'chunks': {'south_north': 120, 'west_east': 120}}
            which then gets passed to xr.open_mfdataset(file, **kwargs)

        Returns
        -------
        dict
            dictionary of feature arrays with integer keys for chunks and str
            keys for features.  e.g. data[chunk_number][feature] = array.
            (spatial_1, spatial_2, temporal)
        """

        data = defaultdict(dict)
        for t, t_slice in enumerate(time_chunks):
            for f in input_features:
                data[t][f] = cls.extract_feature(file_paths, raster_index, f,
                                                 t_slice, **kwargs)
            interval = int(np.ceil(len(time_chunks) / 10))
            if t % interval == 0:
                logger.debug(f'{t+1} out of {len(time_chunks)} feature '
                             'chunks extracted.')
        return data

    @classmethod
    def parallel_extract(cls, file_paths, raster_index, time_chunks,
                         input_features, max_workers=None, **kwargs):
        """Extract features using parallel subprocesses

        Parameters
        ----------
        file_paths : list
            list of file paths
        raster_index : ndarray | list
            raster index for spatial domain
        time_chunks : list
            List of slices to chunk data feature extraction along time
            dimension
        input_features : list
            list of input feature strings
        max_workers : int | None
            Number of max workers to use for extraction.  If equal to 1 then
            method is run in serial
        kwargs : dict
            kwargs passed to source handler for data extraction. e.g. This
            could be {'parallel': True,
                      'chunks': {'south_north': 120, 'west_east': 120}}
            which then gets passed to xr.open_mfdataset(file, **kwargs)

        Returns
        -------
        dict
            dictionary of feature arrays with integer keys for chunks and str
            keys for features.  e.g. data[chunk_number][feature] = array.
            (spatial_1, spatial_2, temporal)
        """
        futures = {}
        data = defaultdict(dict)

        with SpawnProcessPool(max_workers=max_workers) as exe:
            for t, t_slice in enumerate(time_chunks):
                for f in input_features:
                    future = exe.submit(cls.extract_feature,
                                        file_paths=file_paths,
                                        raster_index=raster_index,
                                        feature=f,
                                        time_slice=t_slice,
                                        **kwargs)
                    meta = {'feature': f, 'chunk': t}
                    futures[future] = meta

            shape = get_raster_shape(raster_index)
            time_shape = time_chunks[0].stop - time_chunks[0].start
            time_shape //= time_chunks[0].step
            logger.info(f'Started extracting {input_features}'
                        f' using {len(time_chunks)}'
                        f' time chunks of shape ({shape[0]}, {shape[1]}, '
                        f'{time_shape}) for {len(input_features)} features')

            interval = int(np.ceil(len(futures) / 10))
            for i, future in enumerate(as_completed(futures)):
                v = futures[future]
                try:
                    data[v['chunk']][v['feature']] = future.result()
                except Exception as e:
                    msg = (f'Error extracting chunk {v["chunk"]} for'
                           f' {v["feature"]}')
                    logger.error(msg)
                    raise RuntimeError(msg) from e
                if i % interval == 0:
                    mem = psutil.virtual_memory()
                    logger.info(f'{i+1} out of {len(futures)} feature '
                                'chunks extracted. Current memory usage is '
                                f'{mem.used / 1e9:.3f} GB out of '
                                f'{mem.total / 1e9:.3f} GB total.')

        return data

    @classmethod
    def recursive_compute(cls, data, feature, handle_features, file_paths,
                          raster_index):
        """Compute intermediate features recursively

        Parameters
        ----------
        data : dict
            dictionary of feature arrays. e.g. data[feature] = array.
            (spatial_1, spatial_2, temporal)
        feature : str
            Name of feature to compute
        handle_features : list
            Features available in raw data
        file_paths : list
            Paths to data files. Used if compute method operates directly on
            source handler instead of input arrays. This is done with features
            without inputs methods like lat_lon and topography.
        raster_index : ndarray
            raster index for spatial domain

        Returns
        -------
        ndarray
            Array of computed feature data
        """
        if feature not in data:
            inputs = cls.lookup(feature, 'inputs',
                                handle_features=handle_features)
            method = cls.lookup(feature, 'compute')
            height = Feature.get_height(feature)
            if inputs is not None:
                if method is None:
                    return data[inputs(feature)[0]]
                elif all(r in data for r in inputs(feature)):
                    data[feature] = method(data, height)
                else:
                    for r in inputs(feature):
                        data[r] = cls.recursive_compute(data, r,
                                                        handle_features,
                                                        file_paths,
                                                        raster_index)
                    data[feature] = method(data, height)
            elif method is not None:
                data[feature] = method(file_paths, raster_index)

        return data[feature]

    @classmethod
    def serial_compute(cls, data, file_paths, raster_index, time_chunks,
                       derived_features, all_features, handle_features):
        """Compute features in series

        Parameters
        ----------
        data : dict
            dictionary of feature arrays with integer keys for chunks and str
            keys for features. e.g. data[chunk_number][feature] = array.
            (spatial_1, spatial_2, temporal)
        file_paths : list
            Paths to data files. Used if compute method operates directly on
            source handler instead of input arrays. This is done with features
            without inputs methods like lat_lon and topography.
        raster_index : ndarray
            raster index for spatial domain
        time_chunks : list
            List of slices to chunk data feature extraction along time
            dimension
        derived_features : list
            list of feature strings which need to be derived
        all_features : list
            list of all features including those requiring derivation from
            input features
        handle_features : list
            Features available in raw data

        Returns
        -------
        data : dict
            dictionary of feature arrays, including computed features, with
            integer keys for chunks and str keys for features.
            e.g. data[chunk_number][feature] = array.
            (spatial_1, spatial_2, temporal)
        """

        if len(derived_features) == 0:
            return data

        for t, _ in enumerate(time_chunks):
            data[t] = data.get(t, {})
            for _, f in enumerate(derived_features):
                tmp = cls.get_input_arrays(data, t, f, handle_features)
                data[t][f] = cls.recursive_compute(
                    data=tmp, feature=f, handle_features=handle_features,
                    file_paths=file_paths, raster_index=raster_index)
            cls.pop_old_data(data, t, all_features)
            interval = int(np.ceil(len(time_chunks) / 10))
            if t % interval == 0:
                logger.debug(f'{t+1} out of {len(time_chunks)} feature '
                             'chunks computed.')

        return data

    @classmethod
    def parallel_compute(cls, data, file_paths, raster_index, time_chunks,
                         derived_features, all_features, handle_features,
                         max_workers=None):
        """Compute features using parallel subprocesses

        Parameters
        ----------
        data : dict
            dictionary of feature arrays with integer keys for chunks and str
            keys for features.
            e.g. data[chunk_number][feature] = array.
            (spatial_1, spatial_2, temporal)
        file_paths : list
            Paths to data files. Used if compute method operates directly on
            source handler instead of input arrays. This is done with features
            without inputs methods like lat_lon and topography.
        raster_index : ndarray
            raster index for spatial domain
        time_chunks : list
            List of slices to chunk data feature extraction along time
            dimension
        derived_features : list
            list of feature strings which need to be derived
        all_features : list
            list of all features including those requiring derivation from
            input features
        handle_features : list
            Features available in raw data
        max_workers : int | None
            Number of max workers to use for computation. If equal to 1 then
            method is run in serial

        Returns
        -------
        data : dict
            dictionary of feature arrays, including computed features, with
            integer keys for chunks and str keys for features. Includes e.g.
            data[chunk_number][feature] = array.
            (spatial_1, spatial_2, temporal)
        """
        if len(derived_features) == 0:
            return data

        futures = {}
        with SpawnProcessPool(max_workers=max_workers) as exe:
            for t, _ in enumerate(time_chunks):
                for f in derived_features:
                    tmp = cls.get_input_arrays(data, t, f, handle_features)
                    future = exe.submit(cls.recursive_compute, data=tmp,
                                        feature=f,
                                        handle_features=handle_features,
                                        file_paths=file_paths,
                                        raster_index=raster_index)
                    meta = {'feature': f, 'chunk': t}
                    futures[future] = meta

                cls.pop_old_data(data, t, all_features)

            shape = get_raster_shape(raster_index)
            time_shape = time_chunks[0].stop - time_chunks[0].start
            time_shape //= time_chunks[0].step
            logger.info(f'Started computing {derived_features}'
                        f' using {len(time_chunks)}'
                        f' time chunks of shape ({shape[0]}, {shape[1]}, '
                        f'{time_shape}) for {len(derived_features)} features')

            interval = int(np.ceil(len(futures) / 10))
            for i, future in enumerate(as_completed(futures)):
                v = futures[future]
                chunk_idx = v['chunk']
                data[chunk_idx] = data.get(chunk_idx, {})
                data[chunk_idx][v['feature']] = future.result()
                if i % interval == 0:
                    mem = psutil.virtual_memory()
                    logger.info(f'{i+1} out of {len(futures)} feature '
                                'chunks computed. Current memory usage is '
                                f'{mem.used / 1e9:.3f} GB out of '
                                f'{mem.total / 1e9:.3f} GB total.')

        return data

    @classmethod
    def get_input_arrays(cls, data, chunk_number, f, handle_features):
        """Get only arrays needed for computations

        Parameters
        ----------
        data : dict
            Dictionary of feature arrays
        chunk_number :
            time chunk for which to get input arrays
        f : str
            feature to compute using input arrays
        handle_features : list
            Features available in raw data

        Returns
        -------
        dict
            Dictionary of arrays with only needed features
        """
        tmp = {}
        if data:
            inputs = cls.get_inputs_recursive(f, handle_features)
            for r in inputs:
                if r in data[chunk_number]:
                    tmp[r] = data[chunk_number][r]
        return tmp

    @classmethod
    def _exact_lookup(cls, feature):
        """Check for exact feature match in feature registry. e.g. check if
        temperature_2m matches a feature registry entry of temperature_2m.
        (Still case insensitive)

        Parameters
        ----------
        feature : str
            Feature to lookup in registry

        Returns
        -------
        out : str
            Matching feature registry entry.
        """

        out = None
        for k, v in cls.feature_registry().items():
            if k.lower() == feature.lower():
                out = v
                break
        return out

    @classmethod
    def _pattern_lookup(cls, feature):
        """Check for pattern feature match in feature registry. e.g. check if
        U_100m matches a feature registry entry of U_(.*)m

        Parameters
        ----------
        feature : str
            Feature to lookup in registry

        Returns
        -------
        out : str
            Matching feature registry entry.
        """

        out = None
        for k, v in cls.feature_registry().items():
            if re.match(k.lower(), feature.lower()):
                out = v
                break
        return out

    @classmethod
    def lookup(cls, feature, attr_name, handle_features=None):
        """Lookup feature in feature registry

        Parameters
        ----------
        feature : str
            Feature to lookup in registry
        attr_name : str
            Type of method to lookup. e.g. inputs or compute
        handle_features : list
            List of feature names (datasets) available in the source file. If
            feature is found explicitly in this list, height/pressure suffixes
            will not be appended to the output.

        Returns
        -------
        method | None
            Feature registry method corresponding to feature
        """

        handle_features = handle_features or []

        out = cls._exact_lookup(feature)
        if out is None:
            out = cls._pattern_lookup(feature)

        if out is None:
            return None

        if not isinstance(out, str):
            return getattr(out, attr_name, None)

        elif attr_name == 'inputs':

            if out in handle_features:
                return lambda x: [out]

            height = Feature.get_height(feature)
            if height is not None:
                out = out.split('(.*)')[0] + f'{height}m'

            pressure = Feature.get_pressure(feature)
            if pressure is not None:
                out = out.split('(.*)')[0] + f'{pressure}pa'

            return lambda x: [out]

    @classmethod
    def get_inputs_recursive(cls, feature, handle_features):
        """Lookup inputs needed to compute feature. Walk through inputs methods
        for each required feature to get all raw features.

        Parameters
        ----------
        feature : str
            Feature for which to get needed inputs for derivation
        handle_features : list
            Features available in raw data

        Returns
        -------
        list
            List of input features
        """
        raw_features = []
        method = cls.lookup(feature, 'inputs', handle_features=handle_features)

        check1 = feature not in raw_features
        check2 = (cls.valid_handle_features([feature], handle_features)
                  or method is None)
        if check1 and check2:
            raw_features.append(feature)

        else:
            for f in method(feature):
                lkup = cls.lookup(f, 'inputs', handle_features=handle_features)
                valid = cls.valid_handle_features([f], handle_features)
                if (lkup is None or valid) and f not in raw_features:
                    raw_features.append(f)
                else:
                    for r in cls.get_inputs_recursive(f, handle_features):
                        if r not in raw_features:
                            raw_features.append(r)

        return raw_features

    @classmethod
    def get_raw_feature_list(cls, features, handle_features):
        """Lookup inputs needed to compute feature

        Parameters
        ----------
        features : list
            Features for which to get needed inputs for derivation
        handle_features : list
            Features available in raw data

        Returns
        -------
        list
            List of input features
        """
        raw_features = []
        for f in features:
            candidate_features = cls.get_inputs_recursive(f, handle_features)
            if candidate_features:
                for r in candidate_features:
                    if r not in raw_features:
                        raw_features.append(r)
            else:
                req = cls.lookup(f, "inputs", handle_features=handle_features)
                req = req(f)
                msg = (f'Cannot compute {f} from the provided data. '
                       f'Requested features: {req}')
                logger.error(msg)
                raise ValueError(msg)
        return raw_features

    @classmethod
    @abstractmethod
    def feature_registry(cls):
        """Registry of methods for computing features

        Returns
        -------
        dict
            Method registry
        """

    @classmethod
    @abstractmethod
    def extract_feature(cls, file_paths, raster_index, feature,
                        time_slice=slice(None), **kwargs):
        """Extract single feature from data source

        Parameters
        ----------
        file_paths : list
            path to data file
        raster_index : ndarray
            Raster index array
        time_slice : slice
            slice of time to extract
        feature : str
            Feature to extract from data

        Returns
        -------
        ndarray
            Data array for extracted feature
            (spatial_1, spatial_2, temporal)
        """
