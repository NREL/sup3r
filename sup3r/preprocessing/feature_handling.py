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
from datetime import datetime as dt
import xarray as xr

from rex import Resource
from rex.utilities.execution import SpawnProcessPool
from sup3r.utilities.utilities import (transform_rotate_wind,
                                       bvf_squared,
                                       get_raster_shape,
                                       inverse_mo_length)


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


class BVFreqSquaredNC(DerivedFeature):
    """BVF Squared feature class with needed inputs method and compute
    method"""

    @classmethod
    def inputs(cls, feature):
        height = Feature.get_height(feature)
        features = [f'T_{height}m',
                    f'T_{int(height) - 100}m']

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
        bvf2 *= (data[f'T_{height}m']
                 - data[f'T_{int(height) - 100}m'])
        bvf2 /= (data[f'T_{height}m'] - 600
                 + data[f'T_{int(height) - 100}m'])
        bvf2 /= np.float32(2)
        return bvf2


class InverseMonNC(DerivedFeature):
    """Inverse MO feature class with needed inputs method and compute method"""

    @classmethod
    def inputs(cls, feature):
        """Required inputs inverse MO from NETCDF data

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
        features = ['U_2m', 'V_2m', 'W_2m', 'T_2m']
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
        return inverse_mo_length(
            data['U_2m'], data['V_2m'],
            data['W_2m'], data['T_2m'])


class BVFreqMonNC(DerivedFeature):
    """BVF MO feature class with needed inputs method and compute method"""

    @classmethod
    def inputs(cls, feature):
        """Required inputs for computing BVF times inverse MO from NETCDF data

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
        features = [f'T_{height}m',
                    f'T_{int(height) - 100}m',
                    'RMOL']
        return features

    @classmethod
    def alternative_inputs(cls, feature):
        """Alternative required inputs for computing BVF times inverse MO from
        NETCDF data

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
        features = [f'T_{height}m', f'T_{int(height) - 100}m']
        features += InverseMonNC.inputs('RMOL')

        return features

    @classmethod
    def compute(cls, data, height):
        """Method to compute BVF MO from NC data

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
        bvf_mo = BVFreqSquaredNC.compute(data, height)

        if 'RMOL' not in data:
            data['RMOL'] = InverseMonNC.compute(data, height)

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
            raw feature name. e.g. BVF_squared_100m

        Returns
        -------
        list
            List of required features for computing BVF_squared
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


class BVFreqMonH5(DerivedFeature):
    """BVF MO feature class with needed inputs method and compute method"""

    @classmethod
    def inputs(cls, feature):
        """Required inputs for computing BVF times inverse MO

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
        features = [f'temperature_{height}m',
                    f'temperature_{int(height) - 100}m',
                    f'pressure_{height}m',
                    f'pressure_{int(height) - 100}m',
                    'inversemoninobukhovlength_2m']
        return features

    @classmethod
    def compute(cls, data, height):
        """Method to compute BVF MO from H5 data

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
        bvf_mo = BVFreqSquaredH5.compute(data, height)
        mask = data['inversemoninobukhovlength_2m'] != 0
        bvf_mo[mask] = (bvf_mo[mask]
                        / data['inversemoninobukhovlength_2m'][mask])

        # making this zero when not both bvf and mo are negative
        bvf_mo[data['inversemoninobukhovlength_2m'] >= 0] = 0
        bvf_mo[bvf_mo < 0] = 0

        return bvf_mo


class UWindH5(DerivedFeature):
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
        features = [f'windspeed_{height}m',
                    f'winddirection_{height}m',
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
        u, _ = transform_rotate_wind(
            data[f'windspeed_{height}m'],
            data[f'winddirection_{height}m'],
            data['lat_lon'])
        return u


class VWindH5(DerivedFeature):
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
        features = [f'windspeed_{height}m',
                    f'winddirection_{height}m',
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
        _, v = transform_rotate_wind(
            data[f'windspeed_{height}m'],
            data[f'winddirection_{height}m'],
            data['lat_lon'])
        return v


class UWindNsrdb(DerivedFeature):
    """U wind component feature class with needed inputs method and compute
    method"""

    @classmethod
    def inputs(cls, feature):
        """Required inputs for computing U wind component

        Parameters
        ----------
        feature : str
            raw feature name. e.g. U

        Returns
        -------
        list
            List of required features for computing U
        """
        features = ['wind_speed', 'wind_direction', 'lat_lon']
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
        u, _ = transform_rotate_wind(data['wind_speed'],
                                     data['wind_direction'],
                                     data['lat_lon'])
        return u


class VWindNsrdb(DerivedFeature):
    """V wind component feature class with needed inputs
    method and compute method"""

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
        features = ['wind_speed', 'wind_direction', 'lat_lon']
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
        _, v = transform_rotate_wind(data['wind_speed'],
                                     data['wind_direction'],
                                     data['lat_lon'])
        return v


class LatLonNC:
    """Lat Lon feature class with compute method"""

    @classmethod
    def compute(cls, file_paths, raster_index):
        """Get lats and lons
<<<<<<< HEAD
=======

>>>>>>> 25f1ea2... init h5 output handling
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
        with xr.open_dataset(file_paths[0]) as handle:
            lats = (handle.XLAT.values if 'Time' not
                    in handle.XLAT.dims else handle.XLAT.values[0])
            lons = (handle.XLONG.values if 'Time' not
                    in handle.XLONG.dims else handle.XLONG.values[0])
            lat_lon = np.concatenate(
                [lats[tuple(raster_index)][..., np.newaxis],
                 lons[tuple(raster_index)][..., np.newaxis]], axis=-1)
        return lat_lon


class LatLonH5:
    """Lat Lon feature class with compute method"""

    @classmethod
    def compute(cls, file_paths, raster_index):
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
                                       raster_index.shape[1],
                                       2))

        return lat_lon


class Feature:
    """Class to simplify feature computations. Stores alternative names,
    feature height, feature basename, name of feature in handle"""

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
        self.alternative_names = {
            'temperature': 'T',
            'pressure': 'P',
            'T': 'temperature',
            'P': 'pressure'
        }
        self.raw_name = feature
        self.height = self.get_height(feature)
        self.basename = self.get_basename(feature)
        self.alt_name = self.check_rename(handle, feature)
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
        if height is not None:
            suffix = feature.split('_')[-1]
            basename = feature.strip(f'_{suffix}')
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
        height = feature.split('_')[-1].strip('m')
        if not height.isdigit():
            height = None
        return height

    def check_rename(self, handle, feature):
        """Method to account for possible alternative feature names. e.g. T for
        temperature

        Parameters
        ----------
        handle : WindX | NSRDBX | xarray
            handle pointing to file data
        feature : str
            Feature name. e.g. temperature_100m

        Returns
        -------
        renamed_feature : str
            New feature name. e.g. T_100m
        """

        for k, v in self.alternative_names.items():
            if k in feature:
                renamed_feature = feature.replace(k, v)
                handle_basenames = [
                    self.get_basename(f) for f in handle]
                if v in handle_basenames:
                    return renamed_feature
        return None


class FeatureHandler:
    """Feature Handler with cache for previously loaded features used in other
    calculations """

    TIME_IND_FEATURES = ('lat_lon',)

    @classmethod
    def valid_input_features(cls, features, handle):
        """Check if features are in handle or have compute methods

        Parameters
        ----------
        features : str | list
            Raw feature names e.g. U_100m
        handle : WindX | xarray
            handle for data file

        Returns
        -------
        bool
            Whether feature basename is in handle
        """

        if features is None:
            return False

        handle_basenames = [Feature.get_basename(r) for r in handle]
        if all(Feature.get_basename(f) in handle_basenames
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
        old_keys = [f for f in data[chunk_number]
                    if f not in all_features]
        for k in old_keys:
            data[chunk_number].pop(k)

    @classmethod
    def serial_extract(cls, file_paths, raster_index, time_chunks,
                       input_features):
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

        Returns
        -------
        dict
            dictionary of feature arrays with integer keys for chunks and str
            keys for features.  e.g. data[chunk_number][feature] = array.
            (spatial_1, spatial_2, temporal)
        """

        time_dep_features = [f for f in input_features
                             if f not in cls.TIME_IND_FEATURES]
        time_ind_features = [f for f in input_features
                             if f in cls.TIME_IND_FEATURES]

        data = defaultdict(dict)

        for t, t_slice in enumerate(time_chunks):
            for f in time_dep_features:
                data[t][f] = cls.extract_feature(
                    file_paths, raster_index, f, t_slice)
        for f in time_ind_features:
            data[-1][f] = cls.extract_feature(
                file_paths, raster_index, f)

        return data

    @classmethod
    def parallel_extract(cls, file_paths, raster_index, time_chunks,
                         input_features, max_workers=None):
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

        Returns
        -------
        dict
            dictionary of feature arrays with integer keys for chunks and str
            keys for features.  e.g. data[chunk_number][feature] = array.
            (spatial_1, spatial_2, temporal)
        """

        logger.info(f'Extracting {input_features}')

        futures = {}
        now = dt.now()

        time_dep_features = [f for f in input_features
                             if f not in cls.TIME_IND_FEATURES]
        time_ind_features = [f for f in input_features
                             if f in cls.TIME_IND_FEATURES]

        data = defaultdict(dict)

        if max_workers == 1:
            return cls.serial_extract(
                file_paths, raster_index, time_chunks, input_features)
        else:
            with SpawnProcessPool(max_workers=max_workers) as exe:
                for t, t_slice in enumerate(time_chunks):
                    for f in time_dep_features:
                        future = exe.submit(cls.extract_feature,
                                            file_paths=file_paths,
                                            raster_index=raster_index,
                                            feature=f,
                                            time_slice=t_slice)
                        meta = {'feature': f,
                                'chunk': t}
                        futures[future] = meta

                for f in time_ind_features:
                    future = exe.submit(cls.extract_feature,
                                        file_paths=file_paths,
                                        raster_index=raster_index,
                                        feature=f)
                    meta = {'feature': f,
                            'chunk': -1}
                    futures[future] = meta

                shape = get_raster_shape(raster_index)
                time_shape = time_chunks[0].stop - time_chunks[0].start
                time_shape //= time_chunks[0].step
                logger.info(
                    f'Started extracting {input_features}'
                    f' in {dt.now() - now}. Using {len(time_chunks)}'
                    f' time chunks of shape ({shape[0]}, {shape[1]}, '
                    f'{time_shape}) for {len(input_features)} features')

                for i, future in enumerate(as_completed(futures)):
                    v = futures[future]
                    data[v['chunk']][v['feature']] = future.result()
                    if i % (len(futures) // 10 + 1) == 0:
                        logger.debug(f'{i+1} out of {len(futures)} feature '
                                     'chunks extracted.')

        return data

    @classmethod
    def serial_compute(cls, data, time_chunks, input_features, all_features):
        """Compute features in series

        Parameters
        ----------
        data : dict
            dictionary of feature arrays with integer keys for chunks and str
            keys for features. e.g. data[chunk_number][feature] = array.
            (spatial_1, spatial_2, temporal)
        time_chunks : list
            List of slices to chunk data feature extraction along time
            dimension
        input_features : list
            list of input feature strings
        all_features : list
            list of all features including those requiring derivation from
            input features

        Returns
        -------
        data : dict
            dictionary of feature arrays, including computed features, with
            integer keys for chunks and str keys for features.
            e.g. data[chunk_number][feature] = array.
            (spatial_1, spatial_2, temporal)
        """

        derived_features = [f for f in all_features if f not in input_features]

        for t, _ in enumerate(time_chunks):
            for _, f in enumerate(derived_features):
                method = cls.lookup(f, 'compute')
                height = Feature.get_height(f)
                tmp = cls.get_input_arrays(data, t, f)
                data[t][f] = method(tmp, height)
            cls.pop_old_data(data, t, all_features)

        return data

    @classmethod
    def parallel_compute(cls, data, raster_index, time_chunks,
                         input_features, all_features, max_workers=None):
        """Compute features using parallel subprocesses

        Parameters
        ----------
        data : dict
            dictionary of feature arrays with integer keys for chunks and str
            keys for features.
            e.g. data[chunk_number][feature] = array.
            (spatial_1, spatial_2, temporal)
        raster_index : ndarray
            raster index for spatial domain
        time_chunks : list
            List of slices to chunk data feature extraction along time
            dimension
        input_features : list
            list of input feature strings
        all_features : list
            list of all features including those requiring derivation from
            input features
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

        derived_features = [f for f in all_features if f not in input_features]
        if len(derived_features) == 0:
            return data
        else:
            logger.info(f'Computing {derived_features}')

        futures = {}
        now = dt.now()
        if max_workers == 1:
            return cls.serial_compute(
                data, time_chunks, input_features, all_features)
        else:
            with SpawnProcessPool(max_workers=max_workers) as exe:
                for t, _ in enumerate(time_chunks):
                    for f in derived_features:
                        method = cls.lookup(f, 'compute')
                        height = Feature.get_height(f)
                        tmp = cls.get_input_arrays(data, t, f)
                        future = exe.submit(
                            method, data=tmp, height=height)

                        meta = {'feature': f,
                                'chunk': t}

                        futures[future] = meta

                    cls.pop_old_data(
                        data, t, all_features)

                shape = get_raster_shape(raster_index)
                time_shape = time_chunks[0].stop - time_chunks[0].start
                time_shape //= time_chunks[0].step
                logger.info(
                    f'Started computing {derived_features}'
                    f' in {dt.now() - now}. Using {len(time_chunks)}'
                    f' time chunks of shape ({shape[0]}, {shape[1]}, '
                    f'{time_shape}) for {len(derived_features)} features')

                for i, future in enumerate(as_completed(futures)):
                    v = futures[future]
                    data[v['chunk']][v['feature']] = future.result()
                    if i % (len(futures) // 10 + 1) == 0:
                        logger.debug(f'{i+1} out of {len(futures)} feature '
                                     'chunks computed')

        return data

    @classmethod
    def get_input_arrays(cls, data, chunk_number, f):
        """Get only arrays needed for computations

        Parameters
        ----------
        data : dict
            Dictionary of feature arrays
        chunk_number :
            time chunk for which to get input arrays
        f : str
            feature to compute using input arrays

        Returns
        -------
        dict
            Dictionary of arrays with only needed features
        """
        inputs = cls.lookup(f, 'inputs')
        if not all(r in data[chunk_number] or r
                   in data[-1] for r in inputs(f)):
            inputs = cls.lookup(f, 'alternative_inputs')

        tmp = {}
        for r in inputs(f):
            if r in data[chunk_number]:
                tmp[r] = data[chunk_number][r]
            else:
                tmp[r] = data[-1][r]
        return tmp

    @classmethod
    def lookup(cls, feature, attr_name):
        """Lookup feature in feature registry

        Parameters
        ----------
        feature : str
            Feature to lookup in registry
        attr_name : str
            Attribute to get from registry. Can be compute, inputs,
            alternative_inputs

        Returns
        -------
        method | None
            Method corresponding to requested attribute
        """

        for k, v in cls.feature_registry().items():
            if re.match(k.lower(), feature.lower()):
                method = getattr(v, attr_name, None)
                if method is not None:
                    return method
        return None

    @classmethod
    def get_raw_feature_list_from_handle(cls, features, handle):
        """Lookup inputs needed to compute feature

        Parameters
        ----------
        features : list
            Features for which to get needed inputs for derivation
        handle : WindX | xarray
            File handle to use for checking whether features can be extracted

        Returns
        -------
        list
            List of input features
        """

        raw_features = []
        for f in features:
            method = cls.lookup(f, 'inputs')
            if method is not None:
                if cls.valid_input_features(method(f), handle):
                    for r in method(f):
                        if r not in raw_features:
                            raw_features.append(r)
                else:
                    method = cls.lookup(f, 'alternative_inputs')
                    if method is not None:
                        for r in method(f):
                            if r not in raw_features:
                                raw_features.append(r)
                    else:
                        msg = (f'Cannot compute {f} from the provided data.')
                        raise ValueError(msg)
            else:
                if f not in raw_features:
                    raw_features.append(f)

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
    def get_raw_feature_list(cls, file_paths, features):
        """Lookup inputs needed to compute features"""

    @classmethod
    @abstractmethod
    def extract_feature(cls, file_paths, raster_index, feature,
                        time_slice=slice(None)) -> np.dtype(np.float32):
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
