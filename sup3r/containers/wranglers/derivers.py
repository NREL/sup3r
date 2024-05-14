"""Sup3r feature handling: extraction / computations.

@author: bbenton
"""

import logging
import re
from abc import abstractmethod
from collections import defaultdict
from concurrent.futures import as_completed
from typing import ClassVar

import numpy as np
import psutil
from rex.utilities.execution import SpawnProcessPool

from sup3r.preprocessing.derived_features import Feature
from sup3r.utilities.utilities import (
    get_raster_shape,
)

np.random.seed(42)

logger = logging.getLogger(__name__)


class FeatureDeriver:
    """Collection of methods used for computing / deriving features from
    available raw features.  """

    FEATURE_REGISTRY: ClassVar[dict] = {}

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

        return all(
            Feature.get_basename(f) in handle_features or f in handle_features
            for f in features)

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

        return all(
            Feature.get_basename(f) in handle_features
            or f in handle_features or cls.lookup(f, 'compute') is not None
            for f in features)

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
            old_keys = [f for f in data[chunk_number] if f not in all_features]
            for k in old_keys:
                data[chunk_number].pop(k)

    @classmethod
    def has_surrounding_features(cls, feature, handle):
        """Check if handle has feature values at surrounding heights. e.g. if
        feature=U_40m check if the handler has u at heights below and above 40m

        Parameters
        ----------
        feature : str
            Raw feature name e.g. U_100m
        handle: xarray.Dataset
            netcdf data object

        Returns
        -------
        bool
            Whether feature has surrounding heights
        """
        basename = Feature.get_basename(feature)
        height = float(Feature.get_height(feature))
        handle_features = list(handle)

        msg = ('Trying to check surrounding heights for multi-level feature '
               f'({feature})')
        assert feature.lower() != basename.lower(), msg
        msg = ('Trying to check surrounding heights for feature already in '
               f'handler ({feature}).')
        assert feature not in handle_features, msg
        surrounding_features = [
            v for v in handle_features
            if Feature.get_basename(v).lower() == basename.lower()
        ]
        heights = [int(Feature.get_height(v)) for v in surrounding_features]
        heights = np.array(heights)
        lower_check = len(heights[heights < height]) > 0
        higher_check = len(heights[heights > height]) > 0
        return lower_check and higher_check

    @classmethod
    def has_exact_feature(cls, feature, handle):
        """Check if exact feature is in handle

        Parameters
        ----------
        feature : str
            Raw feature name e.g. U_100m
        handle: xarray.Dataset
            netcdf data object

        Returns
        -------
        bool
            Whether handle contains exact feature or not
        """
        return feature in handle or feature.lower() in handle

    @classmethod
    def has_multilevel_feature(cls, feature, handle):
        """Check if exact feature is in handle

        Parameters
        ----------
        feature : str
            Raw feature name e.g. U_100m
        handle: xarray.Dataset
            netcdf data object

        Returns
        -------
        bool
            Whether handle contains multilevel data for given feature
        """
        basename = Feature.get_basename(feature)
        return basename in handle or basename.lower() in handle

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
            logger.debug(f'{t + 1} out of {len(time_chunks)} feature '
                         'chunks extracted.')
        return data

    @classmethod
    def parallel_extract(cls,
                         file_paths,
                         raster_index,
                         time_chunks,
                         input_features,
                         max_workers=None,
                         **kwargs):
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

            for i, future in enumerate(as_completed(futures)):
                v = futures[future]
                try:
                    data[v['chunk']][v['feature']] = future.result()
                except Exception as e:
                    msg = (f'Error extracting chunk {v["chunk"]} for'
                           f' {v["feature"]}')
                    logger.error(msg)
                    raise RuntimeError(msg) from e
                mem = psutil.virtual_memory()
                logger.info(f'{i + 1} out of {len(futures)} feature '
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
            inputs = cls.lookup(feature,
                                'inputs',
                                handle_features=handle_features)
            method = cls.lookup(feature, 'compute')
            height = Feature.get_height(feature)
            if inputs is not None:
                if method is None:
                    return data[inputs(feature)[0]]
                if all(r in data for r in inputs(feature)):
                    data[feature] = method(data, height)
                else:
                    for r in inputs(feature):
                        data[r] = cls.recursive_compute(
                            data, r, handle_features, file_paths, raster_index)
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
                    data=tmp,
                    feature=f,
                    handle_features=handle_features,
                    file_paths=file_paths,
                    raster_index=raster_index)
            cls.pop_old_data(data, t, all_features)
            logger.debug(f'{t + 1} out of {len(time_chunks)} feature '
                         'chunks computed.')

        return data

    @classmethod
    def parallel_compute(cls,
                         data,
                         file_paths,
                         raster_index,
                         time_chunks,
                         derived_features,
                         all_features,
                         handle_features,
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
                    future = exe.submit(cls.recursive_compute,
                                        data=tmp,
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

            for i, future in enumerate(as_completed(futures)):
                v = futures[future]
                chunk_idx = v['chunk']
                data[chunk_idx] = data.get(chunk_idx, {})
                data[chunk_idx][v['feature']] = future.result()
                mem = psutil.virtual_memory()
                logger.info(f'{i + 1} out of {len(futures)} feature '
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
        if isinstance(feature, str):
            for k, v in cls.FEATURE_REGISTRY.items():
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
        if isinstance(feature, str):
            for k, v in cls.FEATURE_REGISTRY.items():
                if re.match(k.lower(), feature.lower()):
                    out = v
                    break
        return out

    @classmethod
    def _lookup(cls, out, feature, handle_features=None):
        """Lookup feature in feature registry

        Parameters
        ----------
        out : None
            Candidate registry method for feature
        feature : str
            Feature to lookup in registry
        handle_features : list
            List of feature names (datasets) available in the source file. If
            feature is found explicitly in this list, height/pressure suffixes
            will not be appended to the output.

        Returns
        -------
        method | None
            Feature registry method corresponding to feature
        """
        if isinstance(out, list):
            for v in out:
                if v in handle_features:
                    return lambda x: [v]

        if out in handle_features:
            return lambda x: [out]

        height = Feature.get_height(feature)
        if height is not None:
            out = out.split('(.*)')[0] + f'{height}m'

        pressure = Feature.get_pressure(feature)
        if pressure is not None:
            out = out.split('(.*)')[0] + f'{pressure}pa'

        return lambda x: [out] if isinstance(out, str) else out

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

        if not isinstance(out, (str, list)):
            return getattr(out, attr_name, None)

        if attr_name == 'inputs':
            return cls._lookup(out, feature, handle_features)

        return None

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
        low_handle_features = [f.lower() for f in handle_features]
        vhf = cls.valid_handle_features([feature.lower()], low_handle_features)

        check1 = feature not in raw_features
        check2 = (vhf or method is None)

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
    def extract_feature(cls,
                        file_paths,
                        raster_index,
                        feature,
                        time_slice=slice(None),
                        **kwargs):
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
        kwargs : dict
            Keyword arguments passed to source handler

        Returns
        -------
        ndarray
            Data array for extracted feature
            (spatial_1, spatial_2, temporal)
        """
