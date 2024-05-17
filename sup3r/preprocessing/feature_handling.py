"""Sup3r feature handling: extraction / computations.

@author: bbenton
"""

import logging
import re
from typing import ClassVar

import numpy as np

from sup3r.utilities.utilities import Feature

np.random.seed(42)

logger = logging.getLogger(__name__)


class FeatureHandler:
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
