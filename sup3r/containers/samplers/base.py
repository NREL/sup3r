"""Abstract sampler objects. These are containers which also can sample from
the underlying data. These interface with Batchers so they also have additional
information about how different features are used by models."""

import logging
from fnmatch import fnmatch
from typing import Dict, Optional, Tuple
from warnings import warn

from sup3r.containers.abstract import Data
from sup3r.containers.base import Container
from sup3r.utilities.utilities import uniform_box_sampler, uniform_time_sampler

logger = logging.getLogger(__name__)


class Sampler(Container):
    """Sampler class for iterating through contained things."""

    def __init__(self, data: Data, sample_shape,
                 feature_sets: Optional[Dict] = None):
        """
        Parameters
        ----------
        data : Data
            wrapped xr.Dataset() object with data that will be sampled from.
            Can be the `.data` attribute of various :class:`Container` objects.
            i.e. :class:`Loader`, :class:`Extracter`, :class:`Deriver`, as long
            as the spatial dimensions are not flattened.
        sample_shape : tuple
            Size of arrays to sample from the contained data.
        feature_sets : Optional[dict]
            Optional dictionary describing how the full set of features is
            split between `lr_only_features` and `hr_exo_features`.

            lr_only_features : list | tuple
                List of feature names or patt*erns that should only be
                included in the low-res training set and not the high-res
                observations.
            hr_exo_features : list | tuple
                List of feature names or patt*erns that should be included
                in the high-resolution observation but not expected to be
                output from the generative model. An example is high-res
                topography that is to be injected mid-network.
        """
        super().__init__(data=data)
        feature_sets = feature_sets or {}
        self._lr_only_features = feature_sets.get('lr_only_features', [])
        self._hr_exo_features = feature_sets.get('hr_exo_features', [])
        self._counter = 0
        self.sample_shape = sample_shape
        self.lr_features = self.features
        self.hr_features = self.features
        self.preflight()

    def get_sample_index(self):
        """Randomly gets spatial sample and time sample

        Parameters
        ----------
        data_shape : tuple
            Size of available region for sampling
            (spatial_1, spatial_2, temporal)
        sample_shape : tuple
            Size of observation to sample
            (spatial_1, spatial_2, temporal)

        Returns
        -------
        sample_index : tuple
            Tuple of latitude slice, longitude slice, time slice, and features.
            Used to get single observation like self.data[sample_index]
        """
        spatial_slice = uniform_box_sampler(self.shape, self.sample_shape[:2])
        time_slice = uniform_time_sampler(self.shape, self.sample_shape[2])
        return (*spatial_slice, time_slice, self.features)

    def preflight(self):
        """Check if the sample_shape is larger than the requested raster
        size"""
        bad_shape = (self.sample_shape[0] > self.shape[0]
                     and self.sample_shape[1] > self.shape[1])
        if bad_shape:
            msg = (f'spatial_sample_shape {self.sample_shape[:2]} is '
                   f'larger than the raster size {self.shape[:2]}')
            logger.warning(msg)
            warn(msg)

        if len(self.sample_shape) == 2:
            logger.info(
                'Found 2D sample shape of {}. Adding temporal dim of 1'.format(
                    self.sample_shape))
            self.sample_shape = (*self.sample_shape, 1)

        msg = (f'sample_shape[2] ({self.sample_shape[2]}) cannot be larger '
               'than the number of time steps in the raw data '
               f'({self.shape[2]}).')

        if self.shape[2] < self.sample_shape[2]:
            logger.warning(msg)
            warn(msg)

    def get_next(self):
        """Get "next" thing in the container. e.g. data observation or batch of
        observations"""
        return self[self.get_sample_index()]

    @property
    def sample_shape(self) -> Tuple:
        """Shape of the data sample to select when `get_next()` is called."""
        return self._sample_shape

    @sample_shape.setter
    def sample_shape(self, sample_shape):
        """Set the shape of the data sample to select when `get_next()` is
        called."""
        self._sample_shape = sample_shape

    @property
    def hr_sample_shape(self) -> Tuple:
        """Shape of the data sample to select when `get_next()` is called. Same
        as sample_shape"""
        return self._sample_shape

    @hr_sample_shape.setter
    def hr_sample_shape(self, hr_sample_shape):
        """Set the sample shape to select when `get_next()` is called. Same
        as sample_shape"""
        self._sample_shape = hr_sample_shape

    def __next__(self):
        """Iterable next method"""
        return self.get_next()

    def __iter__(self):
        self._counter = 0
        return self

    def __len__(self):
        return self._size

    def _parse_features(self, unparsed_feats):
        """Return a list of parsed feature names without wildcards."""
        if isinstance(unparsed_feats, str):
            parsed_feats = [unparsed_feats]
        elif isinstance(unparsed_feats, tuple):
            parsed_feats = list(unparsed_feats)
        elif unparsed_feats is None:
            parsed_feats = []
        else:
            parsed_feats = unparsed_feats

        if any('*' in fn for fn in parsed_feats):
            out = []
            for feature in self.features:
                match = any(fnmatch(feature.lower(), pattern.lower())
                            for pattern in parsed_feats)
                if match:
                    out.append(feature)
            parsed_feats = out
        return [f.lower() for f in parsed_feats]

    @property
    def lr_only_features(self):
        """List of feature names or patt*erns that should only be included in
        the low-res training set and not the high-res observations."""
        return self._parse_features(self._lr_only_features)

    @property
    def hr_exo_features(self):
        """Get a list of exogenous high-resolution features that are only used
        for training e.g., mid-network high-res topo injection. These must come
        at the end of the high-res feature set. These can also be input to the
        model as low-res features."""
        self._hr_exo_features = self._parse_features(self._hr_exo_features)

        if len(self._hr_exo_features) > 0:
            msg = (f'High-res train-only features "{self._hr_exo_features}" '
                   f'do not come at the end of the full high-res feature set: '
                   f'{self.features}')
            last_feat = self.features[-len(self._hr_exo_features):]
            assert list(self._hr_exo_features) == list(last_feat), msg

        return self._hr_exo_features

    @property
    def hr_out_features(self):
        """Get a list of high-resolution features that are intended to be
        output by the GAN. Does not include high-resolution exogenous
        features"""

        out = []
        for feature in self.features:
            lr_only = any(fnmatch(feature.lower(), pattern.lower())
                          for pattern in self.lr_only_features)
            ignore = lr_only or feature in self.hr_exo_features
            if not ignore:
                out.append(feature)

        if len(out) == 0:
            msg = (f'It appears that all handler features "{self.features}" '
                   'were specified as `hr_exo_features` or `lr_only_features` '
                   'and therefore there are no output features!')
            logger.error(msg)
            raise RuntimeError(msg)

        return out
