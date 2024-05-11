"""Abstract sampler objects. These are containers which also can sample from
the underlying data. These interface with Batchers so they also have additional
information about how different features are used by models."""

import logging
from abc import ABC, abstractmethod
from fnmatch import fnmatch
from typing import List, Tuple

from sup3r.containers.base import Container
from sup3r.containers.collections.base import Collection

logger = logging.getLogger(__name__)


class AbstractSampler(Container, ABC):
    """Sampler class for iterating through contained things."""

    def __init__(self, data, sample_shape, lr_only_features=(),
                 hr_exo_features=()):
        """
        Parameters
        ----------
        data : DataObject
            Object with data that will be sampled from.
        data_shape : tuple
            Size of extent available for sampling
        sample_shape : tuple
            Size of arrays to sample from the contained data.
        lr_only_features : list | tuple
            List of feature names or patt*erns that should only be included in
            the low-res training set and not the high-res observations.
        hr_exo_features : list | tuple
            List of feature names or patt*erns that should be included in the
            high-resolution observation but not expected to be output from the
            generative model. An example is high-res topography that is to be
            injected mid-network.
        """
        super().__init__(data)
        self._lr_only_features = lr_only_features
        self._hr_exo_features = hr_exo_features
        self._counter = 0
        self._sample_shape = sample_shape

    @abstractmethod
    def get_sample_index(self):
        """Get index used to select sample from contained data. e.g.
        self[index]."""

    def get_next(self):
        """Get "next" thing in the container. e.g. data observation or batch of
        observations"""
        return self[self.get_sample_index()]

    @property
    def sample_shape(self) -> Tuple:
        """Shape of the data sample to select when `get_next()` is called."""
        return self._sample_shape

    def __next__(self):
        """Iterable next method"""
        return self.get_next()

    def __iter__(self):
        self._counter = 0
        return self

    def __len__(self):
        return self._size

    @property
    def lr_only_features(self):
        """List of feature names or patt*erns that should only be included in
        the low-res training set and not the high-res observations."""
        if isinstance(self._lr_only_features, str):
            self._lr_only_features = [self._lr_only_features]

        elif isinstance(self._lr_only_features, tuple):
            self._lr_only_features = list(self._lr_only_features)

        elif self._lr_only_features is None:
            self._lr_only_features = []

        return self._lr_only_features

    @property
    def lr_features(self):
        """Get a list of low-resolution features. It is assumed that all
        features are used in the low-resolution observations for single
        container objects. For container pairs this is overridden."""
        return self.features

    @property
    def hr_exo_features(self):
        """Get a list of exogenous high-resolution features that are only used
        for training e.g., mid-network high-res topo injection. These must come
        at the end of the high-res feature set. These can also be input to the
        model as low-res features."""

        if isinstance(self._hr_exo_features, str):
            self._hr_exo_features = [self._hr_exo_features]

        elif isinstance(self._hr_exo_features, tuple):
            self._hr_exo_features = list(self._hr_exo_features)

        elif self._hr_exo_features is None:
            self._hr_exo_features = []

        if any('*' in fn for fn in self._hr_exo_features):
            hr_exo_features = []
            for feature in self.features:
                match = any(fnmatch(feature.lower(), pattern.lower())
                            for pattern in self._hr_exo_features)
                if match:
                    hr_exo_features.append(feature)
            self._hr_exo_features = hr_exo_features

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

    @property
    def hr_features(self):
        """Same as features since this is a single data object container."""
        return self.features


class AbstractSamplerCollection(Collection, ABC):
    """Abstract collection of sampler containers with methods for sampling
    across the containers."""

    def __init__(self, containers: List[AbstractSampler], s_enhance,
                 t_enhance):
        super().__init__(containers)
        self.container_weights = None
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance

    @abstractmethod
    def get_container_weights(self):
        """List of normalized container sizes used to weight them when randomly
        sampling."""

    @abstractmethod
    def get_container_index(self) -> int:
        """Get random container index based on weights."""

    @abstractmethod
    def get_random_container(self) -> Container:
        """Get random container based on weights."""

    def __getitem__(self, index):
        """Get data sample from sampled container."""
        container = self.get_random_container()
        return container.get_next()

    @property
    def sample_shape(self):
        """Get shape of sample to select when sampling container collection."""
        return self.containers[0].sample_shape

    @property
    def lr_sample_shape(self):
        """Get shape of low resolution samples"""
        return self.containers[0].lr_sample_shape

    @property
    def hr_sample_shape(self):
        """Get shape of high resolution samples"""
        return self.containers[0].hr_sample_shape

    @property
    def lr_shape(self):
        """Shape of low resolution sample in a low-res / high-res pair.  (e.g.
        (spatial_1, spatial_2, temporal, features)) """
        return (*self.lr_sample_shape, len(self.lr_features))

    @property
    def hr_shape(self):
        """Shape of high resolution sample in a low-res / high-res pair.  (e.g.
        (spatial_1, spatial_2, temporal, features)) """
        return (*self.hr_sample_shape, len(self.hr_features))
