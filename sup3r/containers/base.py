"""Base Container classes. These are general objects that contain data. Data
wranglers, data samplers, data loaders, batch handlers, etc are all
containers."""

import copy
import logging
from fnmatch import fnmatch
from typing import Tuple

from sup3r.containers.abstract import (
    AbstractContainer,
)

logger = logging.getLogger(__name__)


class Container(AbstractContainer):
    """Base container class."""

    def __init__(self, features, lr_only_features, hr_exo_features):
        """
        Parameters
        ----------
        features : list
            list of all features extracted or to extract.
        lr_only_features : list | tuple
            List of feature names or patt*erns that should only be included in
            the low-res training set and not the high-res observations.
        hr_exo_features : list | tuple
            List of feature names or patt*erns that should be included in the
            high-resolution observation but not expected to be output from the
            generative model. An example is high-res topography that is to be
            injected mid-network.
        """
        self.features = features
        self._lr_only_features = lr_only_features
        self._hr_exo_features = hr_exo_features

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


class ContainerPair(Container):
    """Pair of two Containers, one for low resolution and one for high
    resolution data."""

    def __init__(self, lr_container: Container, hr_container: Container):
        self.lr_container = lr_container
        self.hr_container = hr_container

    @property
    def data(self) -> Tuple[Container, Container]:
        """Raw data."""
        return (self.lr_container, self.hr_container)

    @property
    def shape(self):
        """Shape of raw data"""
        return (self.lr_container.shape, self.hr_container.shape)

    def __getitem__(self, keys):
        """Method for accessing self.data."""
        lr_key, hr_key = keys
        return (self.lr_container[lr_key], self.hr_container[hr_key])

    @property
    def features(self):
        """Get a list of data features including features from both the lr and
        hr data handlers"""
        out = list(copy.deepcopy(self.lr_container.features))
        out += [fn for fn in self.hr_container.features if fn not in out]
        return out

    @property
    def lr_only_features(self):
        """Features to use for training only and not output"""
        tof = [fn for fn in self.lr_container.features
               if fn not in self.hr_out_features
               and fn not in self.hr_exo_features]
        return tof

    @property
    def lr_features(self):
        """Get a list of low-resolution features. All low-resolution features
        are used for training."""
        return self.lr_container.features

    @property
    def hr_features(self):
        """Get a list of high-resolution features. This is hr_exo_features plus
        hr_out_features."""
        return self.hr_container.features

    @property
    def hr_exo_features(self):
        """Get a list of high-resolution features that are only used for
        training e.g., mid-network high-res topo injection. These must come at
        the end of the high-res feature set."""
        return self.hr_container.hr_exo_features

    @property
    def hr_out_features(self):
        """Get a list of high-resolution features that are intended to be
        output by the GAN. Does not include high-resolution exogenous features
        """
        return self.hr_container.hr_out_features
