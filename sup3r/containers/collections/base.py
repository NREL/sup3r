"""Base collection classes. These are objects that contain sets / lists of
containers like batch handlers. Of course these also contain data so they're
containers also!."""

from typing import List

import numpy as np

from sup3r.containers.base import Container, ContainerPair
from sup3r.containers.collections.abstract import (
    AbstractCollection,
)


class Collection(AbstractCollection):
    """Base collection class."""

    def __init__(self, containers: List[Container]):
        super().__init__(containers)
        self.all_container_pairs = self.check_all_container_pairs()

    @property
    def features(self):
        """Get set of features available in the container collection."""
        return self.containers[0].features

    @property
    def shape(self):
        """Get full available shape to sample from when selecting sample_size
        samples."""
        return self.containers[0].shape

    def check_all_container_pairs(self):
        """Check if all containers are pairs of low and high res or single
        containers"""
        return all(isinstance(container, ContainerPair)
                   for container in self.containers)

    @property
    def lr_features(self):
        """Get a list of low-resolution features. All low-resolution features
        are used for training."""
        return self.containers[0].lr_features

    @property
    def hr_exo_features(self):
        """Get a list of high-resolution features that are only used for
        training e.g., mid-network high-res topo injection."""
        return self.containers[0].hr_exo_features

    @property
    def hr_out_features(self):
        """Get a list of low-resolution features that are intended to be output
        by the GAN."""
        return self.containers[0].hr_out_features

    @property
    def hr_features_ind(self):
        """Get the high-resolution feature channel indices that should be
        included for training. Any high-resolution features that are only
        included in the data handler to be coarsened for the low-res input are
        removed"""
        hr_features = list(self.hr_out_features) + list(self.hr_exo_features)
        if list(self.features) == hr_features:
            return np.arange(len(self.features))
        return [i for i, feature in enumerate(self.features)
                if feature in hr_features]

    @property
    def hr_features(self):
        """Get the high-resolution features corresponding to
        `hr_features_ind`"""
        return [self.features[ind] for ind in self.hr_features_ind]
