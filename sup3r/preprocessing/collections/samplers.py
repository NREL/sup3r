"""Collection objects consisting of lists of :class:`Sampler` instances"""

import logging
from typing import List, Union

import numpy as np

from sup3r.preprocessing.collections.base import Collection
from sup3r.preprocessing.samplers.base import Sampler
from sup3r.preprocessing.samplers.dual import DualSampler

logger = logging.getLogger(__name__)

np.random.seed(42)


class SamplerCollection(Collection):
    """Collection of :class:`Sampler` containers with methods for
    sampling across the containers."""

    def __init__(
        self,
        samplers: Union[List[Sampler], List[DualSampler]],
        s_enhance,
        t_enhance,
    ):
        super().__init__(containers=samplers)
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        self.container_index = self.get_container_index()
        _ = self.check_shared_attr('sample_shape')

    def __getattr__(self, attr):
        """Get attributes from self or the first container in the
        collection."""
        if attr in dir(self):
            return self.__getattribute__(attr)
        return self.check_shared_attr(attr)

    def check_shared_attr(self, attr):
        """Check if all containers have the same value for `attr`."""
        msg = ('Not all containers in the collection have the same value for '
               f'{attr}')
        out = getattr(self.containers[0], attr, None)
        assert all(getattr(c, attr, None) == out for c in self.containers), msg
        return out

    def get_container_index(self):
        """Get random container index based on weights"""
        indices = np.arange(0, len(self.containers))
        return np.random.choice(indices, p=self.container_weights)

    def get_random_container(self):
        """Get random container based on container weights"""
        if self._sample_counter % self.batch_size == 0:
            self.container_index = self.get_container_index()
        return self.containers[self.container_index]

    def __getitem__(self, keys):
        """Get data sample from sampled container."""
        container = self.get_random_container()
        return container.get_next()

    @property
    def lr_shape(self):
        """Shape of low resolution sample in a low-res / high-res pair.  (e.g.
        (spatial_1, spatial_2, temporal, features))"""
        return (*self.lr_sample_shape, len(self.lr_features))

    @property
    def hr_shape(self):
        """Shape of high resolution sample in a low-res / high-res pair.  (e.g.
        (spatial_1, spatial_2, temporal, features))"""
        return (*self.hr_sample_shape, len(self.hr_features))

    @property
    def hr_features_ind(self):
        """Get the high-resolution feature channel indices that should be
        included for training. Any high-resolution features that are only
        included in the data handler to be coarsened for the low-res input are
        removed"""
        hr_features = list(self.hr_out_features) + list(self.hr_exo_features)
        if list(self.features) == hr_features:
            return np.arange(len(self.features))
        return [
            i
            for i, feature in enumerate(self.features)
            if feature in hr_features
        ]

    @property
    def hr_features(self):
        """Get the high-resolution features corresponding to
        `hr_features_ind`"""
        return [self.features[ind].lower() for ind in self.hr_features_ind]
