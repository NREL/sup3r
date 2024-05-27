"""Collection objects consisting of lists of :class:`Sampler` instances"""

import logging
from typing import List, Union

import numpy as np

from sup3r.containers.collections.base import Collection
from sup3r.containers.samplers.base import Sampler
from sup3r.containers.samplers.dual import DualSampler

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
        self.check_shape_consistency()
        self.all_container_pairs = self.check_all_container_pairs()

    def __getattr__(self, attr):
        """Get attributes from self or the first container in the
        collection."""
        if attr in dir(self):
            return self.__getattribute__(attr)
        return self.get_multi_attr(attr)

    def get_multi_attr(self, attr):
        """Check if all containers have the same value for `attr`."""
        msg = (
            f'Requested {attr} attribute from a collection with '
            f'{len(self.containers)} container objects but these objects do '
            f'not all have the same value for {attr}.'
        )
        attr = getattr(self.containers[0], attr, None)
        check = all(getattr(c, attr, None) == attr for c in self.containers)
        if not check:
            logger.error(msg)
            raise ValueError(msg)
        return attr

    def check_shape_consistency(self):
        """Make sure all samplers in the collection have the same sample
        shape."""
        sample_shapes = [c.sample_shape for c in self.containers]
        msg = (
            'All samplers must have the same sample_shape. Received '
            'inconsistent collection.'
        )
        assert all(s == sample_shapes[0] for s in sample_shapes), msg

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
