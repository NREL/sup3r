"""Sampler objects. These take in data objects / containers and can them sample
from them. These samples can be used to build batches."""

import copy
import logging
from typing import Dict, Optional

from sup3r.preprocessing.base import Container
from sup3r.preprocessing.samplers.base import Sampler

logger = logging.getLogger(__name__)


class DualSampler(Sampler):
    """Pair of sampler objects, one for low resolution and one for high
    resolution, initialized from a :class:`Container` object with low and high
    resolution :class:`Data` objects."""

    def __init__(
        self,
        container: Container,
        sample_shape,
        s_enhance,
        t_enhance,
        feature_sets: Optional[Dict] = None,
    ):
        """
        Parameters
        ----------
        container : Container
            Container instance with `.data = (low_res, high_res)`, with each
            tuple member a :class:`Data` instance.
        sample_shape : tuple
            Size of arrays to sample from the high-res data. The sample shape
            for the low-res sampler will be determined from the enhancement
            factors.
        s_enhance : int
            Spatial enhancement factor
        t_enhance : int
            Temporal enhancement factor
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
        feature_sets = feature_sets or {}
        self.hr_sample_shape = sample_shape
        self.lr_sample_shape = (
            sample_shape[0] // s_enhance,
            sample_shape[1] // s_enhance,
            sample_shape[2] // t_enhance,
        )
        self._lr_only_features = feature_sets.get('lr_only_features', [])
        self._hr_exo_features = feature_sets.get('hr_exo_features', [])
        msg = (
            'DualSampler requires a low-res and high-res Data object. '
            'Recieved an inconsistent Container.'
        )
        assert container.data.n_members == 2, msg
        self.lr_data, self.hr_data = container.data
        self.lr_sampler = Sampler(
            self.lr_data, sample_shape=self.lr_sample_shape
        )
        features = list(copy.deepcopy(self.lr_data.features))
        features += [fn for fn in self.hr_data.features if fn not in features]
        self.features = features
        self.lr_features = self.lr_data.features
        self.hr_features = self.hr_data.features
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        self.check_for_consistent_shapes()
        super().__init__(container.data, sample_shape=sample_shape)

    def check_for_consistent_shapes(self):
        """Make sure container shapes are compatible with enhancement
        factors."""
        enhanced_shape = (
            self.lr_data.shape[0] * self.s_enhance,
            self.lr_data.shape[1] * self.s_enhance,
            self.lr_data.shape[2] * self.t_enhance,
        )
        msg = (
            f'hr_data.shape {self.hr_data.shape} and enhanced '
            f'lr_data.shape {enhanced_shape} are not compatible with '
            'the given enhancement factors'
        )
        assert self.hr_data.shape[:3] == enhanced_shape, msg

    def get_sample_index(self):
        """Get paired sample index, consisting of index for the low res sample
        and the index for the high res sample with the same spatiotemporal
        extent."""
        lr_index = self.lr_sampler.get_sample_index()
        hr_index = [
            slice(s.start * self.s_enhance, s.stop * self.s_enhance)
            for s in lr_index[:2]
        ]
        hr_index += [
            slice(s.start * self.t_enhance, s.stop * self.t_enhance)
            for s in lr_index[2:-1]
        ]
        hr_index = (*hr_index, self.hr_features)
        return (lr_index, hr_index)