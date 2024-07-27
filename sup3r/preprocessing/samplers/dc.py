"""Data centric sampler. This samples container data according to weights
which are updated during training based on performance of the model."""

import logging
from typing import Dict, List, Optional, Union

from sup3r.preprocessing.samplers.base import Sampler
from sup3r.preprocessing.samplers.utilities import (
    uniform_box_sampler,
    uniform_time_sampler,
    weighted_box_sampler,
    weighted_time_sampler,
)
from sup3r.typing import T_Array, T_Dataset

logger = logging.getLogger(__name__)


class SamplerDC(Sampler):
    """DataCentric Sampler class used for sampling based on weights which can
    be updated during training."""

    def __init__(
        self,
        data: T_Dataset,
        sample_shape,
        batch_size: int = 16,
        feature_sets: Optional[Dict] = None,
        spatial_weights: Optional[Union[T_Array, List]] = None,
        temporal_weights: Optional[Union[T_Array, List]] = None,
    ):
        self.spatial_weights = spatial_weights or [1]
        self.temporal_weights = temporal_weights or [1]
        super().__init__(
            data=data,
            sample_shape=sample_shape,
            batch_size=batch_size,
            feature_sets=feature_sets,
        )

    def update_weights(self, spatial_weights, temporal_weights):
        """Update spatial and temporal sampling weights."""
        self.spatial_weights = spatial_weights
        self.temporal_weights = temporal_weights

    def get_sample_index(self, n_obs=None):
        """Randomly gets weighted spatial sample and time sample indices

        Returns
        -------
        observation_index : tuple
            Tuple of sampled spatial grid, time slice, and features indices.
            Used to get single observation like self.data[observation_index]
        """
        n_obs = n_obs or self.batch_size
        if self.spatial_weights is not None:
            spatial_slice = weighted_box_sampler(
                self.shape, self.sample_shape[:2], weights=self.spatial_weights
            )
        else:
            spatial_slice = uniform_box_sampler(
                self.shape, self.sample_shape[:2]
            )
        if self.temporal_weights is not None:
            time_slice = weighted_time_sampler(
                self.shape,
                self.sample_shape[2] * n_obs,
                weights=self.temporal_weights,
            )
        else:
            time_slice = uniform_time_sampler(
                self.shape, self.sample_shape[2] * n_obs
            )
        return (*spatial_slice, time_slice, self.features)
