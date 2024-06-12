"""Data centric sampler. This samples container data according to weights
which are updated during training based on performance of the model."""

import logging

from sup3r.preprocessing.samplers.base import Sampler
from sup3r.utilities.utilities import (
    uniform_box_sampler,
    uniform_time_sampler,
    weighted_box_sampler,
    weighted_time_sampler,
)

logger = logging.getLogger(__name__)


class DataCentricSampler(Sampler):
    """DataCentric Sampler class used for sampling based on weights which can
    be updated during training."""

    def __init__(
        self,
        data,
        sample_shape,
        feature_sets,
        space_weights=None,
        time_weights=None,
    ):
        self.space_weights = space_weights or [1]
        self.time_weights = time_weights or [1]
        super().__init__(
            data=data, sample_shape=sample_shape, feature_sets=feature_sets
        )

    def update_weights(self, space_weights, time_weights):
        """Update spatial and temporal sampling weights."""
        self.space_weights = space_weights
        self.time_weights = time_weights

    def get_sample_index(self, time_weights=None, space_weights=None):
        """Randomly gets weighted spatial sample and time sample indices

        Parameters
        ----------
        time_weights : array
            Weights used to select time slice
            (n_time_chunks)
        space_weights : array
            Weights used to select spatial chunks
            (n_lat_chunks * n_lon_chunks)

        Returns
        -------
        observation_index : tuple
            Tuple of sampled spatial grid, time slice, and features indices.
            Used to get single observation like self.data[observation_index]
        """
        if space_weights is not None:
            spatial_slice = weighted_box_sampler(
                self.shape, self.sample_shape[:2], weights=space_weights
            )
        else:
            spatial_slice = uniform_box_sampler(
                self.shape, self.sample_shape[:2]
            )
        if time_weights is not None:
            time_slice = weighted_time_sampler(
                self.shape, self.sample_shape[2], weights=time_weights
            )
        else:
            time_slice = uniform_time_sampler(self.shape, self.sample_shape[2])

        return (*spatial_slice, time_slice, self.features)

    def __next__(self):
        """Get data for observation using weighted random observation index.
        Loops repeatedly over randomized time index.

        Parameters
        ----------
        time_weights : array
            Weights used to select time slice
            (n_time_chunks)
        space_weights : array
            Weights used to select spatial chunks
            (n_lat_chunks * n_lon_chunks)

        Returns
        -------
        observation : T_Array
            4D array
            (spatial_1, spatial_2, temporal, features)
        """
        return self.data[
            self.get_sample_index(
                time_weights=self.time_weights,
                space_weights=self.space_weights,
            )
        ]
