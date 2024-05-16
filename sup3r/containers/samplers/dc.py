"""Sampler objects. These take in data objects / containers and can them sample
from them. These samples can be used to build batches."""

import logging

import numpy as np

from sup3r.containers import Sampler
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

    def __init__(self, data, sample_shape, feature_sets):
        super().__init__(
            data=data, sample_shape=sample_shape, feature_sets=feature_sets
        )

    def get_sample_index(self, temporal_weights=None, spatial_weights=None):
        """Randomly gets weighted spatial sample and time sample indices

        Parameters
        ----------
        temporal_weights : array
            Weights used to select time slice
            (n_time_chunks)
        spatial_weights : array
            Weights used to select spatial chunks
            (n_lat_chunks * n_lon_chunks)

        Returns
        -------
        observation_index : tuple
            Tuple of sampled spatial grid, time slice, and features indices.
            Used to get single observation like self.data[observation_index]
        """
        if spatial_weights is not None:
            spatial_slice = weighted_box_sampler(
                self.shape, self.sample_shape[:2], weights=spatial_weights
            )
        else:
            spatial_slice = uniform_box_sampler(
                self.shape, self.sample_shape[:2]
            )
        if temporal_weights is not None:
            time_slice = weighted_time_sampler(
                self.shape, self.sample_shape[2], weights=temporal_weights
            )
        else:
            time_slice = uniform_time_sampler(
                self.shape, self.sample_shape[2]
            )

        return (*spatial_slice, time_slice, np.arange(len(self.features)))

    def get_next(self, temporal_weights=None, spatial_weights=None):
        """Get data for observation using weighted random observation index.
        Loops repeatedly over randomized time index.

        Parameters
        ----------
        temporal_weights : array
            Weights used to select time slice
            (n_time_chunks)
        spatial_weights : array
            Weights used to select spatial chunks
            (n_lat_chunks * n_lon_chunks)

        Returns
        -------
        observation : np.ndarray
            4D array
            (spatial_1, spatial_2, temporal, features)
        """
        return self[
            self.get_sample_index(
                temporal_weights=temporal_weights,
                spatial_weights=spatial_weights,
            )
        ]
