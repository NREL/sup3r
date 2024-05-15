"""Base data handling classes.
@author: bbenton
"""
import logging
from typing import ClassVar

import numpy as np

from sup3r.preprocessing.data_handling.base import DataHandler
from sup3r.preprocessing.derived_features import (
    LatLonNC,
    PressureNC,
    UWind,
    VWind,
    WinddirectionNC,
    WindspeedNC,
)
from sup3r.utilities.utilities import (
    uniform_box_sampler,
    uniform_time_sampler,
    weighted_box_sampler,
    weighted_time_sampler,
)

np.random.seed(42)

logger = logging.getLogger(__name__)


# pylint: disable=W0223
class DataHandlerDC(DataHandler):
    """Data-centric data handler"""

    FEATURE_REGISTRY: ClassVar[dict] = {
        'U_(.*)': UWind,
        'V_(.*)': VWind,
        'Windspeed_(.*)m': WindspeedNC,
        'Winddirection_(.*)m': WinddirectionNC,
        'lat_lon': LatLonNC,
        'Pressure_(.*)m': PressureNC,
        'topography': ['HGT', 'orog']
    }

    def get_observation_index(self,
                              temporal_weights=None,
                              spatial_weights=None):
        """Randomly gets weighted spatial sample and time sample

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
            spatial_slice = weighted_box_sampler(self.data.shape,
                                                 self.sample_shape[:2],
                                                 weights=spatial_weights)
        else:
            spatial_slice = uniform_box_sampler(self.data.shape,
                                                self.sample_shape[:2])
        if temporal_weights is not None:
            temporal_slice = weighted_time_sampler(self.data.shape,
                                                   self.sample_shape[2],
                                                   weights=temporal_weights)
        else:
            temporal_slice = uniform_time_sampler(self.data.shape,
                                                  self.sample_shape[2])

        return (*spatial_slice, temporal_slice, np.arange(len(self.features)))

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
        self.current_obs_index = self.get_observation_index(
            temporal_weights=temporal_weights, spatial_weights=spatial_weights)
        return self.data[self.current_obs_index]
