"""Data handling for H5 files.
@author: bbenton
"""

import logging

import numpy as np

from sup3r.preprocessing.samplers.base import Sampler
from sup3r.utilities.utilities import (
    uniform_box_sampler,
)

np.random.seed(42)

logger = logging.getLogger(__name__)


class SamplerH5CC(Sampler):
    """Special sampling for h5 wtk or nsrdb data for climate change
    applications"""

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args : list
            Same positional args as Sampler
        **kwargs : dict
            Same keyword args as Sampler
        """
        sample_shape = kwargs.get('sample_shape', (10, 10, 24))
        t_shape = sample_shape[-1]

        if len(sample_shape) == 2:
            logger.info(
                'Found 2D sample shape of {}. Adding spatial dim of 24'.format(
                    sample_shape))
            sample_shape = (*sample_shape, 24)
            t_shape = sample_shape[-1]
            kwargs['sample_shape'] = sample_shape

        if t_shape < 24 or t_shape % 24 != 0:
            msg = ('Climate Change DataHandler can only work with temporal '
                   'sample shapes that are one or more days of hourly data '
                   '(e.g. 24, 48, 72...). The requested temporal sample '
                   'shape was: {}'.format(t_shape))
            logger.error(msg)
            raise RuntimeError(msg)

        super().__init__(*args, **kwargs)

    def get_sample_index(self):
        """Randomly gets spatial sample and time sample

        Returns
        -------
        obs_ind_hourly : tuple
            Tuple of sampled spatial grid, time slice, and features indices.
            Used to get single observation like self.data[observation_index].
            This is for hourly high-res data slicing.
        obs_ind_daily : tuple
            Same as obs_ind_hourly but the temporal index (i=2) is a slice of
            the daily data (self.daily_data) with day integers.
        """
        spatial_slice = uniform_box_sampler(self.data.shape,
                                            self.sample_shape[:2])

        n_days = int(self.sample_shape[2] / 24)
        rand_day_ind = np.random.choice(len(self.daily_data_slices) - n_days)
        t_slice_0 = self.container.daily_data_slices[rand_day_ind]
        t_slice_1 = self.container.daily_data_slices[rand_day_ind + n_days - 1]
        t_slice_hourly = slice(t_slice_0.start, t_slice_1.stop)
        t_slice_daily = slice(rand_day_ind, rand_day_ind + n_days)

        obs_ind_hourly = (*spatial_slice, t_slice_hourly,
                          np.arange(len(self.features)))

        obs_ind_daily = (*spatial_slice, t_slice_daily,
                         np.arange(len(self.features)))

        return obs_ind_hourly, obs_ind_daily

    def get_next(self):
        """Get data for observation using random observation index. Loops
        repeatedly over randomized time index

        Returns
        -------
        obs_hourly : np.ndarray
            4D array
            (spatial_1, spatial_2, temporal_hourly, features)
        obs_daily_avg : np.ndarray
            4D array but the temporal axis is temporal_hourly//24
            (spatial_1, spatial_2, temporal_daily, features)
        """
        obs_ind_hourly, obs_ind_daily = self.get_sample_index()
        obs_hourly = self.data[obs_ind_hourly]
        obs_daily_avg = self.container.daily_data[obs_ind_daily]
        return obs_hourly, obs_daily_avg
