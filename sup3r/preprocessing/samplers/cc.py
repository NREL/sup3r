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


class SamplerH5forCC(Sampler):
    """Special sampling for h5 wtk or nsrdb data for climate change
    applications

    TODO: refactor according to DualSampler pattern. Maybe create base
    MixedSampler class since this wont be lr + hr but still has two data
    objects to sample from.
    """

    def __init__(self, container, sample_shape=None, feature_sets=None):
        """
        Parameters
        ----------
        container : DataHandler
            DataHandlerH5 type container. Needs to have `.daily_data` and
            `.daily_data_slices`. See `sup3r.preprocessing.data_handlers.h5_cc`
        """
        self.data = (container.data, container.daily_data)
        sample_shape = (
            sample_shape if sample_shape is not None else (10, 10, 24)
        )
        sample_shape = self.check_sample_shape(sample_shape)

        super().__init__(
            data=self.data,
            sample_shape=sample_shape,
            feature_sets=feature_sets,
        )

    @staticmethod
    def check_sample_shape(sample_shape):
        """Make sure sample_shape is consistent with required number of time
        steps in the sample data."""
        t_shape = sample_shape[-1]
        if len(sample_shape) == 2:
            logger.info(
                'Found 2D sample shape of {}. Adding spatial dim of 24'.format(
                    sample_shape
                )
            )
            sample_shape = (*sample_shape, 24)
            t_shape = sample_shape[-1]

        if t_shape < 24 or t_shape % 24 != 0:
            msg = (
                'Climate Change DataHandler can only work with temporal '
                'sample shapes that are one or more days of hourly data '
                '(e.g. 24, 48, 72...). The requested temporal sample '
                'shape was: {}'.format(t_shape)
            )
            logger.error(msg)
            raise RuntimeError(msg)
        return sample_shape

    def get_sample_index(self):
        """Randomly gets spatial sample and time sample.

        Notes
        -----
        This pair of hourly
        and daily observation indices will be used to sample from self.data =
        (hourly_data, daily_data) through the standard
        :meth:`Container.__getitem__((obs_ind_hourly, obs_ind_daily))`

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
        spatial_slice = uniform_box_sampler(
            self.data.shape, self.sample_shape[:2]
        )

        n_days = int(self.sample_shape[2] / 24)
        rand_day_ind = np.random.choice(
            len(self.container.daily_data_slices) - n_days
        )
        t_slice_0 = self.container.daily_data_slices[rand_day_ind]
        t_slice_1 = self.container.daily_data_slices[rand_day_ind + n_days - 1]
        t_slice_hourly = slice(t_slice_0.start, t_slice_1.stop)
        t_slice_daily = slice(rand_day_ind, rand_day_ind + n_days)

        obs_ind_hourly = (
            *spatial_slice,
            t_slice_hourly,
            np.arange(len(self.features)),
        )

        obs_ind_daily = (
            *spatial_slice,
            t_slice_daily,
            np.arange(len(self.features)),
        )

        return obs_ind_hourly, obs_ind_daily