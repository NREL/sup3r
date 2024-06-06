"""Data handling for H5 files.
@author: bbenton
"""

import logging
from typing import Dict, Optional

import numpy as np

from sup3r.preprocessing.base import Sup3rDataset
from sup3r.preprocessing.samplers.base import Sampler
from sup3r.utilities.utilities import (
    uniform_box_sampler,
)

np.random.seed(42)

logger = logging.getLogger(__name__)


class DualSamplerCC(Sampler):
    """Special sampling of WTK or NSRDB data for climate change applications

    Note
    ----
    This is a similar pattern to :class:`DualSampler` but different in
    important ways. We are grouping `daily_data` and `hourly_data` like
    `low_res` and `high_res` but `daily_data` is only the temporal low_res
    version of the hourly data. It will ultimately be coarsened spatially
    before constructing batches. Here we are constructing a sampler to sample
    the daily / hourly pairs so we use an "lr_sample_shape" which is only
    temporally low resolution.

    TODO: With the dalyed computations from dask we could spatially coarsen
    the daily data here and then use the standard `DualSampler` methods (most,
    anyway, `get_sample_index` would need to be slightly different, I
    think.). The call to `spatial_coarsening` in `coarsen` could then be
    removed and only temporal down sampling for the hourly data would be
    performed there.
    """

    def __init__(
        self,
        data: Sup3rDataset,
        sample_shape=None,
        s_enhance=1,
        t_enhance=24,
        feature_sets: Optional[Dict] = None,
    ):
        """
        Parameters
        ----------
        data : Sup3rDataset
            A tuple of xr.Dataset instances wrapped in the
            :class:`Sup3rDataset` interface. The first must be daily and the
            second must be hourly data
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
        msg = (f'{self.__class__.__name__} requires a Sup3rDataset object '
               'with `.daily` and `.hourly` data members, in that order')
        assert hasattr(data, 'daily') and hasattr(data, 'hourly'), msg
        assert data.daily == data[0] and data.hourly == data[1], msg
        n_hours = data.hourly.sizes['time']
        n_days = data.daily.sizes['time']
        self.daily_data_slices = [
            slice(x[0], x[-1] + 1)
            for x in np.array_split(np.arange(n_hours), n_days)
        ]
        sample_shape = (
            sample_shape if sample_shape is not None else (10, 10, 24)
        )
        sample_shape = self.check_sample_shape(sample_shape)
        self.hr_sample_shape = sample_shape
        self.lr_sample_shape = (
            sample_shape[0],
            sample_shape[1],
            sample_shape[2] // t_enhance,
        )
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        super().__init__(
            data=data,
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

        Note
        ----
        This pair of hourly and daily observation indices will be used to
        sample from self.data = (daily_data, hourly_data) through the standard
        :meth:`Container.__getitem__((obs_ind_daily, obs_ind_hourly))` This
        follows the pattern of (low-res, high-res) ordering.

        Returns
        -------
        obs_ind_daily : tuple
            Tuple of sampled spatial grid, time slice, and feature names.
            Used to get single observation like self.data[observation_index].
            Temporal index (i=2) is a slice of the daily data (self.daily_data)
            with day integers.
        obs_ind_hourly : tuple
            Tuple of sampled spatial grid, time slice, and feature names.
            Used to get single observation like self.data[observation_index].
            This is for hourly high-res data slicing.
        """
        spatial_slice = uniform_box_sampler(
            self.data.shape, self.sample_shape[:2]
        )

        n_days = int(self.sample_shape[2] / 24)
        rand_day_ind = np.random.choice(len(self.daily_data_slices) - n_days)
        t_slice_0 = self.daily_data_slices[rand_day_ind]
        t_slice_1 = self.daily_data_slices[rand_day_ind + n_days - 1]
        t_slice_hourly = slice(t_slice_0.start, t_slice_1.stop)
        t_slice_daily = slice(rand_day_ind, rand_day_ind + n_days)
        obs_ind_daily = (
            *spatial_slice,
            t_slice_daily,
            self.data.daily.features,
        )
        obs_ind_hourly = (
            *spatial_slice,
            t_slice_hourly,
            self.data.hourly.features,
        )

        return (obs_ind_daily, obs_ind_hourly)
