"""Data handling for H5 files.
@author: bbenton
"""

import copy
import logging

import numpy as np
from rex import MultiFileNSRDBX, MultiFileWindX

from sup3r.containers import DataHandlerH5
from sup3r.utilities.utilities import (
    daily_temporal_coarsening,
    uniform_box_sampler,
)

np.random.seed(42)

logger = logging.getLogger(__name__)


class DataHandlerH5WindCC(DataHandlerH5):
    """Special data handling and batch sampling for h5 wtk or nsrdb data for
    climate change applications"""

    # the handler from rex to open h5 data.
    REX_HANDLER = MultiFileWindX

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args : list
            Same positional args as DataHandlerH5
        **kwargs : dict
            Same keyword args as DataHandlerH5
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

        # validation splits not enabled for solar CC model.
        kwargs['val_split'] = 0.0

        super().__init__(*args, **kwargs)

        self.daily_data = None
        self.daily_data_slices = None
        self.run_daily_averages()

    def run_daily_averages(self):
        """Calculate daily average data and store as attribute."""
        msg = ('Data needs to be hourly with at least 24 hours, but data '
               'shape is {}.'.format(self.data.shape))
        assert self.data.shape[2] % 24 == 0, msg
        assert self.data.shape[2] > 24, msg

        n_data_days = int(self.data.shape[2] / 24)
        daily_data_shape = (*self.data.shape[0:2], n_data_days,
                            self.data.shape[3])

        logger.info('Calculating daily average datasets for {} training '
                    'data days.'.format(n_data_days))

        self.daily_data = np.zeros(daily_data_shape, dtype=np.float32)

        self.daily_data_slices = np.array_split(np.arange(self.data.shape[2]),
                                                n_data_days)
        self.daily_data_slices = [
            slice(x[0], x[-1] + 1) for x in self.daily_data_slices
        ]
        for idf, fname in enumerate(self.features):
            for d, t_slice in enumerate(self.daily_data_slices):
                if '_max_' in fname:
                    tmp = np.max(self.data[:, :, t_slice, idf], axis=2)
                    self.daily_data[:, :, d, idf] = tmp[:, :]
                elif '_min_' in fname:
                    tmp = np.min(self.data[:, :, t_slice, idf], axis=2)
                    self.daily_data[:, :, d, idf] = tmp[:, :]
                else:
                    tmp = daily_temporal_coarsening(self.data[:, :, t_slice,
                                                              idf],
                                                    temporal_axis=2)
                    self.daily_data[:, :, d, idf] = tmp[:, :, 0]

        logger.info('Finished calculating daily average datasets for {} '
                    'training data days.'.format(n_data_days))

    def get_observation_index(self):
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
        t_slice_0 = self.daily_data_slices[rand_day_ind]
        t_slice_1 = self.daily_data_slices[rand_day_ind + n_days - 1]
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
        obs_ind_hourly, obs_ind_daily = self.get_observation_index()
        self.current_obs_index = obs_ind_hourly
        obs_hourly = self.data[obs_ind_hourly]
        obs_daily_avg = self.daily_data[obs_ind_daily]
        return obs_hourly, obs_daily_avg


class DataHandlerH5SolarCC(DataHandlerH5WindCC):
    """Special data handling and batch sampling for h5 NSRDB solar data for
    climate change applications"""

    # the handler from rex to open h5 data.
    REX_HANDLER = MultiFileNSRDBX

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args : list
            Same positional args as DataHandlerH5
        **kwargs : dict
            Same keyword args as DataHandlerH5
        """

        args = copy.deepcopy(args)  # safe copy for manipulation
        required = ['ghi', 'clearsky_ghi', 'clearsky_ratio']
        missing = [dset for dset in required if dset not in args[1]]
        if any(missing):
            msg = ('Cannot initialize DataHandlerH5SolarCC without required '
                   'features {}. All three are necessary to get the daily '
                   'average clearsky ratio (ghi sum / clearsky ghi sum), '
                   'even though only the clearsky ratio will be passed to the '
                   'GAN.'.format(required))
            logger.error(msg)
            raise KeyError(msg)

        super().__init__(*args, **kwargs)

    def run_daily_averages(self):
        """Calculate daily average data and store as attribute.

        Note that the H5 clearsky ratio feature requires special logic to match
        the climate change dataset of daily average GHI / daily average CS_GHI.
        This target climate change dataset is not equivalent to the average of
        instantaneous hourly clearsky ratios
        """

        msg = ('Data needs to be hourly with at least 24 hours, but data '
               'shape is {}.'.format(self.data.shape))
        assert self.data.shape[2] % 24 == 0, msg
        assert self.data.shape[2] > 24, msg

        n_data_days = int(self.data.shape[2] / 24)
        daily_data_shape = (*self.data.shape[0:2], n_data_days,
                            self.data.shape[3])

        logger.info('Calculating daily average datasets for {} training '
                    'data days.'.format(n_data_days))

        self.daily_data = np.zeros(daily_data_shape, dtype=np.float32)

        self.daily_data_slices = np.array_split(np.arange(self.data.shape[2]),
                                                n_data_days)
        self.daily_data_slices = [
            slice(x[0], x[-1] + 1) for x in self.daily_data_slices
        ]

        i_ghi = self.features.index('ghi')
        i_cs = self.features.index('clearsky_ghi')
        i_ratio = self.features.index('clearsky_ratio')

        for d, t_slice in enumerate(self.daily_data_slices):
            for idf in range(self.data.shape[-1]):
                self.daily_data[:, :, d, idf] = daily_temporal_coarsening(
                    self.data[:, :, t_slice, idf], temporal_axis=2)[:, :, 0]

            # note that this ratio of daily irradiance sums is not the same as
            # the average of hourly ratios.
            total_ghi = np.nansum(self.data[:, :, t_slice, i_ghi], axis=2)
            total_cs_ghi = np.nansum(self.data[:, :, t_slice, i_cs], axis=2)
            avg_cs_ratio = total_ghi / total_cs_ghi
            self.daily_data[:, :, d, i_ratio] = avg_cs_ratio

        # remove ghi and clearsky ghi from feature set. These shouldn't be used
        # downstream for solar cc and keeping them confuses the batch handler
        logger.info('Finished calculating daily average clearsky_ratio, '
                    'removing ghi and clearsky_ghi from the '
                    'DataHandlerH5SolarCC feature list.')
        ifeats = np.array(
            [i for i in range(len(self.features)) if i not in (i_ghi, i_cs)])
        self.data = self.data[..., ifeats]
        self.daily_data = self.daily_data[..., ifeats]
        self.features.remove('ghi')
        self.features.remove('clearsky_ghi')

        logger.info('Finished calculating daily average datasets for {} '
                    'training data days.'.format(n_data_days))
