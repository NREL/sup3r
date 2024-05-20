"""
Sup3r batch_handling module.
@author: bbenton
"""

import logging

import numpy as np
from scipy.ndimage import gaussian_filter

from sup3r.containers import BatchHandler
from sup3r.utilities.utilities import (
    nn_fill_array,
    nsrdb_reduce_daily_data,
    spatial_coarsening,
)

np.random.seed(42)

logger = logging.getLogger(__name__)


class BatchHandlerCC(BatchHandler):
    """Batch handling class for climate change data with daily averages as the
    coarse dataset."""

    def __init__(self, *args, sub_daily_shape=None, **kwargs):
        """
        Parameters
        ----------
        *args : list
            Same positional args as BatchHandler
        sub_daily_shape : int
            Number of hours to use in the high res sample output. This is the
            shape of the temporal dimension of the high res batch observation.
            This time window will be sampled for the daylight hours on the
            middle day of the data handler observation.
        **kwargs : dict
            Same keyword args as BatchHandler
        """
        super().__init__(*args, **kwargs)
        self.sub_daily_shape = sub_daily_shape

    def __next__(self):
        """Get the next iterator output.

        Returns
        -------
        batch : Batch
            Batch object with batch.low_res and batch.high_res attributes
            with the appropriate coarsening.
        """
        self.current_batch_indices = []

        if self._i >= self.n_batches:
            raise StopIteration

        handler = self.get_random_container()

        low_res = None
        high_res = None

        for i in range(self.batch_size):
            obs_hourly, obs_daily_avg = handler.get_next()
            self.current_batch_indices.append(handler.current_obs_index)

            obs_hourly = obs_hourly[..., self.hr_features_ind]

            if low_res is None:
                lr_shape = (self.batch_size, *obs_daily_avg.shape)
                hr_shape = (self.batch_size, *obs_hourly.shape)
                low_res = np.zeros(lr_shape, dtype=np.float32)
                high_res = np.zeros(hr_shape, dtype=np.float32)

            low_res[i] = obs_daily_avg
            high_res[i] = obs_hourly

        high_res = self.reduce_high_res_sub_daily(high_res)
        low_res = spatial_coarsening(low_res, self.s_enhance)

        if (
            self.hr_out_features is not None
            and 'clearsky_ratio' in self.hr_out_features
        ):
            i_cs = self.hr_out_features.index('clearsky_ratio')
            if np.isnan(high_res[..., i_cs]).any():
                high_res[..., i_cs] = nn_fill_array(high_res[..., i_cs])

        if self.smoothing is not None:
            feat_iter = [
                j
                for j in range(low_res.shape[-1])
                if self.features[j] not in self.smoothing_ignore
            ]
            for i in range(low_res.shape[0]):
                for j in feat_iter:
                    low_res[i, ..., j] = gaussian_filter(
                        low_res[i, ..., j], self.smoothing, mode='nearest'
                    )

        batch = self.BATCH_CLASS(low_res, high_res)

        self._i += 1
        return batch

    def reduce_high_res_sub_daily(self, high_res):
        """Take an hourly high-res observation and reduce the temporal axis
        down to the self.sub_daily_shape using only daylight hours on the
        center day.

        Parameters
        ----------
        high_res : np.ndarray
            5D array with dimensions (n_obs, spatial_1, spatial_2, temporal,
            n_features) where temporal >= 24 (set by the data handler).

        Returns
        -------
        high_res : np.ndarray
            5D array with dimensions (n_obs, spatial_1, spatial_2, temporal,
            n_features) where temporal has been reduced down to the integer
            self.sub_daily_shape. For example if the input temporal shape is 72
            (3 days) and sub_daily_shape=9, the center daylight 9 hours from
            the second day will be returned in the output array.
        """

        if self.sub_daily_shape is not None:
            n_days = int(high_res.shape[3] / 24)
            if n_days > 1:
                ind = np.arange(high_res.shape[3])
                day_slices = np.array_split(ind, n_days)
                day_slices = [slice(x[0], x[-1] + 1) for x in day_slices]
                assert n_days % 2 == 1, 'Need odd days'
                i_mid = int((n_days - 1) / 2)
                high_res = high_res[:, :, :, day_slices[i_mid], :]

            high_res = nsrdb_reduce_daily_data(high_res, self.sub_daily_shape)

        return high_res
