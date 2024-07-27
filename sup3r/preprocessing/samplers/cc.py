"""Sampler for climate change applications."""

import logging
from typing import Dict, Optional

import numpy as np

from sup3r.preprocessing.base import Sup3rDataset
from sup3r.preprocessing.samplers.dual import DualSampler
from sup3r.preprocessing.samplers.utilities import nsrdb_reduce_daily_data
from sup3r.preprocessing.utilities import Dimension, _compute_if_dask
from sup3r.utilities.utilities import nn_fill_array

logger = logging.getLogger(__name__)


class DualSamplerCC(DualSampler):
    """Special sampling of WTK or NSRDB data for climate change applications

    Note
    ----
    This will always give daily / hourly data if `t_enhance != 1`. The number
    of days / hours in the samples is determined by t_enhance. For example, if
    `t_enhance = 8` and `sample_shape = (..., 24)` there will be 3 days in the
    low res sample: `lr_sample_shape = (..., 3)`. If `t_enhance != 24` and > 1
    :meth:`reduce_high_res_sub_daily` will be used to reduce a high res sample
    shape from `(..., sample_shape[2] * 24 // t_enhance)` to `(...,
    sample_shape[2])`
    """

    def __init__(
        self,
        data: Sup3rDataset,
        sample_shape,
        batch_size: int = 16,
        s_enhance: int = 1,
        t_enhance: int = 24,
        feature_sets: Optional[Dict] = None,
    ):
        """
        See Also
        --------
        :class:`DualSampler` for argument descriptions.
        """
        msg = (
            f'{self.__class__.__name__} requires a Sup3rDataset object '
            'with `.daily` and `.hourly` data members, in that order'
        )
        assert hasattr(data, 'daily') and hasattr(data, 'hourly'), msg
        lr, hr = data.daily, data.hourly
        assert lr == data[0] and hr == data[1], msg
        if t_enhance == 1:
            hr = data.daily
        if s_enhance > 1:
            lr = lr.coarsen(
                {
                    Dimension.SOUTH_NORTH: s_enhance,
                    Dimension.WEST_EAST: s_enhance,
                }
            ).mean()
        data = Sup3rDataset(low_res=lr, high_res=hr)
        super().__init__(
            data=data,
            sample_shape=sample_shape,
            batch_size=batch_size,
            t_enhance=t_enhance,
            s_enhance=s_enhance,
            feature_sets=feature_sets,
        )

    def check_for_consistent_shapes(self):
        """Make sure container shapes are compatible with enhancement
        factors."""
        enhanced_shape = (
            self.lr_data.shape[0] * self.s_enhance,
            self.lr_data.shape[1] * self.s_enhance,
            self.lr_data.shape[2] * (1 if self.t_enhance == 1 else 24),
        )
        msg = (
            f'hr_data.shape {self.hr_data.shape} and enhanced '
            f'lr_data.shape {enhanced_shape} are not compatible with '
            f'the given enhancement factors t_enhance = {self.t_enhance}, '
            f's_enhance = {self.s_enhance}'
        )
        assert self.hr_data.shape[:3] == enhanced_shape, msg

    def reduce_high_res_sub_daily(self, high_res, csr_ind=0):
        """Take an hourly high-res observation and reduce the temporal axis
        down to lr_sample_shape[2] * t_enhance time steps, using only daylight
        hours on the middle part of the high res data.

        Parameters
        ----------
        high_res : T_Array
            5D array with dimensions (n_obs, spatial_1, spatial_2, temporal,
            n_features) where temporal >= 24 (set by the data handler).
        csr_ind : int
            Feature index of clearsky_ratio. e.g. self.data[..., csr_ind] ->
            cs_ratio

        Returns
        -------
        high_res : T_Array
            5D array with dimensions (n_obs, spatial_1, spatial_2, temporal,
            n_features) where temporal has been reduced down to the integer
            lr_sample_shape[2] * t_enhance. For example if hr_sample_shape[2]
            is 9 and t_enhance = 8, 72 hourly time steps will be reduced to 9
            using the center daylight 9 hours from the second day.

        Note
        ----
        This only does something when `1 < t_enhance < 24.` If t_enhance = 24
        there is no need for reduction since every daily time step will have 24
        hourly time steps in the high_res batch data. Of course, if t_enhance =
        1, we are running for a spatial only model so this routine is
        unnecessary.

        *Needs review from @grantbuster
        """
        if self.t_enhance not in (24, 1):
            high_res = self.get_middle(high_res, self.hr_sample_shape)
            high_res = nsrdb_reduce_daily_data(
                high_res, self.hr_sample_shape[-1], csr_ind=csr_ind
            )

        return high_res

    @staticmethod
    def get_middle(high_res, sample_shape):
        """Get middle chunk of high_res data that will then be reduced to day
        time steps. This has n_time_steps = 24 if sample_shape[-1] <= 24
        otherwise n_time_steps = sample_shape[-1]."""
        n_days = int(high_res.shape[3] / 24)
        if n_days > 1:
            mid = int(np.ceil(high_res.shape[3] / 2))
            start = mid - np.max((sample_shape[-1] // 2, 12))
            t_slice = slice(start, start + np.max((sample_shape[-1], 12)))
            high_res = high_res[..., t_slice, :]
        return high_res

    def get_sample_index(self, n_obs=None):
        """Get sample index for expanded hourly chunk which will be reduced to
        the given sample shape."""
        lr_ind, hr_ind = super().get_sample_index(n_obs=n_obs)
        upsamp_factor = 1 if self.t_enhance == 1 else 24
        hr_ind = (
            *hr_ind[:2],
            slice(
                upsamp_factor * lr_ind[2].start, upsamp_factor * lr_ind[2].stop
            ),
            hr_ind[-1],
        )
        return lr_ind, hr_ind

    def _fast_batch_possible(self):
        upsamp_factor = 1 if self.t_enhance == 1 else 24
        return (
            upsamp_factor * self.lr_sample_shape[2] * self.batch_size
            <= self.data.shape[2]
        )

    def __next__(self):
        """Slight modification of `super().__next__()` to first get a sample of
        `shape = (..., hr_sample_shape[2] * 24 // t_enhance)` and then reduce
        this to `(..., hr_sample_shape[2])` with
        :func:`nsrdb_reduce_daily_data.` If this is for a spatial only model
        this subroutine is skipped."""
        low_res, high_res = super().__next__()
        if (
            self.hr_out_features is not None
            and 'clearsky_ratio' in self.hr_out_features
            and self.t_enhance != 1
        ):
            i_cs = self.hr_out_features.index('clearsky_ratio')
            high_res = self.reduce_high_res_sub_daily(high_res, csr_ind=i_cs)

            if np.isnan(high_res[..., i_cs]).any():
                high_res[..., i_cs] = nn_fill_array(
                    _compute_if_dask(high_res[..., i_cs])
                )

        return low_res, high_res
