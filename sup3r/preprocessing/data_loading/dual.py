"""Dual data handler class for using separate low_res and high_res datasets"""
import logging

import numpy as np

from sup3r.preprocessing.mixin import DualMixIn

logger = logging.getLogger(__name__)


class LazyDualLoader(DualMixIn):
    """Lazy loading dual data handler. Matches sample regions for low res and
    high res lazy data handlers."""

    def __init__(self, lr_handler, hr_handler, s_enhance=1, t_enhance=1):
        self.lr_dh = lr_handler
        self.hr_dh = hr_handler
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        self.current_obs_index = None
        self._means = None
        self._stds = None
        self.check_shapes()
        DualMixIn.__init__(self, lr_handler, hr_handler)

        logger.info(f'Finished initializing {self.__class__.__name__}.')

    @property
    def means(self):
        """Get dictionary of means for all features available in low-res and
        high-res handlers."""
        if self._means is None:
            lr_features = self.lr_dh.features
            hr_only_features = [f for f in self.hr_dh.features
                                if f not in lr_features]
            self._means = dict(zip(lr_features,
                                   self.lr_dh.data[lr_features].mean(axis=0)))
            hr_means = dict(zip(hr_only_features,
                                self.hr_dh[hr_only_features].mean(axis=0)))
            self._means.update(hr_means)
        return self._means

    @property
    def stds(self):
        """Get dictionary of standard deviations for all features available in
        low-res and high-res handlers."""
        if self._stds is None:
            lr_features = self.lr_dh.features
            hr_only_features = [f for f in self.hr_dh.features
                                if f not in lr_features]
            self._stds = dict(zip(lr_features,
                              self.lr_dh.data[lr_features].std(axis=0)))
            hr_stds = dict(zip(hr_only_features,
                               self.hr_dh[hr_only_features].std(axis=0)))
            self._stds.update(hr_stds)
        return self._stds

    def __iter__(self):
        self._i = 0
        return self

    def __len__(self):
        return self.epoch_samples

    @property
    def size(self):
        """'Size' of data handler. Used to compute handler weights for batch
        sampling."""
        return np.prod(self.lr_dh.shape)

    def check_shapes(self):
        """Make sure data handler shapes are compatible with enhancement
        factors."""
        hr_shape = self.hr_dh.shape
        lr_shape = self.lr_dh.shape
        enhanced_shape = (lr_shape[0] * self.s_enhance,
                          lr_shape[1] * self.s_enhance,
                          lr_shape[2] * self.t_enhance)
        msg = (f'hr_dh.shape {hr_shape} and enhanced lr_dh.shape '
               f'{enhanced_shape} are not compatible')
        assert hr_shape == enhanced_shape, msg

    def get_next(self):
        """Get next pair of low-res / high-res samples ensuring that low-res
        and high-res sampling regions match.

        Returns
        -------
        tuple
            (low_res, high_res) pair
        """
        lr_obs_idx, hr_obs_idx = self.get_index_pair(self.lr_dh.shape,
                                                     self.lr_sample_shape,
                                                     s_enhance=self.s_enhance,
                                                     t_enhance=self.t_enhance)

        out = (self.lr_dh.get_observation(lr_obs_idx[:-1]),
               self.hr_dh.get_observation(hr_obs_idx[:-1]))
        return out

