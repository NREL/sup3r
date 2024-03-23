"""Quantile Delta Mapping
"""

import logging
from glob import glob

import numpy as np

from .distribution import EmpiricalDistribution
from .bias_calc import DataRetrievalBase

logger = logging.getLogger(__name__)


class QuantileDeltaMapping(DataRetrievalBase):
    def __init__(self, base_fps, bias_fps, bias_fut_fps, *args, **kwargs):
        super().__init__(base_fps, bias_fps, *args, **kwargs)

        self.bias_fut_fps = bias_fps

        if isinstance(self.bias_fut_fps, str):
            self.bias_fut_fps = sorted(glob(self.bias_fut_fps))

        self.bias_fut_dh = self.bias_handler(self.bias_fut_fps,
                                             [self.bias_feature],
                                             target=self.target,
                                             shape=self.shape,
                                             val_split=0.0,
                                             **self.bias_handler_kwargs)

        assert np.allclose(self.bias_dh.lat_lon, self.bias_fut_dh.lat_lon)


    def get_base_data(self, base_gid, daily_reduction):
        """Get base data for given GID

        For now, there is no need to change most of the arguments, thus
        this wrapper simplifies the DataRetrievalBase.get_base_data()
        by using its context.

        Parameters
        ----------

        Return
        ------

        """
        base_data, _ = super().get_base_data(self.base_fps,
                                             self.base_dset,
                                             base_gid,
                                             self.base_handler,
                                             base_handler_kwargs=None,
                                             daily_reduction=daily_reduction,
                                             decimals=self.decimals,
                                             base_dh_inst=self.base_dh
                                             )
        return base_data

    def run(self, daily_reduction='avg'):
        """

        Reminders:
        - Options between relative or absolute correction
        """
        logger.debug("Starting linear correction calculation...")

        logger.debug('Running serial calculation.')
        self.NT = 51
        keys = ["base_CDF", "bias_CDF", "bias_fut_CDF"]
        self.out = {k: np.full((*self.bias_gid_raster.shape, self.NT),
                               np.nan, np.float32)
                    for k in keys}

        for i, bias_gid in enumerate(self.bias_meta.index):
            raster_loc = np.where(self.bias_gid_raster == bias_gid)
            _, base_gid = self.get_base_gid(bias_gid)

            if not base_gid.any():
                self.bad_bias_gids.append(bias_gid)
                logger.debug(f"No base data for bias_gid: {bias_gid}. "
                             "Adding it to bad_bias_gids")
            else:
                bias_data = self.get_bias_data(bias_gid)
                bias_fut_data = self.get_bias_data(bias_gid, self.bias_fut_dh)
                base_data = self.get_base_data(base_gid, daily_reduction)

                D_base = EmpiricalDistribution.from_fit(base_data, self.NT)
                self.out['base_CDF'][raster_loc] = D_base.cut_point

                D_bias = EmpiricalDistribution.from_fit(bias_data, self.NT)
                self.out['bias_CDF'][raster_loc] = D_bias.cut_point
                D_bias_fut = EmpiricalDistribution.from_fit(bias_fut_data, self.NT)
                self.out['bias_fut_CDF'][raster_loc] = D_bias_fut.cut_point

            logger.info('Completed bias calculations for {} out of {} '
                            'sites'.format(i + 1, len(self.bias_meta)))
