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

    def run(self):
        logger.debug('Starting linear correction calculation...')
