"""Quantile Delta Mapping
"""

import logging

from .bias_calc import DataRetrievalBase

logger = logging.getLogger(__name__)


class QuantileDeltaMapping(DataRetrievalBase):
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
