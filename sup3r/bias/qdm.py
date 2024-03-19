"""Quantile Delta Mapping
"""

import logging

from .bias_calc import DataRetrievalBase

logger = logging.getLogger(__name__)


class QuantileDeltaMapping(DataRetrievalBase):
    def run(self):
        logger.debug('Starting linear correction calculation...')
