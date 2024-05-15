"""Collection object with methods to compute and save stats."""

import json
import logging
import os
from typing import List

import numpy as np
from rex import safe_json_load

from sup3r.containers.collections.base import Collection
from sup3r.containers.wranglers import Wrangler

logger = logging.getLogger(__name__)


class StatsCollection(Collection):
    """Extended collection object with methods for computing means and stds and
    saving these to files."""

    def __init__(
        self, containers: List[Wrangler], means_file=None, stds_file=None
    ):
        super().__init__(containers)
        self.means = self.get_means(means_file)
        self.stds = self.get_stds(stds_file)
        self.save_stats(stds_file=stds_file, means_file=means_file)

    def get_means(self, means_file):
        """Dictionary of means for each feature, computed across all data
        handlers."""
        if means_file is None or not os.path.exists(means_file):
            means = {}
            for fidx, feat in enumerate(self.containers[0].features):
                means[feat] = np.sum(
                    [
                        self.data[cidx][..., fidx].mean() * wgt
                        for cidx, wgt in enumerate(self.container_weights)
                    ]
                )
        else:
            means = safe_json_load(means_file)
        return means

    def get_stds(self, stds_file):
        """Dictionary of standard deviations for each feature, computed across
        all data handlers."""
        if stds_file is None or not os.path.exists(stds_file):
            stds = {}
            for fidx, feat in enumerate(self.containers[0].features):
                stds[feat] = np.sqrt(
                    np.sum(
                        [
                            self.data[cidx][..., fidx].std() ** 2 * wgt
                            for cidx, wgt in enumerate(self.container_weights)
                        ]
                    )
                )
        else:
            stds = safe_json_load(stds_file)
        return stds

    def save_stats(self, stds_file, means_file):
        """Save stats to json files."""
        if stds_file is not None and not os.path.exists(stds_file):
            with open(stds_file, 'w') as f:
                f.write(json.dumps(self.stds))
                logger.info(f'Saved standard deviations to {stds_file}.')
        if means_file is not None and not os.path.exists(means_file):
            with open(means_file, 'w') as f:
                f.write(json.dumps(self.means))
                logger.info(f'Saved means to {means_file}.')
