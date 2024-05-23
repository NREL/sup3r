"""Collection object with methods to compute and save stats."""

import json
import logging
import os
from typing import List

import numpy as np
from rex import safe_json_load

from sup3r.containers.collections.base import Collection
from sup3r.containers.extracters import Extracter

logger = logging.getLogger(__name__)


class StatsCollection(Collection):
    """Extended collection object with methods for computing means and stds and
    saving these to files.

    Notes
    -----
    We write stats as float64 because float32 is not json serializable
    """

    def __init__(
        self, containers: List[Extracter], means_file=None, stds_file=None
    ):
        super().__init__(containers)
        self.means = self.get_means(means_file)
        self.stds = self.get_stds(stds_file)
        self.save_stats(stds_file=stds_file, means_file=means_file)

    @staticmethod
    def container_mean(container, feature):
        """Method for computing means on containers, accounting for possible
        multi-dataset containers."""
        if container.is_multi_container:
            return container.data[0][feature].mean()
        return container.data[feature].mean()

    @staticmethod
    def container_std(container, feature):
        """Method for computing stds on containers, accounting for possible
        multi-dataset containers."""
        if container.is_multi_container:
            return container.data[0][feature].std()
        return container.data[feature].std()

    def get_means(self, means_file):
        """Dictionary of means for each feature, computed across all data
        handlers."""
        if means_file is None or not os.path.exists(means_file):
            means = {}
            for f in self.containers[0].features:
                cmeans = [
                    w * self.container_mean(c, f)
                    for c, w in zip(self.containers, self.container_weights)
                ]
                means[f] = np.float64(np.sum(cmeans))
        else:
            means = safe_json_load(means_file)
        return means

    def get_stds(self, stds_file):
        """Dictionary of standard deviations for each feature, computed across
        all data handlers."""
        if stds_file is None or not os.path.exists(stds_file):
            stds = {}
            for f in self.containers[0].features:
                cstds = [
                    w * self.container_std(c, f) ** 2
                    for c, w in zip(self.containers, self.container_weights)
                ]
                stds[f] = np.float64(np.sqrt(np.sum(cstds)))
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
