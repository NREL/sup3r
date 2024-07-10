"""Collection object with methods to compute and save stats."""

import logging
import os

import numpy as np
from rex import safe_json_load

from sup3r.utilities.utilities import safe_serialize

from .base import Collection

logger = logging.getLogger(__name__)


class StatsCollection(Collection):
    """Extended collection object with methods for computing means and stds and
    saving these to files.

    Note
    ----
    We write stats as float64 because float32 is not json serializable
    """

    def __init__(self, containers, means=None, stds=None):
        """
        Parameters
        ----------
        containers: List[Extracter]
            List of containers to compute stats for.
        means : str | dict | None
            Usually a file path for saving results, or None for just
            calculating stats and not saving. Can also be a dict, which will
            just get returned as the "result".
        stds : str | dict | None
            Usually a file path for saving results, or None for just
            calculating stats and not saving. Can also be a dict, which will
            just get returned as the "result".
        """
        super().__init__(containers=containers)
        self.means = self.get_means(means)
        self.stds = self.get_stds(stds)
        self.save_stats(stds=stds, means=means)

    @staticmethod
    def container_mean(container, feature):
        """Method for computing means on containers, accounting for possible
        multi-dataset containers."""
        return container.data.high_res[feature].mean(skipna=True)

    @staticmethod
    def container_std(container, feature):
        """Method for computing stds on containers, accounting for possible
        multi-dataset containers."""
        return container.data.high_res[feature].std(skipna=True)

    def get_means(self, means):
        """Dictionary of means for each feature, computed across all data
        handlers."""
        if means is None or (
            isinstance(means, str) and not os.path.exists(means)
        ):
            means = {}
            for f in self.containers[0].data_vars:
                cmeans = [
                    w * self.container_mean(c, f)
                    for c, w in zip(self.containers, self.container_weights)
                ]
                means[f] = np.float32(np.sum(cmeans))
        elif isinstance(means, str):
            means = safe_json_load(means)
        return means

    def get_stds(self, stds):
        """Dictionary of standard deviations for each feature, computed across
        all data handlers."""
        if stds is None or (
            isinstance(stds, str) and not os.path.exists(stds)
        ):
            stds = {}
            for f in self.containers[0].data_vars:
                cstds = [
                    w * self.container_std(c, f) ** 2
                    for c, w in zip(self.containers, self.container_weights)
                ]
                stds[f] = np.float32(np.sqrt(np.sum(cstds)))
        elif isinstance(stds, str):
            stds = safe_json_load(stds)
        return stds

    def save_stats(self, stds, means):
        """Save stats to json files."""
        if isinstance(stds, str) and not os.path.exists(stds):
            with open(stds, 'w') as f:
                f.write(safe_serialize(self.stds))
                logger.info(
                    f'Saved standard deviations {self.stds} to {stds}.'
                )
        if isinstance(means, str) and not os.path.exists(means):
            with open(means, 'w') as f:
                f.write(safe_serialize(self.means))
                logger.info(f'Saved means {self.means} to {means}.')
