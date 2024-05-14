"""Collection object with methods to compute and save stats."""
import json
import os

import numpy as np
from rex import safe_json_load

from sup3r.containers.collections import Collection


class StatsCollection(Collection):
    """Extended collection object with methods for computing means and stds and
    saving these to files."""

    def __init__(self, containers, means_file=None, stdevs_file=None):
        super().__init__(containers)
        self.means = self.get_means(means_file)
        self.stds = self.get_stds(stdevs_file)
        self.lr_means = np.array([self.means[k] for k in self.lr_features])
        self.lr_stds = np.array([self.stds[k] for k in self.lr_features])
        self.hr_means = np.array([self.means[k] for k in self.hr_features])
        self.hr_stds = np.array([self.stds[k] for k in self.hr_features])

    def get_means(self, means_file):
        """Dictionary of means for each feature, computed across all data
        handlers."""
        if means_file is None or not os.path.exists(means_file):
            means = {}
            for k in self.containers[0].features:
                means[k] = np.sum(
                    [c.means[k] * wgt for (wgt, c)
                     in zip(self.handler_weights, self.containers)])
        else:
            means = safe_json_load(means_file)
        return means

    def get_stds(self, stdevs_file):
        """Dictionary of standard deviations for each feature, computed across
        all data handlers."""
        if stdevs_file is None or not os.path.exists(stdevs_file):
            stds = {}
            for k in self.containers[0].features:
                stds[k] = np.sqrt(np.sum(
                    [c.stds[k]**2 * wgt for (wgt, c)
                     in zip(self.handler_weights, self.containers)]))
        else:
            stds = safe_json_load(stdevs_file)
        return stds

    def save_stats(self, stdevs_file, means_file):
        """Save stats to json files."""
        with open(stdevs_file) as f:
            json.dumps(f, self.stds)
        with open(means_file) as f:
            json.dumps(f, self.means)
