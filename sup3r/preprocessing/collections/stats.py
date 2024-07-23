"""Collection object with methods to compute and save stats."""

import logging
import os

import numpy as np
import xarray as xr
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
        self.normalize(means=self.means, stds=self.stds)

    def _get_stat(self, stat_type):
        """Get either mean or std for all features and all containers."""
        all_feats = self.containers[0].data_vars
        hr_feats = self.containers[0].data.high_res.data_vars
        lr_feats = [f for f in all_feats if f not in hr_feats]
        cstats = [
            getattr(c.data.high_res[hr_feats], stat_type)(skipna=True)
            for c in self.containers
        ]
        if any(lr_feats):
            cstats_lr = [
                getattr(c.data.low_res[lr_feats], stat_type)(skipna=True)
                for c in self.containers
            ]
            cstats = [
                xr.merge([c._ds, c_lr._ds])
                for c, c_lr in zip(cstats, cstats_lr)
            ]
        return cstats

    def get_means(self, means):
        """Dictionary of means for each feature, computed across all data
        handlers."""
        if means is None or (
            isinstance(means, str) and not os.path.exists(means)
        ):
            all_feats = self.containers[0].data_vars
            means = dict.fromkeys(all_feats, 0)
            logger.info(f'Computing means for {all_feats}.')
            cmeans = [
                cm * w
                for cm, w in zip(
                    self._get_stat('mean'), self.container_weights
                )
            ]
            for f in all_feats:
                logger.info(f'Computing mean for {f}.')
                means[f] = np.float32(np.sum(cm[f] for cm in cmeans))
        elif isinstance(means, str):
            means = safe_json_load(means)
        return means

    def get_stds(self, stds):
        """Dictionary of standard deviations for each feature, computed across
        all data handlers."""
        if stds is None or (
            isinstance(stds, str) and not os.path.exists(stds)
        ):
            all_feats = self.containers[0].data_vars
            stds = dict.fromkeys(all_feats, 0)
            logger.info(f'Computing stds for {all_feats}.')
            cstds = [
                w * cm**2
                for cm, w in zip(self._get_stat('std'), self.container_weights)
            ]
            for f in all_feats:
                logger.info(f'Computing std for {f}.')
                stds[f] = np.float32(np.sqrt(np.sum(cs[f] for cs in cstds)))
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

    def normalize(self, stds, means):
        """Normalize container data with computed stats."""
        logger.info(
            f'Normalizing container data with means: {means}, stds: {stds}.'
        )
        _ = [c.normalize(means=means, stds=stds) for c in self.containers]
