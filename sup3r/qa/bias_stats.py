"""Compute bias stats for Sup3r vs Reference (typically WTK)"""

import logging
import os

import pandas as pd
from rex import Resource
from rex.temporal_stats import TemporalStats
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


class Sup3rBiasStats:
    """Compute bias stats for Sup3r vs Reference (typically WTK). Computes
    coordinate index map from sup3r to reeference and requested temporal stats.
    Then uses this coordinate index map to compare stats of sup3r and
    reference."""

    RESOURCE_CLASS = Resource

    def __init__(self, sup3r_file, ref_file, dsets, stats='mean',
                 do_months=True, max_workers=None, out_pattern=None,
                 threshold=1e-2, overwrite=False):
        """Initialize Sup3rBiasPlotter.

        sup3r_file: str
            Name of sup3r data file.
        ref_file: str | list
            Name of reference data file.
        dsets : str | list
            str or list of dsets to compute biases for.
        stats: str | list
            str or list of stats to compute.
        do_months: bool
            Whether to compute monthly stats.
        threshold : float
            Max distance to between sup3r and reference coordinates.
        out_pattern : str | None
            Output pattern for stats files and bias files. Must include {res},
            {dset}, {stat} format keys if not None.
        max_workers: int | None
            Number of workers to use for computing stats.
        overwrite : bool
            Whether to overwrite existing files.
        """

        self.sup3r_file = sup3r_file
        self.ref_file = ref_file
        self.dsets = (dsets if isinstance(dsets, list) else [dsets])
        self.stats = stats if isinstance(stats, list) else [stats]
        self.do_months = do_months
        self.max_workers = max_workers
        self.threshold = threshold
        self.out_pattern = out_pattern
        self.overwrite = overwrite
        if out_pattern is not None:
            os.makedirs(os.path.dirname(out_pattern), exist_ok=True)

        msg = ('Initializing Sup3rBiasStats with '
               f'sup3r_file={self.sup3r_file}, ref_file={self.ref_file}, '
               f'dsets={self.dsets}, stats={self.stats}, '
               f'do_months={self.do_months}, threshold={self.threshold}, '
               f'out_pattern={self.out_pattern}, '
               f'max_workers={self.max_workers}.')
        logger.info(msg)

        logger.info(f'Computing KDTree for reference files: {self.ref_file}')
        self.tree = self.get_ref_tree()
        logger.info('Computing coordinate map for sup3r_file, ref_file: '
                    f'{self.sup3r_file}, {self.ref_file}.')
        self.indices, self.mask = self.get_coord_map()

    @property
    def sup3r_sites(self):
        """Get list of matched sup3r site indices."""
        with self.RESOURCE_CLASS(self.sup3r_file) as res:
            gids = res.meta.index.values
            return gids[self.mask]

    @property
    def ref_sites(self):
        """Get list of matched reference site indices"""
        return self.indices[self.mask]

    def get_ref_tree(self):
        """Get KDTree for reference coordinates."""
        with self.RESOURCE_CLASS(self.ref_file) as res:
            lat_lon = res.meta[['latitude', 'longitude']].values
            tree = KDTree(lat_lon)
            return tree

    def get_coord_map(self):
        """Get list of indices to map from reference coordinates to sup3r
        coordinates with mask to remove coordinates exceeding distance
        threshold."""
        with self.RESOURCE_CLASS(self.sup3r_file) as res:
            dists, indices = self.tree.query(
                res.meta[['latitude', 'longitude']].values)
            mask = dists < self.threshold
        return indices, mask

    def _compute_stats(self, files, sites=None, res_name=None):
        """Compute stats for given files.

        Parameters
        ----------
        files : list
            List of files to compute stats for.
        sites : list
            List of sites to compute stats for.
        res_name : str | None
            Name of resource to compute stats for. Used to name output files.

        Returns
        -------
        dfs : dict
            Dictionary of pandas dataframes with keys for each dataset and
            stat for given files. e.g. dfs['windspeed_10m']['mean'].
        """
        dfs = {}
        for dset in self.dsets:
            if dset not in dfs:
                dfs[dset] = {}
            for stat in self.stats:
                out_path = None
                if self.out_pattern is not None:
                    out_path = self.out_pattern.format(res=res_name, dset=dset,
                                                       stat=stat)
                check = (out_path is not None and not self.overwrite
                         and os.path.exists(out_path))
                if check:
                    dfs[dset][stat] = pd.read_csv(out_path)

                else:
                    logger.info(f'Computing {stat} for {dset} for {files}.')
                    dfs[dset][stat] = TemporalStats.run(
                        files, dset, sites=sites, statistics=stat,
                        diurnal=False, month=self.do_months,
                        combinations=False, res_cls=self.RESOURCE_CLASS,
                        max_workers=self.max_workers, chunks_per_worker=5,
                        lat_lon_only=True, mask_zeros=False, out_path=out_path)
        return dfs

    def compute_stats(self):
        """Compute stats for sup3r and reference files.

        Returns
        -------
        sup3r_dfs : dict
            Dictionary of pandas dataframes with keys for each dataset and
            stat for sup3r files. e.g. sup3r_dfs['windspeed_10m']['mean'].
        ref_dfs : dict
            Dictionary of pandas dataframes with keys for each dataset and
            stat for reference files. e.g. ref_dfs['windspeed_10m']['mean'].
        """

        logger.info(f'Computing stats for sup3r_file: {self.sup3r_file}.')
        sup3r_dfs = self._compute_stats(self.sup3r_file, self.sup3r_sites,
                                        res_name='sup3r')
        logger.info(f'Computing stats for ref_file: {self.ref_file}.')
        ref_dfs = self._compute_stats(self.ref_file, self.ref_sites,
                                      res_name='ref')
        return sup3r_dfs, ref_dfs

    def compute_biases(self, sup3r_dfs, ref_dfs):
        """Compute biases between sup3r and reference stats

        Parameters
        ----------
        sup3r_dfs : dict
            Dictionary of pandas dataframes with keys for each dataset and
            stat for sup3r files. e.g. sup3r_dfs['windspeed_10m']['mean'].
        ref_dfs : dict
            Dictionary of pandas dataframes with keys for each dataset and
            stat for reference files. e.g. ref_dfs['windspeed_10m']['mean'].

        Returns
        -------
        bias_dfs : dict
            Dictionary of pandas dataframes with bias values with keys for each
            dataset and stat. e.g. sup3r_dfs['windspeed_10m']['mean'][col] -
            ref_dfs['windspeed_10m']['mean'][col]
        """
        bias_dfs = {}
        for dset in sup3r_dfs:
            bias_dfs[dset] = {}
            for stat, df in sup3r_dfs[dset].items():
                bias_dfs[dset][stat] = df.copy()
                cols = sup3r_dfs[dset][stat].columns
                cols = set(cols).intersection(ref_dfs[dset][stat].columns)
                cols = [col for col in cols if col
                        not in ('latitude', 'longitude')]
                for col in cols:
                    bias_dfs[dset][stat][col] -= ref_dfs[dset][stat][col]
                if self.out_pattern is not None:
                    out_file = self.out_pattern.format(res='bias', dset=dset,
                                                       stat=stat)
                    bias_dfs[dset][stat].to_csv(out_file)
                    logger.info(f'Saved bias file: {out_file}.')
        return bias_dfs

    @classmethod
    def run(cls, sup3r_file, ref_file, dsets, stats='mean', do_months=True,
            max_workers=None, out_pattern=None, threshold=1e-2):
        """Run bias compute routine.

        sup3r_file: str | list
            glob-able str pointing to sup3r files or list of sup3r files.
        ref_file: str | list
            glob-able str pointing to reference files or list of reference
            files.
        dsets : str | list
            str or list of dsets to compute biases for.
        stats: str | list
            str or list of stats to compute.
        do_months: bool
            Whether to compute monthly stats.
        threshold : float
            Max distance to between sup3r and reference coordinates.
        out_pattern : str | None
            Output pattern for stats files. Must include {res}, {dset}, {stat}
            format keys if not None.
        max_workers: int | None
            Number of workers to use for computing stats.

        Returns
        -------
        bias_dfs : dict
            Dictionary of pandas dataframes with bias values with keys for each
            dataset and stat. e.g. sup3r_dfs['windspeed_10m']['mean'][col] -
            ref_dfs['windspeed_10m']['mean'][col]
        """
        bc_plot = cls(sup3r_file=sup3r_file, ref_file=ref_file,
                      dsets=dsets, stats=stats, do_months=do_months,
                      max_workers=max_workers, out_pattern=out_pattern,
                      threshold=threshold)
        sup3r_dfs, ref_dfs = bc_plot.compute_stats()
        return bc_plot.compute_biases(sup3r_dfs, ref_dfs)
