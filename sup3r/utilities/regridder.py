"""Code for regridding data from one list of coordinates to another"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime as dt
from typing import Optional

import dask.array as da
import numpy as np
import pandas as pd
import psutil
from sklearn.neighbors import BallTree

from sup3r.preprocessing.utilities import log_args

logger = logging.getLogger(__name__)


@dataclass
class Regridder:
    """Regridder class. Builds ball tree and runs all queries to create full
    arrays of indices and distances for neighbor points. Computes array of
    weights used to interpolate from old grid to new grid.

    Parameters
    ----------
    source_meta : pd.DataFrame
        Set of coordinates for source grid
    target_meta : pd.DataFrame
        Set of coordinates for target grid
    leaf_size : int, optional
        leaf size for BallTree
    k_neighbors : int, optional
        number of nearest neighbors to use for interpolation
    n_chunks : int
        Number of spatial chunks to use for tree queries. The total number
        of points in the target_meta will be split into n_chunks and the
        points in each chunk will be queried at the same time.
    max_distance : float | None
        Max distance to new grid points from original points before filling
        with nans.
    max_workers : int | None
        Max number of workers to use for running all tree queries needed
        to building full set of indices and distances for each target_meta
        coordinate.
    """

    source_meta: pd.DataFrame
    target_meta: pd.DataFrame
    k_neighbors: Optional[int] = 4
    n_chunks: Optional[int] = 100
    max_workers: Optional[int] = None
    min_distance: Optional[float] = 1e-12
    max_distance: Optional[float] = 0.01
    leaf_size: Optional[int] = 4

    @log_args
    def __post_init__(self):
        self._tree = None
        self._distances = None
        self._indices = None
        self._weights = None

    @property
    def distances(self):
        """Get distances for all tree queries."""
        if self._distances is None:
            self.init_queries()
        return self._distances

    @property
    def indices(self):
        """Get indices for all tree queries."""
        if self._indices is None:
            self.init_queries()
        return self._indices

    def init_queries(self):
        """Initialize arrays for tree queries and perform all queries"""
        self._indices = [None] * len(self.target_meta)
        self._distances = [None] * len(self.target_meta)
        self.get_all_queries(self.max_workers)

    @classmethod
    def run(
        cls,
        source_meta,
        target_meta,
        leaf_size=4,
        k_neighbors=4,
        n_chunks=100,
        max_workers=None,
    ):
        """Query tree for every point in target_meta to get full set of indices
        and distances for the neighboring points in the source_meta.

        Parameters
        ----------
        source_meta : pd.DataFrame
            Set of coordinates for source grid
        target_meta : pd.DataFrame
            Set of coordinates for target grid
        leaf_size : int, optional
            leaf size for BallTree
        k_neighbors : int, optional
            number of nearest neighbors to use for interpolation
        n_chunks : int
            Number of spatial chunks to use for tree queries. The total number
            of points in the target_meta will be split into n_chunks and the
            points in each chunk will be queried at the same time.
        max_workers : int | None
            Max number of workers to use for running all tree queries needed
            to building full set of indices and distances for each target_meta
            coordinate.
        """
        regridder = cls(
            source_meta=source_meta,
            target_meta=target_meta,
            leaf_size=leaf_size,
            k_neighbors=k_neighbors,
            n_chunks=n_chunks,
            max_workers=max_workers,
        )
        regridder.get_all_queries(max_workers)

    @property
    def weights(self):
        """Get weights used for regridding"""
        if self._weights is None:
            dists = np.array(self.distances, dtype=np.float32)
            mask = dists < self.min_distance
            if mask.sum() > 0:
                logger.info(
                    f'{np.sum(mask)} of {np.prod(mask.shape)} '
                    f'neighbor distances are within {self.min_distance}.'
                )
            weights = 1 / dists
            weights[mask.any(axis=1), :] = np.eye(
                1, self.k_neighbors
            ).flatten()
            self._weights = weights / np.sum(weights, axis=-1)[:, None]
        return self._weights

    @property
    def tree(self):
        """Build ball tree from source_meta"""
        if self._tree is None:
            logger.info('Building ball tree for regridding.')
            ll2 = self.source_meta[['latitude', 'longitude']].values
            ll2 = np.radians(ll2)
            self._tree = BallTree(
                ll2, leaf_size=self.leaf_size, metric='haversine'
            )
        return self._tree

    def get_all_queries(self, max_workers=None):
        """Query ball tree for all coordinates in the target_meta and store
        results"""

        if max_workers == 1:
            logger.info('Querying all coordinates in serial.')
            self.save_query(slice(None))

        else:
            logger.info('Querying all coordinates in parallel.')
            self._parallel_queries(max_workers=max_workers)
        logger.info('Finished querying all coordinates.')

    def _parallel_queries(self, max_workers=None):
        """Get indices and distances for all points in target_meta, in
        serial"""
        futures = {}
        now = dt.now()
        slices = np.arange(len(self.target_meta))
        slices = np.array_split(slices, min(self.n_chunks, len(slices)))
        slices = [slice(s[0], s[-1] + 1) for s in slices]
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            for i, s_slice in enumerate(slices):
                future = exe.submit(self.save_query, s_slice=s_slice)
                futures[future] = i
                mem = psutil.virtual_memory()
                msg = (
                    'Query futures submitted: {} out of {}. Current '
                    'memory usage is {:.3f} GB out of {:.3f} GB '
                    'total.'.format(
                        i + 1, len(slices), mem.used / 1e9, mem.total / 1e9
                    )
                )
                logger.info(msg)

            logger.info(f'Submitted all query futures in {dt.now() - now}.')

            for i, future in enumerate(as_completed(futures)):
                idx = futures[future]
                mem = psutil.virtual_memory()
                msg = (
                    'Query futures completed: {} out of '
                    '{}. Current memory usage is {:.3f} '
                    'GB out of {:.3f} GB total.'.format(
                        i + 1, len(futures), mem.used / 1e9, mem.total / 1e9
                    )
                )
                logger.info(msg)
                try:
                    future.result()
                except Exception as e:
                    msg = (
                        'Failed to query coordinate chunk with '
                        'index={index}'.format(index=idx)
                    )
                    logger.exception(msg)
                    raise RuntimeError(msg) from e

    def save_query(self, s_slice):
        """Save tree query for coordinates specified by given spatial slice"""
        out = self.tree.query(
            self.get_spatial_chunk(s_slice), k=self.k_neighbors
        )
        self.distances[s_slice] = out[0]
        self.indices[s_slice] = out[1]

    def get_spatial_chunk(self, s_slice):
        """Get list of coordinates in target_meta specified by the given
        spatial slice

        Parameters
        ----------
        s_slice : slice
            slice specifying which spatial indices in the target grid should be
            selected. This selects n_points from the target grid

        Returns
        -------
        ndarray
            Array of n_points in target_meta selected by s_slice.
        """
        out = self.target_meta.iloc[s_slice][['latitude', 'longitude']].values
        return np.radians(out)

    def __call__(self, data):
        """Regrid given spatiotemporal data over entire grid

        Parameters
        ----------
        data : ndarray
            Spatiotemporal data to regrid to target_meta. Data can be flattened
            in the spatial dimension to match the target_meta or be in a 2D
            spatial grid, e.g.:
            (spatial, temporal) or (spatial_1, spatial_2, temporal)

        Returns
        -------
        out : ndarray
            Flattened regridded spatiotemporal data
            (spatial, temporal)
        """
        if len(data.shape) == 3:
            data = data.reshape((data.shape[0] * data.shape[1], -1))
        msg = 'Input data must be 2D (spatial, temporal)'
        assert len(data.shape) == 2, msg
        vals = data[da.concatenate(self.indices)].reshape(
            (len(self.indices), self.k_neighbors, -1)
        )
        vals = da.transpose(vals, axes=(2, 0, 1))
        return da.einsum('ijk,jk->ij', vals, self.weights).T
