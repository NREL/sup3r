"""Code for regridding data from one list of coordinates to another"""

import logging
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime as dt
from glob import glob

import dask
import numpy as np
import pandas as pd
import psutil
from rex import MultiFileResource
from sklearn.neighbors import BallTree

from sup3r.postprocessing.file_handling import OutputMixIn, RexOutputs
from sup3r.utilities.execution import DistributedProcess

dask.config.set({'array.slicing.split_large_chunks': True})

logger = logging.getLogger(__name__)


class Regridder:
    """Basic Regridder class. Builds ball tree and runs all queries to
    create full arrays of indices and distances for neighbor points. Computes
    array of weights used to interpolate from old grid to new grid.
    """

    MIN_DISTANCE = 1e-12
    MAX_DISTANCE = 0.01

    def __init__(
        self,
        source_meta,
        target_meta,
        cache_pattern=None,
        leaf_size=4,
        k_neighbors=4,
        n_chunks=100,
        max_distance=None,
        max_workers=None,
    ):
        """Get weights and indices used to map from source grid to target grid

        Parameters
        ----------
        source_meta : pd.DataFrame
            Set of coordinates for source grid
        target_meta : pd.DataFrame
            Set of coordinates for target grid
        cache_pattern : str | None
            Pattern for cached indices and distances for ball tree. Will load
            these if provided. Should be of the form './{array_name}.pkl' where
            array_name will be replaced with either 'indices' or 'distances'.
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
        logger.info('Initializing Regridder.')

        self.cache_pattern = cache_pattern
        self.target_meta = target_meta
        self.source_meta = source_meta
        self.k_neighbors = k_neighbors
        self.n_chunks = n_chunks
        self.max_workers = max_workers
        self.max_distance = max_distance or self.MAX_DISTANCE
        self.leaf_size = leaf_size
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
        """Initialize arrays for tree queries and either load query cache or
        perform all queries"""
        self._indices = [None] * len(self.target_meta)
        self._distances = [None] * len(self.target_meta)

        if self.cache_exists:
            self.load_cache()
        else:
            self.get_all_queries(self.max_workers)
            self.cache_all_queries()

    @classmethod
    def run(
        cls,
        source_meta,
        target_meta,
        cache_pattern=None,
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
        cache_pattern : str | None
            Pattern for cached indices and distances for ball tree. Will load
            these if provided. Should be of the form './{array_name}.pkl' where
            array_name will be replaced with either 'indices' or 'distances'.
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
            cache_pattern=cache_pattern,
            leaf_size=leaf_size,
            k_neighbors=k_neighbors,
            n_chunks=n_chunks,
            max_workers=max_workers,
        )
        if not regridder.cache_exists:
            regridder.get_all_queries(max_workers)
            regridder.cache_all_queries()

    @property
    def weights(self):
        """Get weights used for regridding"""
        if self._weights is None:
            dists = np.array(self.distances, dtype=np.float32)
            mask = dists < self.MIN_DISTANCE
            if mask.sum() > 0:
                logger.info(
                    f'{np.sum(mask)} of {np.prod(mask.shape)} '
                    'distances are zero.'
                )
            dists[mask] = self.MIN_DISTANCE
            weights = 1 / dists
            self._weights = weights / np.sum(weights, axis=-1)[:, None]
        return self._weights

    @property
    def cache_exists(self):
        """Check if cache exists before building tree."""
        cache_exists_check = (
            self.index_file is not None
            and os.path.exists(self.index_file)
            and self.distance_file is not None
            and os.path.exists(self.distance_file)
        )
        return cache_exists_check

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
            self._serial_queries()

        else:
            logger.info('Querying all coordinates in parallel.')
            self._parallel_queries(max_workers=max_workers)
        logger.info('Finished querying all coordinates.')

    def _serial_queries(self):
        """Get indices and distances for all points in target_meta, in
        serial"""
        self.save_query(slice(None))

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
        out = self.query_tree(s_slice)
        self.distances[s_slice] = out[0]
        self.indices[s_slice] = out[1]

    def load_cache(self):
        """Load cached indices and distances from ball tree query"""
        with open(self.index_file, 'rb') as f:
            self._indices = pickle.load(f)
        with open(self.distance_file, 'rb') as f:
            self._distances = pickle.load(f)
        logger.info(
            f'Loaded cache files: {self.index_file}, ' f'{self.distance_file}'
        )

    def cache_all_queries(self):
        """Cache indices and distances from ball tree query"""
        if self.cache_pattern is not None:
            with open(self.index_file, 'wb') as f:
                pickle.dump(self.indices, f, protocol=4)
            with open(self.distance_file, 'wb') as f:
                pickle.dump(self.distances, f, protocol=4)
            logger.info(
                f'Saved cache files: {self.index_file}, '
                f'{self.distance_file}'
            )

    @property
    def index_file(self):
        """Get name of cache indices file"""
        if self.cache_pattern is not None:
            return self.cache_pattern.format(array_name='indices')
        return None

    @property
    def distance_file(self):
        """Get name of cache distances file"""
        if self.cache_pattern is not None:
            return self.cache_pattern.format(array_name='distances')
        return None

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

    def query_tree(self, s_slice):
        """Get indices and distances for points specified by the given spatial
        slice

        Parameters
        ----------
        s_slice : slice
            slice specifying which spatial indices in the target grid should be
            selected. This selects n_points from the target grid

        Returns
        -------
        distances : ndarray
            Array of distances for neighboring points for each point selected
            by s_slice. (n_ponts, k_neighbors)
        indices : ndarray
            Array of indices for neighboring points for each point selected
            by s_slice. (n_ponts, k_neighbors)
        """
        return self.tree.query(
            self.get_spatial_chunk(s_slice), k=self.k_neighbors
        )

    @property
    def dist_mask(self):
        """Mask for points too far from original grid

        Returns
        -------
        mask : ndarray
            Bool array for points outside original grid extent
        """
        return np.array(self.distances)[:, -1] > self.max_distance

    @classmethod
    def interpolate(cls, distance_chunk, values):
        """Interpolate to new coordinates based on distances from those
        coordinates and the values of the points at those distances

        Parameters
        ----------
        distance_chunk : ndarray
            Chunk of the full array of distances where distances[i] gives the
            list of k_neighbors distances to the source coordinates to be used
            for interpolation for the i-th coordinate in the target data.
            (n_points, k_neighbors)
        values : ndarray
            Array of values corresponding to the point distances with shape
            (temporal, n_points, k_neighbors)

        Returns
        -------
        ndarray
            Time series of values at interpolated points with shape
            (temporal, n_points)
        """
        dists = np.array(distance_chunk, dtype=np.float32)
        mask = dists < cls.MIN_DISTANCE
        if mask.sum() > 0:
            logger.info(
                f'{np.sum(mask)} of {np.prod(mask.shape)} '
                'distances are zero.'
            )
        dists[mask] = cls.MIN_DISTANCE
        weights = 1 / dists
        norm = np.sum(weights, axis=-1)
        out = np.einsum('ijk,jk->ij', values, weights) / norm
        return out

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
        vals = data[np.concatenate(self.indices)].reshape(
            (len(self.indices), self.k_neighbors, -1)
        )
        vals = np.transpose(vals, axes=(2, 0, 1))
        return np.einsum('ijk,jk->ij', vals, self.weights).T


class RegridOutput(OutputMixIn, DistributedProcess):
    """Output regridded data as it is interpolated. Takes source data from
    windspeed and winddirection h5 files and uses this data to interpolate onto
    a new target grid. The interpolated data is then written to new files, with
    one file for each field (e.g. windspeed_100m)."""

    def __init__(
        self,
        source_files,
        out_pattern,
        target_meta,
        heights,
        cache_pattern=None,
        leaf_size=4,
        k_neighbors=4,
        incremental=False,
        n_chunks=100,
        max_nodes=1,
        worker_kwargs=None,
    ):
        """
        Parameters
        ----------
        source_files : str | list
            Path to source files to regrid to target_meta
        out_pattern : str
            Pattern to use for naming outputs file to store the regridded data.
            This must include a {file_id} format key. e.g.
            ./chunk_{file_id}.h5
        target_meta : str
            Path to dataframe of final grid coordinates on which to regrid
        heights : list
            List of wind field heights to regrid. e.g if heights = [100] then
            windspeed_100m and winddirection_100m will be regridded and stored
            in the output_file.
        cache_pattern : str
            Pattern for cached indices and distances for ball tree
        leaf_size : int, optional
            leaf size for BallTree
        k_neighbors : int, optional
            number of nearest neighbors to use for interpolation
        incremental : bool
            Whether to keep already written output chunks or overwrite them
        n_chunks : int
            Number of spatial chunks to use for interpolation. The total number
            of points in the target_meta will be split into n_chunks and the
            points in each chunk will be interpolated at the same time.
        max_nodes : int
            Number of nodes to distribute chunks across.
        worker_kwargs : dict | None
            Dictionary of workers args. Optional keys include regrid_workers
            (max number of workers to use for regridding and output)
        """
        worker_kwargs = worker_kwargs or {}
        self.regrid_workers = worker_kwargs.get('regrid_workers', None)
        self.query_workers = worker_kwargs.get('query_workers', None)
        self.source_files = (
            source_files
            if isinstance(source_files, list)
            else glob(source_files)
        )
        self.target_meta_path = target_meta
        self.target_meta = pd.read_csv(self.target_meta_path)
        self.target_meta['gid'] = np.arange(len(self.target_meta))
        self.target_meta = self.target_meta.sort_values(
            ['latitude', 'longitude'], ascending=[False, True]
        )
        self.heights = heights
        self.incremental = incremental
        self.out_pattern = out_pattern
        os.makedirs(os.path.dirname(self.out_pattern), exist_ok=True)

        with MultiFileResource(source_files) as res:
            self.time_index = res.time_index
            self.source_meta = res.meta
            self.global_attrs = res.global_attrs

        self.regridder = Regridder(
            self.source_meta,
            self.target_meta,
            leaf_size=leaf_size,
            k_neighbors=k_neighbors,
            cache_pattern=cache_pattern,
            n_chunks=n_chunks,
            max_workers=self.query_workers,
        )
        DistributedProcess.__init__(
            self,
            max_nodes=max_nodes,
            n_chunks=n_chunks,
            max_chunks=len(self.regridder.indices),
            incremental=incremental,
        )

        logger.info(
            'Initializing RegridOutput with '
            f'source_files={self.source_files}, '
            f'out_pattern={self.out_pattern}, '
            f'heights={self.heights}, '
            f'target_meta={target_meta}, '
            f'k_neighbors={k_neighbors}, and '
            f'n_chunks={n_chunks}.'
        )
        logger.info(f'Max memory usage: {self.max_memory:.3f} GB.')

    @property
    def spatial_slices(self):
        """Get the list of slices which select index and distance chunks"""
        slices = np.arange(len(self.regridder.indices))
        slices = np.array_split(slices, self.chunks)
        return [slice(s[0], s[-1] + 1) for s in slices]

    @property
    def max_memory(self):
        """Check max memory usage (in GB)"""
        chunk_mem = 8 * len(self.time_index) * len(self.index_chunks[0])
        chunk_mem *= len(self.index_chunks[0][0])
        return self.regrid_workers * chunk_mem / 1e9

    @property
    def index_chunks(self):
        """Get list of index chunks to use for chunking data extraction and
        interpolation. indices[i] is the set of indices for the i-th coordinate
        in the target grid which select the neighboring points in the source
        grid"""
        return [self.regridder.indices[s] for s in self.spatial_slices]

    @property
    def distance_chunks(self):
        """Get list of distance chunks to use for chunking data extraction and
        interpolation. distances[i] is the set of distances from the i-th
        coordinate in the target grid to the neighboring points in the source
        grid"""
        return [self.regridder.distances[s] for s in self.spatial_slices]

    @property
    def meta_chunks(self):
        """Get meta chunks corresponding to the spatial chunks of the
        target_meta"""
        return [self.regridder.target_meta[s] for s in self.spatial_slices]

    @property
    def out_files(self):
        """Get list of output files for each spatial chunk"""
        return [
            self.out_pattern.format(file_id=str(i).zfill(6))
            for i in range(self.chunks)
        ]

    @property
    def output_features(self):
        """Get list of dsets to write to output files"""
        out = []
        for height in self.heights:
            out.append(f'windspeed_{height}m')
            out.append(f'winddirection_{height}m')
        return out

    def run(self, node_index):
        """Run regridding and output write in either serial or parallel

        Parameters
        ----------
        node_index : int
            Node index to run. e.g. if node_index=0 then only the chunks for
            node_chunks[0] will be run.
        """
        if self.node_finished(node_index):
            return

        if self.regrid_workers == 1:
            self._run_serial(
                source_files=self.source_files, node_index=node_index
            )
        else:
            self._run_parallel(
                source_files=self.source_files,
                node_index=node_index,
                max_workers=self.regrid_workers,
            )

    def _run_serial(self, source_files, node_index):
        """Regrid data and write to output file, in serial.

        Parameters
        ----------
        source_files : list
            List of paths to source files
        node_index : int
            Node index to run. e.g. if node_index=0 then the chunks for
            node_chunks[0] will be run.
        """
        logger.info('Regridding all coordinates in serial.')
        for i, chunk_index in enumerate(self.node_chunks[node_index]):
            self.write_coordinates(
                source_files=source_files, chunk_index=chunk_index
            )

            mem = psutil.virtual_memory()
            msg = (
                'Coordinate chunks regridded: {} out of {}. '
                'Current memory usage is {:.3f} GB out of {:.3f} '
                'GB total.'.format(
                    i + 1,
                    len(self.node_chunks[node_index]),
                    mem.used / 1e9,
                    mem.total / 1e9,
                )
            )
            logger.info(msg)

    def _run_parallel(self, source_files, node_index, max_workers=None):
        """Regrid data and write to output file, in parallel.

        Parameters
        ----------
        source_files : list
            List of paths to source files
        node_index : int
            Node index to run. e.g. if node_index=0 then the chunks for
            node_chunks[0] will be run.
        max_workers : int | None
            Max number of workers to use for regridding in parallel
        """
        futures = {}
        now = dt.now()
        logger.info('Regridding all coordinates in parallel.')
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            for i, chunk_index in enumerate(self.node_chunks[node_index]):
                future = exe.submit(
                    self.write_coordinates,
                    source_files=source_files,
                    chunk_index=chunk_index,
                )
                futures[future] = chunk_index
                mem = psutil.virtual_memory()
                msg = 'Regrid futures submitted: {} out of {}'.format(
                    i + 1, len(self.node_chunks[node_index])
                )
                logger.info(msg)

            logger.info(f'Submitted all regrid futures in {dt.now() - now}.')

            for i, future in enumerate(as_completed(futures)):
                idx = futures[future]
                mem = psutil.virtual_memory()
                msg = (
                    'Regrid futures completed: {} out of {}, in {}. '
                    'Current memory usage is {:.3f} GB out of {:.3f} GB '
                    'total.'.format(
                        i + 1,
                        len(futures),
                        dt.now() - now,
                        mem.used / 1e9,
                        mem.total / 1e9,
                    )
                )
                logger.info(msg)

                try:
                    future.result()
                except Exception as e:
                    msg = (
                        'Falied to regrid coordinate chunks with '
                        'index={index}'.format(index=idx)
                    )
                    logger.exception(msg)
                    raise RuntimeError(msg) from e

    def write_coordinates(self, source_files, chunk_index):
        """Write regridded coordinate data to the output file

        Parameters
        ----------
        source_files : list
            List of paths to source files
        chunk_index : int
            Index of spatial chunk to regrid and write to output file
        """
        index_chunk = self.index_chunks[chunk_index]
        distance_chunk = self.distance_chunks[chunk_index]
        s_slice = self.spatial_slices[chunk_index]
        out_file = self.out_files[chunk_index]
        meta = self.meta_chunks[chunk_index]
        if self.chunk_finished(chunk_index):
            return

        tmp_file = out_file.replace('.h5', '.h5.tmp')
        with RexOutputs(tmp_file, 'w') as fh:
            fh.meta = meta
            fh.time_index = self.time_index
            fh.run_attrs = self.global_attrs
            for height in self.heights:
                ws, wd = self.regridder.regrid_coordinates(
                    index_chunk=index_chunk,
                    distance_chunk=distance_chunk,
                    height=height,
                    source_files=source_files,
                )

                features = [f'windspeed_{height}m', f'winddirection_{height}m']

                for dset, data in zip(features, [ws, wd]):
                    attrs, dtype = self.get_dset_attrs(dset)
                    fh.add_dataset(
                        tmp_file,
                        dset,
                        data,
                        dtype=dtype,
                        attrs=attrs,
                        chunks=attrs['chunks'],
                    )

                logger.info(f'Added {features} to {out_file}')
        os.replace(tmp_file, out_file)
        logger.info(f'Finished regridding chunk with s_slice={s_slice}')
