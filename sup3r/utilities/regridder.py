"""Code for regridding data from one list of coordinates to another"""
import numpy as np
from sklearn.neighbors import BallTree
import logging
import psutil
from glob import glob
import pickle
import os
import pandas as pd
from datetime import datetime as dt
from concurrent.futures import as_completed, ThreadPoolExecutor

from rex.utilities.fun_utils import get_fun_call_str
from rex import MultiFileResource

from sup3r.postprocessing.file_handling import OutputMixIn, RexOutputs
from sup3r.utilities import ModuleName
from sup3r.utilities.execution import DistributedProcess
from sup3r.utilities.cli import BaseCLI

logger = logging.getLogger(__name__)


class TreeBuilder:
    """TreeBuilder class for building ball tree and running all queries to
    create full arrays of indices and distances for neighbor points
    """

    def __init__(self, source_meta, target_meta, cache_pattern=None,
                 leaf_size=4, k_neighbors=4, n_chunks=100, max_workers=None):
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
        max_workers : int | None
            Max number of workers to use for running all tree queries needed
            to building full set of indices and distances for each target_meta
            coordinate.
        """
        self.cache_pattern = cache_pattern
        self.target_meta = target_meta
        self.source_meta = source_meta
        self.k_neighbors = k_neighbors
        self.n_chunks = n_chunks
        self.max_workers = max_workers
        self.tree = None
        self.leaf_size = leaf_size
        self.distances = [None] * len(self.target_meta)
        self.indices = [None] * len(self.target_meta)

        if self.cache_exists:
            self.load_cache()
        else:
            self.build_tree()
            self.get_all_queries(max_workers)
            self.cache_all_queries()

    @classmethod
    def run(cls, source_meta, target_meta, cache_pattern=None,
            leaf_size=4, k_neighbors=4, n_chunks=100, max_workers=None):
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
        tree_builder = cls(source_meta=source_meta, target_meta=target_meta,
                           cache_pattern=cache_pattern, leaf_size=leaf_size,
                           k_neighbors=k_neighbors, n_chunks=n_chunks,
                           max_workers=max_workers)
        if not tree_builder.cache_exists:
            tree_builder.get_all_queries(max_workers)
            tree_builder.cache_all_queries()

    @property
    def cache_exists(self):
        """Check if cache exists before building tree."""
        cache_exists_check = (self.index_file is not None
                              and os.path.exists(self.index_file)
                              and self.distance_file is not None
                              and os.path.exists(self.distance_file))
        return cache_exists_check

    def build_tree(self):
        """Build ball tree from source_meta"""

        logger.info("Building ball tree for regridding.")
        ll2 = self.source_meta[['latitude', 'longitude']].values
        ll2 = np.radians(ll2)
        self.tree = BallTree(ll2, leaf_size=self.leaf_size, metric='haversine')

    def get_all_queries(self, max_workers=None):
        """Query ball tree for all coordinates in the target_meta and store
        results"""

        if max_workers == 1:
            logger.info('Querying all coordinates in serial.')
            self._serial_queries()

        else:
            logger.info('Querying all coordinates in parallel.')
            self._parallel_queries(max_workers=max_workers)

    def _serial_queries(self):
        """Get indices and distances for all points in target_meta, in serial
        """
        self.save_query(slice(None))

    def _parallel_queries(self, max_workers=None):
        """Get indices and distances for all points in target_meta, in serial
        """
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
                msg = ('Query futures submitted: {0} out of {1}. Current '
                       'memory usage is {2:.3f} GB out of {3:.3f} GB '
                       'total.'.format(i + 1, len(slices), mem.used / 1e9,
                                       mem.total / 1e9))
                logger.info(msg)

            logger.info(f'Submitted all query futures in {dt.now() - now}.')

            for i, future in enumerate(as_completed(futures)):
                idx = futures[future]
                mem = psutil.virtual_memory()
                msg = ('Query futures completed: {0} out of '
                       '{1}. Current memory usage is {2:.3f} '
                       'GB out of {3:.3f} GB total.'.format(i + 1,
                                                            len(futures),
                                                            mem.used / 1e9,
                                                            mem.total / 1e9))
                logger.info(msg)
                try:
                    future.result()
                except Exception as e:
                    msg = ('Failed to query coordinate chunk with '
                           'index={index}'.format(index=idx))
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
            self.indices = pickle.load(f)
        with open(self.distance_file, 'rb') as f:
            self.distances = pickle.load(f)
        logger.info(f'Loaded cache files: {self.index_file}, '
                    f'{self.distance_file}')

    def cache_all_queries(self):
        """Cache indices and distances from ball tree query"""
        if self.cache_pattern is not None:
            with open(self.index_file, 'wb') as f:
                pickle.dump(self.indices, f, protocol=4)
            with open(self.distance_file, 'wb') as f:
                pickle.dump(self.distances, f, protocol=4)
            logger.info(f'Saved cache files: {self.index_file}, '
                        f'{self.distance_file}')

    @property
    def index_file(self):
        """Get name of cache indices file"""
        if self.cache_pattern is not None:
            return self.cache_pattern.format(array_name='indices')
        else:
            return None

    @property
    def distance_file(self):
        """Get name of cache distances file"""
        if self.cache_pattern is not None:
            return self.cache_pattern.format(array_name='distances')
        else:
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
        return self.tree.query(self.get_spatial_chunk(s_slice),
                               k=self.k_neighbors)


class Regridder(TreeBuilder):
    """Regridder class for mapping list of coordinates to another.
    Includes weights and indicies used to map from source grid to each point in
    the new grid"""

    @staticmethod
    def interpolate(distance_chunk, values):
        """Interpolate to a new coordinate based on distances from that
        coordinate and the values of the points at those distances

        Parameters
        ----------
        distance_chunk : ndarray
            Chunk of the full array of distances where distances[i] gives the
            list of distances to the source coordinates to be used for
            interpolation for the i-th coordinate in the target data.
        values : ndarray
            Array of values corresponding to the point distances with shape
            (temporal, n_points, k_neighbors)

        Returns
        -------
        ndarray
            Time series of values at interpolated point with shape
            (temporal, n_points)
        """
        weights = 1 / np.array(distance_chunk)
        norm = np.sum(weights, axis=-1)
        out = np.einsum('ijk,jk->ij', values, weights) / norm
        return out

    @classmethod
    def get_source_values(cls, index_chunk, feature, source_files):
        """Get values to use for interpolation

        Parameters
        ----------
        index_chunk : ndarray
            Chunk of the full array of indices where indices[i] gives the
            list of coordinate indices in the source data to be used for
            interpolation for the i-th coordinate in the target data.
        feature : str
            Name of feature to interpolate
        source_files : list
            List of paths to source files

        Returns
        -------
        ndarray
            Array of values to use for interpolation with shape
            (temporal, n_points, k_neighbors)
        """
        with MultiFileResource(source_files) as res:
            shape = (len(res.time_index), len(index_chunk),
                     len(index_chunk[0]))
            tmp = np.array(index_chunk).flatten()
            out = res[feature, :, tmp]
            out = out.reshape(shape)
        return out


class WindRegridder(Regridder):
    """Class to regrid windspeed and winddirection. Includes methods for
    converting windspeed and winddirection to U and V and inverting after
    interpolation"""

    @classmethod
    def get_source_uv(cls, index_chunk, height, source_files):
        """Get u/v wind components from windspeed and winddirection

        Parameters
        ----------
        index_chunk : ndarray
            Chunk of the full array of indices where indices[i] gives the
            list of coordinate indices in the source data to be used for
            interpolation for the i-th coordinate in the target data.
        height : int
            Wind height level
        source_files : list
            List of paths to source files

        Returns
        -------
        u: ndarray
            Array of zonal wind values to use for interpolation with shape
            (temporal, n_points, k_neighbors)
        v: ndarray
            Array of meridional wind values to use for interpolation with shape
            (temporal, n_points, k_neighbors)
        """
        ws = cls.get_source_values(index_chunk, f'windspeed_{height}m',
                                   source_files)
        wd = cls.get_source_values(index_chunk, f'winddirection_{height}m',
                                   source_files)
        u = ws * np.sin(np.radians(wd))
        v = ws * np.cos(np.radians(wd))

        return u, v

    @classmethod
    def invert_uv(cls, u, v):
        """Get u/v wind components from windspeed and winddirection

        Parameters
        ----------
        u: ndarray
            Array of interpolated zonal wind values with shape
            (temporal, n_points)
        v: ndarray
            Array of interpolated meridional wind values with shape
            (temporal, n_points)

        Returns
        -------
        ws: ndarray
            Array of interpolated windspeed values with shape
            (temporal, n_points)
        wd: ndarray
            Array of winddirection values with shape (temporal, n_points)
        """
        ws = np.hypot(u, v)
        wd = np.rad2deg(np.arctan2(u, v))
        wd = (wd + 360) % 360

        return ws, wd

    @classmethod
    def regrid_coordinates(cls, index_chunk, distance_chunk, height,
                           source_files):
        """Regrid wind fields at given height for the requested coordinate
        index

        Parameters
        ----------
        index_chunk : ndarray
            Chunk of the full array of indices where indices[i] gives the
            list of coordinate indices in the source data to be used for
            interpolation for the i-th coordinate in the target data.
        distance_chunk : ndarray
            Chunk of the full array of distances where distances[i] gives the
            list of distances to the source coordinates to be used for
            interpolation for the i-th coordinate in the target data.
        height : int
            Wind height level
        source_files : list
            List of paths to source files

        Returns
        -------
        ws: ndarray
            Array of interpolated windspeed values with shape
            (temporal, n_points)
        wd: ndarray
            Array of winddirection values with shape (temporal, n_points)

        """
        u, v = cls.get_source_uv(index_chunk, height, source_files)
        u = cls.interpolate(distance_chunk, u)
        v = cls.interpolate(distance_chunk, v)
        ws, wd = cls.invert_uv(u, v)
        return ws, wd


class RegridOutput(OutputMixIn, DistributedProcess):
    """Output regridded data as it is interpolated. Takes source data from
    windspeed and winddirection h5 files and uses this data to interpolate onto
    a new target grid. The interpolated data is then written to new files, with
    one file for each field (e.g. windspeed_100m)."""

    def __init__(self, source_files, out_pattern, target_meta, heights,
                 cache_pattern=None, leaf_size=4, k_neighbors=4,
                 incremental=False, n_chunks=100, max_nodes=1,
                 worker_kwargs=None):
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
        self.source_files = (source_files if isinstance(source_files, list)
                             else glob(source_files))
        self.target_meta_path = target_meta
        self.target_meta = pd.read_csv(self.target_meta_path)
        self.target_meta['gid'] = np.arange(len(self.target_meta))
        self.target_meta = self.target_meta.sort_values(
            ['latitude', 'longitude'], ascending=[False, True])
        self.heights = heights
        self.incremental = incremental
        self.out_pattern = out_pattern
        os.makedirs(os.path.dirname(self.out_pattern), exist_ok=True)

        with MultiFileResource(source_files) as res:
            self.time_index = res.time_index
            self.source_meta = res.meta
            self.global_attrs = res.global_attrs

        self.regridder = WindRegridder(self.source_meta,
                                       self.target_meta,
                                       leaf_size=leaf_size,
                                       k_neighbors=k_neighbors,
                                       cache_pattern=cache_pattern,
                                       n_chunks=n_chunks,
                                       max_workers=self.query_workers)
        DistributedProcess.__init__(self, max_nodes=max_nodes,
                                    n_chunks=n_chunks,
                                    max_chunks=len(self.regridder.indices),
                                    incremental=incremental)

        logger.info('Initializing RegridOutput with '
                    f'source_files={self.source_files}, '
                    f'out_pattern={self.out_pattern}, '
                    f'heights={self.heights}, '
                    f'target_meta={target_meta}, '
                    f'k_neighbors={k_neighbors}, and '
                    f'n_chunks={n_chunks}.')
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
        return [self.out_pattern.format(file_id=str(i).zfill(6))
                for i in range(self.chunks)]

    @property
    def output_features(self):
        """Get list of dsets to write to output files"""
        out = []
        for height in self.heights:
            out.append(f'windspeed_{height}m')
            out.append(f'winddirection_{height}m')
        return out

    @classmethod
    def get_node_cmd(cls, config):
        """Get a CLI call to regrid data.

        Parameters
        ----------
        config : dict
            sup3r collection config with all necessary args and kwargs to
            run regridding.
        """

        import_str = ('from sup3r.utilities.regridder import RegridOutput;\n'
                      'from rex import init_logger;\n'
                      'import time;\n'
                      'from reV.pipeline.status import Status;\n')

        regrid_fun_str = get_fun_call_str(cls, config)

        node_index = config['node_index']
        log_file = config.get('log_file', None)
        log_level = config.get('log_level', 'INFO')
        log_arg_str = (f'"sup3r", log_level="{log_level}"')
        if log_file is not None:
            log_arg_str += f', log_file="{log_file}"'

        cmd = (f"python -c \'{import_str}\n"
               "t0 = time.time();\n"
               f"logger = init_logger({log_arg_str});\n"
               f"regrid_output = {regrid_fun_str};\n"
               f"regrid_output.run({node_index});\n"
               "t_elap = time.time() - t0;\n"
               )

        cmd = BaseCLI.add_status_cmd(config, ModuleName.REGRID, cmd)
        cmd += (";\'\n")

        return cmd.replace('\\', '/')

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
            self._run_serial(source_files=self.source_files,
                             node_index=node_index)
        else:
            self._run_parallel(source_files=self.source_files,
                               node_index=node_index,
                               max_workers=self.regrid_workers)

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
            self.write_coordinates(source_files=source_files,
                                   chunk_index=chunk_index)

            mem = psutil.virtual_memory()
            msg = ('Coordinate chunks regridded: {0} out of {1}. '
                   'Current memory usage is {2:.3f} GB out of {3:.3f} '
                   'GB total.'.format(i + 1,
                                      len(self.node_chunks[node_index]),
                                      mem.used / 1e9, mem.total / 1e9))
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
                future = exe.submit(self.write_coordinates,
                                    source_files=source_files,
                                    chunk_index=chunk_index)
                futures[future] = chunk_index
                mem = psutil.virtual_memory()
                msg = ('Regrid futures submitted: {0} out of {1}'.format(
                       i + 1, len(self.node_chunks[node_index])))
                logger.info(msg)

            logger.info(f'Submitted all regrid futures in {dt.now() - now}.')

            for i, future in enumerate(as_completed(futures)):
                idx = futures[future]
                mem = psutil.virtual_memory()
                msg = ('Regrid futures completed: {0} out of {1}, in {2}. '
                       'Current memory usage is {3:.3f} GB out of {4:.3f} GB '
                       'total.'.format(i + 1, len(futures), dt.now() - now,
                                       mem.used / 1e9, mem.total / 1e9))
                logger.info(msg)

                try:
                    future.result()
                except Exception as e:
                    msg = ('Falied to regrid coordinate chunks with '
                           'index={index}'.format(index=idx))
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
                    index_chunk=index_chunk, distance_chunk=distance_chunk,
                    height=height, source_files=source_files)

                features = [f'windspeed_{height}m', f'winddirection_{height}m']

                for dset, data in zip(features, [ws, wd]):
                    attrs, dtype = self.get_dset_attrs(dset)
                    fh.add_dataset(tmp_file, dset, data, dtype=dtype,
                                   attrs=attrs, chunks=attrs['chunks'])

                logger.info(f'Added {features} to {out_file}')
        os.replace(tmp_file, out_file)
        logger.info(f'Finished regridding chunk with s_slice={s_slice}')
