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

from rex import MultiFileResourceX

from sup3r.postprocessing.file_handling import RexOutputs, OutputMixIn


logger = logging.getLogger(__name__)


class TreeBuilder:
    """TreeBuilder class for building ball tree and running all queries to
    create full arrays of indices and distances for neighbor points
    """

    def __init__(self, source_meta, target_meta, cache_pattern=None,
                 leaf_size=3, k_neighbors=3, n_chunks=100, max_workers=None):
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
            leaf_size=3, k_neighbors=3, n_chunks=100, max_workers=None):
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
            logger.info(f'Querying all coordinates in serial')
            self._serial_queries()

        else:
            logger.info(f'Querying all coordinates in parallel')
            self._parallel_queries(max_workers=max_workers)

    def _serial_queries(self):
        """Get indices and distances for all points in target_meta, in serial
        """
        out = self.tree.query(self.target_meta[['latitude', 'longitude']],
                              k=self.k_neighbors)
        self.distances, self.indices = out

    def _parallel_queries(self, max_workers=None):
        """Get indices and distances for all points in target_meta, in serial
        """
        futures = {}
        now = dt.now()
        slices = np.arange(len(self.target_meta))
        slices = np.array_split(slices, min(self.n_chunks, len(slices)))
        slices = [slice(s[0], s[-1] + 1) for s in slices]
        interval = min(10, int(np.ceil(len(slices) / 100)))
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            for i, s_slice in enumerate(slices):
                future = exe.submit(self.save_query, s_slice=s_slice)
                futures[future] = i
                if i % interval == 0:
                    mem = psutil.virtual_memory()
                    msg = ('Query futures submitted: {0} out of {1}. Current '
                           'memory usage is {2:.3f} GB out of {3:.3f} GB '
                           'total.'.format(i + 1, len(slices),
                                           mem.used / 1e9, mem.total / 1e9))
                    logger.info(msg)

            logger.info(f'Submitted all query futures in {dt.now() - now}.')

            interval = int(np.ceil(len(futures) / 10))
            for i, future in enumerate(as_completed(futures)):
                idx = futures[future]
                if interval > 0 and i % interval == 0:
                    mem = psutil.virtual_memory()
                    msg = ('Query futures completed: {0} out of '
                           '{1}. Current memory usage is {2:.3f} '
                           'GB out of {3:.3f} GB total.'.format(
                               i + 1, len(futures), mem.used / 1e9,
                               mem.total / 1e9))
                    logger.info(msg)
                try:
                    future.result()
                except Exception as e:
                    msg = ('Falied to query coordinate chunk with '
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
    def interpolate(distances, values):
        """Interpolate to a new coordinate based on distances from that
        coordinate and the values of the points at those distances

        Parameters
        ----------
        distances : ndarray
            Array of distances from interpolation point with shape
            (n_points, k_neighbors)
        values : ndarray
            Array of values corresponding to the point distances with shape
            (temporal, n_points, k_neighbors)

        Returns
        -------
        ndarray
            Time series of values at interpolated point with shape
            (temporal, n_points)
        """
        weights = 1 / np.array(distances)
        norm = np.sum(weights, axis=-1)
        out = np.einsum('ijk,jk->ij', values, weights) / norm
        return out

    def get_source_values(self, s_slice, feature, resource):
        """Get values to use for interpolation

        Parameters
        ----------
        s_slice : slice
            slice specifying which spatial indices in the target grid should be
            used for interpolation. This selects n_points from the target grid
        feature : str
            Name of feature to interpolate
        resource : ResourceX
            ResourceX data handler for source data

        Returns
        -------
        ndarray
            Array of values to use for interpolation with shape
            (temporal, n_points, k_neighbors)
        """
        shape = (len(resource.time_index), len(self.indices[s_slice]), -1)
        out = resource[feature, :, np.array(self.indices[s_slice]).flatten()]
        out = out.reshape(shape)
        return out

    def get_interpolated_values(self, s_slice, src_values):
        """Get interpolated values using values from source grid

        Parameters
        ----------
        s_slice : slice
            slice specifying which spatial indices in the target grid should be
            used for interpolation. This selects n_points from the target grid
        src_values : ndarray
            Array of values from source data to use for interpolation with
            shape (temporal, n_points, k_neighbors)

        Returns
        -------
        ndarray
            Array of interpolated time series values with shape (temporal)
        """
        return self.interpolate(self.distances[s_slice], src_values)

    def saved_query_check(self, s_slice):
        """Make sure ball tree query has been stored in index and distance
        arrays"""
        check_stored_query = (
            all(idx is not None for idx in self.indices[s_slice])
            and all(dist is not None for dist in self.distances[s_slice]))
        if not check_stored_query:
            self.save_query(s_slice)


class WindRegridder(Regridder):
    """Class to regrid windspeed and winddirection. Includes methods for
    converting windspeed and winddirection to U and V and inverting after
    interpolation"""

    def get_source_uv(self, s_slice, height, resource):
        """Get u/v wind components from windspeed and winddirection

        Parameters
        ----------
        s_slice : slice
            slice specifying target grid indices to use for interpolation
        height : int
            Wind height level
        resource : MultiFileResourceX
            Resource handler for source data

        Returns
        -------
        u: ndarray
            Array of zonal wind values to use for interpolation with shape
            (temporal, n_points, k_neighbors)
        v: ndarray
            Array of meridional wind values to use for interpolation with shape
            (temporal, n_points, k_neighbors)
        """
        ws = self.get_source_values(s_slice, f'windspeed_{height}m',
                                    resource)
        wd = self.get_source_values(s_slice, f'winddirection_{height}m',
                                    resource)
        u = ws * np.sin(np.radians(wd))
        v = ws * np.cos(np.radians(wd))

        return u, v

    def invert_uv(self, u, v):
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

    def regrid_coordinates(self, s_slice, height, resource):
        """Regrid wind fields at given height for the requested coordinate
        index

        Parameters
        ----------
        s_slice : slice
            slice specifying range of indices in the target grid to interpolate
        height : int
            Wind height level
        resource : ResourceX
            ResourceX data handler for source data

        Returns
        -------
        ws: ndarray
            Array of interpolated windspeed values with shape
            (temporal, n_points)
        wd: ndarray
            Array of winddirection values with shape (temporal, n_points)

        """
        u, v = self.get_source_uv(s_slice, height, resource)
        u = self.get_interpolated_values(s_slice, u)
        v = self.get_interpolated_values(s_slice, v)
        ws, wd = self.invert_uv(u, v)
        return ws, wd


class RegridOutput(OutputMixIn):
    """Output regridded data as it is interpolated"""

    def __init__(self, source_files, output_pattern, target_meta, heights,
                 cache_pattern=None, leaf_size=40, k_neighbors=3,
                 overwrite=False, n_chunks=100, worker_kwargs=None):
        """
        Parameters
        ----------
        source_files : str | list
            Path to source files to regrid to target_meta
        output_pattern : str
            Pattern to use for naming outputs file to store the regridded data.
            This must include a {feature} format key. e.g. ./{feature}.h5
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
        overwrite : bool
            Whether to overwrite previously saved output files
        n_chunks : int
            Number of spatial chunks to use for interpolation. The total number
            of points in the target_meta will be split into n_chunks and the
            points in each chunk will be interpolated at the same time.
        worker_kwargs : dict | None
            Dictionary of workers args. Optional keys include regrid_workers
            (max number of workers to use for regridding and output)
        """
        worker_kwargs = worker_kwargs or {}
        self.regrid_workers = worker_kwargs.get('regrid_workers', None)
        self.query_workers = worker_kwargs.get('query_workers', None)
        self.source_files = (source_files if isinstance(source_files, list)
                             else glob(source_files))
        self.output_pattern = output_pattern
        self.target_meta = pd.read_csv(target_meta)
        self.heights = heights
        if 'gid' in self.target_meta.columns:
            self.target_meta = self.target_meta.drop(['gid'], axis=1)

        logger.info('Initializing RegridOutput with '
                    f'source_files={self.source_files} and '
                    f'output_pattern={self.output_pattern}.')

        with MultiFileResourceX(source_files) as res:
            self.time_index = res.time_index
            self.source_meta = res.meta
            self.attrs = res.attrs
            self.global_attrs = res.global_attrs

        self.regridder = WindRegridder(self.source_meta, self.target_meta,
                                       leaf_size=leaf_size,
                                       k_neighbors=k_neighbors,
                                       cache_pattern=cache_pattern,
                                       n_chunks=n_chunks,
                                       max_workers=self.query_workers)
        for out_file in self.output_files:
            if os.path.exists(out_file) and overwrite:
                logger.info(f'{out_file} already exists but overwrite=True. '
                            'Proceeding with overwrite.')
                os.remove(out_file)
            self._init_h5(out_file, self.time_index, self.target_meta,
                          self.global_attrs)

    @property
    def output_files(self):
        """Get list of output files"""
        out = []
        for height in self.heights:
            out.append(self.output_pattern.format(
                feature=f'windspeed_{height}m'))
            out.append(self.output_pattern.format(
                feature=f'winddirection_{height}m'))
        return out

    def get_height_output_files(self, height):
        """Get output files for a given height. e.g. if height = 100 this
        returns the files for windspeed_100m and winddirection_100m

        Parameters
        ----------
        height : int
            Wind level height. e.g. 100 for 100 meter level.
        """
        index = self.heights.index(height)
        return self.output_files[2 * index: 2 * (index + 1)]

    @classmethod
    def run(cls, source_files, output_pattern, target_meta, heights,
            n_chunks=100, k_neighbors=3, cache_pattern=None,
            worker_kwargs=None, overwrite=False):
        """
        Parameters
        ----------
        source_files : str | list
            Path to source files to regrid to target_meta
        output_pattern : str
            Pattern to use for naming outputs file to store the regridded data
        target_meta : pd.DataFrame
            Dataframe of final grid coordinates on which to regrid
        heights : list
            List of wind field heights to regrid. e.g if heights = [100] then
            windspeed_100m and winddirection_100m will be regridded and stored
            in the output_file.
        n_chunks : int
            Number of spatial chunks to use for interpolation. The total number
            of points in the target_meta will be split into n_chunks and the
            points in each chunk will be interpolated at the same time.
        k_neighbors : int, optional
            number of nearest neighbors to use for interpolation
        cache_pattern : str | None
            File name pattern for ball tree indices and distances
        worker_kwargs : dict | None
            Dictionary of workers args. Optional keys include regrid_workers
            (max number of workers to use for regridding and output)
        overwrite : bool
            Whether to overwrite previously saved output files
        """
        regrid_output = cls(source_files=source_files,
                            output_pattern=output_pattern,
                            target_meta=target_meta,
                            cache_pattern=cache_pattern,
                            heights=heights,
                            overwrite=overwrite,
                            n_chunks=n_chunks,
                            worker_kwargs=worker_kwargs,
                            k_neighbors=k_neighbors)

        with MultiFileResourceX(regrid_output.source_files) as src_res:
            for height in heights:
                output_files = regrid_output.get_height_output_files(height)
                with RexOutputs(output_files[0], 'a') as ws_res:
                    with RexOutputs(output_files[1], 'a') as wd_res:
                        cls._ensure_dset_in_output(output_files[0],
                                                   f'windspeed_{height}m',
                                                   data=None)
                        cls._ensure_dset_in_output(output_files[1],
                                                   f'winddirection_{height}m',
                                                   data=None)
                        regrid_output.regrid(src_res=src_res, ws_res=ws_res,
                                             wd_res=wd_res, height=height,
                                             n_chunks=n_chunks)
        logger.info(f'Finished writing output files: {output_files}')

    def regrid(self, src_res, ws_res, wd_res, height, n_chunks):
        """Regrid data and write to output file.

        Parameters
        ----------
        src_res : RexOutputs
            Resource handler for source data
        ws_res : RexOutputs
            Resource handler for windspeed output data
        wd_res : RexOutputs
            Resource handler for winddirection output data
        height : int
            Wind level height to write to output file
        n_chunks : int
            Number of chunks to split target_meta coordinates into to perform
            interpolation in chunks.
        """
        if self.regrid_workers == 1:
            self._run_serial(src_res=src_res, ws_res=ws_res, wd_res=wd_res,
                             height=height, n_chunks=n_chunks)

        else:
            self._run_parallel(src_res=src_res, ws_res=ws_res, wd_res=wd_res,
                               height=height, n_chunks=n_chunks,
                               max_workers=self.regrid_workers)

    def _run_serial(self, src_res, ws_res, wd_res, height, n_chunks):
        """Regrid data and write to output file, in serial.

        Parameters
        ----------
        src_res : RexOutputs
            Resource handler for source data
        ws_res : RexOutputs
            Resource handler for windspeed output data
        wd_res : RexOutputs
            Resource handler for winddirection output data
        height : int
            Wind level height to write to output file
        n_chunks : int
            Number of chunks to split target_meta coordinates into to perform
            interpolation in chunks.
        """
        logger.info('Regridding all coordinates in serial.')
        slices = np.arange(len(self.target_meta))
        slices = np.array_split(slices, min(n_chunks, len(slices)))
        slices = [slice(s[0], s[-1] + 1) for s in slices]
        interval = min(10, int(np.ceil(len(slices) / 100)))
        for i, s_slice in enumerate(slices):
            self.write_coordinates(src_res, ws_res, wd_res, height, s_slice)
            if i % interval == 0:
                mem = psutil.virtual_memory()
                msg = ('Coordinate chunks regridded: {0} out of {1}. Current '
                       'memory usage is {2:.3f} GB out of {3:.3f} GB '
                       'total.'.format(i + 1, len(slices),
                                       mem.used / 1e9, mem.total / 1e9))
                logger.info(msg)

    def _run_parallel(self, src_res, ws_res, wd_res, height, n_chunks,
                      max_workers=None):
        """Regrid data and write to output file, in parallel.

        Parameters
        ----------
        src_res : RexOutputs
            Resource handler for source data
        ws_res : RexOutputs
            Resource handler for windspeed output data
        wd_res : RexOutputs
            Resource handler for winddirection output data
        height : int
            Wind level height to write to output file
        n_chunks : int
            Number of chunks to split target_meta coordinates into to perform
            interpolation in chunks.
        max_workers : int | None
            Max number of workers to use for regridding in parallel
        """
        futures = {}
        now = dt.now()
        logger.info('Regridding all coordinates in parallel.')
        slices = np.arange(len(self.target_meta))
        slices = np.array_split(slices, min(n_chunks, len(slices)))
        slices = [slice(s[0], s[-1] + 1) for s in slices]
        interval = min(10, int(np.ceil(len(slices) / 100)))
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            for i, s_slice in enumerate(slices):
                future = exe.submit(self.write_coordinates, src_res=src_res,
                                    ws_res=ws_res, wd_res=wd_res,
                                    height=height, s_slice=s_slice)
                futures[future] = i
                if i % interval == 0:
                    mem = psutil.virtual_memory()
                    msg = ('Regrid futures submitted: {0} out of {1}. Current '
                           'memory usage is {2:.3f} GB out of {3:.3f} GB '
                           'total.'.format(i + 1, len(slices),
                                           mem.used / 1e9, mem.total / 1e9))
                    logger.info(msg)

            logger.info(f'Submitted all regrid futures in {dt.now() - now}.')

            for i, future in enumerate(as_completed(futures)):
                idx = futures[future]
                mem = psutil.virtual_memory()
                msg = ('Regrid futures completed: {0} out of {1}. Current '
                       'memory usage is {2:.3f} GB out of {3:.3f} GB '
                       'total.'.format(i + 1, len(futures), mem.used / 1e9,
                                       mem.total / 1e9))
                logger.info(msg)

                try:
                    future.result()
                except Exception as e:
                    msg = ('Falied to regrid coordinate chunks with '
                           'index={index}'.format(index=idx))
                    logger.exception(msg)
                    raise RuntimeError(msg) from e

    def write_coordinates(self, src_res, ws_res, wd_res, height, s_slice):
        """Write regridded coordinate data to the output file

        Parameters
        ----------
        src_res : MultiFileResourceX
            Resource handler for source data
        ws_res : RexOutputs
            Resource handler for windspeed output data
        wd_res : RexOutputs
            Resource handler for winddirection output data
        height : int
            Wind level height to write to output file
        s_slice : s_slice
            slice specifying indices of coordinates to regrid and write to
            output file
        """
        out = self.regridder.regrid_coordinates(s_slice, height, src_res)
        ws_res[f'windspeed_{height}m', :, s_slice] = out[0]
        wd_res[f'winddirection_{height}m', :, s_slice] = out[1]
