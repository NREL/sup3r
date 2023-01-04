"""Code for regridding data from one list of coordinates to another"""
import numpy as np
from sklearn.neighbors import BallTree
import logging
import psutil
from glob import glob
import pickle
import os
from concurrent.futures import as_completed, ThreadPoolExecutor

from rex import MultiFileResourceX

from sup3r.postprocessing.file_handling import RexOutputs, OutputMixIn


logger = logging.getLogger(__name__)


class Regridder:
    """Regridder class for mapping list of coordinates to another.
    Includes weights and indicies used to map from source grid to each point in
    the new grid"""

    def __init__(self, source_meta, target_meta, cache_pattern=None,
                 leaf_size=3, k=3):
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
        k : int, optional
            number of nearest neighbors to use for interpolation
        """
        self.cache_pattern = cache_pattern
        cache_exists_check = (self.index_file is not None
                              and os.path.exists(self.index_file)
                              and self.distance_file is not None
                              and os.path.exists(self.distance_file))
        if cache_exists_check:
            with open(self.index_file, 'rb') as f:
                self.indices = pickle.load(f)
            with open(self.distance_file, 'rb') as f:
                self.distances = pickle.load(f)
        else:
            logger.info("Building ball tree for regridding")
            self.tree = BallTree(source_meta[['latitude', 'longitude']],
                                 leaf_size=leaf_size, metric='haversine')
            logger.info("Getting points and distances from ball tree.")
            out = self.tree.query(target_meta[['latitude', 'longitude']], k=k)
            self.distances, self.indices = out

            self.cache_ball_tree()

    def cache_ball_tree(self):
        """Cache indices and distances from ball tree query"""
        if self.cache_pattern is not None:
            with open(self.index_file, 'wb') as f:
                pickle.dump(self.indices, f, protocol=4)
            with open(self.distance_file, 'wb') as f:
                pickle.dump(self.distances, f, protocol=4)

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

    @staticmethod
    def interpolate(distances, values):
        """Interpolate to a new coordinate based on distances from that
        coordinate and the values of the points at those distances

        Parameters
        ----------
        distances : ndarray
            Array of distances from interpolation point with shape (n_points)
        values : ndarray
            Array of values corresponding to the point distances with shape
            (temporal, n_points)

        Returns
        -------
        ndarray
            Time series of values at interpolated point with shape (temporal)
        """
        if any(w == 0 for w in distances):
            return values[..., np.where(distances == 0)[0][0]]

        weights = 1 / distances
        return np.inner(values, weights) / np.sum(weights)

    def get_source_values(self, index, feature, resource):
        """Get values to use for interpolation

        Parameters
        ----------
        index : int
            Index of the interpolated point in the target grid
        feature : str
            Name of feature to interpolate
        resource : ResourceX
            ResourceX data handler for source data

        Returns
        -------
        ndarray
            Array of values to use for interpolation with shape
            (temporal, n_points)
        """
        src_indices = self.indices[index]
        return resource[feature, :, src_indices]

    def get_interpolated_values(self, index, src_values):
        """Get interpolated values using values from source grid

        Parameters
        ----------
        index : int
            Index of the interpolated point in the target grid
        src_values : ndarray
            Array of values from source data to use for interpolation with
            shape (temporal, n_points)

        Returns
        -------
        ndarray
            Array of interpolated time series values with shape (temporal)
        """
        distances = self.distances[index]
        return self.interpolate(distances, src_values)


class WindRegridder(Regridder):
    """Class to regrid windspeed and winddirection. Includes methods for
    converting windspeed and winddirection to U and V and inverting after
    interpolation"""

    def get_source_uv(self, index, height, resource):
        """Get u/v wind components from windspeed and winddirection

        Parameters
        ----------
        index : int
            Index of the interpolated point in the target grid
        height : int
            Wind height level
        resource : MultiFileResourceX
            Resource handler for source data

        Returns
        -------
        u: ndarray
            Array of zonal wind values to use for interpolation with shape
            (temporal, n_points)
        v: ndarray
            Array of meridional wind values to use for interpolation with shape
            (temporal, n_points)
        """
        ws = self.get_source_values(index, f'windspeed_{height}m',
                                    resource)
        wd = self.get_source_values(index, f'winddirection_{height}m',
                                    resource)
        u = ws * np.sin(np.radians(wd))
        v = ws * np.cos(np.radians(wd))

        return u, v

    def invert_uv(self, u, v):
        """Get u/v wind components from windspeed and winddirection

        Parameters
        ----------
        u: ndarray
            Array of interpolated zonal wind values with shape (temporal)
        v: ndarray
            Array of interpolated meridional wind values with shape (temporal)

        Returns
        -------
        ws: ndarray
            Array of interpolated windspeed values with shape (temporal)
        wd: ndarray
            Array of winddirection values with shape (temporal)
        """
        ws = np.hypot(u, v)
        wd = np.rad2deg(np.arctan2(u, v))
        wd = (wd + 360) % 360

        return ws, wd

    def regrid_coordinate(self, index, height, resource):
        """Regrid wind fields at given height for the requested coordinate
        index

        Parameters
        ----------
        index : int
            Index of the interpolated point in the target grid
        height : int
            Wind height level
        resource : ResourceX
            ResourceX data handler for source data

        Returns
        -------
        ws: ndarray
            Array of interpolated windspeed values with shape (temporal)
        wd: ndarray
            Array of winddirection values with shape (temporal)

        """
        u, v = self.get_source_uv(index, height, resource)
        u = self.get_interpolated_values(index, u)
        v = self.get_interpolated_values(index, v)
        ws, wd = self.invert_uv(u, v)
        return ws, wd


class RegridOutput(OutputMixIn):
    """Output regridded data as it is interpolated"""

    def __init__(self, source_files, output_pattern, target_meta, heights,
                 cache_pattern=None, leaf_size=3, k=3):
        """
        Parameters
        ----------
        source_files : str | list
            Path to source files to regrid to target_meta
        output_pattern : str
            Pattern to use for naming outputs file to store the regridded data.
            This must include a {feature} format key. e.g. ./{feature}.h5
        target_meta : pd.DataFrame
            Dataframe of final grid coordinates on which to regrid
        heights : list
            List of wind field heights to regrid. e.g if heights = [100] then
            windspeed_100m and winddirection_100m will be regridded and stored
            in the output_file.
        cache_pattern : str
            Pattern for cached indices and distances for ball tree
        leaf_size : int, optional
            leaf size for BallTree
        k : int, optional
            number of nearest neighbors to use for interpolation
        """
        self.source_files = (source_files if isinstance(source_files, list)
                             else glob(source_files))
        self.output_pattern = output_pattern
        self.target_meta = target_meta
        self.heights = heights

        with MultiFileResourceX(source_files) as res:
            self.time_index = res.time_index
            self.source_meta = res.meta
            self.attrs = res.attrs
            self.global_attrs = res.global_attrs

        self.regridder = WindRegridder(self.source_meta, self.target_meta,
                                       leaf_size=leaf_size, k=k,
                                       cache_pattern=cache_pattern)
        for out_file in self.output_files:
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
    def regrid(cls, source_files, output_pattern, target_meta, heights,
               cache_pattern=None, max_workers=None):
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
        cache_pattern : str | None
            File name pattern for ball tree indices and distances
        max_workers : int | None
            Max number of workers to use for regridding and output
        """
        regrid_output = cls(source_files=source_files,
                            output_pattern=output_pattern,
                            target_meta=target_meta,
                            cache_pattern=cache_pattern,
                            heights=heights)

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

                        if max_workers == 1:
                            regrid_output._run_serial(
                                src_res=src_res, ws_res=ws_res, wd_res=wd_res,
                                height=height)

                        else:
                            regrid_output._run_parallel(
                                src_res=src_res, ws_res=ws_res, wd_res=wd_res,
                                height=height, max_workers=max_workers)
        logger.info(f'Finished writing output files: {output_files}')

    def _run_serial(self, src_res, ws_res, wd_res, height):
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
        """
        for index in range(len(self.target_meta)):
            self.write_coordinate(src_res, ws_res, wd_res, height, index)

    def _run_parallel(self, src_res, ws_res, wd_res, height, max_workers=None):
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
        max_workers : int | None
            Max number of workers to use for regridding in parallel
        """
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            for index in range(len(self.target_meta)):
                future = exe.submit(self.write_coordinate, src_res=src_res,
                                    ws_res=ws_res, wd_res=wd_res,
                                    height=height, index=index)
                futures[future] = index

            interval = int(np.ceil(len(futures) / 10))
            for i, future in enumerate(as_completed(futures)):
                if interval > 0 and i % interval == 0:
                    mem = psutil.virtual_memory()
                    msg = ('Regrid futures completed: {0} out of '
                           '{1}. Current memory usage is {2:.3f} '
                           'GB out of {3:.3f} GB total.'.format(
                               i + 1, len(futures), mem.used / 1e9,
                               mem.total / 1e9))
                    logger.info(msg)
                try:
                    idx = futures[future]
                    future.result()
                except Exception as e:
                    msg = ('Falied to regrid coordinate with '
                           'index={}'.format(index=idx))
                    logger.exception(msg)
                    raise RuntimeError(msg) from e

    def write_coordinate(self, src_res, ws_res, wd_res, height, index):
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
        index : int
            Index of coordinate to regrid and write to output file
        """
        out = self.regridder.regrid_coordinate(index, height, src_res)
        ws_res[f'windspeed_{height}m', :, index] = out[0]
        wd_res[f'winddirection_{height}m', :, index] = out[1]
