"""Sup3r topography utilities"""

import logging
import os
import shutil
from abc import ABC, abstractmethod
from warnings import warn

import dask.array as da
import numpy as np
import pandas as pd
from rex import Resource
from rex.utilities.solar_position import SolarPosition
from scipy.spatial import KDTree

from sup3r.containers import (
    Cacher,
    LoaderH5,
    LoaderNC,
)
from sup3r.postprocessing.file_handling import OutputHandler
from sup3r.utilities.utilities import (
    generate_random_string,
    get_class_kwargs,
    get_input_handler_class,
    nn_fill_array,
)

logger = logging.getLogger(__name__)


class ExoExtract(ABC):
    """Class to extract high-res (4km+) data rasters for new
    spatially-enhanced datasets (e.g. GCM files after spatial enhancement)
    using nearest neighbor mapping and aggregation from NREL datasets
    (e.g. WTK or NSRDB)
    """

    def __init__(
        self,
        file_paths,
        exo_source,
        s_enhance,
        t_enhance,
        t_agg_factor,
        target=None,
        shape=None,
        time_slice=None,
        raster_file=None,
        max_delta=20,
        input_handler=None,
        cache_data=True,
        cache_dir='./exo_cache/',
        distance_upper_bound=None,
        res_kwargs=None,
    ):
        """Parameters
        ----------
        file_paths : str | list
            A single source h5 file to extract raster data from or a list
            of netcdf files with identical grid. The string can be a unix-style
            file path which will be passed through glob.glob. This is
            typically low-res WRF output or GCM netcdf data files that is
            source low-resolution data intended to be sup3r resolved.
        exo_source : str
            Filepath to source data file to get hi-res elevation data from
            which will be mapped to the enhanced grid of the file_paths input.
            Pixels from this exo_source will be mapped to their nearest low-res
            pixel in the file_paths input. Accordingly, exo_source should be a
            significantly higher resolution than file_paths. Warnings will be
            raised if the low-resolution pixels in file_paths do not have
            unique nearest pixels from exo_source. File format can be .h5 for
            TopoExtractH5 or .nc for TopoExtractNC
        s_enhance : int
            Factor by which the Sup3rGan model will enhance the spatial
            dimensions of low resolution data from file_paths input. For
            example, if getting topography data, file_paths has 100km data, and
            s_enhance is 4, this class will output a topography raster
            corresponding to the file_paths grid enhanced 4x to ~25km
        t_enhance : int
            Factor by which the Sup3rGan model will enhance the temporal
            dimension of low resolution data from file_paths input. For
            example, if getting sza data, file_paths has hourly data, and
            t_enhance is 4, this class will output a sza raster
            corresponding to the file_paths temporally enhanced 4x to 15 min
        t_agg_factor : int
            Factor by which to aggregate / subsample the exo_source data to the
            resolution of the file_paths input enhanced by t_enhance. For
            example, if getting sza data, file_paths have hourly data, and
            t_enhance is 4 resulting in a target resolution of 15 min and
            exo_source has a resolution of 5 min, the t_agg_factor should be 3
            so that only timesteps that are a multiple of 15min are selected
            e.g., [0, 5, 10, 15, 20, 25, 30][slice(0, None, 3)] = [0, 15, 30]
        target : tuple
            (lat, lon) lower left corner of raster. Either need target+shape or
            raster_file.
        shape : tuple
            (rows, cols) grid size. Either need target+shape or raster_file.
        time_slice : slice | None
            slice used to extract interval from temporal dimension for input
            data and source data
        raster_file : str | None
            File for raster_index array for the corresponding target and shape.
            If specified the raster_index will be loaded from the file if it
            exists or written to the file if it does not yet exist.  If None
            raster_index will be calculated directly. Either need target+shape
            or raster_file.
        max_delta : int, optional
            Optional maximum limit on the raster shape that is retrieved at
            once. If shape is (20, 20) and max_delta=10, the full raster will
            be retrieved in four chunks of (10, 10). This helps adapt to
            non-regular grids that curve over large distances, by default 20
        input_handler : str
            data handler class to use for input data. Provide a string name to
            match a :class:`Extracter`. If None the correct handler will
            be guessed based on file type and time series properties.
        cache_data : bool
            Flag to cache exogeneous data in <cache_dir>/exo_cache/ this can
            speed up forward passes with large temporal extents when the exo
            data is time independent.
        cache_dir : str
            Directory for storing cache data. Default is './exo_cache'
        distance_upper_bound : float | None
            Maximum distance to map high-resolution data from exo_source to the
            low-resolution file_paths input. None (default) will calculate this
            based on the median distance between points in exo_source
        res_kwargs : dict | None
            Dictionary of kwargs passed to lowest level resource handler. e.g.
            xr.open_dataset(file_paths, **res_kwargs)
        """
        logger.info(f'Initializing {self.__class__.__name__} utility.')

        self._exo_source = exo_source
        self._source_data = None
        self._s_enhance = s_enhance
        self._t_enhance = t_enhance
        self._t_agg_factor = t_agg_factor
        self._tree = None
        self._hr_lat_lon = None
        self._source_lat_lon = None
        self._hr_time_index = None
        self._src_time_index = None
        self._distance_upper_bound = distance_upper_bound
        self.cache_data = cache_data
        self.cache_dir = cache_dir
        self.time_slice = time_slice
        self.target = target
        self.shape = shape
        self.res_kwargs = res_kwargs
        self._source_handler = None
        InputHandler = get_input_handler_class(file_paths, input_handler)
        kwargs = {
            'file_paths': file_paths,
            'target': target,
            'shape': shape,
            'time_slice': time_slice,
            'raster_file': raster_file,
            'max_delta': max_delta,
            'res_kwargs': self.res_kwargs,
        }
        self.input_handler = InputHandler(
            **get_class_kwargs(InputHandler, kwargs)
        )
        self.lr_lat_lon = self.input_handler.lat_lon

    @property
    @abstractmethod
    def source_data(self):
        """Get the 1D array of source data from the exo_source_h5"""

    def get_cache_file(self, feature, s_enhance, t_enhance, t_agg_factor):
        """Get cache file name

        Parameters
        ----------
        feature : str
            Name of feature to get cache file for
        s_enhance : int
            Spatial enhancement for this exogeneous data step (cumulative for
            all model steps up to the current step).
        t_enhance : int
            Temporal enhancement for this exogeneous data step (cumulative for
            all model steps up to the current step).
        t_agg_factor : int
            Factor by which to aggregate the exo_source data to the temporal
            resolution of the file_paths input enhanced by t_enhance.

        Returns
        -------
        cache_fp : str
            Name of cache file. This is a netcdf files which will be saved with
            :class:`Cacher` and loaded with :class:`LoaderNC`
        """
        tsteps = (
            None
            if self.time_slice is None
            or self.time_slice.start is None
            or self.time_slice.stop is None
            else self.time_slice.stop - self.time_slice.start
        )
        fn = f'exo_{feature}_{self.target}_{self.shape},{tsteps}'
        fn += f'_tagg{t_agg_factor}_{s_enhance}x_'
        fn += f'{t_enhance}x.nc'
        fn = fn.replace('(', '').replace(')', '')
        fn = fn.replace('[', '').replace(']', '')
        fn = fn.replace(',', 'x').replace(' ', '')
        cache_fp = os.path.join(self.cache_dir, fn)
        if self.cache_data:
            os.makedirs(self.cache_dir, exist_ok=True)
        return cache_fp

    @property
    def source_lat_lon(self):
        """Get the 2D array (n, 2) of lat, lon data from the exo_source_h5"""
        if self._source_lat_lon is None:
            with LoaderH5(self._exo_source) as res:
                self._source_lat_lon = res.lat_lon
        return self._source_lat_lon

    @property
    def lr_shape(self):
        """Get the low-resolution spatial shape tuple"""
        return (
            self.lr_lat_lon.shape[0],
            self.lr_lat_lon.shape[1],
            len(self.input_handler.time_index),
        )

    @property
    def hr_shape(self):
        """Get the high-resolution spatial shape tuple"""
        return (
            self._s_enhance * self.lr_lat_lon.shape[0],
            self._s_enhance * self.lr_lat_lon.shape[1],
            self._t_enhance * len(self.input_handler.time_index),
        )

    @property
    def hr_lat_lon(self):
        """Lat lon grid for data in format (spatial_1, spatial_2, 2) Lat/Lon
        array with same ordering in last dimension. This corresponds to the
        enhanced meta data from the file_paths input * s_enhance.

        Returns
        -------
        ndarray
        """
        if self._hr_lat_lon is None:
            if self._s_enhance > 1:
                self._hr_lat_lon = OutputHandler.get_lat_lon(
                    self.lr_lat_lon, self.hr_shape[:-1]
                )
            else:
                self._hr_lat_lon = self.lr_lat_lon
        return self._hr_lat_lon

    @property
    def source_time_index(self):
        """Get the full time index of the exo_source data"""
        if self._src_time_index is None:
            if self._t_agg_factor > 1:
                self._src_time_index = OutputHandler.get_times(
                    self.input_handler.time_index,
                    self.hr_shape[-1] * self._t_agg_factor,
                )
            else:
                self._src_time_index = self.hr_time_index
        return self._src_time_index

    @property
    def hr_time_index(self):
        """Get the full time index for aggregated source data"""
        if self._hr_time_index is None:
            if self._t_enhance > 1:
                self._hr_time_index = OutputHandler.get_times(
                    self.input_handler.time_index, self.hr_shape[-1]
                )
            else:
                self._hr_time_index = self.input_handler.time_index
        return self._hr_time_index

    @property
    def distance_upper_bound(self):
        """Maximum distance (float) to map high-resolution data from exo_source
        to the low-resolution file_paths input."""
        if self._distance_upper_bound is None:
            diff = da.diff(self.source_lat_lon, axis=0)
            diff = da.median(diff, axis=0).max()
            self._distance_upper_bound = diff
            logger.info(
                'Set distance upper bound to {:.4f}'.format(
                    self._distance_upper_bound.compute()
                )
            )
        return self._distance_upper_bound

    @property
    def tree(self):
        """Get the KDTree built on the target lat lon data from the file_paths
        input with s_enhance"""
        if self._tree is None:
            self._tree = KDTree(self.hr_lat_lon.reshape((-1, 2)))
        return self._tree

    @property
    def nn(self):
        """Get the nearest neighbor indices"""
        _, nn = self.tree.query(
            self.source_lat_lon,
            k=1,
            distance_upper_bound=self.distance_upper_bound,
        )
        return nn

    @property
    def data(self):
        """Get a raster of source values corresponding to the
        high-resolution grid (the file_paths input grid * s_enhance *
        t_enhance). The shape is (lats, lons, temporal, 1)
        """
        cache_fp = self.get_cache_file(
            feature=self.__class__.__name__,
            s_enhance=self._s_enhance,
            t_enhance=self._t_enhance,
            t_agg_factor=self._t_agg_factor,
        )
        tmp_fp = cache_fp + f'.{generate_random_string(10)}.tmp'
        if os.path.exists(cache_fp):
            data = LoaderNC(cache_fp)[self.__class__.__name__]

        else:
            data = self.get_data()

            if self.cache_data:
                coords = {
                    'latitude': (
                        ('south_north', 'west_east'),
                        self.hr_lat_lon[..., 0],
                    ),
                    'longitude': (
                        ('south_north', 'west_east'),
                        self.hr_lat_lon[..., 1],
                    ),
                    'time': self.hr_time_index.values,
                }
                Cacher.write_netcdf(
                    tmp_fp,
                    feature=self.__class__.__name__,
                    data=da.broadcast_to(data, self.hr_shape),
                    coords=coords,
                )
                shutil.move(tmp_fp, cache_fp)

        if data.shape[-1] == 1 and self.hr_shape[-1] > 1:
            data = da.repeat(data, self.hr_shape[-1], axis=-1)

        return data[..., None]

    @abstractmethod
    def get_data(self):
        """Get a raster of source values corresponding to the
        high-resolution grid (the file_paths input grid * s_enhance *
        t_enhance). The shape is (lats, lons, temporal)
        """


class TopoExtractH5(ExoExtract):
    """TopoExtract for H5 files"""

    @property
    def source_data(self):
        """Get the 1D array of elevation data from the exo_source_h5"""
        if self._source_data is None:
            with LoaderH5(self._exo_source) as res:
                self._source_data = res['topography'][..., None]
        return self._source_data

    @property
    def source_time_index(self):
        """Time index of the source exo data"""
        if self._src_time_index is None:
            with Resource(self._exo_source) as res:
                self._src_time_index = res.time_index
        return self._src_time_index

    def get_data(self):
        """Get a raster of source values corresponding to the
        high-resolution grid (the file_paths input grid * s_enhance *
        t_enhance). The shape is (lats, lons, 1)
        """

        assert len(self.source_data.shape) == 2
        assert self.source_data.shape[1] == 1

        df = pd.DataFrame(
            {'topo': self.source_data.flatten(), 'gid_target': self.nn}
        )
        n_target = np.prod(self.hr_shape[:-1])
        df = df[df['gid_target'] != n_target]
        df = df.sort_values('gid_target')
        df = df.groupby('gid_target').mean()

        missing = set(np.arange(n_target)) - set(df.index)
        if any(missing):
            msg = (
                f'{len(missing)} target pixels did not have unique '
                'high-resolution source data to map from. If there are a '
                'lot of target pixels missing source data this probably '
                'means the source data is not high enough resolution. '
                'Filling raster with NN.'
            )
            logger.warning(msg)
            warn(msg)
            temp_df = pd.DataFrame({'topo': np.nan}, index=sorted(missing))
            df = pd.concat((df, temp_df)).sort_index()

        hr_data = df['topo'].values.reshape(self.hr_shape[:-1])
        if np.isnan(hr_data).any():
            hr_data = nn_fill_array(hr_data)

        hr_data = np.expand_dims(hr_data, axis=-1)

        logger.info('Finished mapping raster from {}'.format(self._exo_source))

        return da.from_array(hr_data)

    def get_cache_file(self, feature, s_enhance, t_enhance, t_agg_factor):
        """Get cache file name. This uses a time independent naming convention.

        Parameters
        ----------
        feature : str
            Name of feature to get cache file for
        s_enhance : int
            Spatial enhancement for this exogeneous data step (cumulative for
            all model steps up to the current step).
        t_enhance : int
            Temporal enhancement for this exogeneous data step (cumulative for
            all model steps up to the current step).
        t_agg_factor : int
            Factor by which to aggregate the exo_source data to the temporal
            resolution of the file_paths input enhanced by t_enhance.

        Returns
        -------
        cache_fp : str
            Name of cache file
        """
        fn = f'exo_{feature}_{self.target}_{self.shape}'
        fn += f'_tagg{t_agg_factor}_{s_enhance}x_'
        fn += f'{t_enhance}x.nc'
        fn = fn.replace('(', '').replace(')', '')
        fn = fn.replace('[', '').replace(']', '')
        fn = fn.replace(',', 'x').replace(' ', '')
        cache_fp = os.path.join(self.cache_dir, fn)
        if self.cache_data:
            os.makedirs(self.cache_dir, exist_ok=True)
        return cache_fp


class TopoExtractNC(TopoExtractH5):
    """TopoExtract for netCDF files"""

    @property
    def source_handler(self):
        """Get the LoaderNC object that handles the .nc source topography
        data file."""
        if self._source_handler is None:
            logger.info(
                'Getting topography for full domain from '
                f'{self._exo_source}'
            )
            self._source_handler = LoaderNC(
                self._exo_source,
                features=['topography'],
            )
        return self._source_handler

    @property
    def source_data(self):
        """Get the 1D array of elevation data from the exo_source_nc"""
        return self.source_handler['topography'].flatten()[..., None]

    @property
    def source_lat_lon(self):
        """Get the 2D array (n, 2) of lat, lon data from the exo_source_nc"""
        source_lat_lon = self.source_handler.lat_lon.reshape((-1, 2))
        return source_lat_lon


class SzaExtract(ExoExtract):
    """SzaExtract for H5 files"""

    @property
    def source_data(self):
        """Get the 1D array of sza data from the exo_source_h5"""
        return SolarPosition(
            self.hr_time_index, self.hr_lat_lon.reshape((-1, 2))
        ).zenith.T

    def get_data(self):
        """Get a raster of source values corresponding to the
        high-resolution grid (the file_paths input grid * s_enhance *
        t_enhance). The shape is (lats, lons, temporal)
        """
        hr_data = self.source_data.reshape(self.hr_shape)
        logger.info('Finished computing SZA data')
        return hr_data
