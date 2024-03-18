"""Sup3r topography utilities"""

import logging
import os
import pickle
import shutil
from abc import ABC, abstractmethod
from warnings import warn

import pandas as pd
import numpy as np
from rex import Resource
from rex.utilities.solar_position import SolarPosition
from scipy.spatial import KDTree

import sup3r.preprocessing.data_handling
from sup3r.postprocessing.file_handling import OutputHandler
from sup3r.preprocessing.data_handling.h5_data_handling import DataHandlerH5
from sup3r.preprocessing.data_handling.nc_data_handling import DataHandlerNC
from sup3r.utilities.utilities import (generate_random_string, get_source_type,
                                       nn_fill_array)

logger = logging.getLogger(__name__)


class ExoExtract(ABC):
    """Class to extract high-res (4km+) data rasters for new
    spatially-enhanced datasets (e.g. GCM files after spatial enhancement)
    using nearest neighbor mapping and aggregation from NREL datasets
    (e.g. WTK or NSRDB)
    """

    def __init__(self,
                 file_paths,
                 exo_source,
                 s_enhance,
                 t_enhance,
                 t_agg_factor,
                 target=None,
                 shape=None,
                 temporal_slice=None,
                 raster_file=None,
                 max_delta=20,
                 input_handler=None,
                 cache_data=True,
                 cache_dir='./exo_cache/',
                 ti_workers=1,
                 distance_upper_bound=None,
                 res_kwargs=None):
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
        temporal_slice : slice | None
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
            match a class in data_handling.py. If None the correct handler will
            be guessed based on file type and time series properties.
        cache_data : bool
            Flag to cache exogeneous data in <cache_dir>/exo_cache/ this can
            speed up forward passes with large temporal extents when the exo
            data is time independent.
        cache_dir : str
            Directory for storing cache data. Default is './exo_cache'
        ti_workers : int | None
            max number of workers to use to get full time index. Useful when
            there are many input files each with a single time step. If this is
            greater than one, time indices for input files will be extracted in
            parallel and then concatenated to get the full time index. If input
            files do not all have time indices or if there are few input files
            this should be set to one.
        distance_upper_bound : float | None
            Maximum distance to map high-resolution data from exo_source to the
            low-resolution file_paths input. None (default) will calculate this
            based on the median distance between points in exo_source
        res_kwargs : dict | None
            Dictionary of kwargs passed to lowest level resource handler. e.g.
            xr.open_dataset(file_paths, **res_kwargs)
        """
        logger.info(f'Initializing {self.__class__.__name__} utility.')

        self.ti_workers = ti_workers
        self._exo_source = exo_source
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
        self.temporal_slice = temporal_slice
        self.target = target
        self.shape = shape
        self.res_kwargs = res_kwargs

        # for subclasses
        self._source_handler = None

        if input_handler is None:
            in_type = get_source_type(file_paths)
            if in_type == 'nc':
                input_handler = DataHandlerNC
            elif in_type == 'h5':
                input_handler = DataHandlerH5
            else:
                msg = (f'Did not recognize input type "{in_type}" for file '
                       f'paths: {file_paths}')
                logger.error(msg)
                raise RuntimeError(msg)
        elif isinstance(input_handler, str):
            input_handler = getattr(sup3r.preprocessing.data_handling,
                                    input_handler, None)
            if input_handler is None:
                msg = ('Could not find requested data handler class '
                       f'"{input_handler}" in '
                       'sup3r.preprocessing.data_handling.')
                logger.error(msg)
                raise KeyError(msg)

        self.input_handler = input_handler(
            file_paths, [],
            target=target,
            shape=shape,
            temporal_slice=temporal_slice,
            raster_file=raster_file,
            max_delta=max_delta,
            worker_kwargs={'ti_workers': ti_workers},
            res_kwargs=self.res_kwargs
        )

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
            Name of cache file
        """
        tsteps = (None if self.temporal_slice is None
                  or self.temporal_slice.start is None
                  or self.temporal_slice.stop is None
                  else self.temporal_slice.stop - self.temporal_slice.start)
        fn = f'exo_{feature}_{self.target}_{self.shape},{tsteps}'
        fn += f'_tagg{t_agg_factor}_{s_enhance}x_'
        fn += f'{t_enhance}x.pkl'
        fn = fn.replace('(', '').replace(')', '')
        fn = fn.replace('[', '').replace(']', '')
        fn = fn.replace(',', 'x').replace(' ', '')
        cache_fp = os.path.join(self.cache_dir, fn)
        if self.cache_data:
            os.makedirs(self.cache_dir, exist_ok=True)
        return cache_fp

    @property
    def source_temporal_slice(self):
        """Get the temporal slice for the exo_source data corresponding to the
        input file temporal slice
        """
        start_index = self.source_time_index.get_indexer(
            [self.input_handler.hr_time_index[0]], method='nearest')[0]
        end_index = self.source_time_index.get_indexer(
            [self.input_handler.hr_time_index[-1]], method='nearest')[0]
        return slice(start_index, end_index + 1, self._t_agg_factor)

    @property
    def source_lat_lon(self):
        """Get the 2D array (n, 2) of lat, lon data from the exo_source_h5"""
        with Resource(self._exo_source) as res:
            source_lat_lon = res.lat_lon
        return source_lat_lon

    @property
    def lr_shape(self):
        """Get the low-resolution spatial shape tuple"""
        return (self.lr_lat_lon.shape[0], self.lr_lat_lon.shape[1],
                len(self.input_handler.time_index))

    @property
    def hr_shape(self):
        """Get the high-resolution spatial shape tuple"""
        return (self._s_enhance * self.lr_lat_lon.shape[0],
                self._s_enhance * self.lr_lat_lon.shape[1],
                self._t_enhance * len(self.input_handler.time_index))

    @property
    def lr_lat_lon(self):
        """Lat lon grid for data in format (spatial_1, spatial_2, 2) Lat/Lon
        array with same ordering in last dimension. This corresponds to the raw
        meta data from the file_paths input.

        Returns
        -------
        ndarray
        """
        return self.input_handler.lat_lon

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
                    self.lr_lat_lon, self.hr_shape[:-1])
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
                    self.hr_shape[-1] * self._t_agg_factor)
            else:
                self._src_time_index = self.hr_time_index
        return self._src_time_index

    @property
    def hr_time_index(self):
        """Get the full time index for aggregated source data"""
        if self._hr_time_index is None:
            if self._t_enhance > 1:
                self._hr_time_index = OutputHandler.get_times(
                    self.input_handler.time_index, self.hr_shape[-1])
            else:
                self._hr_time_index = self.input_handler.time_index
        return self._hr_time_index

    @property
    def distance_upper_bound(self):
        """Maximum distance (float) to map high-resolution data from exo_source
        to the low-resolution file_paths input."""
        if self._distance_upper_bound is None:
            diff = np.diff(self.source_lat_lon, axis=0)
            diff = np.max(np.median(diff, axis=0))
            self._distance_upper_bound = diff
            logger.info('Set distance upper bound to {:.4f}'
                        .format(self._distance_upper_bound))
        return self._distance_upper_bound

    @property
    def tree(self):
        """Get the KDTree built on the target lat lon data from the file_paths
        input with s_enhance"""
        if self._tree is None:
            lat = self.hr_lat_lon[..., 0].flatten()
            lon = self.hr_lat_lon[..., 1].flatten()
            hr_meta = np.vstack((lat, lon)).T
            self._tree = KDTree(hr_meta)
        return self._tree

    @property
    def nn(self):
        """Get the nearest neighbor indices"""
        _, nn = self.tree.query(self.source_lat_lon, k=1,
                                distance_upper_bound=self.distance_upper_bound)
        return nn

    @property
    def data(self):
        """Get a raster of source values corresponding to the
        high-resolution grid (the file_paths input grid * s_enhance *
        t_enhance). The shape is (lats, lons, temporal, 1)
        """
        cache_fp = self.get_cache_file(feature=self.__class__.__name__,
                                       s_enhance=self._s_enhance,
                                       t_enhance=self._t_enhance,
                                       t_agg_factor=self._t_agg_factor)
        tmp_fp = cache_fp + f'.{generate_random_string(10)}.tmp'
        if os.path.exists(cache_fp):
            with open(cache_fp, 'rb') as f:
                data = pickle.load(f)

        else:
            data = self.get_data()

            if self.cache_data:
                with open(tmp_fp, 'wb') as f:
                    pickle.dump(data, f)
                shutil.move(tmp_fp, cache_fp)

        if data.shape[-1] == 1 and self.hr_shape[-1] > 1:
            data = np.repeat(data, self.hr_shape[-1], axis=-1)

        return data[..., np.newaxis]

    @abstractmethod
    def get_data(self):
        """Get a raster of source values corresponding to the
        high-resolution grid (the file_paths input grid * s_enhance *
        t_enhance). The shape is (lats, lons, temporal)
        """

    @classmethod
    def get_exo_raster(cls,
                       file_paths,
                       s_enhance,
                       t_enhance,
                       t_agg_factor,
                       exo_source=None,
                       target=None,
                       shape=None,
                       temporal_slice=None,
                       raster_file=None,
                       max_delta=20,
                       input_handler=None,
                       cache_data=True,
                       cache_dir='./exo_cache/'):
        """Get the exo feature raster corresponding to the spatially enhanced
        grid from the file_paths input

        Parameters
        ----------
        file_paths : str | list
            A single source h5 file to extract raster data from or a list
            of netcdf files with identical grid. The string can be a unix-style
            file path which will be passed through glob.glob
        s_enhance : int
            Factor by which the Sup3rGan model will enhance the spatial
            dimensions of low resolution data from file_paths input. For
            example, if file_paths has 100km data and s_enhance is 4, this
            class will output a topography raster corresponding to the
            file_paths grid enhanced 4x to ~25km
        t_enhance : int
            Factor by which the Sup3rGan model will enhance the temporal
            dimension of low resolution data from file_paths input. For
            example, if getting sza data, file_paths has hourly data, and
            t_enhance is 4, this class will output a sza raster
            corresponding to the file_paths temporally enhanced 4x to 15 min
        t_agg_factor : int
            Factor by which to aggregate the exo_source data to the resolution
            of the file_paths input enhanced by t_enhance. For example, if
            getting sza data, file_paths have hourly data, and t_enhance
            is 4 resulting in a desired resolution of 5 min and exo_source
            has a resolution of 5 min, the t_agg_factor should be 4 so that
            every fourth timestep in the exo_source data is skipped.
        exo_source : str
            Filepath to source wtk, nsrdb, or netcdf file to get hi-res (2km or
            4km) data from which will be mapped to the enhanced grid of the
            file_paths input
        target : tuple
            (lat, lon) lower left corner of raster. Either need target+shape or
            raster_file.
        shape : tuple
            (rows, cols) grid size. Either need target+shape or raster_file.
        temporal_slice : slice | None
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
            match a class in data_handling.py. If None the correct handler will
            be guessed based on file type and time series properties.
        cache_data : bool
            Flag to cache exogeneous data in <cache_dir>/exo_cache/ this can
            speed up forward passes with large temporal extents when the exo
            data is time independent.
        cache_dir : str
            Directory for storing cache data. Default is './exo_cache'

        Returns
        -------
        exo_raster : np.ndarray
            Exo feature raster with shape (hr_rows, hr_cols, h_temporal)
            corresponding to the shape of the spatiotemporally enhanced data
            from file_paths * s_enhance * t_enhance.  The data units correspond
            to the source units in exo_source_h5. This is usually meters when
            feature='topography'
        """
        exo = cls(file_paths,
                  s_enhance,
                  t_enhance,
                  t_agg_factor,
                  exo_source=exo_source,
                  target=target,
                  shape=shape,
                  temporal_slice=temporal_slice,
                  raster_file=raster_file,
                  max_delta=max_delta,
                  input_handler=input_handler,
                  cache_data=cache_data,
                  cache_dir=cache_dir)
        return exo.data


class TopoExtractH5(ExoExtract):
    """TopoExtract for H5 files"""

    @property
    def source_data(self):
        """Get the 1D array of elevation data from the exo_source_h5"""
        with Resource(self._exo_source) as res:
            elev = res.get_meta_arr('elevation')
        return elev[:, np.newaxis]

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

        df = pd.DataFrame({'topo': self.source_data.flatten(),
                           'gid_target': self.nn})
        n_target = np.prod(self.hr_shape[:-1])
        df = df[df['gid_target'] != n_target]
        df = df.sort_values('gid_target')
        df = df.groupby('gid_target').mean()

        missing = set(np.arange(n_target)) - set(df.index)
        if any(missing):
            msg = (f'{len(missing)} target pixels did not have unique '
                   'high-resolution source data to map from. If there are a '
                   'lot of target pixels missing source data this probably '
                   'means the source data is not high enough resolution. '
                   'Filling raster with NN.')
            logger.warning(msg)
            warn(msg)
            temp_df = pd.DataFrame({'topo': np.nan}, index=sorted(missing))
            df = pd.concat((df, temp_df)).sort_index()

        hr_data = df['topo'].values.reshape(self.hr_shape[:-1])
        if np.isnan(hr_data).any():
            hr_data = nn_fill_array(hr_data)

        hr_data = np.expand_dims(hr_data, axis=-1)

        logger.info('Finished mapping raster from {}'.format(self._exo_source))

        return hr_data

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
        fn += f'{t_enhance}x.pkl'
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
        """Get the DataHandlerNC object that handles the .nc source topography
        data file."""
        if self._source_handler is None:
            logger.info('Getting topography for full domain from '
                        f'{self._exo_source}')
            self._source_handler = DataHandlerNC(
                self._exo_source,
                features=['topography'],
                worker_kwargs={'ti_workers': self.ti_workers},
                val_split=0.0,
            )
        return self._source_handler

    @property
    def source_data(self):
        """Get the 1D array of elevation data from the exo_source_nc"""
        elev = self.source_handler.data[..., 0, 0].flatten()
        return elev[..., np.newaxis]

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
        return SolarPosition(self.hr_time_index,
                             self.hr_lat_lon.reshape((-1, 2))).zenith.T

    def get_data(self):
        """Get a raster of source values corresponding to the
        high-resolution grid (the file_paths input grid * s_enhance *
        t_enhance). The shape is (lats, lons, temporal)
        """
        hr_data = self.source_data.reshape(self.hr_shape)
        logger.info('Finished computing SZA data')
        return hr_data
