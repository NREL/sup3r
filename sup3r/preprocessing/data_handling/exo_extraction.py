"""Sup3r topography utilities"""

import logging
from abc import ABC, abstractmethod

import numpy as np
from rex import Resource
from rex.utilities.solar_position import SolarPosition
from scipy.spatial import KDTree

import sup3r.preprocessing.data_handling
from sup3r.postprocessing.file_handling import OutputHandler
from sup3r.preprocessing.data_handling.h5_data_handling import DataHandlerH5
from sup3r.preprocessing.data_handling.nc_data_handling import DataHandlerNC
from sup3r.utilities.utilities import get_source_type

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
                 s_agg_factor,
                 t_agg_factor,
                 target=None,
                 shape=None,
                 temporal_slice=None,
                 raster_file=None,
                 max_delta=20,
                 input_handler=None,
                 ti_workers=1):
        """
        Parameters
        ----------
        file_paths : str | list
            A single source h5 file to extract raster data from or a list
            of netcdf files with identical grid. The string can be a unix-style
            file path which will be passed through glob.glob. This is
            typically low-res WRF output or GCM netcdf data files that is
            source low-resolution data intended to be sup3r resolved.
        exo_source : str
            Filepath to source wtk or nsrdb file to get hi-res (2km or 4km)
            elevation data from which will be mapped to the enhanced grid of
            the file_paths input
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
        s_agg_factor : int
            Factor by which to aggregate the exo_source data to the resolution
            of the file_paths input enhanced by s_enhance. For example, if
            getting topography data, file_paths have 100km data, and s_enhance
            is 4 resulting in a desired resolution of ~25km and topo_source_h5
            has a resolution of 4km, the s_agg_factor should be 36 so that 6x6
            4km cells are averaged to the ~25km enhanced grid.
        t_agg_factor : int
            Factor by which to aggregate the exo_source data to the resolution
            of the file_paths input enhanced by t_enhance. For example, if
            getting sza data, file_paths have hourly data, and t_enhance
            is 4 resulting in a desired resolution of 5 min and exo_source
            has a resolution of 5 min, the t_agg_factor should be 4 so that
            every fourth timestep in the exo_source data is skipped.
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
        ti_workers : int | None
            max number of workers to use to get full time index. Useful when
            there are many input files each with a single time step. If this is
            greater than one, time indices for input files will be extracted in
            parallel and then concatenated to get the full time index. If input
            files do not all have time indices or if there are few input files
            this should be set to one.
        """

        logger.info(f'Initializing {self.__class__.__name__} utility.')

        self.ti_workers = ti_workers
        self._exo_source = exo_source
        self._s_enhance = s_enhance
        self._t_enhance = t_enhance
        self._s_agg_factor = s_agg_factor
        self._t_agg_factor = t_agg_factor
        self._tree = None
        self._hr_lat_lon = None
        self._source_lat_lon = None
        self._hr_time_index = None
        self._src_time_index = None

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
            worker_kwargs=dict(ti_workers=ti_workers),
        )

    @property
    @abstractmethod
    def source_data(self):
        """Get the 1D array of source data from the exo_source_h5"""

    @property
    def source_temporal_slice(self):
        """Get the temporal slice for the exo_source data corresponding to the
        input file temporal slice"""
        start_index = self.source_time_index.get_indexer(
            [self.input_handler.hr_time_index[0]], method='nearest')[0]
        end_index = self.source_time_index.get_indexer(
            [self.input_handler.hr_time_index[-1]], method='nearest')[0]
        return slice(start_index, end_index + 1, self._t_agg_factor)

    @property
    def source_lat_lon(self):
        """Get the 2D array (n, 2) of lat, lon data for the exo source"""
        if self._source_lat_lon is None:
            src_enhance = int(np.sqrt(self._s_agg_factor))
            src_shape = (self.hr_shape[0] * src_enhance,
                         self.hr_shape[1] * src_enhance)
            self._source_lat_lon = OutputHandler.get_lat_lon(
                self.lr_lat_lon, src_shape).reshape((-1, 2))
        return self._source_lat_lon

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
        low-resolution meta data from the file_paths input.

        Returns
        -------
        ndarray
        """
        return self.input_handler.lat_lon

    @property
    def hr_lat_lon(self):
        """Lat lon grid for data in format (spatial_1, spatial_2, 2) Lat/Lon
        array with same ordering in last dimension. This corresponds to the
        enhanced high-res meta data from the file_paths input * s_enhance.

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
    def tree(self):
        """Get the KDTree built on the source lat lon data"""
        if self._tree is None:
            self._tree = KDTree(self.source_lat_lon)
        return self._tree

    @property
    def nn(self):
        """Get the nearest neighbor indices"""
        ll2 = np.vstack(
            (self.hr_lat_lon[:, :, 0].flatten(),
             self.hr_lat_lon[:, :, 1].flatten())).T
        _, nn = self.tree.query(ll2, k=self._s_agg_factor)
        if len(nn.shape) == 1:
            nn = np.expand_dims(nn, 1)
        return nn

    @property
    def data(self):
        """Get a raster of source values corresponding to the
        high-resolution grid (the file_paths input grid * s_enhance *
        t_enhance). The shape is (lats, lons, temporal, 1)
        """
        nn = self.nn
        hr_data = []
        for j in range(self._s_agg_factor):
            out = self.source_data[nn[:, j], self.source_temporal_slice]
            out = out.reshape(self.hr_shape)
            hr_data.append(out[..., np.newaxis])
        hr_data = np.concatenate(hr_data, axis=-1).mean(axis=-1)
        logger.info('Finished mapping raster from {}'.format(self._exo_source))
        return hr_data[..., np.newaxis]

    @classmethod
    def get_exo_raster(cls,
                       file_paths,
                       s_enhance,
                       t_enhance,
                       s_agg_factor,
                       t_agg_factor,
                       exo_source=None,
                       target=None,
                       shape=None,
                       temporal_slice=None,
                       raster_file=None,
                       max_delta=20,
                       input_handler=None):
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
        s_agg_factor : int
            Factor by which to aggregate the exo_source data to the resolution
            of the file_paths input enhanced by s_enhance. For example, if
            getting topography data, file_paths have 100km data, and s_enhance
            is 4 resulting in a desired resolution of ~25km and topo_source_h5
            has a resolution of 4km, the s_agg_factor should be 36 so that 6x6
            4km cells are averaged to the ~25km enhanced grid.
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
                  s_agg_factor,
                  t_agg_factor,
                  exo_source=exo_source,
                  target=target,
                  shape=shape,
                  temporal_slice=temporal_slice,
                  raster_file=raster_file,
                  max_delta=max_delta,
                  input_handler=input_handler)
        return exo.data


class TopoExtractH5(ExoExtract):
    """TopoExtract for H5 files"""

    @property
    def source_data(self):
        """Get the 1D array of elevation data from the exo_source_h5"""
        with Resource(self._exo_source) as res:
            elev = res.get_meta_arr('elevation')
            elev = np.repeat(elev[:, np.newaxis], self.hr_shape[-1], axis=-1)
        return elev

    @property
    def source_lat_lon(self):
        """Get the 2D array (n, 2) of lat, lon data from the exo_source_h5"""
        with Resource(self._exo_source) as res:
            source_lat_lon = res.lat_lon
        return source_lat_lon

    @property
    def source_time_index(self):
        """Time index of the source exo data"""
        if self._src_time_index is None:
            with Resource(self._exo_source) as res:
                self._src_time_index = res.time_index
        return self._src_time_index

    @property
    def data(self):
        """Get a raster of source values corresponding to the
        high-resolution grid (the file_paths input grid * s_enhance *
        t_enhance). The shape is (lats, lons, temporal, 1)
        """
        nn = self.nn
        hr_data = []
        for j in range(self._s_agg_factor):
            out = self.source_data[nn[:, j]]
            out = out.reshape(self.hr_shape)
            hr_data.append(out[..., np.newaxis])
        hr_data = np.concatenate(hr_data, axis=-1).mean(axis=-1)
        logger.info('Finished mapping raster from {}'.format(self._exo_source))
        return hr_data[..., np.newaxis]


class TopoExtractNC(TopoExtractH5):
    """TopoExtract for netCDF files"""

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        args : list
            Same positional arguments as TopoExtract
        kwargs : dict
            Same keyword arguments as TopoExtract
        """

        super().__init__(*args, **kwargs)
        logger.info('Getting topography for full domain from '
                    f'{self._exo_source}')
        self.source_handler = DataHandlerNC(
            self._exo_source,
            features=['topography'],
            worker_kwargs=dict(ti_workers=self.ti_workers),
            val_split=0.0,
        )

    @property
    def source_data(self):
        """Get the 1D array of elevation data from the exo_source_nc"""
        elev = self.source_handler.data.reshape((-1, self.lr_shape[-1]))
        return elev

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

    @property
    def data(self):
        """Get a raster of source values corresponding to the
        high-resolution grid (the file_paths input grid * s_enhance *
        t_enhance). The shape is (lats, lons, temporal, 1)
        """
        hr_data = self.source_data.reshape(self.hr_shape)
        logger.info('Finished computing SZA data')
        return hr_data[..., np.newaxis]
