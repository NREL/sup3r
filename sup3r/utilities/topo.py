"""Sup3r topography utilities"""

import numpy as np
import logging
from scipy.spatial import KDTree
from rex import Resource
from abc import ABC, abstractmethod

import sup3r.preprocessing.data_handling
from sup3r.preprocessing.data_handling import DataHandlerNC, DataHandlerH5
from sup3r.postprocessing.file_handling import OutputHandler
from sup3r.utilities.utilities import get_source_type


logger = logging.getLogger(__name__)


class TopoExtract(ABC):
    """Class to extract high-res (4km+) topography rasters for new
    spatially-enhanced datasets (e.g. GCM files after spatial enhancement)
    using nearest neighbor mapping and aggregation from NREL datasets
    (e.g. WTK or NSRDB)
    """

    def __init__(self, file_paths, topo_source, s_enhance, agg_factor,
                 target=None, shape=None, raster_file=None, max_delta=20,
                 input_handler=None, ti_workers=1):
        """
        Parameters
        ----------
        file_paths : str | list
            A single source h5 file to extract raster data from or a list
            of netcdf files with identical grid. The string can be a unix-style
            file path which will be passed through glob.glob. This is
            typically low-res WRF output or GCM netcdf data files that is
            source low-resolution data intended to be sup3r resolved.
        topo_source : str
            Filepath to source wtk or nsrdb file to get hi-res (2km or 4km)
            elevation data from which will be mapped to the enhanced grid of
            the file_paths input
        s_enhance : int
            Factor by which the Sup3rGan model will enhance the spatial
            dimensions of low resolution data from file_paths input. For
            example, if file_paths has 100km data and s_enhance is 4, this
            class will output a topography raster corresponding to the
            file_paths grid enhanced 4x to ~25km
        agg_factor : int
            Factor by which to aggregate the topo_source_h5 elevation
            data to the resolution of the file_paths input enhanced by
            s_enhance. For example, if file_paths has 100km data and s_enhance
            is 4 resulting in a desired resolution of ~25km and topo_source_h5
            has a resolution of 4km, the agg_factor should be 36 so that 6x6
            4km cells are averaged to the ~25km enhanced grid.
        target : tuple
            (lat, lon) lower left corner of raster. Either need target+shape or
            raster_file.
        shape : tuple
            (rows, cols) grid size. Either need target+shape or raster_file.
        raster_file : str | None
            File for raster_index array for the corresponding target and shape.
            If specified the raster_index will be loaded from the file if it
            exists or written to the file if it does not yet exist.  If None
            raster_index will be calculated directly. Either need target+shape
            or raster_file.
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

        logger.info('Initializing TopoExtract utility.')

        self._topo_source = topo_source
        self._s_enhance = s_enhance
        self._agg_factor = agg_factor
        self._tree = None
        self.ti_workers = ti_workers
        self._hr_lat_lon = None

        if input_handler is None:
            in_type = get_source_type(file_paths)
            if in_type == 'nc':
                input_handler = DataHandlerNC
            elif in_type == 'h5':
                input_handler = DataHandlerH5
            else:
                msg = ('Did not recognize input type "{}" for file paths: {}'
                       .format(in_type, file_paths))
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
            file_paths, [], target=target, shape=shape,
            raster_file=raster_file, max_delta=max_delta,
            worker_kwargs=dict(ti_workers=ti_workers))

    @property
    @abstractmethod
    def source_elevation(self):
        """Get the 1D array of elevation data from the topo_source_h5"""

    @property
    @abstractmethod
    def source_lat_lon(self):
        """Get the 2D array (n, 2) of lat, lon data from the topo_source_h5"""

    @property
    def lr_shape(self):
        """Get the low-resolution spatial shape tuple"""
        return (self.lr_lat_lon.shape[0], self.lr_lat_lon.shape[1])

    @property
    def hr_shape(self):
        """Get the high-resolution spatial shape tuple"""
        return (self._s_enhance * self.lr_lat_lon.shape[0],
                self._s_enhance * self.lr_lat_lon.shape[1])

    @property
    def lr_lat_lon(self):
        """lat lon grid for data in format (spatial_1, spatial_2, 2) Lat/Lon
        array with same ordering in last dimension. This corresponds to the raw
        low-resolution meta data from the file_paths input.

        Returns
        -------
        ndarray
        """
        return self.input_handler.lat_lon

    @property
    def hr_lat_lon(self):
        """lat lon grid for data in format (spatial_1, spatial_2, 2) Lat/Lon
        array with same ordering in last dimension. This corresponds to the
        enhanced high-res meta data from the file_paths input * s_enhance.

        Returns
        -------
        ndarray
        """
        if self._hr_lat_lon is None:
            if self._s_enhance > 1:
                self._hr_lat_lon = OutputHandler.get_lat_lon(self.lr_lat_lon,
                                                             self.hr_shape)
            else:
                self._hr_lat_lon = self.lr_lat_lon
        return self._hr_lat_lon

    @property
    def tree(self):
        """Get the KDTree built on the source lat lon data"""
        if self._tree is None:
            self._tree = KDTree(self.source_lat_lon)
        return self._tree

    @property
    def nn(self):
        """Get the nearest neighbor indices """
        ll2 = np.vstack((self.hr_lat_lon[:, :, 0].flatten(),
                         self.hr_lat_lon[:, :, 1].flatten())).T
        _, nn = self.tree.query(ll2, k=self._agg_factor)
        if len(nn.shape) == 1:
            nn = np.expand_dims(nn, 1)
        return nn

    @property
    def hr_elev(self):
        """Get a raster of elevation values corresponding to the
        high-resolution grid (the file_paths input grid * s_enhance). The shape
        is (rows, cols)
        """
        nn = self.nn
        hr_elev = []
        for j in range(self._agg_factor):
            elev = self.source_elevation[nn[:, j]]
            elev = elev.reshape(self.hr_shape)
            hr_elev.append(elev)
        hr_elev = np.dstack(hr_elev).mean(axis=-1)
        logger.info('Finished mapping topo raster from {}'
                    .format(self._topo_source))
        return hr_elev

    @classmethod
    def get_topo_raster(cls, file_paths, topo_source, s_enhance,
                        agg_factor, target=None, shape=None, raster_file=None,
                        max_delta=20, input_handler=None):
        """Get the topography raster corresponding to the spatially enhanced
        grid from the file_paths input

        Parameters
        ----------
        file_paths : str | list
            A single source h5 wind file to extract raster data from or a list
            of netcdf files with identical grid. The string can be a unix-style
            file path which will be passed through glob.glob
        topo_source : str
            Filepath to source wtk, nsrdb, or netcdf file to get hi-res (2km or
            4km) data from which will be mapped to the enhanced grid of the
            file_paths input
        s_enhance : int
            Factor by which the Sup3rGan model will enhance the spatial
            dimensions of low resolution data from file_paths input. For
            example, if file_paths has 100km data and s_enhance is 4, this
            class will output a topography raster corresponding to the
            file_paths grid enhanced 4x to ~25km
        agg_factor : int
            Factor by which to aggregate the topo_source_h5 elevation data to
            the resolution of the file_paths input enhanced by s_enhance. For
            example, if file_paths has 100km data and s_enhance is 4 resulting
            in a desired resolution of ~25km and topo_source_h5 has a
            resolution of 4km, the agg_factor should be 36 so that 6x6 4km
            cells are averaged to the ~25km enhanced grid.
        target : tuple
            (lat, lon) lower left corner of raster. Either need target+shape or
            raster_file.
        shape : tuple
            (rows, cols) grid size. Either need target+shape or raster_file.
        raster_file : str | None
            File for raster_index array for the corresponding target and shape.
            If specified the raster_index will be loaded from the file if it
            exists or written to the file if it does not yet exist.  If None
            raster_index will be calculated directly. Either need target+shape
            or raster_file.
        input_handler : str
            data handler class to use for input data. Provide a string name to
            match a class in data_handling.py. If None the correct handler will
            be guessed based on file type and time series properties.

        Returns
        -------
        topo_raster : np.ndarray
            Elevation raster with shape (hr_rows, hr_cols) corresponding to the
            shape of the spatially enhanced grid from file_paths * s_enhance.
            The elevation units correspond to the source units in
            topo_source_h5, usually meters.
        """

        te = cls(file_paths, topo_source, s_enhance, agg_factor,
                 target=target, shape=shape, raster_file=raster_file,
                 max_delta=max_delta, input_handler=input_handler)

        return te.hr_elev


class TopoExtractH5(TopoExtract):
    """TopoExtract for H5 files"""

    @property
    def source_elevation(self):
        """Get the 1D array of elevation data from the topo_source_h5"""
        with Resource(self._topo_source) as res:
            elev = res.get_meta_arr('elevation')
        return elev

    @property
    def source_lat_lon(self):
        """Get the 2D array (n, 2) of lat, lon data from the topo_source_h5"""
        with Resource(self._topo_source) as res:
            source_lat_lon = res.lat_lon
        return source_lat_lon


class TopoExtractNC(TopoExtract):
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
                    f'{self._topo_source}')
        self.source_handler = DataHandlerNC(
            self._topo_source, features=['topography'],
            worker_kwargs=dict(ti_workers=self.ti_workers), val_split=0.0)

    @property
    def source_elevation(self):
        """Get the 1D array of elevation data from the topo_source_h5"""
        elev = self.source_handler.data.reshape((-1))
        return elev

    @property
    def source_lat_lon(self):
        """Get the 2D array (n, 2) of lat, lon data from the topo_source_h5"""
        source_lat_lon = self.source_handler.lat_lon.reshape((-1, 2))
        return source_lat_lon
