# -*- coding: utf-8 -*-
"""
Sup3r data handling module.
"""
import logging
import xarray
import numpy as np
import os
import json
import time
import sys

from rex import init_logger
from rex.resource_extraction.resource_extraction import ResourceX
from rex.utilities.loggers import create_dirs
from rex import WindX
from reV.utilities.exceptions import ConfigError
from wtk.utilities import get_wrf_files
from sup3r.utilities import utilities
from sup3r.pipeline import Status
from sup3r import __version__


logger = logging.getLogger(__name__)


class Sup3rData:
    """Sup3r data handling framework."""

    def __init__(self, out_dir,
                 make_out_dirs=True):
        """
        Parameters
        ----------
        out_dir : str
            Project directory.
        make_out_dirs : bool
            Flag to make output directories for logs
        """

        self._out_dir = out_dir
        self._log_dir = os.path.join(out_dir, 'logs/')

        if make_out_dirs:
            for d in [self._out_dir, self._log_dir]:
                create_dirs(d)

    @staticmethod
    def _log_version():
        """Check Sup3r and python version"""

        logger.info(f'Running Sup3r version: {__version__}')
        logger.info(f'Running python version: {sys.version_info}')

        is_64bits = sys.maxsize > 2 ** 32
        if is_64bits:
            msg = f'Running on 64-bit python, sys.maxsize: {sys.maxsize}'
            logger.info(msg)
        else:
            msg = f'Running on 32-bit python, sys.maxsize: {sys.maxsize}'
            logger.warning(msg)

    @classmethod
    def _get_high_res_data(cls, data_files, target, shape, features):
        """Concatenate files along time dimension

        Parameters
        ----------
        data_files : list
            list of strings of file paths
        target : tuple
            lat lon tuple for lower left coordinate of raster
        shape : tuple
            grid size of raster
        features : list
            list of fields to extract from data

        Returns
        -------
        y : np.ndarray
            4D array of high res data
            (spatial_1, spatial_2, temporal, features)

        lat_lon : np.ndarray
            3D array of lat lon
            (spatial_1, spatial_2, 2)
            lat (lon) first channel (second channel)
        """

        raster_index = cls.get_raster_index(data_files[0],
                                            target,
                                            shape)

        y, lat_lon = cls._get_file_data(data_files[0],
                                        raster_index,
                                        features)
        if len(data_files) > 1:
            for f in data_files[1:]:
                tmp = cls._get_file_data(f, raster_index,
                                         features,
                                         get_coords=False)
                y = np.concatenate((y, tmp), axis=2)

        return y, lat_lon

    @classmethod
    def reshape_data(cls, x, y, n_observations):
        """Reshape high and low resolution
        data so the first dimension of x and y
        are the same - a requirement for batching

        Parameters
        ----------
        x : np.ndarray
            4D array of low resolution data
            (spatial_1, spatial_2, temporal, features)
        y : np.ndarray
            4D array of high resolution data
            (spatial_1, spatial_2, temporal, features)

        Returns
        -------
        x : np.ndarray
            5D array of low resolution data
            (n_observations, spatial_1, spatial_2, temporal, features)
        y : np.ndarray
            5D array of high resolution data
            (n_observations, spatial_1, spatial_2, temporal, features)
        """

        y = y.reshape((n_observations,
                       y.shape[0],
                       y.shape[1],
                       -1, y.shape[3]))

        msg = f'Reshaping high resolution data: {y.shape}'
        logger.info(msg)

        x = x.reshape((n_observations,
                       x.shape[0],
                       x.shape[1],
                       -1, x.shape[3]))

        msg = f'Reshaping low resolution data: {x.shape}'
        logger.info(msg)

        return x, y

    @classmethod
    def get_raster_index(cls, file_path, target, shape):
        """Get raster index for file data

        Parameters
        ----------
        file_path : str
            path to data file
        target : tuple
            (lat, lon) for lower left corner
        shape : tuple
            (n_rows, n_cols) grid size

        Returns
        -------
        raster_index : np.ndarray
            2D array of grid indices

        """

        _, file_ext = os.path.splitext(file_path)
        if file_ext == '.h5':
            resourceX = ResourceX(file_path)
            logger.info('Getting raster index. '
                        f'target={target}, shape={shape}')
            raster_index = resourceX.get_raster_index(target, shape,
                                                      max_delta=20)

        elif file_ext == '.nc':
            pass
        else:
            raise ConfigError('Data must be either h5 or netcdf '
                              f'but received file extension: {file_ext}')
        return raster_index

    @classmethod
    def run_data_model(cls, out_dir, var_kwargs,
                       factory_kwargs=None, log_level='DEBUG',
                       log_file='data_model.log',
                       job_name=None):
        """Run data model for preprocessing

        Parameters
        ----------
        out_dir : str
            Project directory.
        var_kwargs : dict
            Namespace of kwargs
        factory_kwargs : dict | None
            Optional namespace of kwargs
        log_level : str | None
            Logging level (DEBUG, INFO). If None,
            no logging will be initialized.
        log_file : str
            File to log to. Will be put in output directory.
        job_name : str
            Optional name for pipeline and status identification.
        """

        t0 = time.time()

        sup3r = cls(out_dir)
        sup3r._init_loggers(log_file=log_file,
                            log_level=log_level)

        if isinstance(factory_kwargs, str):
            factory_kwargs = json.loads(factory_kwargs)
        if isinstance(var_kwargs, str):
            var_kwargs = json.loads(var_kwargs)

        logger.info('Initializing variables')

        data_files = var_kwargs['data_files']
        target = var_kwargs['target']
        shape = var_kwargs['shape']
        features = var_kwargs['features']
        n_observations = var_kwargs.get('n_observations', 1)
        spatial_res = var_kwargs.get('spatial_res', None)
        temporal_res = var_kwargs.get('temporal_res', None)

        if shape[0] % spatial_res != 0:
            msg = 'spatial_res must evenly divide grid size. '
            msg += f'Received spatial_res: {spatial_res} '
            msg += f'with grid size: ({shape})'
            raise ValueError(msg)

        msg = 'Getting training data. '
        msg += f'target={target}, '
        msg += f'shape={shape}, '
        msg += f'features={features}'

        logger.info(msg)

        y, lat_lon = sup3r._get_high_res_data(data_files, target,
                                              shape, features)

        if y.shape[2] % n_observations != 0:
            msg = 'n_observations must evenly divide time '
            msg += 'dimension. Received n_observations='
            msg += f'{n_observations} and time_steps={y.shape[2]}'
            raise ValueError(msg)

        logger.info('Checking features for wind and transforming')

        y = sup3r.transform_rotate_wind(y, lat_lon, features)

        msg = 'Coarsening high resolution data '
        msg += f'spatial_res={spatial_res}, '
        msg += f'temporal_res={temporal_res}'
        logger.info(msg)

        x, _ = sup3r.get_coarse_data(y, lat_lon,
                                     spatial_res,
                                     temporal_res)

        x, y = sup3r.reshape_data(x, y, n_observations)

        logger.info('Finished getting training data')

        if job_name is not None:
            runtime = (time.time() - t0) / 60
            status = {'out_dir': out_dir,
                      'job_status': 'successful',
                      'runtime': runtime}
            Status.make_job_file(out_dir, 'data-model',
                                 job_name, status)

        return x, y

    @classmethod
    def _get_file_data(cls, file_path,
                       raster_index,
                       features,
                       get_coords=True):
        """Extract fields from file for region
        given by target and shape

        Parameters
        ----------
        file_path : str
            File path
        target : tuple
            (lat, lon) for lower left corner of region
        shape : tuple
            (n_lat, n_lon) grid size for region
        features : list
            list of fields to extract from file
        get_coords : bool
            get coordinates

        Returns
        -------
        y : np.ndarray
            4D array of extracted data
            (spatial_1, spatial_2, temporal, features)
        lat_lon : np.ndarray
            3D array of (spatial_1, spatial_2, 2)
            with 2 channels as lat/lon in that order.
            Only returned if get_coords=True
        """

        _, file_ext = os.path.splitext(file_path)
        if file_ext == '.h5':
            return cls._get_h5_data(file_path, raster_index,
                                    features,
                                    get_coords=get_coords)
        elif file_ext == '.nc':
            return cls._get_nc_data(file_path, raster_index,
                                    features,
                                    get_coords=get_coords)
        else:
            raise ConfigError('Data must be either h5 or netcdf '
                              f'but received file extension: {file_ext}')

    @staticmethod
    def _get_h5_data(file_path, raster_index,
                     features, get_coords=True):
        """Get chunk of h5 data based on raster_indices
        and features

        Parameters
        ----------
        file_path : str
            h5 file path.
        raster_index : np.ndarray
            2D array of grid indices
        features : str list
            List of fields to extract from dataset
        get_coords : bool
            get coordinates

        Returns
        -------
        data : np.ndarray
            Real high-resolution data in a 4D array:
            (spatial_1, spatial_2, temporal, features)
        lat_lon : np.ndarray
            3D array of (spatial_1, spatial_2, 2)
            with 2 channels as lat/lon in that order.
            Only returned if get_coords=True
        """

        logger.info(f'Opening data file: {file_path}')

        with WindX(file_path, hsds=False) as handle:

            data = np.zeros((raster_index.shape[0],
                             raster_index.shape[1],
                             len(handle.time_index),
                             len(features)), dtype=np.float32)

            logger.info('Populating data array')
            for j, f in enumerate(features):

                logger.info(f'Extracting {f} from file')

                data[:, :, :, j] = np.transpose(handle[f, :, raster_index],
                                                (1, 2, 0))
            if get_coords:
                lat_lon = np.zeros((raster_index.shape[0],
                                    raster_index.shape[1], 2))
                logger.info('Populating lat_lon array')
                lat_lon = handle.lat_lon[raster_index]

        if get_coords:
            return data, lat_lon
        else:
            return data

    @staticmethod
    def _get_nc_data(file_path, raster_index,
                     features, get_coords=True):
        """Get chunk of netcdf data based on raster_indices
        and features

        Parameters
        ----------
        file_path : str
            netcdf file path.
        raster_index : np.ndarray
            2D array of grid indices
        features : str list
            List of fields to extract from dataset
        get_coords : bool
            get coordinates

        Returns
        -------
        data : np.ndarray
            Real high-resolution data in a 4D array:
            (spatial_1, spatial_2, temporal, features)

        lat_lon : np.ndarray
            3D array of (spatial_1, spatial_2, 2)
            with 2 channels as lat/lon in that order.
            Only returned if get_coords=True
        """

        logger.info(f'Opening data file: {file_path}')

        return xarray.open_dataset(file_path)

    def _init_loggers(self, loggers=None,
                      log_dir='./logs',
                      log_file='sup3r.log',
                      log_level='DEBUG',
                      log_version=True,
                      use_log_dir=True):
        """Initialize sup3r loggers.
        Parameters
        ----------
        loggers : None | list | tuple
            List of logger names to initialize. None defaults to all Sup3r
            loggers.
        log_file : str
            Log file name. Will be placed in the sup3r out dir.
        log_level : str | None
            Logging level (DEBUG, INFO). If None, no logging will be
            initialized.
        date : None | datetime
            Optional date to put in the log file name.
        use_log_dir : bool
            Flag to use the class log directory (self._log_dir = ./logs/)
        """

        if log_level in ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'):

            if loggers is None:
                loggers = ('sup3r.sup3r', 'sup3r.data_model')

            if log_file is not None and use_log_dir:
                log_file = os.path.join(log_dir, log_file)
                create_dirs(os.path.dirname(log_file))

            for name in loggers:
                init_logger(name, log_level=log_level,
                            log_file=log_file)
        if log_version:
            self._log_version()

    @staticmethod
    def transform_wind(y, i, j):
        """Maps windspeed and direction to u v

        Parameters
        ----------
        y : np.ndarray
            4D array (spatial_1, spatial_2, temporal, features)
        i : int
            index of windspeed feature
        j : int
            index of winddirection feature

        Returns
        -------
        u : np.ndarray
            3D array of zonal wind components

        v : np.ndarray
            3D array of meridional wind components
        """
        # convert from windspeed and direction to u v
        windspeed = y[:, :, :, i]
        direction = y[:, :, :, j]

        y[:, :, :, i] = windspeed * np.cos(np.radians(direction - 180.0))
        y[:, :, :, j] = windspeed * np.sin(np.radians(direction - 180.0))

        return y

    @staticmethod
    def rotate_u_v(y, i, j, lat_lon):
        """aligns u v with grid

        Parameters
        ----------
        y : np.ndarray
            4D array (spatial_1, spatial_2, temporal, features)
        i : int
            index of u feature
        j : int
            index of v feature
        lat_lon : np.ndarray
            3D array (spatial_1, spatial_2, 2)
            2 channels are lat and lon in that
            order

        Returns
        -------
        u_rot : np.ndarray
            3D array of zonal wind components

        v_rot : np.ndarray
            3D array of meridional wind components
        """

        u = y[:, :, :, i]
        v = y[:, :, :, j]
        lats = lat_lon[:, :, 0]
        lons = lat_lon[:, :, 1]

        # get the dy/dx to the nearest vertical neighbor
        dy = lats - np.roll(lats, 1, axis=0)
        dx = lons - np.roll(lons, 1, axis=0)

        # calculate the angle from the vertical
        theta = (np.pi / 2) - np.arctan2(dy, dx)
        theta[0] = theta[1]  # fix the roll row

        sin2 = np.sin(theta)
        cos2 = np.cos(theta)

        y[:, :, :, i] = np.einsum('ij,ijk->ijk', sin2, v) \
            + np.einsum('ij,ijk->ijk', cos2, u)
        y[:, :, :, j] = np.einsum('ij,ijk->ijk', cos2, v) \
            - np.einsum('ij,ijk->ijk', sin2, u)

        return y

    @staticmethod
    def get_coarse_data(data, lat_lon,
                        spatial_res=2,
                        temporal_res=None):
        """"Coarsen data according to spatial_res resolution
        and temporal_res temporal sample frequency

        Parameters
        ----------
        data : np.ndarray
            4D array with dimensions
            (spatial_1, spatial_2, temporal, features)

        lat_lon : np.ndarray
            2D array with dimensions
            (spatial_1, spatial_2)

        spatial_res : int
            factor by which to coarsen spatial dimensions

        temporal_res : (int, int)
            factor by which to coarsen temporal dimension

        Returns
        -------
        coarse_data : np.ndarray
            4D array with same dimensions as data
            with new coarse resolution

        coarse_lat_lon : np.ndarray
            3D array (spatial_1, spatial_2, 2) with
            lat and lon as the 2 channels in that order
            with same resolution as coarse_data
        """

        if temporal_res is not None:
            tmp = data[:, :, ::temporal_res, :]
        else:
            tmp = data

        if spatial_res is not None:
            if data.shape[1] % spatial_res != 0:
                msg = 'spatial_res must evenly divide grid size. '
                msg += f'Received spatial_res: {spatial_res} '
                msg += f'with grid size: ({data.shape[0]}, '
                msg += f'{data.shape[1]})'
                raise ValueError(msg)

            coarse_data = tmp.reshape(-1, spatial_res,
                                      data.shape[1] // spatial_res,
                                      spatial_res,
                                      tmp.shape[2],
                                      tmp.shape[3]).sum((1, 3)) \
                / (spatial_res * spatial_res)

            coarse_lat_lon = lat_lon.reshape(-1, spatial_res,
                                             data.shape[1] // spatial_res,
                                             spatial_res, 2).sum((3, 1)) \
                / (spatial_res * spatial_res)
        else:
            coarse_data = data.copy()
            coarse_lat_lon = lat_lon.copy()

        return coarse_data, coarse_lat_lon

    @staticmethod
    def transform_rotate_wind(y, lat_lon, features):
        """Transform windspeed/direction to
        u and v and align u and v with grid

        Parameters
        ----------
        y : np.ndarray
            4D array of high res data
        lat_lon : np.ndarray
            3D array of lat lon
        features : list
            list of extracted features

        Returns
        -------
        y : np.ndarray
            4D array of high res data with
            (windspeed, direction) -> (u, v)
        """

        for i, f in enumerate(features):
            if f.split('_')[0] == 'windspeed':
                if len(f.split('_')) > 1:
                    height = f.split('_')[1]
                    msg = f'Converting windspeed/direction to u/v ({height})'
                    j = features.index(f'winddirection_{height}')
                    features[i] = f'u_{height}'
                    features[j] = f'v_{height}'
                else:
                    msg = 'Converting windspeed/direction to u/v'
                    j = features.index('winddirection')
                    features[i] = 'u'
                    features[j] = 'v'

                logger.info(msg)

                y = Sup3rData.transform_wind(y, i, j)

            if features[i].split('_')[0] == 'u':
                if len(features[i].split('_')) > 1:
                    height = features[i].split('_')[1]
                    j = features.index(f'v_{height}')
                    msg = f'Aligning u/v to grid ({height})'
                else:
                    j = features.index('v')
                    msg = 'Aligning u/v to grid'

                logger.info(msg)

                y = Sup3rData.rotate_u_v(y, i, j, lat_lon)

        return y
