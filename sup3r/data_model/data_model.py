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
from rex.multi_year_resource import MultiYearResource
from rex.utilities.loggers import create_dirs
from rex import WindX
from wtk.utilities import get_wrf_files
from phygnn import CustomNetwork
from sup3r.utilities import utilities
from sup3r.pipeline import Status
from sup3r import __version__


logger = logging.getLogger(__name__)


class Sup3rData:
    """Sup3r data handling framework."""

    def __init__(self, out_dir, year,
                 make_out_dirs=True):
        """
        Parameters
        ----------
        out_dir : str
            Project directory.
        year : int | str
            Processing year.
        make_out_dirs : bool
            Flag to make output directories for logs
        """

        self._out_dir = out_dir
        self._log_dir = os.path.join(out_dir, 'logs/')
        self._year = int(year)

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
    def get_high_res_data(cls, data_files, target, shape, features):
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

        y, lat_lon = cls.get_h5_data(data_files[0],
                                     target, shape,
                                     features)
        for f in data_files[1:]:

            tmp, _ = cls.get_h5_data(f, target, shape, features)
            y = np.concatenate((y, tmp), axis=2)

        return y, lat_lon

    @classmethod
    def transform_wind(cls, y, lat_lon, features):
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
                height = f.split('_')[1]

                msg = f'Converting windspeed/direction to u/v ({height})'
                logger.info(msg)

                j = features.index(f'winddirection_{height}')
                y[:, :, :, i], y[:, :, :, j] = utilities.get_u_v(y[:, :, :, i],
                                                                 y[:, :, :, j],
                                                                 lat_lon)
        return y

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
    def run_data_model(cls, out_dir, year, var_kwargs,
                       factory_kwargs=None, log_level='DEBUG',
                       log_file='data_model.log',
                       job_name=None):
        """Run data model for preprocessing

        Parameters
        ----------
        out_dir : str
            Project directory.
        year : int
            data year
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

        sup3r = cls(out_dir, year)
        sup3r._init_loggers(year=year, log_file=log_file,
                            log_level=log_level)

        if isinstance(factory_kwargs, str):
            factory_kwargs = json.loads(factory_kwargs)
        if isinstance(var_kwargs, str):
            var_kwargs = json.loads(var_kwargs)

        logger.info('Initializing variables')

        file_path = var_kwargs['file_path']
        target = var_kwargs['target']
        shape = var_kwargs['shape']
        features = var_kwargs['features']
        n_observations = var_kwargs.get('n_observations', 1)
        spatial_res = var_kwargs.get('spatial_res', None)
        temporal_res = var_kwargs.get('temporal_res', None)
        n_batch = var_kwargs.get('n_batch', 16)
        batch_size = var_kwargs.get('batch_size', None)
        shuffle = var_kwargs.get('shuffle', True)

        msg = 'Getting training data. '
        msg += f'target={target}, '
        msg += f'target={shape}, '
        msg += f'target={features}'

        logger.info(msg)

        multiResource = MultiYearResource(file_path)

        logger.info('Concatenating along time dimension')

        y, lat_lon = sup3r.get_high_res_data(multiResource.h5_files,
                                             target, shape, features)

        logger.info('Checking features for wind and transforming')

        y = sup3r.transform_wind(y, lat_lon, features)

        msg = 'Coarsening high resolution data'
        msg += f'Spatial coarsening factor: {spatial_res}'
        msg += f'Temporal coarsening factor: {temporal_res}'
        logger.info(msg)

        x, _ = utilities.get_coarse_data(y, lat_lon,
                                         spatial_res,
                                         temporal_res)

        x, y = sup3r.reshape_data(x, y, n_observations)

        yield sup3r.batch_data(x, y, n_batch, batch_size, shuffle)

        logger.info('Finished getting training data')

        if job_name is not None:
            runtime = (time.time() - t0) / 60
            status = {'out_dir': out_dir,
                      'job_status': 'successful',
                      'runtime': runtime}
            Status.make_job_file(out_dir, 'data-model',
                                 job_name, status)

    @staticmethod
    def get_h5_data(file_path, target, shape, features):
        """Get chunk of h5 data based on raster_indices
        and features

        Parameters
        ----------
        file_path : str
            h5 file path.
        target : tuple
            Starting coordinate (latitude, longitude) in decimal degrees for
            the bottom left hand corner of the raster grid.
        shape : tuple
            Desired raster shape in format (number_rows, number_cols)
        features : str list
            List of fields to extract from dataset

        Returns
        -------
        data : np.ndarray
            Real high-resolution data in a 4D array:
            (spatial_1, spatial_2, temporal, features)

        lat_lon : np.ndarray
            3D array (spatial_1, spatial_2, 2) with
            lat and lon as the 2 channels in that order
        """

        logger.info(f'Opening data file: {file_path}')

        with WindX(file_path, hsds=False) as handle:
            resourceX = ResourceX(file_path)
            raster_index = resourceX.get_raster_index(target, shape)
            lat_lon = np.zeros((raster_index.shape[0],
                                raster_index.shape[1], 2))
            data = np.zeros((raster_index.shape[0],
                             raster_index.shape[1],
                             len(handle.time_index),
                             len(features)), dtype=np.float32)

            for j, f in enumerate(features):

                logger.info(f'Extracting {f} from file')

                for i in range(data.shape[0]):
                    data[i, :, :, j] = handle[f, :,
                                              raster_index[i]].transpose()

            for i in range(data.shape[0]):
                lat_lon[i, :, :] = handle.lat_lon[raster_index[i]]

        return data, lat_lon

    @staticmethod
    def get_nc_data(res_nc):
        """
        Open nc File instance

        Parameters
        ----------
        res_nc : str
            Path to source .nc file of interest

        Returns
        -------
        nc : xarray.Dataset
        """

        logger.info(f'Opening data file: {res_nc}')

        return xarray.open_dataset(res_nc)

    @classmethod
    def batch_data(cls, x, y, n_batch=16, batch_size=None, shuffle=True):
        """Make lists of unique data batches for training

        Parameters
        ----------
        x : np.ndarray
            5D array of low resolution training data
            (n_observation, spatial_1, spatial_2, temporal, features)
        y : np.ndarray
            5D array of high resolution target for training

        Returns
        -------
        batches : GeneratorType
        """

        msg = 'Batching training data. '
        msg += f'n_batch={n_batch}, '
        msg += f'batch_size={batch_size}, '
        msg += f'shuffle={shuffle}'
        logger.info(msg)

        yield CustomNetwork.make_batches(x, y,
                                         n_batch,
                                         batch_size,
                                         shuffle)

    def _init_loggers(self, loggers=None,
                      year=None,
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

            if year is not None:
                log_file = log_file.replace('.log', f'_{year}.log')

            for name in loggers:
                init_logger(name, log_level=log_level,
                            log_file=log_file)
        if log_version:
            self._log_version()
