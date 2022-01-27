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

from rex import init_logger
from rex.resource_extraction.resource_extraction import ResourceX
from rex.multi_year_resource import MultiYearResource
from rex.utilities.loggers import create_dirs
from rex import WindX
from wtk.utilities import get_wrf_files
from phygnn import CustomNetwork
from sup3r.utilities import utilities
from sup3r.pipeline import Status


logger = logging.getLogger(__name__)


def run_data_model(out_dir, year, var_kwargs, factory_kwargs=None,
                   log_level='DEBUG', log_file='data_model.log',
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

    if isinstance(factory_kwargs, str):
        factory_kwargs = json.loads(factory_kwargs)
    if isinstance(var_kwargs, str):
        var_kwargs = json.loads(var_kwargs)

    sup3rData = Sup3rData(var_kwargs['file_path'])
    sup3rData._init_loggers(year=year,
                            log_dir=os.path.join(out_dir, 'logs/'),
                            log_file=log_file,
                            log_level=log_level)
    msg = 'Getting training data. '
    msg += f'target={var_kwargs["target"]}, '
    msg += f'target={var_kwargs["shape"]}, '
    msg += f'target={var_kwargs["features"]}, '
    logger.info(msg)
    sup3rData.get_training_data(var_kwargs['target'],
                                var_kwargs['shape'],
                                var_kwargs['features'],
                                var_kwargs.get('n_observations', 1),
                                var_kwargs.get('temporal_res', None),
                                var_kwargs.get('spatial_res', None))

    if job_name is not None:
        runtime = (time.time() - t0) / 60
        status = {'out_dir': out_dir,
                  'job_status': 'successful',
                  'runtime': runtime}
        Status.make_job_file(out_dir, 'data-model',
                             job_name, status)


class Sup3rData():
    """Sup3r data handling framework."""

    def __init__(self, h5_path=None,
                 nc_path=None):
        """
        Parameters
        ----------
        h5_path : str
            path to h5 files
        nc_path : str
            path to netcdf files with
            wildcard filename prefix
        """

        self.resource = None
        self.multiResource = None
        self.h5_files = None
        self.h5_file = None
        self.nc_files = None
        self.nc_file = None

        if h5_path is not None:
            self.initialize_h5_multiresource(h5_path)

        if nc_path is not None:
            self.nc_files = get_wrf_files(os.path.dirname(nc_path),
                                          os.path.basename(nc_path))

    def initialize_h5_multiresource(self, h5_path):
        """Use MultiYearResource to handle
        multiple h5 files

        Parameters
        ----------
        h5_path : str
            Directory containing h5 files
            or single file path

        Returns
        -------
        h5_files : str list
            List of file names
        """

        logger.info('Initializing MultiYearResource')

        self.multiResource = MultiYearResource(h5_path)
        self.h5_files = self.multiResource.h5_files
        return self.h5_files

    def initialize_h5_resource(self, res_h5):
        """Use ResourceX class to
        open h5 file

        Parameters
        ----------
        res_h5 : str
            Path to resource .h5 file of interest

        Returns
        -------
        h5 : h5py.File | h5py.Group
        """

        logger.info('Initializing ResourceX')

        self.resource = ResourceX(res_h5)
        self.h5_file = self.resource.h5
        return self.h5_file

    def get_h5_data(self, target, shape, features, h5_file=None):
        """Get chunk of h5 data based on raster_indices
        and features

        Parameters
        ----------
        h5_file : str
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

        if h5_file is None:
            h5_file = self.multiResource.h5_files[0]

        logger.info(f'Opening data file: {h5_file}')

        with WindX(h5_file, hsds=False) as handle:
            self.initialize_h5_resource(h5_file)
            raster_index = self.resource.get_raster_index(target, shape)
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

    def get_nc_data(self, res_nc):
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

    def get_training_data(self, target, shape, features,
                          n_batch=16, batch_size=None,
                          shuffle=True,
                          n_observations=1,
                          spatial_res=None,
                          temporal_res=None):
        """Build full arrays for training

        Parameters
        ----------
        target : tuple
            Starting coordinate (latitude, longitude) in decimal degrees for
            the bottom left hand corner of the raster grid.
        shape : tuple
            Desired raster shape in format (number_rows, number_cols)
        features : str list
            List of fields to extract from dataset

        Returns
        -------
        x : np.ndarray
            5D array of low res data
            (n_observations, spatial_1, spatial_2, temporal, features)
        y : np.ndarray
            5D array of high res data
            (n_observations, spatial_1, spatial_2, temporal, features)
        """

        y, lat_lon = self.get_h5_data(target, shape,
                                      features, self.h5_files[0])
        for f in self.h5_files[1:]:
            tmp, _ = self.get_h5_data(target, shape, features, f)
            y = np.concatenate((y, tmp), axis=2)

        for i, f in enumerate(features):
            if f.split('_')[0] == 'windspeed':
                height = f.split('_')[1]
                j = features.index(f'winddirection_{height}')
                # features[i] = f'u_{height}'
                # features[j] = f'v_{height}'
                y[:, :, :, i], y[:, :, :, j] = utilities.get_u_v(y[:, :, :, i],
                                                                 y[:, :, :, j],
                                                                 lat_lon)
        msg = 'Coarsening high resolution data'
        msg += f'Spatial coarsening factor: {spatial_res}'
        msg += f'Temporal coarsening factor: {temporal_res}'
        logger.info(msg)

        x, _ = utilities.get_coarse_data(y, lat_lon,
                                         spatial_res,
                                         temporal_res)

        y = y.reshape((n_observations,
                       y.shape[0],
                       y.shape[1],
                       -1, len(features)))

        x = x.reshape((n_observations,
                       x.shape[0],
                       x.shape[1],
                       -1, len(features)))

        msg = 'Batching training data. '
        msg += f'n_batch={n_batch}, '
        msg += f'batch_size={batch_size}, '
        msg += f'shuffle={shuffle}'
        logger.info(msg)

        yield self.batch_data(x, y, n_batch, batch_size, shuffle)

    def batch_data(self, x, y, n_batch=16, batch_size=None, shuffle=True):
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

        yield CustomNetwork.make_batches(x, y,
                                         n_batch,
                                         batch_size,
                                         shuffle)

    def _init_loggers(self, loggers=None,
                      year=None,
                      log_dir='./logs',
                      log_file='sup3r.log',
                      log_level='DEBUG',
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
