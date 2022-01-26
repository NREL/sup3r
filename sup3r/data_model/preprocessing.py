# -*- coding: utf-8 -*-
"""
Sup3r data handling module.
"""
import logging
import xarray
import numpy as np
import os

from reV.pipeline.pipeline import Pipeline
# from rex.utilities.loggers import init_logger
from rex.resource_extraction.resource_extraction import ResourceX
from rex.multi_year_resource import MultiYearResource
from rex import WindX
from wtk.utilities import get_wrf_files
from phygnn import CustomNetwork
from sup3r.utilities import utilities

# from sup3r.pipeline.config import Sup3rPipelineConfig

logger = logging.getLogger(__name__)


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
