# -*- coding: utf-8 -*-
"""
Sup3r data pipeline architecture.
"""
import logging
import xarray
import numpy as np

from reV.pipeline.pipeline import Pipeline
from rex.utilities.loggers import init_logger
from rex.resource_extraction.resource_extraction import ResourceX
from rex.multi_year_resource import MultiYearResource
from rex import WindX

from sup3r.pipeline.config import Sup3rPipelineConfig

logger = logging.getLogger(__name__)


class Sup3rPipeline(Pipeline):
    """Sup3r pipeline execution framework."""

    CMD_BASE = 'python -m nsrdb.cli config -c {fp_config} -cmd {command}'

    COMMANDS = ()

    def __init__(self, pipeline, monitor=True, verbose=False):
        """
        Parameters
        ----------
        pipeline : str | dict
            Pipeline config file path or dictionary.
        monitor : bool
            Flag to perform continuous monitoring of the pipeline.
        verbose : bool
            Flag to submit pipeline steps with -v flag for debug logging
        """
        self.monitor = monitor
        self.verbose = verbose
        self._config = Sup3rPipelineConfig(pipeline)
        self._run_list = self._config.pipeline
        self.resource = None
        self.multiResource = None
        self.h5_files = None
        self.h5_file = None
        self._init_status()

        # init logger for pipeline module if requested in input config
        if 'logging' in self._config:
            init_logger('sup3r.pipeline', **self._config.logging)
            init_logger('reV.pipeline', **self._config.logging)

    def initialize_h5_multiresource(self, res_h5_path):
        """Use MultiYearResource to handle
        multiple h5 files

        Parameters
        ----------
        res_h5_path : str
            Directory containing h5 files
            or single file path

        Returns
        -------
        h5_files : str list
            List of file names
        """

        self.multiResource = MultiYearResource(res_h5_path)
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
            h5 file path. Uses first file from multiResource
            file list if not specified.
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
        """

        if h5_file is None:
            h5_file = self.multiResource.h5_files[0]

        with WindX(h5_file, hsds=False) as handle:
            self.initialize_h5_resource(h5_file)
            raster_index = self.resource.get_raster_index(target, shape)
            lat_lon = np.zeros((raster_index.shape[0],
                                raster_index.shape[1]))
            data = np.zeros((raster_index.shape[0],
                             raster_index.shape[1],
                             len(handle.time_index),
                             len(features)), dtype=np.float32)

            for j, f in enumerate(features):
                for i in range(data.shape[0]):
                    data[i, :, :, j] = handle[f, :,
                                              raster_index[i]].transpose()

            for i in range(data.shape[0]):
                lat_lon[i, :] = handle.lat_lon[raster_index[i]]

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

    def get_u_v(self, data, lats, lons):
        """Maps windspeed and direction to u v
        and aligns u v with grid

        Parameters
        ----------
        data : np.ndarray
            4D array (spatial_1, spatial_2, temporal, 2)
            2 channels are windspeed and direction
            in that order

        Returns
        -------
        data : np.ndarray
            Same dimensions as input but new channels
            are u and v in that order
        """

        # convert from windspeed and direction to u v
        u = data[:, :, :, 0] * np.cos(np.radians(data[:, :, :, 1] - 180.0))
        v = data[:, :, :, 0] * np.sin(np.radians(data[:, :, :, 1] - 180.0))

        # get the dy/dx to the nearest vertical neighbor
        dy = lats - np.roll(lats, 1, axis=0)
        dx = lons - np.roll(lons, 1, axis=0)

        # calculate the angle from the vertical
        theta = (np.pi / 2) - np.arctan2(dy, dx)
        theta[0] = theta[1]  # fix the roll row

        sin2 = np.sin(theta)
        cos2 = np.cos(theta)

        u_rot = v * sin2 + u * cos2
        v_rot = v * cos2 - u * sin2

        data[:, :, :, 0] = u_rot
        data[:, :, :, 1] = v_rot

        return data

    def get_coarse_data(self, data, spatial_res=None, temporal_res=None):
        """"Coarsen data according to spatial_res resolution
        and temporal_res temporal sample frequency

        Parameters
        ----------
        data : np.ndarray
            4D array with dimensions
            (spatial_1, spatial_2, temporal, features)

        spatial_res : int
            factor by which to coarsen spatial dimensions

        temporal_res : (int, int)
            factor by which to coarsen temporal dimension

        Returns
        -------
        coarse_data : np.ndarray
            4D array with same dimensions as data
            with new coarse resolution
        """

        if temporal_res is not None:
            tmp = data[:, :, ::temporal_res, :]
        else:
            tmp = data

        if spatial_res is not None:
            coarse_data = tmp.reshape(-1, spatial_res,
                                      data.shape[1] // spatial_res,
                                      spatial_res,
                                      tmp.shape[2],
                                      tmp.shape[3]).sum((1, 3)) \
                / (spatial_res * spatial_res)

        return coarse_data
