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

    def initialize_h5(self, res_h5_path):
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

    def load_h5_data(self, res_h5):
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

    def get_h5_data(self, h5_data, target, shape, features):
        """Get chunk of h5 data based on raster_indices
        and features

        Parameters
        ----------
        h5_file : h5py.File | h5py.Group
            returned from load_h5_data method
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

        raster_index = self.resource.get_raster_index(target, shape)
        data = np.zeros((raster_index.shape[0],
                         raster_index.shape[1],
                         h5_data['time_index'].shape[0],
                         len(features)))
        for i in range(data.shape[0]):
            for j, f in enumerate(features):
                data[i, :, :, j] = h5_data[f][:, raster_index[i]].transpose()

        return data

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
