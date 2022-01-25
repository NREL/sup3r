# -*- coding: utf-8 -*-
"""
Sup3r data pipeline architecture.
"""
import logging
import xarray

from reV.pipeline.pipeline import Pipeline
from rex.utilities.loggers import init_logger
from rex.resource_extraction.resource_extraction import ResourceX

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
        self._init_status()

        # init logger for pipeline module if requested in input config
        if 'logging' in self._config:
            init_logger('sup3r.pipeline', **self._config.logging)
            init_logger('reV.pipeline', **self._config.logging)

    def get_h5_data(self, res_h5):
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
        return self.resource.h5

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

    def get_raster_index(self, target, shape, meta=None, max_delta=50):
        """Get meta data index values that correspond to a 2D rectangular grid
        of the requested shape starting with the target coordinate in the
        bottom left hand corner. Note that this can break down if a target is
        requested outside of the main grid area.
        Parameters
        ----------
        target : tuple
            Starting coordinate (latitude, longitude) in decimal degrees for
            the bottom left hand corner of the raster grid.
        shape : tuple
            Desired raster shape in format (number_rows, number_cols)
        meta : pd.DataFrame | None
            Optional meta data input with latitude, longitude fields. Default
            is None which extracts self.meta from the resource data.
        max_delta : int
            Optional maximum limit on the raster shape that is retrieved at
            once. If shape is (20, 20) and max_delta=10, the full raseter will
            be retrieved in four chunks of (10, 10). This helps adapt to
            non-regular grids that curve over large distances.
        Returns
        -------
        raster_index : np.ndarray
            2D array of meta data index values that form a 2D rectangular grid
            with latitudes descending from top to bottom and longitudes
            ascending from left to right.
        """

        return self.resource.get_raster_index(target, shape, meta, max_delta)

    def get_coarse_data(self, data, spatial_res=None, temporal_res=None):
        """"Coarsen data according to spatial_res resolution
        and temporal_res temporal sample frequency

        Parameters
        ----------
        data : np.ndarray
            3D array with dimensions (time, lat, lon)

        spatial_res : (int, int)
            tuple with first element the spatial resolution
            of input data and second element the spatial
            resolution of coarse data

        temporal_res : (int, int)
            tuple with first element the temporal resolution
            of input data and second element the temporal
            resolution of coarse data

        Returns
        -------
        coarse_data : np.ndarray
            3D array with same dimensions as data
            with new coarse resolution
        """

        if temporal_res is not None:
            n = temporal_res[1] // temporal_res[0]
            coarse_data = data[::n, :, :]
        else:
            coarse_data = data

        if spatial_res is not None:
            n = spatial_res[1] // spatial_res[0]
            coarse_data = coarse_data.reshape(-1, n,
                                              coarse_data.shape[1] // n,
                                              n).sum((-1, -3)) / n

        return coarse_data
