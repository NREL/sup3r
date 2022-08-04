"""Sup3r exogenous data handling"""

import logging
import numpy as np

from sup3r.utilities.topo import TopoExtract


logger = logging.getLogger(__name__)


class ExogenousDataHandler:
    """Class to extract exogenous features for multistep forward passes. e.g.
    Multiple topography arrays at different resolutions for multiple spatial
    enhancement steps."""

    def __init__(self, file_paths, features, source_h5, s_enhancements,
                 agg_factors, target=None, shape=None, raster_file=None,
                 max_delta=20, input_handler=None):
        """
        Parameters
        ----------
        file_paths : str | list
            A single source h5 wind file to extract raster data from or a list
            of netcdf files with identical grid. The string can be a unix-style
            file path which will be passed through glob.glob
        features : list
            List of exogenous features to extract from source_h5
        source_h5 : str
            Filepath to source wtk or nsrdb file to get hi-res (2km or 4km)
            data from which will be mapped to the enhanced grid of
            the file_paths input
        s_enhancements : list
            List of factor by which the Sup3rGan model will enhance the spatial
            dimensions of low resolution data from file_paths input. For
            example, if file_paths has 100km data and s_enhance is 4, this
            class will output a feature raster corresponding to the
            file_paths grid enhanced 4x to ~25km
        agg_factor : list
            List of factos by which to aggregate the topo_source_h5 elevation
            data to the resolution of the file_paths input enhanced by
            s_enhance. For example, if file_paths has 100km data and s_enhance
            is 4 resulting in a desired resolution of ~25km and topo_source_h5
            has a resolution of 4km, the agg_factor should be 6 so that 6x 4km
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
        """

        self.features = features
        self.s_enhancements = s_enhancements
        self.agg_factors = agg_factors
        self.data = []
        for s_enhance, agg_factor in zip(s_enhancements, agg_factors):
            for f in features:
                fdata = []
                if f == 'elevation':
                    data = TopoExtract(file_paths, source_h5, s_enhance,
                                       agg_factor, target=target, shape=shape,
                                       raster_file=raster_file,
                                       max_delta=max_delta,
                                       input_handler=input_handler)
                    data = data.hr_elev
                    fdata.append(data)
                else:
                    msg = (f"Can only extract topography. Recived {f}.")
                    raise NotImplementedError(msg)
            self.data.append(np.stack(fdata, axis=-1))
