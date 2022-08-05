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
                 max_delta=20, input_handler=None, exo_steps=None):
        """
        Parameters
        ----------
        file_paths : str | list
            A single source h5 file to extract raster data from or a list
            of netcdf files with identical grid. The string can be a unix-style
            file path which will be passed through glob.glob. This is
            typically low-res WRF output or GCM netcdf data files that is
            source low-resolution data intended to be sup3r resolved.
        features : list
            List of exogenous features to extract from source_h5
        source_h5 : str
            Filepath to source wtk or nsrdb file to get hi-res (2km or 4km)
            data from which will be mapped to the enhanced grid of
            the file_paths input
        s_enhancements : list
            List of factors by which the Sup3rGan model will enhance the
            spatial dimensions of low resolution data from file_paths input.
            For example, if file_paths has 100km data and s_enhance is 4, this
            class will output a feature raster corresponding to the file_paths
            grid enhanced 4x to ~25km. The length of this list should be equal
            to the number of model steps. e.g. if using a model with 2 5x
            spatial enhancement steps and a single temporal enhancement step
            s_enhancements should be [5, 5, 1]
        agg_factors : list
            List of factors by which to aggregate the topo_source_h5 elevation
            data to the resolution of the file_paths input enhanced by
            s_enhance. For example, if file_paths has 100km data and s_enhance
            is 4 resulting in a desired resolution of ~25km and topo_source_h5
            has a resolution of 4km, the agg_factor should be 36 so that 6x6
            4km cells are averaged to the ~25km enhanced grid. The length of
            this list should be equal to the number of model steps. e.g. if
            using a model with 2 spatial enhancement steps which require
            exogenous data and and a single temporal enhancement which does not
            require exogenous data then step agg_factors should have integer
            values for the first two entries and None for the third.
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
        exo_steps : list
            List of model step indices for which exogenous data is required.
            e.g. If we have two model steps which take exo data and one which
            does not exo_steps = [0, 1].
        """

        self.features = features
        self.s_enhancements = [1] + s_enhancements
        self.agg_factors = [1] + agg_factors
        self.data = []
        exo_steps = exo_steps or np.arange(len(self.s_enhancements))
        msg = ('Need to provide the same number of enhancement factors and '
               f'agg factors. Received s_enhancements={s_enhancements} and '
               f'agg_factors={agg_factors}.')
        assert len(s_enhancements) == len(agg_factors), msg
        msg = ('Need to provide an integer enhancement factor for each model'
               'step. If the step is temporal enhancement then s_enhance=1')
        assert not any(s is None for s in s_enhancements), msg
        for i in range(len(s_enhancements)):
            s_enhance = np.product(s_enhancements[:i + 1])
            agg_factor = agg_factors[i]
            fdata = []
            if i in exo_steps:
                for f in features:
                    if f == 'topography':
                        data = TopoExtract(file_paths, source_h5, s_enhance,
                                           agg_factor, target=target,
                                           shape=shape,
                                           raster_file=raster_file,
                                           max_delta=max_delta,
                                           input_handler=input_handler)
                        data = data.hr_elev
                        fdata.append(data)
                    else:
                        msg = (f"Can only extract topography. Recived {f}.")
                        raise NotImplementedError(msg)
                self.data.append(np.stack(fdata, axis=-1))
            else:
                self.data.append(None)
