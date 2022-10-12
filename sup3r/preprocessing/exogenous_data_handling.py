"""Sup3r exogenous data handling"""

import logging
import numpy as np
from warnings import warn

from sup3r.utilities.topo import TopoExtractH5, TopoExtractNC
import sup3r.preprocessing.data_handling
import sup3r.utilities.topo
from sup3r.utilities.utilities import get_source_type

logger = logging.getLogger(__name__)


class ExogenousDataHandler:
    """Class to extract exogenous features for multistep forward passes. e.g.
    Multiple topography arrays at different resolutions for multiple spatial
    enhancement steps."""

    def __init__(self, file_paths, features, source_file, s_enhancements,
                 agg_factors, target=None, shape=None, raster_file=None,
                 max_delta=20, input_handler=None, topo_handler=None,
                 exo_steps=None):
        """
        Parameters
        ----------
        file_paths : str | list
            A single source h5 file or netcdf file to extract raster data from.
            The string can be a unix-style file path which will be passed
            through glob.glob. This is typically low-res WRF output or GCM
            netcdf data that is source low-resolution data intended to be
            sup3r resolved.
        features : list
            List of exogenous features to extract from source_h5
        source_file : str
            Filepath to source wtk, nsrdb, or netcdf file to get hi-res (2km or
            4km) data from which will be mapped to the enhanced grid of the
            file_paths input
        s_enhancements : list
            List of factors by which the Sup3rGan model will enhance the
            spatial dimensions of low resolution data from file_paths input
            where the total spatial enhancement is the product of these
            factors. For example, if file_paths has 100km data and there are 2
            spatial enhancement steps of 4x and 5x to a nominal resolution of
            5km, s_enhancements should be [1, 4, 5] and exo_steps should be
            [0, 1, 2] so that the input to the 4x model gets exogenous data
            at 100km (s_enhance=1, exo_step=0), the input to the 5x model gets
            exogenous data at 25km (s_enhance=4, exo_step=1), and there is a
            20x (1*4*5) exogeneous data layer available if the second model can
            receive a high-res input feature (e.g. WindGan). The length of this
            list should be equal to the number of agg_factors and the number of
            exo_steps
        agg_factors : list
            List of factors by which to aggregate the topo_source_h5 elevation
            data to the resolution of the file_paths input enhanced by
            s_enhance. The length of this list should be equal to the number of
            s_enhancements and the number of exo_steps
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
        topo_handler : str
            topo extract class to use for source data. Provide a string name to
            match a class in topo.py. If None the correct handler will
            be guessed based on file type and time series properties.
        exo_steps : list
            List of model step indices for which exogenous data is required.
            e.g. If we have two model steps which take exo data and one which
            does not exo_steps = [0, 1]. The length of this list should be
            equal to the number of s_enhancements and the number of agg_factors
        """

        self.features = features
        self.s_enhancements = s_enhancements
        self.agg_factors = agg_factors
        self.data = []
        exo_steps = exo_steps or np.arange(len(self.s_enhancements))

        if self.s_enhancements[0] != 1:
            msg = ('s_enhancements typically starts with 1 so the first '
                   'exogenous data input matches the spatial resolution of '
                   'the source low-res input data, but received '
                   's_enhancements: {}'.format(self.s_enhancements))
            logger.warning(msg)
            warn(msg)

        msg = ('Need to provide the same number of enhancement factors and '
               f'agg factors. Received s_enhancements={s_enhancements} and '
               f'agg_factors={agg_factors}.')
        assert len(self.s_enhancements) == len(self.agg_factors), msg

        msg = ('Need to provide an integer enhancement factor for each model'
               'step. If the step is temporal enhancement then s_enhance=1')
        assert not any(s is None for s in self.s_enhancements), msg

        for i in range(len(self.s_enhancements)):
            s_enhance = np.product(self.s_enhancements[:i + 1])
            agg_factor = self.agg_factors[i]
            fdata = []
            if i in exo_steps:
                for f in features:
                    if f == 'topography':
                        topo_handler = self.get_topo_handler(source_file,
                                                             topo_handler)
                        data = topo_handler(file_paths, source_file, s_enhance,
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

    @staticmethod
    def get_topo_handler(source_file, topo_handler):
        """Get topo extraction class for source file

        Parameters
        ----------
        source_file : str
            Filepath to source wtk, nsrdb, or netcdf file to get hi-res (2km or
            4km) data from which will be mapped to the enhanced grid of the
            file_paths input
        topo_handler : str
            topo extract class to use for source data. Provide a string name to
            match a class in topo.py. If None the correct handler will
            be guessed based on file type and time series properties.

        Returns
        -------
        topo_handler : str
            topo extract class to use for source data.
        """
        if topo_handler is None:
            in_type = get_source_type(source_file)
            if in_type == 'nc':
                topo_handler = TopoExtractNC
            elif in_type == 'h5':
                topo_handler = TopoExtractH5
            else:
                msg = ('Did not recognize input type "{}" for file paths: {}'
                       .format(in_type, source_file))
                logger.error(msg)
                raise RuntimeError(msg)
        elif isinstance(topo_handler, str):
            topo_handler = getattr(sup3r.utilities.topo, topo_handler, None)
            if topo_handler is None:
                msg = ('Could not find requested topo handler class '
                       f'"{topo_handler}" in '
                       'sup3r.utilities.topo.')
                logger.error(msg)
                raise KeyError(msg)

        return topo_handler
