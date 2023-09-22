"""Sup3r exogenous data handling"""
import logging
import os
import pickle
import shutil
from typing import ClassVar
from warnings import warn

from sup3r.preprocessing.data_handling.exo_extraction import (
    SzaExtractH5,
    SzaExtractNC,
    TopoExtractH5,
    TopoExtractNC,
)
from sup3r.utilities.utilities import get_source_type

logger = logging.getLogger(__name__)


class ExogenousDataHandler:
    """Class to extract exogenous features for multistep forward passes. e.g.
    Multiple topography arrays at different resolutions for multiple spatial
    enhancement steps."""

    AVAILABLE_HANDLERS: ClassVar[dict] = {
        'topography': {
            'h5': TopoExtractH5,
            'nc': TopoExtractNC
        },
        'sza': {
            'h5': SzaExtractH5,
            'nc': SzaExtractNC
        }
    }

    def __init__(self,
                 file_paths,
                 feature,
                 source_file,
                 s_enhancements,
                 t_enhancements,
                 s_agg_factors,
                 t_agg_factors,
                 target=None,
                 shape=None,
                 temporal_slice=None,
                 raster_file=None,
                 max_delta=20,
                 input_handler=None,
                 exo_handler=None,
                 cache_data=True,
                 cache_dir='./exo_cache'):
        """
        Parameters
        ----------
        file_paths : str | list
            A single source h5 file or netcdf file to extract raster data from.
            The string can be a unix-style file path which will be passed
            through glob.glob. This is typically low-res WRF output or GCM
            netcdf data that is source low-resolution data intended to be
            sup3r resolved.
        feature : str
            Exogenous feature to extract from source_h5
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
            receive a high-res input feature. The length of this list should be
            equal to the number of s_agg_factors
        t_enhancements : list
            List of factors by which the Sup3rGan model will enhance the
            temporal dimension of low resolution data from file_paths input
            where the total temporal enhancement is the product of these
            factors.
        s_agg_factors : list
            List of factors by which to aggregate the exo_source
            data to the spatial resolution of the file_paths input enhanced by
            s_enhance. The length of this list should be equal to the number of
            s_enhancements
        t_agg_factors : list
            List of factors by which to aggregate the exo_source
            data to the temporal resolution of the file_paths input enhanced by
            t_enhance. The length of this list should be equal to the number of
            t_enhancements
        target : tuple
            (lat, lon) lower left corner of raster. Either need target+shape or
            raster_file.
        shape : tuple
            (rows, cols) grid size. Either need target+shape or raster_file.
        temporal_slice : slice | None
            slice used to extract interval from temporal dimension for input
            data and source data
        raster_file : str | None
            File for raster_index array for the corresponding target and shape.
            If specified the raster_index will be loaded from the file if it
            exists or written to the file if it does not yet exist.  If None
            raster_index will be calculated directly. Either need target+shape
            or raster_file.
        max_delta : int, optional
            Optional maximum limit on the raster shape that is retrieved at
            once. If shape is (20, 20) and max_delta=10, the full raster will
            be retrieved in four chunks of (10, 10). This helps adapt to
            non-regular grids that curve over large distances, by default 20
        input_handler : str
            data handler class to use for input data. Provide a string name to
            match a class in data_handling.py. If None the correct handler will
            be guessed based on file type and time series properties.
        exo_handler : str
            Feature extract class to use for source data. For example, if
            feature='topography' this should be either TopoExtractH5 or
            TopoExtractNC. If None the correct handler will be guessed based on
            file type and time series properties.
        cache_data : bool
            Flag to cache exogeneous data in <cache_dir>/exo_cache/ this can
            speed up forward passes with large temporal extents
        cache_dir : str
            Directory for storing cache data. Default is './exo_cache'
        """

        self.feature = feature
        self.s_enhancements = s_enhancements
        self.t_enhancements = t_enhancements
        self.s_agg_factors = s_agg_factors
        self.t_agg_factors = t_agg_factors
        self.source_file = source_file
        self.file_paths = file_paths
        self.exo_handler = exo_handler
        self.temporal_slice = temporal_slice
        self.target = target
        self.shape = shape
        self.raster_file = raster_file
        self.max_delta = max_delta
        self.input_handler = input_handler
        self.cache_data = cache_data
        self.cache_dir = cache_dir
        self.data = []

        if self.s_enhancements[0] != 1:
            msg = ('s_enhancements typically starts with 1 so the first '
                   'exogenous data input matches the spatial resolution of '
                   'the source low-res input data, but received '
                   's_enhancements: {}'.format(self.s_enhancements))
            logger.warning(msg)
            warn(msg)
        if self.t_enhancements[0] != 1:
            msg = ('t_enhancements typically starts with 1 so the first '
                   'exogenous data input matches the temporal resolution of '
                   'the source low-res input data, but received '
                   't_enhancements: {}'.format(self.t_enhancements))
            logger.warning(msg)
            warn(msg)

        msg = ('Need to provide the same number of enhancement factors and '
               f'agg factors. Received s_enhancements={self.s_enhancements}, '
               f'and s_agg_factors={self.s_agg_factors}.')
        assert len(self.s_enhancements) == len(self.s_agg_factors), msg
        msg = ('Need to provide the same number of enhancement factors and '
               f'agg factors. Received t_enhancements={self.t_enhancements}, '
               f'and t_agg_factors={self.t_agg_factors}.')
        assert len(self.t_enhancements) == len(self.t_agg_factors), msg

        msg = ('Need to provide an integer enhancement factor for each model'
               'step. If the step is temporal enhancement then s_enhance=1')
        assert not any(s is None for s in self.s_enhancements), msg

        for i, _ in enumerate(self.s_enhancements):
            s_enhance = self.s_enhancements[i]
            t_enhance = self.t_enhancements[i]
            s_agg_factor = self.s_agg_factors[i]
            t_agg_factor = self.t_agg_factors[i]
            if feature in list(self.AVAILABLE_HANDLERS):
                data = self.get_exo_data(feature=feature,
                                         s_enhance=s_enhance,
                                         t_enhance=t_enhance,
                                         s_agg_factor=s_agg_factor,
                                         t_agg_factor=t_agg_factor)
                self.data.append(data)
            else:
                msg = (f"Can only extract {list(self.AVAILABLE_HANDLERS)}."
                       f" Received {feature}.")
                raise NotImplementedError(msg)

    def get_cache_file(self, feature, s_enhance, t_enhance, s_agg_factor,
                       t_agg_factor):
        """Get cache file name

        Parameters
        ----------
        feature : str
            Name of feature to get cache file for
        s_enhance : int
            Spatial enhancement for this exogeneous data step (cumulative for
            all model steps up to the current step).
        t_enhance : int
            Temporal enhancement for this exogeneous data step (cumulative for
            all model steps up to the current step).
        s_agg_factor : int
            Factor by which to aggregate the exo_source data to the spatial
            resolution of the file_paths input enhanced by s_enhance.
        t_agg_factor : int
            Factor by which to aggregate the exo_source data to the temporal
            resolution of the file_paths input enhanced by t_enhance.

        Returns
        -------
        cache_fp : str
            Name of cache file
        """
        fn = f'exo_{feature}_{self.target}_{self.shape}_sagg{s_agg_factor}_'
        fn += f'tagg{t_agg_factor}_{s_enhance}x_{t_enhance}x.pkl'
        fn = fn.replace('(', '').replace(')', '')
        fn = fn.replace('[', '').replace(']', '')
        fn = fn.replace(',', 'x').replace(' ', '')
        cache_fp = os.path.join(self.cache_dir, fn)
        if self.cache_data:
            os.makedirs(self.cache_dir, exist_ok=True)
        return cache_fp

    def get_exo_data(self, feature, s_enhance, t_enhance, s_agg_factor,
                     t_agg_factor):
        """Get the exogenous topography data

        Parameters
        ----------
        feature : str
            Name of feature to get exo data for
        s_enhance : int
            Spatial enhancement for this exogeneous data step (cumulative for
            all model steps up to the current step).
        t_enhance : int
            Temporal enhancement for this exogeneous data step (cumulative for
            all model steps up to the current step).
        s_agg_factor : int
            Factor by which to aggregate the exo_source data to the spatial
            resolution of the file_paths input enhanced by s_enhance.
        t_agg_factor : int
            Factor by which to aggregate the exo_source data to the temporal
            resolution of the file_paths input enhanced by t_enhance.

        Returns
        -------
        data : np.ndarray
            2D or 3D array of exo data with shape (lat, lon) or (lat,
            lon, temporal)
        """

        cache_fp = self.get_cache_file(feature=feature,
                                       s_enhance=s_enhance,
                                       t_enhance=t_enhance,
                                       s_agg_factor=s_agg_factor,
                                       t_agg_factor=t_agg_factor)
        tmp_fp = cache_fp + '.tmp'
        if os.path.exists(cache_fp):
            with open(cache_fp, 'rb') as f:
                data = pickle.load(f)

        else:
            exo_handler = self.get_exo_handler(feature, self.source_file,
                                               self.exo_handler)
            data = exo_handler(self.file_paths,
                               self.source_file,
                               s_enhance=s_enhance,
                               t_enhance=t_enhance,
                               s_agg_factor=s_agg_factor,
                               t_agg_factor=t_agg_factor,
                               target=self.target,
                               shape=self.shape,
                               temporal_slice=self.temporal_slice,
                               raster_file=self.raster_file,
                               max_delta=self.max_delta,
                               input_handler=self.input_handler).data
            if self.cache_data:
                with open(tmp_fp, 'wb') as f:
                    pickle.dump(data, f)
                shutil.move(tmp_fp, cache_fp)
        return data

    @classmethod
    def get_exo_handler(cls, feature, source_file, exo_handler):
        """Get exogenous feature extraction class for source file

        Parameters
        ----------
        feature : str
            Name of feature to get exo handler for
        source_file : str
            Filepath to source wtk, nsrdb, or netcdf file to get hi-res (2km or
            4km) data from which will be mapped to the enhanced grid of the
            file_paths input
        exo_handler : str
            Feature extract class to use for source data. For example, if
            feature='topography' this should be either TopoExtractH5 or
            TopoExtractNC. If None the correct handler will be guessed based on
            file type and time series properties.

        Returns
        -------
        exo_handler : str
            Exogenous feature extraction class to use for source data.
        """
        if exo_handler is None:
            in_type = get_source_type(source_file)
            if in_type not in ('h5', 'nc'):
                msg = ('Did not recognize input type "{}" for file paths: {}'.
                       format(in_type, source_file))
                logger.error(msg)
                raise RuntimeError(msg)
            check = (feature in cls.AVAILABLE_HANDLERS
                     and in_type in cls.AVAILABLE_HANDLERS[feature])
            if check:
                exo_handler = cls.AVAILABLE_HANDLERS[feature][in_type]
            else:
                msg = ('Could not find exo handler class for '
                       f'feature={feature} and input_type={in_type}.')
                logger.error(msg)
                raise KeyError(msg)
        return exo_handler
