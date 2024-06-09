"""Sup3r exogenous data handling

TODO: More cleaning. This does not yet fit the new style of composition and
lazy loading.
"""

import logging
import pathlib
import re
from dataclasses import dataclass
from typing import ClassVar, List

import numpy as np

import sup3r.preprocessing
from sup3r.preprocessing.common import (
    get_class_kwargs,
    get_source_type,
    log_args,
)
from sup3r.preprocessing.data_handlers.base import SingleExoDataStep
from sup3r.preprocessing.extracters import (
    SzaExtract,
    TopoExtractH5,
    TopoExtractNC,
)

logger = logging.getLogger(__name__)


@dataclass
class ExogenousDataHandler:
    """Class to extract exogenous features for multistep forward passes. e.g.
    Multiple topography arrays at different resolutions for multiple spatial
    enhancement steps.

    This takes a list of models and information about model
    steps and uses that info to compute needed enhancement factors for each
    step and extract exo data corresponding to those enhancement factors. The
    list of steps are then updated with the exo data for each step.

    Parameters
    ----------
    file_paths : str | list
        A single source h5 file or netcdf file to extract raster data from.
        The string can be a unix-style file path which will be passed
        through glob.glob. This is typically low-res WRF output or GCM
        netcdf data that is source low-resolution data intended to be
        sup3r resolved.
    feature : str
        Exogenous feature to extract from file_paths
    models : list
        List of models used with the given steps list. This list of models
        is used to determine the input and output resolution and
        enhancement factors for each model step which is then used to
        determine aggregation factors. If agg factors and enhancement
        factors are provided in the steps list the model list is not
        needed.
    steps : list
        List of dictionaries containing info on which models to use for a
        given step index and what type of exo data the step requires. e.g.
        [{'model': 0, 'combine_type': 'input'},
         {'model': 0, 'combine_type': 'layer'}]
        Each step entry can also contain s_enhance, t_enhance,
        s_agg_factor, t_agg_factor. e.g.
        [{'model': 0, 'combine_type': 'input', 's_agg_factor': 900,
          's_enhance': 1, 't_agg_factor': 5, 't_enhance': 1},
         {'model': 0, 'combine_type': 'layer', 's_agg_factor', 100,
          's_enhance': 3, 't_agg_factor': 5, 't_enhance': 1}]
        If they are not included they will be computed using exo_resolution
        and model attributes.
    exo_resolution : dict
        Dictionary of spatiotemporal resolution for the given exo data
        source. e.g. {'spatial': '4km', 'temporal': '60min'}. This is used
        only if agg factors are not provided in the steps list.
    source_file : str
        Filepath to source wtk, nsrdb, or netcdf file to get hi-res data
        from which will be mapped to the enhanced grid of the file_paths
        input. Pixels from this file will be mapped to their nearest
        low-res pixel in the file_paths input. Accordingly, the input
        should be a significantly higher resolution than file_paths.
        Warnings will be raised if the low-resolution pixels in file_paths
        do not have unique nearest pixels from this exo source data.
    target : tuple
        (lat, lon) lower left corner of raster. Either need target+shape or
        raster_file.
    shape : tuple
        (rows, cols) grid size. Either need target+shape or raster_file.
    time_slice : slice | None
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
    res_kwargs : dict | None
        Dictionary of kwargs passed to lowest level resource handler. e.g.
        xr.open_dataset(file_paths, **res_kwargs)
    """

    AVAILABLE_HANDLERS: ClassVar[dict] = {
        'topography': {'h5': TopoExtractH5, 'nc': TopoExtractNC},
        'sza': {'h5': SzaExtract, 'nc': SzaExtract},
    }

    file_paths: str | list | pathlib.Path
    feature: str
    steps: List[dict]
    models: list = None
    exo_resolution: dict = None
    source_file: str = None
    target: tuple = None
    shape: tuple = None
    time_slice: slice = None
    raster_file: str = None
    max_delta: int = 20
    input_handler: str = None
    exo_handler: str = None
    cache_data: bool = True
    cache_dir: str = './exo_cache'
    res_kwargs: dict = None

    @log_args
    def __post_init__(self):
        self.data = {self.feature: {'steps': []}}
        self.input_check()
        agg_enhance = self._get_all_agg_and_enhancement()
        self.s_enhancements = agg_enhance['s_enhancements']
        self.t_enhancements = agg_enhance['t_enhancements']
        self.s_agg_factors = agg_enhance['s_agg_factors']
        self.t_agg_factors = agg_enhance['t_agg_factors']
        self.step_number_check()
        self.get_all_step_data()

    def input_check(self):
        """Make sure agg factors are provided or exo_resolution and models are
        provided. Make sure enhancement factors are provided or models are
        provided"""
        agg_check = all('s_agg_factor' in v for v in self.steps)
        agg_check = agg_check and all('t_agg_factor' in v for v in self.steps)
        agg_check = agg_check or (
            self.models is not None and self.exo_res is not None
        )
        msg = (
            'ExogenousDataHandler needs s_agg_factor and t_agg_factor '
            'provided in each step in steps list or models and '
            'exo_resolution'
        )
        assert agg_check, msg
        en_check = all('s_enhance' in v for v in self.steps)
        en_check = en_check and all('t_enhance' in v for v in self.steps)
        en_check = en_check or self.models is not None
        msg = (
            'ExogenousDataHandler needs s_enhance and t_enhance '
            'provided in each step in steps list or models'
        )
        assert en_check, msg

    def step_number_check(self):
        """Make sure the number of enhancement factors / agg factors provided
        is internally consistent and consistent with number of model steps."""
        msg = (
            'Need to provide the same number of enhancement factors and '
            f'agg factors. Received s_enhancements={self.s_enhancements}, '
            f'and s_agg_factors={self.s_agg_factors}.'
        )
        assert len(self.s_enhancements) == len(self.s_agg_factors), msg
        msg = (
            'Need to provide the same number of enhancement factors and '
            f'agg factors. Received t_enhancements={self.t_enhancements}, '
            f'and t_agg_factors={self.t_agg_factors}.'
        )
        assert len(self.t_enhancements) == len(self.t_agg_factors), msg

        msg = (
            'Need to provide an integer enhancement factor for each model'
            'step. If the step is temporal enhancement then s_enhance=1'
        )
        assert not any(s is None for s in self.s_enhancements), msg

    def get_all_step_data(self):
        """Get exo data for each model step."""
        for i, _ in enumerate(self.s_enhancements):
            s_enhance = self.s_enhancements[i]
            t_enhance = self.t_enhancements[i]
            s_agg_factor = self.s_agg_factors[i]
            t_agg_factor = self.t_agg_factors[i]
            if self.feature in list(self.AVAILABLE_HANDLERS):
                data = self.get_single_step_data(
                    feature=self.feature,
                    s_enhance=s_enhance,
                    t_enhance=t_enhance,
                    s_agg_factor=s_agg_factor,
                    t_agg_factor=t_agg_factor,
                )
                step = SingleExoDataStep(
                    self.feature,
                    self.steps[i]['combine_type'],
                    self.steps[i]['model'],
                    data,
                )
                self.data[self.feature]['steps'].append(step)
            else:
                msg = (
                    f'Can only extract {list(self.AVAILABLE_HANDLERS)}. '
                    f'Received {self.feature}.'
                )
                raise NotImplementedError(msg)
        shapes = [
            None if step is None else step.shape
            for step in self.data[self.feature]['steps']
        ]
        logger.info(
            'Got exogenous_data of length {} with shapes: {}'.format(
                len(self.data[self.feature]['steps']), shapes
            )
        )

    def _get_res_ratio(self, input_res, exo_res):
        """Compute resolution ratio given input and output resolution

        Parameters
        ----------
        input_res : str | None
            Input resolution. e.g. '30km' or '60min'
        exo_res : str | None
            Exo resolution. e.g. '1km' or '5min'

        Returns
        -------
        res_ratio : int | None
            Ratio of input / exo resolution
        """
        ires_num = (
            None
            if input_res is None
            else int(re.search(r'\d+', input_res).group(0))
        )
        eres_num = (
            None
            if exo_res is None
            else int(re.search(r'\d+', exo_res).group(0))
        )
        i_units = (
            None if input_res is None else input_res.replace(str(ires_num), '')
        )
        e_units = (
            None if exo_res is None else exo_res.replace(str(eres_num), '')
        )
        msg = 'Received conflicting units for input and exo resolution'
        if e_units is not None:
            assert i_units == e_units, msg
        if ires_num is not None and eres_num is not None:
            res_ratio = int(ires_num / eres_num)
        else:
            res_ratio = None
        return res_ratio

    def get_agg_factors(self, input_res, exo_res):
        """Compute aggregation ratio for exo data given input and output
        resolution

        Parameters
        ----------
        input_res : dict | None
            Input resolution. e.g. {'spatial': '30km', 'temporal': '60min'}
        exo_res : dict | None
            Exogenous data resolution. e.g.
            {'spatial': '1km', 'temporal': '5min'}

        Returns
        -------
        s_agg_factor : int
            Spatial aggregation factor for exogenous data extraction.
        t_agg_factor : int
            Temporal aggregation factor for exogenous data extraction.
        """
        input_s_res = None if input_res is None else input_res['spatial']
        exo_s_res = None if exo_res is None else exo_res['spatial']
        s_res_ratio = self._get_res_ratio(input_s_res, exo_s_res)
        s_agg_factor = None if s_res_ratio is None else int(s_res_ratio) ** 2
        input_t_res = None if input_res is None else input_res['temporal']
        exo_t_res = None if exo_res is None else exo_res['temporal']
        t_agg_factor = self._get_res_ratio(input_t_res, exo_t_res)
        return s_agg_factor, t_agg_factor

    def _get_single_step_agg(self, step):
        """Compute agg factors for exogenous data extraction
        using exo_kwargs single model step. These factors are computed using
        exo_resolution and the input/output resolution of each model step. If
        agg factors are already provided in step they are not overwritten.

        Parameters
        ----------
        step : dict
            Model step dictionary. e.g. {'model': 0, 'combine_type': 'input'}

        Returns
        -------
        updated_step : dict
            Same as input dictionary with s_agg_factor, t_agg_factor added
        """
        if all(key in step for key in ['s_agg_factor', 't_agg_factor']):
            return step

        model_step = step['model']
        combine_type = step.get('combine_type', None)
        msg = (
            f'Model index from exo_kwargs ({model_step} exceeds number '
            f'of model steps ({len(self.models)})'
        )
        assert len(self.models) > model_step, msg
        model = self.models[model_step]
        input_res = model.input_resolution
        output_res = model.output_resolution
        if combine_type.lower() == 'input':
            s_agg_factor, t_agg_factor = self.get_agg_factors(
                input_res, self.exo_res
            )

        elif combine_type.lower() in ('output', 'layer'):
            s_agg_factor, t_agg_factor = self.get_agg_factors(
                output_res, self.exo_res
            )

        else:
            msg = (
                'Received exo_kwargs entry without valid combine_type '
                '(input/layer/output)'
            )
            raise OSError(msg)

        step.update(
            {'s_agg_factor': s_agg_factor, 't_agg_factor': t_agg_factor}
        )
        return step

    def _get_single_step_enhance(self, step):
        """Get enhancement factors for exogenous data extraction
        using exo_kwargs single model step. These factors are computed using
        stored enhance attributes of each model and the model step provided.
        If enhancement factors are already provided in step they are not
        overwritten.

        Parameters
        ----------
        step : dict
            Model step dictionary. e.g. {'model': 0, 'combine_type': 'input'}

        Returns
        -------
        updated_step : dict
            Same as input dictionary with s_enhance, t_enhance added
        """
        if all(key in step for key in ['s_enhance', 't_enhance']):
            return step

        model_step = step['model']
        combine_type = step.get('combine_type', None)
        msg = (
            f'Model index from exo_kwargs ({model_step} exceeds number '
            f'of model steps ({len(self.models)})'
        )
        assert len(self.models) > model_step, msg

        s_enhancements = [model.s_enhance for model in self.models]
        t_enhancements = [model.t_enhance for model in self.models]
        if combine_type.lower() == 'input':
            if model_step == 0:
                s_enhance = 1
                t_enhance = 1
            else:
                s_enhance = np.prod(s_enhancements[:model_step])
                t_enhance = np.prod(t_enhancements[:model_step])

        elif combine_type.lower() in ('output', 'layer'):
            s_enhance = np.prod(s_enhancements[: model_step + 1])
            t_enhance = np.prod(t_enhancements[: model_step + 1])

        else:
            msg = (
                'Received exo_kwargs entry without valid combine_type '
                '(input/layer/output)'
            )
            raise OSError(msg)

        step.update({'s_enhance': s_enhance, 't_enhance': t_enhance})
        return step

    def _get_all_agg_and_enhancement(self):
        """Compute agg and enhancement factors for all model steps for all
        features.

        Returns
        -------
        agg_enhance_dict : dict
            Dictionary with list of agg and enhancement factors for each model
            step
        """
        agg_enhance_dict = {}
        for i, step in enumerate(self.steps):
            out = self._get_single_step_agg(step)
            out = self._get_single_step_enhance(out)
            self.steps[i] = out
        agg_enhance_dict['s_agg_factors'] = [
            step['s_agg_factor'] for step in self.steps
        ]
        agg_enhance_dict['t_agg_factors'] = [
            step['t_agg_factor'] for step in self.steps
        ]
        agg_enhance_dict['s_enhancements'] = [
            step['s_enhance'] for step in self.steps
        ]
        agg_enhance_dict['t_enhancements'] = [
            step['t_enhance'] for step in self.steps
        ]
        return agg_enhance_dict

    def get_single_step_data(
        self, feature, s_enhance, t_enhance, s_agg_factor, t_agg_factor
    ):
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
        data : T_Array
            2D or 3D array of exo data with shape (lat, lon) or (lat,
            lon, temporal)
        """

        exo_handler = self.get_exo_handler(
            feature, self.source_file, self.exo_handler
        )
        kwargs = {
            'file_paths': self.file_paths,
            'exo_source': self.source_file,
            's_enhance': s_enhance,
            't_enhance': t_enhance,
            's_agg_factor': s_agg_factor,
            't_agg_factor': t_agg_factor,
            'target': self.target,
            'shape': self.shape,
            'time_slice': self.time_slice,
            'raster_file': self.raster_file,
            'max_delta': self.max_delta,
            'input_handler': self.input_handler,
            'cache_data': self.cache_data,
            'cache_dir': self.cache_dir,
            'res_kwargs': self.res_kwargs,
        }
        data = exo_handler(**get_class_kwargs(exo_handler, kwargs)).data
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
                msg = (
                    f'Did not recognize input type "{in_type}" for file '
                    f'paths: {source_file}'
                )
                logger.error(msg)
                raise RuntimeError(msg)
            check = (
                feature in cls.AVAILABLE_HANDLERS
                and in_type in cls.AVAILABLE_HANDLERS[feature]
            )
            if check:
                exo_handler = cls.AVAILABLE_HANDLERS[feature][in_type]
            else:
                msg = (
                    'Could not find exo handler class for '
                    f'feature={feature} and input_type={in_type}.'
                )
                logger.error(msg)
                raise KeyError(msg)
        elif isinstance(exo_handler, str):
            exo_handler = getattr(sup3r.preprocessing, exo_handler, None)
        return exo_handler
