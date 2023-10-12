"""Sup3r exogenous data handling"""
import logging
import os
import pickle
import re
import shutil
from typing import ClassVar

import numpy as np

from sup3r.preprocessing.data_handling import exo_extraction
from sup3r.preprocessing.data_handling.exo_extraction import (
    SzaExtract,
    TopoExtractH5,
    TopoExtractNC,
)
from sup3r.utilities.utilities import get_source_type

logger = logging.getLogger(__name__)


class SingleExoDataStep(dict):
    """Special dictionary class for exogenous_data step"""

    def __init__(self, feature, combine_type, model, data):
        """exogenous_data step dictionary for a given model step

        Parameters
        ----------
        feature : str
            Name of feature corresponding to `data`.
        combine_type : str
            Specifies how the exogenous_data should be used for this step. e.g.
            "input", "layer", "output". For example, if tis equals "input" the
            `data` will be used as input to the forward pass for the model step
            given by `model`
        model : int
            Specifies the model index which will use the `data`. For example,
            if `model` == 1 then the `data` will be used according to
            `combine_type` in the 2nd model step in a MultiStepGan.
        data : tf.Tensor | np.ndarray
            The data to be used for the given model step.
        """
        step = {'model': model, 'combine_type': combine_type, 'data': data}
        for k, v in step.items():
            self.__setitem__(k, v)
        self.feature = feature

    @property
    def shape(self):
        """Shape of data array for this model step."""
        return self['data'].shape


class ExoData(dict):
    """Special dictionary class for multiple exogenous_data steps"""

    def __init__(self, steps):
        """Combine multiple SingleExoDataStep objects

        Parameters
        ----------
        steps : dict
            Dictionary with feature keys each with entries describing whether
            features should be combined at input, a mid network layer, or with
            output. e.g.
            {'topography': {'steps': [
                {'combine_type': 'input', 'model': 0, 'data': ...,
                 'resolution': ...},
                {'combine_type': 'layer', 'model': 0, 'data': ...,
                 'resolution': ...}]}}
            Each array in in 'data' key has 3D or 4D shape:
            (spatial_1, spatial_2, 1)
            (spatial_1, spatial_2, n_temporal, 1)
        """
        if isinstance(steps, dict):
            for k, v in steps.items():
                self.__setitem__(k, v)
        else:
            msg = 'ExoData must be initialized with a dictionary of features.'
            logger.error(msg)
            raise ValueError(msg)

    def append(self, feature, step):
        """Append steps list for given feature"""
        tmp = self.get(feature, {'steps': []})
        tmp['steps'].append(step)
        self[feature] = tmp

    def get_model_step_exo(self, model_step):
        """Get the exogenous data for the given model_step from the full list
        of steps

        Parameters
        ----------
        model_step : int
            Index of the model to get exogenous data for.

        Returns
        -------
        model_step_exo : dict
            Dictionary of features each with list of steps which match the
            given model_step
        """
        model_step_exo = {}
        for feature, entry in self.items():
            steps = [step for step in entry['steps']
                     if step['model'] == model_step]
            if steps:
                model_step_exo[feature] = {'steps': steps}
        return ExoData(model_step_exo)

    def split_exo_dict(self, split_step):
        """Split exogenous_data into two dicts based on split_step. The first
        dict has only model steps less than split_step. The second dict has
        only model steps greater than or equal to split_step.

        Parameters
        ----------
        split_step : int
            Step index to use for splitting. To split this into exo data for
            spatial models and temporal models split_step should be
            len(spatial_models). If this is for a TemporalThenSpatial model
            split_step should be len(temporal_models).

        Returns
        -------
        split_exo_1 : dict
            Same as input dictionary but with only entries with 'model':
            model_step where model_step is less than split_step
        split_exo_2 : dict
            Same as input dictionary but with only entries with 'model':
            model_step where model_step is greater than or equal to split_step
        """
        split_exo_1 = {}
        split_exo_2 = {}
        for feature, entry in self.items():
            steps = [step for step in entry['steps']
                     if step['model'] < split_step]
            if steps:
                split_exo_1[feature] = {'steps': steps}
            steps = [step for step in entry['steps']
                     if step['model'] >= split_step]
            for step in steps:
                step.update({'model': step['model'] - split_step})
            if steps:
                split_exo_2[feature] = {'steps': steps}
        return ExoData(split_exo_1), ExoData(split_exo_2)

    def get_combine_type_data(self, feature, combine_type, model_step=None):
        """Get exogenous data for given feature which is used according to the
        given combine_type (input/output/layer) for this model_step.

        Parameters
        ----------
        feature : str
            Name of exogenous feature to get data for
        combine_type : str
            Usage type for requested data. e.g input/output/layer
        model_step : int | None
            Model step the data will be used for. If this is not None then
            only steps with self[feature]['steps'][:]['model'] == model_step
            will be searched for data.

        Returns
        -------
        data : tf.Tensor | np.ndarray
            Exogenous data for given parameters
        """
        tmp = self[feature]
        if model_step is not None:
            tmp = {k: v for k, v in tmp.items() if v['model'] == model_step}
        combine_types = [step['combine_type'] for step in tmp['steps']]
        msg = ('Received exogenous_data without any combine_type '
               f'= "{combine_type}" steps')
        assert combine_type in combine_types, msg
        idx = combine_types.index(combine_type)
        return tmp['steps'][idx]['data']


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
            'h5': SzaExtract,
            'nc': SzaExtract
        }
    }

    def __init__(self,
                 file_paths,
                 feature,
                 steps,
                 models=None,
                 exo_resolution=None,
                 source_file=None,
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
            Filepath to source wtk, nsrdb, or netcdf file to get hi-res (2km or
            4km) data from which will be mapped to the enhanced grid of the
            file_paths input
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
        self.steps = steps
        self.models = models
        self.exo_res = exo_resolution
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
        self.data = {feature: {'steps': []}}

        self.input_check()
        agg_enhance = self._get_all_agg_and_enhancement()
        self.s_enhancements = agg_enhance['s_enhancements']
        self.t_enhancements = agg_enhance['t_enhancements']
        self.s_agg_factors = agg_enhance['s_agg_factors']
        self.t_agg_factors = agg_enhance['t_agg_factors']

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
                step = SingleExoDataStep(feature, steps[i]['combine_type'],
                                         steps[i]['model'], data)
                self.data[feature]['steps'].append(step)
            else:
                msg = (f"Can only extract {list(self.AVAILABLE_HANDLERS)}."
                       f" Received {feature}.")
                raise NotImplementedError(msg)
        shapes = [None if step is None else step.shape
                  for step in self.data[feature]['steps']]
        logger.info(
            'Got exogenous_data of length {} with shapes: {}'.format(
                len(self.data[feature]['steps']), shapes))

    def input_check(self):
        """Make sure agg factors are provided or exo_resolution and models are
        provided. Make sure enhancement factors are provided or models are
        provided"""
        agg_check = all('s_agg_factor' in v for v in self.steps)
        agg_check = agg_check and all('t_agg_factor' in v for v in self.steps)
        agg_check = (agg_check
                     or self.models is not None and self.exo_res is not None)
        msg = ("ExogenousDataHandler needs s_agg_factor and t_agg_factor "
               "provided in each step in steps list or models and "
               "exo_resolution")
        assert agg_check, msg
        en_check = all('s_enhance' in v for v in self.steps)
        en_check = en_check and all('t_enhance' in v for v in self.steps)
        en_check = en_check or self.models is not None
        msg = ("ExogenousDataHandler needs s_enhance and t_enhance "
               "provided in each step in steps list or models")
        assert en_check, msg

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
        ires_num = (None if input_res is None
                    else int(re.search(r'\d+', input_res).group(0)))
        eres_num = (None if exo_res is None
                    else int(re.search(r'\d+', exo_res).group(0)))
        i_units = (None if input_res is None
                   else input_res.replace(str(ires_num), ''))
        e_units = (None if exo_res is None
                   else exo_res.replace(str(eres_num), ''))
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
        s_agg_factor = None if s_res_ratio is None else int(s_res_ratio)**2
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
        msg = (f'Model index from exo_kwargs ({model_step} exceeds number '
               f'of model steps ({len(self.models)})')
        assert len(self.models) > model_step, msg
        model = self.models[model_step]
        input_res = model.input_resolution
        output_res = model.output_resolution
        if combine_type.lower() == 'input':
            s_agg_factor, t_agg_factor = self.get_agg_factors(
                input_res, self.exo_res)

        elif combine_type.lower() in ('output', 'layer'):
            s_agg_factor, t_agg_factor = self.get_agg_factors(
                output_res, self.exo_res)

        else:
            msg = ('Received exo_kwargs entry without valid combine_type '
                   '(input/layer/output)')
            raise OSError(msg)

        step.update({'s_agg_factor': s_agg_factor,
                     't_agg_factor': t_agg_factor})
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
        msg = (f'Model index from exo_kwargs ({model_step} exceeds number '
               f'of model steps ({len(self.models)})')
        assert len(self.models) > model_step, msg

        s_enhancements = [model.s_enhance for model in self.models]
        t_enhancements = [model.t_enhance for model in self.models]
        if combine_type.lower() == 'input':
            if model_step == 0:
                s_enhance = 1
                t_enhance = 1
            else:
                s_enhance = np.product(s_enhancements[:model_step])
                t_enhance = np.product(t_enhancements[:model_step])

        elif combine_type.lower() in ('output', 'layer'):
            s_enhance = np.product(s_enhancements[:model_step + 1])
            t_enhance = np.product(t_enhancements[:model_step + 1])

        else:
            msg = ('Received exo_kwargs entry without valid combine_type '
                   '(input/layer/output)')
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
        agg_enhance_dict['s_agg_factors'] = [step['s_agg_factor']
                                             for step in self.steps]
        agg_enhance_dict['t_agg_factors'] = [step['t_agg_factor']
                                             for step in self.steps]
        agg_enhance_dict['s_enhancements'] = [step['s_enhance']
                                              for step in self.steps]
        agg_enhance_dict['t_enhancements'] = [step['t_enhance']
                                              for step in self.steps]
        return agg_enhance_dict

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
        elif isinstance(exo_handler, str):
            exo_handler = getattr(exo_extraction, exo_handler, None)
        return exo_handler
