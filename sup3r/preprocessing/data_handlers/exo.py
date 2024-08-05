"""Exogenous data handler. This performs exo extraction for one or more model
steps for requested features."""

import logging
import pathlib
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

from sup3r.preprocessing.names import Dimension
from sup3r.preprocessing.rasterizers import ExoRasterizer
from sup3r.preprocessing.utilities import log_args

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
    """Special dictionary class for multiple exogenous_data steps

    TODO: Can we simplify this by relying more on xr.Dataset meta data instead
    of storing enhancement factors for each step? Seems like we could take the
    highest res data and coarsen based on s/t enhance, also.
    """

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
            self.update(steps)

        else:
            msg = 'ExoData must be initialized with a dictionary of features.'
            logger.error(msg)
            raise ValueError(msg)

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
            steps = [
                step for step in entry['steps'] if step['model'] == model_step
            ]
            if steps:
                model_step_exo[feature] = {'steps': steps}
        return ExoData(model_step_exo)

    @staticmethod
    def _get_bounded_steps(steps, min_step, max_step=None):
        """Get the steps within `steps` which have a model index between min
        and max step."""
        if max_step is not None:
            return [
                s
                for s in steps
                if (s['model'] < max_step and min_step <= s['model'])
            ]
        return [s for s in steps if min_step <= s['model']]

    def split(self, split_steps):
        """Split `self` into multiple `ExoData` objects based on split_steps.
        The splits are done such that the steps in the ith entry of the
        returned list all have a `model number < split_steps[i].`

        Note
        ----
        This is used for multi-step models to correctly distribute the set of
        all exo data steps to the appropriate models. For example,
        `TemporalThenSpatial` models or models with some spatial steps followed
        by some temporal steps. The temporal (spatial) models might take the
        first N exo data steps and then the spatial (temporal) models will take
        the remaining exo data steps.

        TODO: lots of nested loops here. simplify the logic.

        Parameters
        ----------
        split_steps : list
            Step index list to use for splitting. To split this into exo data
            for spatial models and temporal models split_steps should be
            [len(spatial_models)]. If this is for a TemporalThenSpatial model
            split_steps should be [len(temporal_models)]. If this is for a
            multi step model composed of more than two models (e.g.
            SolarMultiStepGan) split_steps should be
            [len(spatial_solar_models), len(spatial_solar_models) +
            len(spatial_wind_models)]

        Returns
        -------
        split_list : List[ExoData]
            List of `ExoData` objects coming from the split of `self`,
            according to `split_steps`
        """
        split_dict = {i: {} for i in range(len(split_steps) + 1)}
        split_steps = [0, *split_steps] if split_steps[0] != 0 else split_steps
        for feature, entry in self.items():
            for i, min_step in enumerate(split_steps):
                max_step = (
                    None if min_step == split_steps[-1] else split_steps[i + 1]
                )
                steps_i = self._get_bounded_steps(
                    steps=entry['steps'], min_step=min_step, max_step=max_step
                )
                for s in steps_i:
                    s.update({'model': s['model'] - min_step})
                if any(steps_i):
                    split_dict[i][feature] = {'steps': steps_i}
        return [ExoData(split) for split in split_dict.values()]

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
        msg = (
            'Received exogenous_data without any combine_type '
            f'= "{combine_type}" steps'
        )
        assert combine_type in combine_types, msg
        return tmp['steps'][combine_types.index(combine_type)]['data']

    @staticmethod
    def _get_enhanced_slices(lr_slices, input_data_shape, exo_data_shape):
        """Get lr_slices enhanced by the ratio of exo_data_shape to
        input_data_shape. Used to slice exo data for each model step."""
        return [
            slice(
                lr_slices[i].start * exo_data_shape[i] // input_data_shape[i],
                lr_slices[i].stop * exo_data_shape[i] // input_data_shape[i],
            )
            for i in range(len(lr_slices))
        ]

    def get_chunk(self, input_data_shape, lr_slices):
        """Get the data for all model steps corresponding to the low res extent
        selected by `lr_slices`

        Parameters
        ----------
        input_data_shape : tuple
            Spatiotemporal shape of the full low-resolution extent.
            (lats, lons, time)
        lr_slices : list List of spatiotemporal slices which specify extent of
        the low-resolution input data.

        Returns
        -------
        exo_data : ExoData
           :class:`ExoData` object composed of multiple
           :class:`SingleExoDataStep` objects. This is the sliced exo data for
           the extent specified by `lr_slices`.
        """
        logger.debug(f'Getting exo data chunk for lr_slices={lr_slices}.')
        exo_chunk = {f: {'steps': []} for f in self}
        for feature in self:
            for step in self[feature]['steps']:
                enhanced_slices = self._get_enhanced_slices(
                    lr_slices=lr_slices,
                    input_data_shape=input_data_shape,
                    exo_data_shape=step['data'].shape,
                )
                chunk_step = {
                    k: step[k]
                    if k != 'data'
                    else step[k][tuple(enhanced_slices)]
                    for k in step
                }
                exo_chunk[feature]['steps'].append(chunk_step)
        return exo_chunk


@dataclass
class ExoDataHandler:
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
        List of models used with the given steps list. This list of models is
        used to determine the input and output resolution and enhancement
        factors for each model step which is then used to determine the target
        shape for rasterized exo data. If enhancement factors are provided in
        the steps list the model list is not needed.
    steps : list
        List of dictionaries containing info on which models to use for a
        given step index and what type of exo data the step requires. e.g.
        [{'model': 0, 'combine_type': 'input'},
         {'model': 0, 'combine_type': 'layer'}]
        Each step entry can also contain enhancement factors. e.g.
        [{'model': 0, 'combine_type': 'input', 's_enhance': 1, 't_enhance': 1},
         {'model': 0, 'combine_type': 'layer', 's_enhance': 3, 't_enhance': 1}]
    source_file : str
        Filepath to source wtk, nsrdb, or netcdf file to get hi-res data
        from which will be mapped to the enhanced grid of the file_paths
        input. Pixels from this file will be mapped to their nearest
        low-res pixel in the file_paths input. Accordingly, the input
        should be a significantly higher resolution than file_paths.
        Warnings will be raised if the low-resolution pixels in file_paths
        do not have unique nearest pixels from this exo source data.
    input_handler_name : str
        data handler class used by the exo handler. Provide a string name to
        match a :class:`~sup3r.preprocessing.rasterizers.Rasterizer`. If None
        the correct handler will be guessed based on file type and time series
        properties. This is passed directly to the exo handler, along with
        input_handler_kwargs
    input_handler_kwargs : dict | None
        Any kwargs for initializing the `input_handler_name` class used by the
        exo handler.
    cache_dir : str | None
        Directory for storing cache data. Default is './exo_cache'. If None
        then no data will be cached.
    distance_upper_bound : float | None
        Maximum distance to map high-resolution data from source_file to the
        low-resolution file_paths input. None (default) will calculate this
        based on the median distance between points in source_file
    """

    file_paths: Union[str, list, pathlib.Path]
    feature: str
    steps: List[dict]
    models: Optional[list] = None
    source_file: Optional[str] = None
    input_handler_name: Optional[str] = None
    input_handler_kwargs: Optional[dict] = None
    cache_dir: str = './exo_cache'
    distance_upper_bound: Optional[int] = None

    @log_args
    def __post_init__(self):
        """Initialize `self.data`, perform checks on enhancement factors, and
        update `self.data` for each model step with rasterized exo data for the
        corresponding enhancement factors."""
        self.data = {self.feature: {'steps': []}}
        en_check = all('s_enhance' in v for v in self.steps)
        en_check = en_check and all('t_enhance' in v for v in self.steps)
        en_check = en_check or self.models is not None
        msg = (
            f'{self.__class__.__name__} needs s_enhance and t_enhance '
            'provided in each step in steps list or models'
        )
        assert en_check, msg
        self.s_enhancements, self.t_enhancements = self._get_all_enhancement()
        msg = (
            'Need to provide s_enhance and t_enhance for each model'
            'step. If the step is temporal only (spatial only) then '
            's_enhance = 1 (t_enhance = 1).'
        )
        assert not any(s is None for s in self.s_enhancements), msg
        assert not any(t is None for t in self.t_enhancements), msg

        self.get_all_step_data()

    def get_all_step_data(self):
        """Get exo data for each model step. We get the maximally enhanced
        exo data and then coarsen this to get the exo data for each enhancement
        step. We get coarsen factors by iterating through enhancement factors
        in reverse.
        """
        hr_exo = ExoRasterizer(
            file_paths=self.file_paths,
            source_file=self.source_file,
            feature=self.feature,
            s_enhance=self.s_enhancements[-1],
            t_enhance=self.t_enhancements[-1],
            input_handler_name=self.input_handler_name,
            input_handler_kwargs=self.input_handler_kwargs,
            cache_dir=self.cache_dir,
            distance_upper_bound=self.distance_upper_bound,
        )
        for i, (s_coarsen, t_coarsen) in enumerate(
            zip(self.s_enhancements[::-1], self.t_enhancements[::-1])
        ):
            coarsen_kwargs = dict(
                zip(Dimension.dims_3d(), [s_coarsen, s_coarsen, t_coarsen])
            )
            step = SingleExoDataStep(
                self.feature,
                self.steps[i]['combine_type'],
                self.steps[i]['model'],
                data=hr_exo.data.coarsen(**coarsen_kwargs).mean().as_array(),
            )
            self.data[self.feature]['steps'].append(step)
        shapes = [
            None if step is None else step.shape
            for step in self.data[self.feature]['steps']
        ]
        logger.info(
            'Got exogenous_data of length {} with shapes: {}'.format(
                len(self.data[self.feature]['steps']), shapes
            )
        )

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
        msg = (
            'Received exo_kwargs entry without valid combine_type '
            '(input/layer/output)'
        )
        assert combine_type.lower() in ('input', 'output', 'layer'), msg
        s_enhancements = [model.s_enhance for model in self.models]
        t_enhancements = [model.t_enhance for model in self.models]
        if combine_type.lower() == 'input':
            if model_step == 0:
                s_enhance = 1
                t_enhance = 1
            else:
                s_enhance = np.prod(s_enhancements[:model_step])
                t_enhance = np.prod(t_enhancements[:model_step])

        else:
            s_enhance = np.prod(s_enhancements[: model_step + 1])
            t_enhance = np.prod(t_enhancements[: model_step + 1])
        step.update({'s_enhance': s_enhance, 't_enhance': t_enhance})
        return step

    def _get_all_enhancement(self):
        """Compute enhancement factors for all model steps for all features.

        Returns
        -------
        s_enhancements: list
            List of s_enhance factors for all model steps
        t_enhancements: list
            List of t_enhance factors for all model steps
        """
        for i, step in enumerate(self.steps):
            out = self._get_single_step_enhance(step)
            self.steps[i] = out
        s_enhancements = [step['s_enhance'] for step in self.steps]
        t_enhancements = [step['t_enhance'] for step in self.steps]
        return s_enhancements, t_enhancements
