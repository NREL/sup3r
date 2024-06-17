"""Exogenous data handler. This performs exo extraction for one or more model
steps for requested features.

TODO: More cleaning. This does not yet fit the new style of composition and
lazy loading.
"""

import logging
import pathlib
from dataclasses import dataclass
from typing import ClassVar, List

import numpy as np

import sup3r.preprocessing
from sup3r.preprocessing.data_handlers.base import SingleExoDataStep
from sup3r.preprocessing.extracters import (
    SzaExtracter,
    TopoExtracterH5,
    TopoExtracterNC,
)
from sup3r.preprocessing.utilities import (
    get_possible_class_args,
    get_source_type,
    log_args,
)

logger = logging.getLogger(__name__)


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
        shape for extracted exo data. If enhancement factors are provided in
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
        feature='topography' this should be either TopoExtracterH5 or
        TopoExtracterNC. If None the correct handler will be guessed based on
        file type and time series properties.
    cache_dir : str | None
        Directory for storing cache data. Default is './exo_cache'. If None
        then no data will be cached.
    res_kwargs : dict | None
        Dictionary of kwargs passed to lowest level resource handler. e.g.
        xr.open_dataset(file_paths, **res_kwargs)
    """

    AVAILABLE_HANDLERS: ClassVar[dict] = {
        'topography': {'h5': TopoExtracterH5, 'nc': TopoExtracterNC},
        'sza': {'h5': SzaExtracter, 'nc': SzaExtracter},
    }

    file_paths: str | list | pathlib.Path
    feature: str
    steps: List[dict]
    models: list = None
    source_file: str = None
    target: tuple = None
    shape: tuple = None
    time_slice: slice = None
    raster_file: str = None
    max_delta: int = 20
    input_handler: str = None
    exo_handler: str = None
    cache_dir: str = './exo_cache'
    res_kwargs: dict = None

    @log_args
    def __post_init__(self):
        """Initialize `self.data`, perform checks on enhancement factors, and
        update `self.data` for each model step with extracted exo data for the
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
        """Get exo data for each model step.

        TODO: I think this could be simplified by getting the highest res data
        first and then calling the xr.Dataset.coarsen() method according to
        enhancement factors for different steps.

        """
        for i, (s_enhance, t_enhance) in enumerate(
            zip(self.s_enhancements, self.t_enhancements)
        ):
            if self.feature in list(self.AVAILABLE_HANDLERS):
                data = self.get_single_step_data(
                    feature=self.feature,
                    s_enhance=s_enhance,
                    t_enhance=t_enhance,
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
        s_enhancements = [
            step['s_enhance'] for step in self.steps
        ]
        t_enhancements = [
            step['t_enhance'] for step in self.steps
        ]
        return s_enhancements, t_enhancements

    def get_single_step_data(self, feature, s_enhance, t_enhance):
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

        Returns
        -------
        data : T_Array
            2D or 3D array of exo data with shape (lat, lon) or (lat,
            lon, temporal)
        """

        ExoHandler = self.get_exo_handler(
            feature, self.source_file, self.exo_handler
        )
        kwargs = {
            's_enhance': s_enhance,
            't_enhance': t_enhance,
        }

        params = get_possible_class_args(ExoHandler)
        kwargs.update(
            {k: getattr(self, k) for k in params if hasattr(self, k)}
        )
        data = ExoHandler(**kwargs).data
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
            feature='topography' this should be either TopoExtracterH5 or
            TopoExtracterNC. If None the correct handler will be guessed based
            on file type and time series properties.

        Returns
        -------
        exo_handler : str
            Exogenous feature extraction class to use for source data.
        """
        if exo_handler is None:
            in_type = get_source_type(source_file)
            msg = (
                f'Did not recognize input type "{in_type}" for file '
                f'paths: {source_file}'
            )
            assert in_type in ('h5', 'nc'), msg
            msg = (
                'Could not find exo handler class for '
                f'feature={feature} and input_type={in_type}.'
            )
            assert (
                feature in cls.AVAILABLE_HANDLERS
                and in_type in cls.AVAILABLE_HANDLERS[feature]
            ), msg
            exo_handler = cls.AVAILABLE_HANDLERS[feature][in_type]
        elif isinstance(exo_handler, str):
            exo_handler = getattr(sup3r.preprocessing, exo_handler, None)
        return exo_handler
