# -*- coding: utf-8 -*-
"""
Sup3r forward pass handling module.

@author: bbenton
"""
import copy
import logging
from concurrent.futures import as_completed
from datetime import datetime as dt
from inspect import signature
from typing import ClassVar

import numpy as np
import psutil
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.fun_utils import get_fun_call_str

import sup3r.bias.bias_transforms
import sup3r.models
from sup3r.pipeline.strategy import ForwardPassStrategy
from sup3r.postprocessing import (
    OutputHandlerH5,
    OutputHandlerNC,
)
from sup3r.preprocessing import (
    ExoData,
    ExogenousDataHandler,
)
from sup3r.utilities import ModuleName
from sup3r.utilities.cli import BaseCLI

np.random.seed(42)

logger = logging.getLogger(__name__)


class StrategyInterface:
    """Object which interfaces with the :class:`Strategy` instance to get
    details for each chunk going through the generator."""

    def __init__(self, strategy):
        """
        Parameters
        ----------
        strategy : ForwardPassStrategy
            ForwardPassStrategy instance with information on data chunks to run
        forward passes on."""

        self.strategy = strategy

    def __call__(self, chunk_index):
        """Get the target, shape, and set of slices for the current chunk."""

        s_chunk_idx = self.strategy._get_spatial_chunk_index(chunk_index)
        t_chunk_idx = self.strategy._get_temporal_chunk_index(chunk_index)
        ti_crop_slice = self.strategy.fwp_slicer.t_lr_crop_slices[t_chunk_idx]
        lr_pad_slice = self.strategy.lr_pad_slices[s_chunk_idx]
        spatial_slice = lr_pad_slice[0], lr_pad_slice[1]
        target = self.strategy.lr_lat_lon[spatial_slice][-1, 0]
        shape = self.strategy.lr_lat_lon[spatial_slice].shape[:-1]
        ti_slice = self.strategy.ti_slices[t_chunk_idx]
        ti_pad_slice = self.strategy.ti_pad_slices[t_chunk_idx]
        lr_slice = self.strategy.lr_slices[s_chunk_idx]
        hr_slice = self.strategy.hr_slices[s_chunk_idx]

        hr_crop_slices = self.strategy.fwp_slicer.hr_crop_slices[t_chunk_idx]
        hr_crop_slice = hr_crop_slices[s_chunk_idx]

        lr_crop_slice = self.strategy.fwp_slicer.s_lr_crop_slices[s_chunk_idx]
        chunk_shape = (lr_pad_slice[0].stop - lr_pad_slice[0].start,
                       lr_pad_slice[1].stop - lr_pad_slice[1].start,
                       ti_pad_slice.stop - ti_pad_slice.start)
        lr_lat_lon = self.strategy.lr_lat_lon[lr_slice[0], lr_slice[1]]
        hr_lat_lon = self.strategy.hr_lat_lon[hr_slice[0], hr_slice[1]]
        pad_width = self.get_pad_width(ti_slice, lr_slice)

        chunk_desc = {
            'target': target,
            'shape': shape,
            'chunk_shape': chunk_shape,
            'ti_slice': ti_slice,
            'ti_pad_slice': ti_pad_slice,
            'ti_crop_slice': ti_crop_slice,
            'lr_slice': lr_slice,
            'lr_pad_slice': lr_pad_slice,
            'lr_crop_slice': lr_crop_slice,
            'hr_slice': hr_slice,
            'hr_crop_slice': hr_crop_slice,
            'lr_lat_lon': lr_lat_lon,
            'hr_lat_lon': hr_lat_lon,
            'pad_width': pad_width}
        return chunk_desc

    def get_pad_width(self, ti_slice, lr_slice):
        """Get padding for the current spatiotemporal chunk

        Returns
        -------
        padding : tuple
            Tuple of tuples with padding width for spatial and temporal
            dimensions. Each tuple includes the start and end of padding for
            that dimension. Ordering is spatial_1, spatial_2, temporal.
        """
        ti_start = ti_slice.start or 0
        ti_stop = ti_slice.stop or self.strategy.raw_tsteps
        pad_t_start = int(
            np.maximum(0, (self.strategy.temporal_pad - ti_start)))
        pad_t_end = (self.strategy.temporal_pad + ti_stop
                     - self.strategy.raw_tsteps)
        pad_t_end = int(np.maximum(0, pad_t_end))

        s1_start = lr_slice[0].start or 0
        s1_stop = lr_slice[0].stop or self.strategy.grid_shape[0]
        pad_s1_start = int(
            np.maximum(0, (self.strategy.spatial_pad - s1_start)))
        pad_s1_end = (self.strategy.spatial_pad + s1_stop
                      - self.strategy.grid_shape[0])
        pad_s1_end = int(np.maximum(0, pad_s1_end))

        s2_start = lr_slice[1].start or 0
        s2_stop = lr_slice[1].stop or self.strategy.grid_shape[1]
        pad_s2_start = int(
            np.maximum(0, (self.strategy.spatial_pad - s2_start)))
        pad_s2_end = (self.strategy.spatial_pad + s2_stop
                      - self.strategy.grid_shape[1])
        pad_s2_end = int(np.maximum(0, pad_s2_end))
        return ((pad_s1_start, pad_s1_end), (pad_s2_start, pad_s2_end),
                (pad_t_start, pad_t_end))


class ForwardPass:
    """Class to run forward passes on all chunks provided by the given
    ForwardPassStrategy. The chunks provided by the strategy are all passed
    through the GAN generator to produce high resolution output.
    """

    OUTPUT_HANDLER_CLASS: ClassVar = {'nc': OutputHandlerNC,
                                      'h5': OutputHandlerH5}

    def __init__(self, strategy, chunk_index=0, node_index=0):
        """Initialize ForwardPass with ForwardPassStrategy. The strategy
        provides the data chunks to run forward passes on

        Parameters
        ----------
        strategy : ForwardPassStrategy
            ForwardPassStrategy instance with information on data chunks to run
            forward passes on.
        chunk_index : int
            Index used to select spatiotemporal chunk on which to run
            forward pass.
        node_index : int
            Index of node used to run forward pass
        """
        self.strategy = strategy
        self.chunk_index = chunk_index
        self.node_index = node_index
        self.output_data = None
        self.strategy_interface = StrategyInterface(strategy)
        chunk_description = self.strategy_interface(chunk_index)
        self.update_attributes(chunk_description)

        msg = (f'Requested forward pass on chunk_index={chunk_index} > '
               f'n_chunks={strategy.chunks}')
        assert chunk_index <= strategy.chunks, msg

        logger.info(f'Initializing ForwardPass for chunk={chunk_index} '
                    f'(temporal_chunk={self.temporal_chunk_index}, '
                    f'spatial_chunk={self.spatial_chunk_index}). {self.chunks}'
                    f' total chunks for the current node.')

        msg = f'Received bad output type {strategy.output_type}'
        if strategy.output_type in list(self.OUTPUT_HANDLER_CLASS):
            self.output_handler_class = self.OUTPUT_HANDLER_CLASS[
                strategy.output_type]

        logger.info(f'Getting input data for chunk_index={chunk_index}.')
        self.input_data, self.exogenous_data = self.get_input_and_exo_data()

    def get_input_and_exo_data(self):
        """Get input and exo data chunks."""
        input_data = self.strategy.extracter.data[
            self.lr_pad_slice[0], self.lr_pad_slice[1], self.ti_pad_slice
        ]
        exo_data = self.load_exo_data()
        input_data = self.bias_correct_source_data(
            input_data, self.strategy.lr_lat_lon
        )
        input_data, exo_data = self.pad_source_data(
            input_data, self.pad_width, exo_data
        )
        return input_data, exo_data

    def update_attrs(self, chunk_desc):
        """Update self attributes with values for the current chunk."""
        for attr, val in chunk_desc.items():
            setattr(self, attr, val)
        for attr in [
            's_enhance',
            't_enhance',
            'model_kwargs',
            'model_class',
            'model',
            'output_features',
            'features',
            'file_paths',
            'pass_workers',
            'output_workers',
            'exo_features'
        ]:
            setattr(self, attr, getattr(self.strategy, attr))

    def load_exo_data(self):
        """Extract exogenous data for each exo feature and store data in
        dictionary with key for each exo feature

        Returns
        -------
        exo_data : ExoData
           :class:`ExoData` object composed of multiple
           :class:`SingleExoDataStep` objects.
        """
        data = {}
        exo_data = None
        if self.exo_kwargs:
            for feature in self.exo_features:
                exo_kwargs = copy.deepcopy(self.exo_kwargs[feature])
                exo_kwargs['feature'] = feature
                exo_kwargs['target'] = self.target
                exo_kwargs['shape'] = self.shape
                exo_kwargs['time_slice'] = self.ti_pad_slice
                exo_kwargs['models'] = getattr(self.model, 'models',
                                               [self.model])
                sig = signature(ExogenousDataHandler)
                exo_kwargs = {k: v for k, v in exo_kwargs.items()
                              if k in sig.parameters}
                data.update(ExogenousDataHandler(**exo_kwargs).data)
            exo_data = ExoData(data)
        return exo_data

    @property
    def hr_times(self):
        """Get high resolution times for the current chunk"""
        lr_times = self.extracter.time_index[self.ti_crop_slice]
        return self.output_handler_class.get_times(
            lr_times, self.t_enhance * len(lr_times))

    @property
    def chunk_specific_meta(self):
        """Meta with chunk specific info. To be included in chunk output file
        global attributes."""
        meta_data = {
            "node_index": self.node_index,
            'creation_date': dt.now().strftime("%d/%m/%Y %H:%M:%S"),
            'fwp_chunk_shape': self.strategy.fwp_chunk_shape,
            'spatial_pad': self.strategy.spatial_pad,
            'temporal_pad': self.strategy.temporal_pad,
        }
        return meta_data

    @property
    def meta(self):
        """Meta data dictionary for the forward pass run (to write to output
        files)."""
        meta_data = {
            'chunk_meta': self.chunk_specific_meta,
            'gan_meta': self.model.meta,
            'gan_params': self.model.model_params,
            'model_kwargs': self.model_kwargs,
            'model_class': self.model_class,
            'spatial_enhance': int(self.s_enhance),
            'temporal_enhance': int(self.t_enhance),
            'input_files': self.file_paths,
            'input_features': self.strategy.features,
            'output_features': self.strategy.output_features,
        }
        return meta_data

    @property
    def gids(self):
        """Get gids for the current chunk"""
        return self.strategy.gids[self.hr_slice[0], self.hr_slice[1]]

    @property
    def chunks(self):
        """Number of chunks for current node"""
        return len(self.strategy.node_chunks[self.node_index])

    @property
    def spatial_chunk_index(self):
        """Spatial index for the current chunk going through forward pass"""
        return self.strategy._get_spatial_chunk_index(self.chunk_index)

    @property
    def temporal_chunk_index(self):
        """Temporal index for the current chunk going through forward pass"""
        return self.strategy._get_temporal_chunk_index(self.chunk_index)

    @property
    def out_file(self):
        """Get output file name for the current chunk"""
        return self.strategy.out_files[self.chunk_index]

    @property
    def cache_pattern(self):
        """Get cache pattern for the current chunk"""
        cache_pattern = self.strategy.cache_pattern
        if cache_pattern is not None:
            if '{temporal_chunk_index}' not in cache_pattern:
                cache_pattern = cache_pattern.replace(
                    '.pkl', '_{temporal_chunk_index}.pkl')
            if '{spatial_chunk_index}' not in cache_pattern:
                cache_pattern = cache_pattern.replace(
                    '.pkl', '_{spatial_chunk_index}.pkl')
            cache_pattern = cache_pattern.replace(
                '{temporal_chunk_index}', str(self.temporal_chunk_index))
            cache_pattern = cache_pattern.replace(
                '{spatial_chunk_index}', str(self.spatial_chunk_index))
        return cache_pattern

    def _get_step_enhance(self, step):
        """Get enhancement factors for a given step and combine type.

        Parameters
        ----------
        step : dict
            Model step dictionary. e.g. {'model': 0, 'combine_type': 'input'}

        Returns
        -------
        s_enhance : int
            Spatial enhancement factor for given step and combine type
        t_enhance : int
            Temporal enhancement factor for given step and combine type
        """
        combine_type = step['combine_type']
        model_step = step['model']
        if combine_type.lower() == 'input':
            if model_step == 0:
                s_enhance = 1
                t_enhance = 1
            else:
                s_enhance = np.prod(
                    self.strategy.s_enhancements[:model_step])
                t_enhance = np.prod(
                    self.strategy.t_enhancements[:model_step])

        elif combine_type.lower() in ('output', 'layer'):
            s_enhance = np.prod(
                self.strategy.s_enhancements[:model_step + 1])
            t_enhance = np.prod(
                self.strategy.t_enhancements[:model_step + 1])
        return s_enhance, t_enhance

    def pad_source_data(self, input_data, pad_width, exo_data, mode='reflect'):
        """Pad the edges of the source data from the data handler.

        Parameters
        ----------
        input_data : np.ndarray
            Source input data from data handler class, shape is:
            (spatial_1, spatial_2, temporal, features)
        pad_width : tuple
            Tuple of tuples with padding width for spatial and temporal
            dimensions. Each tuple includes the start and end of padding for
            that dimension. Ordering is spatial_1, spatial_2, temporal.
        exo_data: dict
            Full exo_kwargs dictionary with all feature entries.
            e.g. {'topography': {'exo_resolution': {'spatial': '1km',
            'temporal': None}, 'steps': [{'model': 0, 'combine_type': 'input'},
            {'model': 0, 'combine_type': 'layer'}]}}
        mode : str
            Mode to use for padding. e.g. 'reflect'.

        Returns
        -------
        out : np.ndarray
            Padded copy of source input data from data handler class, shape is:
            (spatial_1, spatial_2, temporal, features)
        exo_data : dict
            Same as input dictionary with s_agg_factor, t_agg_factor,
            s_enhance, t_enhance added to each step entry for all features

        """
        out = np.pad(input_data, (*pad_width, (0, 0)), mode=mode)

        logger.info('Padded input data shape from {} to {} using mode "{}" '
                    'with padding argument: {}'.format(input_data.shape,
                                                       out.shape,
                                                       mode,
                                                       pad_width))

        if exo_data is not None:
            for feature in exo_data:
                for i, step in enumerate(exo_data[feature]['steps']):
                    s_enhance, t_enhance = self._get_step_enhance(step)
                    exo_pad_width = ((s_enhance * pad_width[0][0],
                                      s_enhance * pad_width[0][1]),
                                     (s_enhance * pad_width[1][0],
                                      s_enhance * pad_width[1][1]),
                                     (t_enhance * pad_width[2][0],
                                      t_enhance * pad_width[2][1]),
                                     (0, 0))
                    new_exo = np.pad(step['data'], exo_pad_width, mode=mode)
                    exo_data[feature]['steps'][i]['data'] = new_exo
        return out, exo_data

    def bias_correct_source_data(self, data, lat_lon):
        """Bias correct data using a method defined by the bias_correct_method
        input to ForwardPassStrategy

        Parameters
        ----------
        data : np.ndarray
            Any source data to be bias corrected, with the feature channel in
            the last axis.
        lat_lon : np.ndarray
            Latitude longitude array for the given data. Used to get the
            correct bc factors for the appropriate domain.
            (n_lats, n_lons, 2)

        Returns
        -------
        data : np.ndarray
            Data corrected by the bias_correct_method ready for input to the
            forward pass through the generative model.
        """
        method = self.strategy.bias_correct_method
        kwargs = self.strategy.bias_correct_kwargs
        if method is not None:
            method = getattr(sup3r.bias.bias_transforms, method)
            logger.info('Running bias correction with: {}'.format(method))
            for feature, feature_kwargs in kwargs.items():
                idf = self.data_handler.features.index(feature)

                if 'lr_padded_slice' in signature(method).parameters:
                    feature_kwargs['lr_padded_slice'] = self.lr_padded_slice
                if 'time_index' in signature(method).parameters:
                    feature_kwargs['time_index'] = self.data_handler.time_index

                logger.debug('Bias correcting feature "{}" at axis index {} '
                             'using function: {} with kwargs: {}'.format(
                                 feature, idf, method, feature_kwargs))

                data[..., idf] = method(data[..., idf],
                                        lat_lon=lat_lon,
                                        **feature_kwargs)

        return data

    @classmethod
    def _run_generator(cls,
                       data_chunk,
                       hr_crop_slices,
                       model=None,
                       model_kwargs=None,
                       model_class=None,
                       s_enhance=None,
                       t_enhance=None,
                       exo_data=None):
        """Run forward pass of the generator on smallest data chunk. Each chunk
        has a maximum shape given by self.strategy.fwp_chunk_shape.

        Parameters
        ----------
        data_chunk : ndarray
            Low res data chunk to go through generator
        hr_crop_slices : list
            List of slices for extracting cropped region of interest from
            output. Output can include an extra overlapping boundary to
            reduce chunking error. The cropping cuts off this padded region
            before stitching chunks.
        model : Sup3rGan
            A loaded Sup3rGan model (any model imported from sup3r.models).
            You need to provide either model or (model_kwargs and model_class)
        model_kwargs : str | list
            Keyword arguments to send to `model_class.load(**model_kwargs)` to
            initialize the GAN. Typically this is just the string path to the
            model directory, but can be multiple models or arguments for more
            complex models.
            You need to provide either model or (model_kwargs and model_class)
        model_class : str
            Name of the sup3r model class for the GAN model to load. The
            default is the basic spatial / spatiotemporal Sup3rGan model. This
            will be loaded from sup3r.models
            You need to provide either model or (model_kwargs and model_class)
        model_path : str
            Path to file for Sup3rGan used to generate high resolution
            data
        t_enhance : int
            Factor by which to enhance temporal resolution
        s_enhance : int
            Factor by which to enhance spatial resolution
        exo_data : dict | None
            Dictionary of exogenous feature data with entries describing
            whether features should be combined at input, a mid network layer,
            or with output. e.g.
            {'topography': {'steps': [
                {'combine_type': 'input', 'model': 0, 'data': ...,
                 'resolution': ...},
                {'combine_type': 'layer', 'model': 0, 'data': ...,
                 'resolution': ...}]}}

        Returns
        -------
        ndarray
            High resolution data generated by GAN
        """
        if model is None:
            msg = 'If model not provided, model_kwargs and model_class must be'
            assert model_kwargs is not None, msg
            assert model_class is not None, msg
            model_class = getattr(sup3r.models, model_class)
            model = model_class.load(**model_kwargs, verbose=False)

        temp = cls._reshape_data_chunk(model, data_chunk, exo_data)
        data_chunk, exo_data, i_lr_t, i_lr_s = temp

        try:
            hi_res = model.generate(data_chunk, exogenous_data=exo_data)
        except Exception as e:
            msg = 'Forward pass failed on chunk with shape {}.'.format(
                data_chunk.shape)
            logger.exception(msg)
            raise RuntimeError(msg) from e

        if len(hi_res.shape) == 4:
            hi_res = np.expand_dims(np.transpose(hi_res, (1, 2, 0, 3)), axis=0)

        if (s_enhance is not None
                and hi_res.shape[1] != s_enhance * data_chunk.shape[i_lr_s]):
            msg = ('The stated spatial enhancement of {}x did not match '
                   'the low res / high res shapes of {} -> {}'.format(
                       s_enhance, data_chunk.shape, hi_res.shape))
            logger.error(msg)
            raise RuntimeError(msg)

        if (t_enhance is not None
                and hi_res.shape[3] != t_enhance * data_chunk.shape[i_lr_t]):
            msg = ('The stated temporal enhancement of {}x did not match '
                   'the low res / high res shapes of {} -> {}'.format(
                       t_enhance, data_chunk.shape, hi_res.shape))
            logger.error(msg)
            raise RuntimeError(msg)

        return hi_res[0][hr_crop_slices]

    @staticmethod
    def _reshape_data_chunk(model, data_chunk, exo_data):
        """Reshape and transpose data chunk and exogenous data before being
        passed to the sup3r model.

        Notes
        -----
        Exo data needs to be different shapes for 5D (Spatiotemporal) /
        4D (Spatial / Surface) models, and different models use different
        indices for spatial and temporal dimensions. These differences are
        handled here.

        Parameters
        ----------
        model : Sup3rGan
            Sup3rGan or similar sup3r model
        data_chunk : np.ndarray
            Low resolution data for a single spatiotemporal chunk that is going
            to be passed to the model generate function.
        exo_data : dict | None
            Dictionary of exogenous feature data with entries describing
            whether features should be combined at input, a mid network layer,
            or with output. e.g.
            {'topography': {'steps': [
                {'combine_type': 'input', 'model': 0, 'data': ...,
                 'resolution': ...},
                {'combine_type': 'layer', 'model': 0, 'data': ...,
                 'resolution': ...}]}}

        Returns
        -------
        data_chunk : np.ndarray
            Same as input but reshaped to (temporal, spatial_1, spatial_2,
            features) if the model is a spatial-first model or
            (n_obs, spatial_1, spatial_2, temporal, features) if the
            model is spatiotemporal
        exo_data : dict | None
            Same reshaping procedure as for data_chunk applied to
            exo_data[feature]['steps'][...]['data']
        i_lr_t : int
            Axis index for the low-resolution temporal dimension
        i_lr_s : int
            Axis index for the low-resolution spatial_1 dimension
        """
        if exo_data is not None:
            for feature in exo_data:
                for i, entry in enumerate(exo_data[feature]['steps']):
                    models = getattr(model, 'models', [model])
                    msg = (f'model index ({entry["model"]}) for exo step {i} '
                           'exceeds the number of model steps')
                    assert entry['model'] < len(models), msg
                    current_model = models[entry['model']]
                    if current_model.is_4d:
                        out = np.transpose(entry['data'], axes=(2, 0, 1, 3))
                    else:
                        out = np.expand_dims(entry['data'], axis=0)
                    exo_data[feature]['steps'][i]['data'] = out

        if model.is_4d:
            i_lr_t = 0
            i_lr_s = 1
            data_chunk = np.transpose(data_chunk, axes=(2, 0, 1, 3))
        else:
            i_lr_t = 3
            i_lr_s = 1
            data_chunk = np.expand_dims(data_chunk, axis=0)

        return data_chunk, exo_data, i_lr_t, i_lr_s

    @classmethod
    def get_node_cmd(cls, config):
        """Get a CLI call to initialize ForwardPassStrategy and run ForwardPass
        on a single node based on an input config.

        Parameters
        ----------
        config : dict
            sup3r forward pass config with all necessary args and kwargs to
            initialize ForwardPassStrategy and run ForwardPass on a single
            node.
        """
        use_cpu = config.get('use_cpu', True)
        import_str = ''
        if use_cpu:
            import_str += 'import os;\n'
            import_str += 'os.environ["CUDA_VISIBLE_DEVICES"] = "-1";\n'
        import_str += 'import time;\n'
        import_str += 'from gaps import Status;\n'
        import_str += 'from rex import init_logger;\n'
        import_str += ('from sup3r.pipeline.forward_pass '
                       f'import ForwardPassStrategy, {cls.__name__};\n')

        fwps_init_str = get_fun_call_str(ForwardPassStrategy, config)

        node_index = config['node_index']
        log_file = config.get('log_file', None)
        log_level = config.get('log_level', 'INFO')
        log_arg_str = f'"sup3r", log_level="{log_level}"'
        if log_file is not None:
            log_arg_str += f', log_file="{log_file}"'

        cmd = (f"python -c \'{import_str}\n"
               "t0 = time.time();\n"
               f"logger = init_logger({log_arg_str});\n"
               f"strategy = {fwps_init_str};\n"
               f"{cls.__name__}.run(strategy, {node_index});\n"
               "t_elap = time.time() - t0;\n")

        pipeline_step = config.get('pipeline_step') or ModuleName.FORWARD_PASS
        cmd = BaseCLI.add_status_cmd(config, pipeline_step, cmd)
        cmd += ";\'\n"

        return cmd.replace('\\', '/')

    def _constant_output_check(self, out_data):
        """Check if forward pass output is constant. This can happen when the
        chunk going through the forward pass is too big.

        Parameters
        ----------
        out_data : ndarray
            Forward pass output corresponding to the given chunk index
        """

        allowed_const = self.strategy.allowed_const
        if allowed_const is True:
            return
        if allowed_const is False:
            allowed_const = []
        elif not isinstance(allowed_const, (list, tuple)):
            allowed_const = [allowed_const]

        for i, f in enumerate(self.strategy.output_features):
            msg = f'All spatiotemporal values are the same for {f} output!'
            value0 = out_data[0, 0, 0, i]
            all_same = (value0 == out_data[..., i]).all()
            if all_same and value0 not in allowed_const:
                self.strategy.failed_chunks = True
                logger.error(msg)
                raise MemoryError(msg)

    @classmethod
    def _single_proc_run(cls, strategy, node_index, chunk_index):
        """Load forward pass object for given chunk and run through generator,
        this method is meant to be called as a single process in a parallel
        pool.

        Parameters
        ----------
        strategy : ForwardPassStrategy
            ForwardPassStrategy instance with information on data chunks to run
            forward passes on.
        node_index : int
            Index of node on which the forward pass for the given chunk will
            be run.
        chunk_index : int
            Index to select chunk specific variables. This index selects the
            corresponding file set, cropped_file_slice, padded_file_slice,
            and padded/overlapping/cropped spatial slice for a spatiotemporal
            chunk

        Returns
        -------
        ForwardPass | None
            If the forward pass for the given chunk is not finished this
            returns an initialized forward pass object, otherwise returns None
        """
        fwp = None
        check = (not strategy.chunk_finished(chunk_index)
                 and not strategy.failed_chunks)

        if strategy.failed_chunks:
            msg = 'A forward pass has failed. Aborting all jobs.'
            logger.error(msg)
            raise MemoryError(msg)

        if check:
            fwp = cls(strategy, chunk_index=chunk_index, node_index=node_index)
            fwp.run_chunk()

    @classmethod
    def run(cls, strategy, node_index):
        """Runs forward passes on all spatiotemporal chunks for the given node
        index.

        Parameters
        ----------
        strategy : ForwardPassStrategy
            ForwardPassStrategy instance with information on data chunks to run
            forward passes on.
        node_index : int
            Index of node on which the forward passes for spatiotemporal chunks
            will be run.
        """
        if strategy.node_finished(node_index):
            return

        if strategy.pass_workers == 1:
            cls._run_serial(strategy, node_index)
        else:
            cls._run_parallel(strategy, node_index)

    @classmethod
    def _run_serial(cls, strategy, node_index):
        """Runs forward passes, on all spatiotemporal chunks for the given node
        index, in serial.

        Parameters
        ----------
        strategy : ForwardPassStrategy
            ForwardPassStrategy instance with information on data chunks to run
            forward passes on.
        node_index : int
            Index of node on which the forward passes for spatiotemporal chunks
            will be run.
        """

        start = dt.now()
        logger.debug(f'Running forward passes on node {node_index} in '
                     'serial.')
        for i, chunk_index in enumerate(strategy.node_chunks[node_index]):
            now = dt.now()
            cls._single_proc_run(strategy=strategy,
                                 node_index=node_index,
                                 chunk_index=chunk_index,
                                 )
            mem = psutil.virtual_memory()
            logger.info('Finished forward pass on chunk_index='
                        f'{chunk_index} in {dt.now() - now}. {i + 1} of '
                        f'{len(strategy.node_chunks[node_index])} '
                        'complete. Current memory usage is '
                        f'{mem.used / 1e9:.3f} GB out of '
                        f'{mem.total / 1e9:.3f} GB total.')

        logger.info('Finished forward passes on '
                    f'{len(strategy.node_chunks[node_index])} chunks in '
                    f'{dt.now() - start}')

    @classmethod
    def _run_parallel(cls, strategy, node_index):
        """Runs forward passes, on all spatiotemporal chunks for the given node
        index, with data extraction and forward pass routines in parallel.

        Parameters
        ----------
        strategy : ForwardPassStrategy
            ForwardPassStrategy instance with information on data chunks to run
            forward passes on.
        node_index : int
            Index of node on which the forward passes for spatiotemporal chunks
            will be run.
        """

        logger.info(f'Running parallel forward passes on node {node_index}'
                    f' with pass_workers={strategy.pass_workers}.')

        futures = {}
        start = dt.now()
        pool_kws = {"max_workers": strategy.pass_workers, "loggers": ['sup3r']}
        with SpawnProcessPool(**pool_kws) as exe:
            now = dt.now()
            for _i, chunk_index in enumerate(strategy.node_chunks[node_index]):
                fut = exe.submit(cls._single_proc_run,
                                 strategy=strategy,
                                 node_index=node_index,
                                 chunk_index=chunk_index,
                                 )
                futures[fut] = {
                    'chunk_index': chunk_index, 'start_time': dt.now(),
                }

            logger.info(f'Started {len(futures)} forward pass runs in '
                        f'{dt.now() - now}.')

            for i, future in enumerate(as_completed(futures)):
                try:
                    future.result()
                    mem = psutil.virtual_memory()
                    msg = ('Finished forward pass on chunk_index='
                           f'{futures[future]["chunk_index"]} in '
                           f'{dt.now() - futures[future]["start_time"]}. '
                           f'{i + 1} of {len(futures)} complete. '
                           f'Current memory usage is {mem.used / 1e9:.3f} GB '
                           f'out of {mem.total / 1e9:.3f} GB total.')
                    logger.info(msg)
                except Exception as e:
                    msg = ('Error running forward pass on chunk_index='
                           f'{futures[future]["chunk_index"]}.')
                    logger.exception(msg)
                    raise RuntimeError(msg) from e

        logger.info('Finished asynchronous forward passes on '
                    f'{len(strategy.node_chunks[node_index])} chunks in '
                    f'{dt.now() - start}')

    def run_chunk(self):
        """Run a forward pass on single spatiotemporal chunk."""

        msg = (f'Running forward pass for chunk_index={self.chunk_index}, '
               f'node_index={self.node_index}, file_paths={self.file_paths}. '
               f'Starting forward pass on chunk_shape={self.chunk_shape} with '
               f'spatial_pad={self.strategy.spatial_pad} and temporal_pad='
               f'{self.strategy.temporal_pad}.')
        logger.info(msg)

        self.output_data = self._run_generator(
            self.input_data,
            hr_crop_slices=self.hr_crop_slice,
            model=self.model,
            model_kwargs=self.model_kwargs,
            model_class=self.model_class,
            s_enhance=self.s_enhance,
            t_enhance=self.t_enhance,
            exo_data=self.exogenous_data,
        )

        self._constant_output_check(self.output_data)

        if self.out_file is not None:
            logger.info(f'Saving forward pass output to {self.out_file}.')
            self.output_handler_class._write_output(
                data=self.output_data,
                features=self.model.hr_out_features,
                lat_lon=self.hr_lat_lon,
                times=self.hr_times,
                out_file=self.out_file,
                meta_data=self.meta,
                max_workers=self.output_workers,
                gids=self.gids,
            )
        return self.output_data
