# -*- coding: utf-8 -*-
"""
Sup3r forward pass handling module.

@author: bbenton
"""

import copy
import logging
import os
import warnings

import numpy as np

from sup3r.pipeline.common import get_model
from sup3r.pipeline.slicer import ForwardPassSlicer
from sup3r.postprocessing import (
    OutputHandler,
)
from sup3r.utilities.execution import DistributedProcess
from sup3r.utilities.utilities import (
    get_input_handler_class,
    get_source_type,
)

logger = logging.getLogger(__name__)


class ForwardPassStrategy(DistributedProcess):
    """Class to prepare data for forward passes through generator.

    A full file list of contiguous times is provided. The corresponding data is
    split into spatiotemporal chunks which can overlap in time and space. These
    chunks are distributed across nodes according to the max nodes input or
    number of temporal chunks. This strategy stores information on these
    chunks, how they overlap, how they are distributed to nodes, and how to
    crop generator output to stich the chunks back togerther.
    """

    def __init__(
        self,
        file_paths,
        model_kwargs,
        fwp_chunk_shape,
        spatial_pad,
        temporal_pad,
        model_class='Sup3rGan',
        out_pattern=None,
        input_handler=None,
        input_handler_kwargs=None,
        incremental=True,
        exo_kwargs=None,
        bias_correct_method=None,
        bias_correct_kwargs=None,
        max_nodes=None,
        allowed_const=False,
        output_workers=None,
        pass_workers=None,
    ):
        """Use these inputs to initialize data handlers on different nodes and
        to define the size of the data chunks that will be passed through the
        generator.

        Parameters
        ----------
        file_paths : list | str
            A list of low-resolution source files to extract raster data from.
            Each file must have the same number of timesteps. Can also pass a
            string with a unix-style file path which will be passed through
            glob.glob
        model_kwargs : str | list
            Keyword arguments to send to `model_class.load(**model_kwargs)` to
            initialize the GAN. Typically this is just the string path to the
            model directory, but can be multiple models or arguments for more
            complex models.
        fwp_chunk_shape : tuple
            Max shape (spatial_1, spatial_2, temporal) of an unpadded coarse
            chunk to use for a forward pass. The number of nodes that the
            ForwardPassStrategy is set to distribute to is calculated by
            dividing up the total time index from all file_paths by the
            temporal part of this chunk shape. Each node will then be
            parallelized accross parallel processes by the spatial chunk shape.
            If temporal_pad / spatial_pad are non zero the chunk sent
            to the generator can be bigger than this shape. If running in
            serial set this equal to the shape of the full spatiotemporal data
            volume for best performance.
        spatial_pad : int
            Size of spatial overlap between coarse chunks passed to forward
            passes for subsequent spatial stitching. This overlap will pad both
            sides of the fwp_chunk_shape. Note that the first and last chunks
            in any of the spatial dimension will not be padded.
        temporal_pad : int
            Size of temporal overlap between coarse chunks passed to forward
            passes for subsequent temporal stitching. This overlap will pad
            both sides of the fwp_chunk_shape. Note that the first and last
            chunks in the temporal dimension will not be padded.
        model_class : str
            Name of the sup3r model class for the GAN model to load. The
            default is the basic spatial / spatiotemporal Sup3rGan model. This
            will be loaded from sup3r.models
        out_pattern : str
            Output file pattern. Must be of form <path>/<name>_{file_id}.<ext>.
            e.g. /tmp/sup3r_job_{file_id}.h5
            Each output file will have a unique file_id filled in and the ext
            determines the output type. Pattern can also include {times}. This
            will be replaced with start_time-end_time. If pattern is None then
            data will be returned in an array and not saved.
        input_handler : str | None
            Class to use for input data. Provide a string name to match an
            extracter or handler class in `sup3r.containers`
        input_handler_kwargs : dict | None
            Any kwargs for initializing the `input_handler` class.
        incremental : bool
            Allow the forward pass iteration to skip spatiotemporal chunks that
            already have an output file (True, default) or iterate through all
            chunks and overwrite any pre-existing outputs (False).
        exo_kwargs : dict | None
            Dictionary of args to pass to :class:`ExogenousDataHandler` for
            extracting exogenous features for multistep foward pass. This
            should be a nested dictionary with keys for each exogeneous
            feature. The dictionaries corresponding to the feature names
            should include the path to exogenous data source, the resolution
            of the exogenous data, and how the exogenous data should be used
            in the model. e.g. {'topography': {'file_paths': 'path to input
            files', 'source_file': 'path to exo data', 'exo_resolution':
            {'spatial': '1km', 'temporal': None}, 'steps': [..]}.
        bias_correct_method : str | None
            Optional bias correction function name that can be imported from
            the :mod:`sup3r.bias.bias_transforms` module. This will transform
            the source data according to some predefined bias correction
            transformation along with the bias_correct_kwargs. As the first
            argument, this method must receive a generic numpy array of data to
            be bias corrected
        bias_correct_kwargs : dict | None
            Optional namespace of kwargs to provide to bias_correct_method.
            If this is provided, it must be a dictionary where each key is a
            feature name and each value is a dictionary of kwargs to correct
            that feature. You can bias correct only certain input features by
            only including those feature names in this dict.
        max_nodes : int | None
            Maximum number of nodes to distribute spatiotemporal chunks across.
            If None then a node will be used for each temporal chunk.
        allowed_const : list | bool
            Tensorflow has a tensor memory limit of 2GB (result of protobuf
            limitation) and when exceeded can return a tensor with a
            constant output. sup3r will raise a ``MemoryError`` in response. If
            your model is allowed to output a constant output, set this to True
            to allow any constant output or a list of allowed possible constant
            outputs. For example, a precipitation model should be allowed to
            output all zeros so set this to ``[0]``. For details on this limit:
            https://github.com/tensorflow/tensorflow/issues/51870
        output_workers : int | None
            Max number of workers to use for writing forward pass output.
        pass_workers : int | None
            Max number of workers to use for performing forward passes on a
            single node. If 1 then all forward passes on chunks distributed to
            a single node will be run in serial. pass_workers=2 is the minimum
            number of workers required to run the ForwardPass initialization
            and ForwardPass.run_chunk() methods concurrently.
        """
        self.input_handler_kwargs = input_handler_kwargs or {}
        self.file_paths = file_paths
        self.model_kwargs = model_kwargs
        self.fwp_chunk_shape = fwp_chunk_shape
        self.spatial_pad = spatial_pad
        self.temporal_pad = temporal_pad
        self.model_class = model_class
        self.out_pattern = out_pattern
        self.exo_kwargs = exo_kwargs or {}
        self.exo_features = (
            [] if not self.exo_kwargs else list(self.exo_kwargs)
        )
        self.incremental = incremental
        self.bias_correct_method = bias_correct_method
        self.bias_correct_kwargs = bias_correct_kwargs or {}
        self.allowed_const = allowed_const
        self.input_type = get_source_type(self.file_paths)
        self.output_type = get_source_type(self.out_pattern)
        self.output_workers = output_workers
        self.pass_workers = pass_workers
        model = get_model(model_class, model_kwargs)
        models = getattr(model, 'models', [model])
        self.s_enhancements = [model.s_enhance for model in models]
        self.t_enhancements = [model.t_enhance for model in models]
        self.s_enhance = np.prod(self.s_enhancements)
        self.t_enhance = np.prod(self.t_enhancements)
        self.input_features = model.lr_features
        self.output_features = model.hr_out_features
        assert len(self.input_features) > 0, 'No input features!'
        assert len(self.output_features) > 0, 'No output features!'
        self.features = [
            f for f in self.input_features if f not in self.exo_features
        ]
        self.input_handler_kwargs.update(
            {'file_paths': self.file_paths, 'features': self.features}
        )
        input_kwargs = copy.deepcopy(self.input_handler_kwargs)
        input_kwargs['features'] = []
        self.input_handler_class = get_input_handler_class(
            file_paths, input_handler
        )
        input_handler = self.input_handler_class(**input_kwargs)
        self.lr_lat_lon = input_handler.lat_lon
        self.time_index = input_handler.time_index
        self.hr_lat_lon = self.get_hr_lat_lon()
        self.raw_tsteps = self.get_raw_tsteps()
        self.gids = np.arange(np.prod(self.hr_lat_lon.shape[:-1]))
        self.gids = self.gids.reshape(self.hr_lat_lon.shape[:-1])
        self.grid_shape = self.lr_lat_lon.shape[:-1]

        self.fwp_slicer = ForwardPassSlicer(
            input_handler.lat_lon.shape[:-1],
            self.raw_tsteps,
            input_handler.time_slice,
            self.fwp_chunk_shape,
            self.s_enhancements,
            self.t_enhancements,
            self.spatial_pad,
            self.temporal_pad,
        )
        DistributedProcess.__init__(
            self,
            max_nodes=max_nodes,
            max_chunks=self.fwp_slicer.n_chunks,
            incremental=self.incremental,
        )
        self.out_files = self.get_out_files(out_files=self.out_pattern)
        self.preflight()

    def preflight(self):
        """Prelight path name formatting and sanity checks"""

        logger.info(
            'Initializing ForwardPassStrategy. '
            f'Using n_nodes={self.nodes} with '
            f'n_spatial_chunks={self.fwp_slicer.n_spatial_chunks}, '
            f'n_temporal_chunks={self.fwp_slicer.n_temporal_chunks}, '
            f'and n_total_chunks={self.chunks}. '
            f'{self.chunks / self.nodes:.3f} chunks per node on '
            'average.'
        )
        logger.info(
            f'pass_workers={self.pass_workers}, '
            f'output_workers={self.output_workers}'
        )

        out = self.fwp_slicer.get_time_slices()
        self.ti_slices, self.ti_pad_slices = out

        msg = (
            'Using a padded chunk size '
            f'({self.fwp_chunk_shape[2] + 2 * self.temporal_pad}) '
            f'larger than the full temporal domain ({self.raw_tsteps}). '
            'Should just run without temporal chunking. '
        )
        if self.fwp_chunk_shape[2] + 2 * self.temporal_pad >= self.raw_tsteps:
            logger.warning(msg)
            warnings.warn(msg)
        out = self.fwp_slicer.get_spatial_slices()
        self.lr_slices, self.lr_pad_slices, self.hr_slices = out

    def _get_spatial_chunk_index(self, chunk_index):
        """Get the spatial index for the given chunk index"""
        return chunk_index % self.fwp_slicer.n_spatial_chunks

    def _get_temporal_chunk_index(self, chunk_index):
        """Get the temporal index for the given chunk index"""
        return chunk_index // self.fwp_slicer.n_spatial_chunks

    def get_raw_tsteps(self):
        """Get number of time steps available in the raw data, which is useful
        for padding the time domain."""
        kwargs = copy.deepcopy(self.input_handler_kwargs)
        _ = kwargs.pop('time_slice', None)
        return len(self.input_handler_class(**kwargs).time_index)

    def get_hr_lat_lon(self):
        """Get high resolution lat lons"""
        logger.info('Getting high-resolution grid for full output domain.')
        lr_lat_lon = self.lr_lat_lon.copy()
        shape = tuple([d * self.s_enhance for d in lr_lat_lon.shape[:-1]])
        return OutputHandler.get_lat_lon(lr_lat_lon, shape)

    def get_file_ids(self):
        """Get file id for each output file

        Returns
        -------
        file_ids : list
            List of file ids for each output file. Will be used to name output
            files of the form filename_{file_id}.ext
        """
        file_ids = []
        for i in range(self.fwp_slicer.n_temporal_chunks):
            for j in range(self.fwp_slicer.n_spatial_chunks):
                file_id = f'{str(i).zfill(6)}_{str(j).zfill(6)}'
                file_ids.append(file_id)
        return file_ids

    @property
    def max_nodes(self):
        """Get the maximum number of nodes that this strategy should distribute
        work to, equal to either the specified max number of nodes or total
        number of temporal chunks"""
        self._max_nodes = (
            self._max_nodes
            if self._max_nodes is not None
            else self.fwp_slicer.n_temporal_chunks
        )
        return self._max_nodes

    def get_out_files(self, out_files):
        """Get output file names for each file chunk forward pass

        Parameters
        ----------
        out_files : str
            Output file pattern. Needs to include a {file_id} format key.
            Each output file will have a unique file_id filled in and the
            extension determines the output type.

        Returns
        -------
        list
            List of output file paths
        """
        file_ids = self.get_file_ids()
        out_file_list = []
        if out_files is not None:
            if '{times}' in out_files:
                out_files = out_files.replace('{times}', '{file_id}')
            if '{file_id}' not in out_files:
                out_files = out_files.split('.')
                tmp = '.'.join(out_files[:-1]) + '_{file_id}'
                tmp += '.' + out_files[-1]
                out_files = tmp
            dirname = os.path.dirname(out_files)
            if not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
            for file_id in file_ids:
                out_file = out_files.replace('{file_id}', file_id)
                out_file_list.append(out_file)
        else:
            out_file_list = [None] * len(file_ids)
        return out_file_list

    def get_chunk_description(self, chunk_index):
        """Get the target, shape, and set of slices for the current chunk."""

        s_chunk_idx = self._get_spatial_chunk_index(chunk_index)
        t_chunk_idx = self._get_temporal_chunk_index(chunk_index)
        lr_pad_slice = self.lr_pad_slices[s_chunk_idx]
        spatial_slice = lr_pad_slice[0], lr_pad_slice[1]
        ti_pad_slice = self.ti_pad_slices[t_chunk_idx]
        lr_slice = self.lr_slices[s_chunk_idx]
        hr_slice = self.hr_slices[s_chunk_idx]
        chunk_shape = (
            lr_pad_slice[0].stop - lr_pad_slice[0].start,
            lr_pad_slice[1].stop - lr_pad_slice[1].start,
            ti_pad_slice.stop - ti_pad_slice.start,
        )

        chunk_desc = {
            'target': self.lr_lat_lon[spatial_slice][-1, 0],
            'shape': self.lr_lat_lon[spatial_slice].shape[:-1],
            'lr_slice': self.lr_slices[s_chunk_idx],
            'hr_slice': self.hr_slices[s_chunk_idx],
            'lr_pad_slice': self.lr_pad_slices[s_chunk_idx],
            'ti_pad_slice': self.ti_pad_slices[t_chunk_idx],
            'ti_slice': self.ti_slices[t_chunk_idx],
            'ti_crop_slice': self.fwp_slicer.t_lr_crop_slices[t_chunk_idx],
            'lr_crop_slice': self.fwp_slicer.s_lr_crop_slices[s_chunk_idx],
            'hr_crop_slice': self.fwp_slicer.hr_crop_slices[t_chunk_idx][
                s_chunk_idx
            ],
            'lr_lat_lon': self.lr_lat_lon[lr_slice[0], hr_slice[1]],
            'hr_lat_lon': self.hr_lat_lon[hr_slice[0], hr_slice[1]],
            'chunk_shape': chunk_shape,
            'pad_width': self.get_pad_width(
                self.ti_slices[t_chunk_idx], self.lr_slices[s_chunk_idx]
            ),
        }
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
        ti_stop = ti_slice.stop or self.raw_tsteps
        pad_t_start = int(np.maximum(0, (self.temporal_pad - ti_start)))
        pad_t_end = self.temporal_pad + ti_stop - self.raw_tsteps
        pad_t_end = int(np.maximum(0, pad_t_end))

        s1_start = lr_slice[0].start or 0
        s1_stop = lr_slice[0].stop or self.grid_shape[0]
        pad_s1_start = int(np.maximum(0, (self.spatial_pad - s1_start)))
        pad_s1_end = self.spatial_pad + s1_stop - self.grid_shape[0]
        pad_s1_end = int(np.maximum(0, pad_s1_end))

        s2_start = lr_slice[1].start or 0
        s2_stop = lr_slice[1].stop or self.grid_shape[1]
        pad_s2_start = int(np.maximum(0, (self.spatial_pad - s2_start)))
        pad_s2_end = self.spatial_pad + s2_stop - self.grid_shape[1]
        pad_s2_end = int(np.maximum(0, pad_s2_end))
        return (
            (pad_s1_start, pad_s1_end),
            (pad_s2_start, pad_s2_end),
            (pad_t_start, pad_t_end),
        )
