""":class:`ForwardPassStrategy` class. This sets up chunks and needed generator
inputs to distribute forward passes."""

import copy
import logging
import os
import pathlib
import pprint
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sup3r.bias.utilities import bias_correct_features
from sup3r.pipeline.slicer import ForwardPassSlicer
from sup3r.pipeline.utilities import get_model
from sup3r.postprocessing import OutputHandler
from sup3r.preprocessing import ExoData, ExoDataHandler
from sup3r.preprocessing.utilities import (
    expand_paths,
    get_class_kwargs,
    get_input_handler_class,
    log_args,
)
from sup3r.typing import T_Array

logger = logging.getLogger(__name__)


@dataclass
class ForwardPassChunk:
    """Structure storing chunk data and attributes for a specific chunk going
    through the generator."""

    input_data: T_Array
    exo_data: Dict
    hr_crop_slice: slice
    lr_pad_slice: slice
    hr_lat_lon: T_Array
    hr_times: pd.DatetimeIndex
    gids: T_Array
    out_file: str
    pad_width: Tuple[tuple, tuple, tuple]
    index: int

    def __post_init__(self):
        self.shape = self.input_data.shape


@dataclass
class ForwardPassStrategy:
    """Class to prepare data for forward passes through generator.

    A full file list of contiguous times is provided. The corresponding data is
    split into spatiotemporal chunks which can overlap in time and space. These
    chunks are distributed across nodes according to the max nodes input or
    number of temporal chunks. This strategy stores information on these
    chunks, how they overlap, how they are distributed to nodes, and how to
    crop generator output to stich the chunks back together.

    Use the following inputs to initialize data handlers on different nodes and
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
        Max shape (spatial_1, spatial_2, temporal) of an unpadded coarse chunk
        to use for a forward pass. The number of nodes that the
        :class:`ForwardPassStrategy` is set to distribute to is calculated by
        dividing up the total time index from all file_paths by the temporal
        part of this chunk shape. Each node will then be parallelized across
        parallel processes by the spatial chunk shape.  If temporal_pad /
        spatial_pad are non zero the chunk sent to the generator can be bigger
        than this shape. If running in serial set this equal to the shape of
        the full spatiotemporal data volume for best performance.
    spatial_pad : int
        Size of spatial overlap between coarse chunks passed to forward passes
        for subsequent spatial stitching. This overlap will pad both sides of
        the fwp_chunk_shape.
    temporal_pad : int
        Size of temporal overlap between coarse chunks passed to forward passes
        for subsequent temporal stitching. This overlap will pad both sides of
        the fwp_chunk_shape.
    model_class : str
        Name of the sup3r model class for the GAN model to load. The default is
        the basic spatial / spatiotemporal Sup3rGan model. This will be loaded
        from sup3r.models
    out_pattern : str
        Output file pattern. Must include {file_id} format key.  Each output
        file will have a unique file_id filled in and the ext determines the
        output type. If pattern is None then data will be returned in an array
        and not saved.
    input_handler_name : str | None
        Class to use for input data. Provide a string name to match an
        extracter or handler class in `sup3r.preprocessing`
    input_handler_kwargs : dict | None
        Any kwargs for initializing the `input_handler_name` class.
    exo_handler_kwargs : dict | None
        Dictionary of args to pass to :class:`ExoDataHandler` for extracting
        exogenous features for multistep foward pass. This should be a nested
        dictionary with keys for each exogenous feature. The dictionaries
        corresponding to the feature names should include the path to exogenous
        data source, the resolution of the exogenous data, and how the
        exogenous data should be used in the model. e.g. {'topography':
        {'file_paths': 'path to input files', 'source_file': 'path to exo
        data', 'steps': [..]}.
    bias_correct_method : str | None
        Optional bias correction function name that can be imported from the
        :mod:`sup3r.bias.bias_transforms` module. This will transform the
        source data according to some predefined bias correction transformation
        along with the bias_correct_kwargs. As the first argument, this method
        must receive a generic numpy array of data to be bias corrected
    bias_correct_kwargs : dict | None
        Optional namespace of kwargs to provide to bias_correct_method.  If
        this is provided, it must be a dictionary where each key is a feature
        name and each value is a dictionary of kwargs to correct that feature.
        You can bias correct only certain input features by only including
        those feature names in this dict.
    allowed_const : list | bool
        Tensorflow has a tensor memory limit of 2GB (result of protobuf
        limitation) and when exceeded can return a tensor with a constant
        output. sup3r will raise a ``MemoryError`` in response. If your model
        is allowed to output a constant output, set this to True to allow any
        constant output or a list of allowed possible constant outputs. For
        example, a precipitation model should be allowed to output all zeros so
        set this to ``[0]``. For details on this limit:
        https://github.com/tensorflow/tensorflow/issues/51870
    incremental : bool
        Allow the forward pass iteration to skip spatiotemporal chunks that
        already have an output file (default = True) or iterate through all
        chunks and overwrite any pre-existing outputs (False).
    output_workers : int | None
        Max number of workers to use for writing forward pass output.
    pass_workers : int | None
        Max number of workers to use for performing forward passes on a single
        node. If 1 then all forward passes on chunks distributed to a single
        node will be run serially. pass_workers=2 is the minimum number of
        workers required to run the ForwardPass initialization and
        :meth:`ForwardPass.run_chunk()` methods concurrently.
    max_nodes : int | None
        Maximum number of nodes to distribute spatiotemporal chunks across. If
        None then a node will be used for each temporal chunk.
    head_node : bool
        Whether initialization is taking place on the head node of a multi node
        job launch. When this is true :class:`ForwardPassStrategy` is only
        partially initialized to provide the head node enough information for
        how to distribute jobs across nodes. Preflight tasks like bias
        correction will be skipped because they will be performed on the nodes
        jobs are distributed to by the head node.
    """

    file_paths: Union[str, list, pathlib.Path]
    model_kwargs: dict
    fwp_chunk_shape: tuple = (None, None, None)
    spatial_pad: int = 0
    temporal_pad: int = 0
    model_class: str = 'Sup3rGan'
    out_pattern: Optional[str] = None
    input_handler_name: Optional[str] = None
    input_handler_kwargs: Optional[dict] = None
    exo_handler_kwargs: Optional[dict] = None
    bias_correct_method: Optional[str] = None
    bias_correct_kwargs: Optional[dict] = None
    allowed_const: Optional[Union[list, bool]] = None
    incremental: bool = True
    output_workers: int = 1
    pass_workers: int = 1
    max_nodes: int = 1
    head_node: bool = False

    @log_args
    def __post_init__(self):
        self.file_paths = expand_paths(self.file_paths)
        self.bias_correct_kwargs = self.bias_correct_kwargs or {}

        model = get_model(self.model_class, self.model_kwargs)
        self.s_enhance, self.t_enhance = model.s_enhance, model.t_enhance
        self.input_features = model.lr_features
        self.output_features = model.hr_out_features
        self.features, self.exo_features = self._init_features(model)
        self.input_handler, self.time_slice = self.init_input_handler()
        self.fwp_chunk_shape = self._get_fwp_chunk_shape()

        self.fwp_slicer = ForwardPassSlicer(
            coarse_shape=self.input_handler.grid_shape,
            time_steps=len(self.input_handler.time_index),
            time_slice=self.time_slice,
            chunk_shape=self.fwp_chunk_shape,
            s_enhance=self.s_enhance,
            t_enhance=self.t_enhance,
            spatial_pad=self.spatial_pad,
            temporal_pad=self.temporal_pad,
        )
        self.node_chunks = self._get_node_chunks()

        if not self.head_node:
            self.out_files = self.get_out_files(out_files=self.out_pattern)
            self.hr_lat_lon = self.get_hr_lat_lon()
            hr_shape = self.hr_lat_lon.shape[:-1]
            self.gids = np.arange(np.prod(hr_shape)).reshape(hr_shape)
            self.exo_data = self.load_exo_data(model)
            self.preflight()

    @property
    def meta(self):
        """Meta data dictionary for the strategy. Used to add info to forward
        pass output meta."""
        meta_data = {
            'fwp_chunk_shape': self.fwp_chunk_shape,
            'spatial_pad': self.spatial_pad,
            'temporal_pad': self.temporal_pad,
            'model_kwargs': self.model_kwargs,
            'model_class': self.model_class,
            'spatial_enhance': int(self.s_enhance),
            'temporal_enhance': int(self.t_enhance),
            'input_files': self.file_paths,
            'input_features': self.features,
            'output_features': self.output_features,
        }
        return meta_data

    def init_input_handler(self):
        """Get input handler instance for given input kwargs. If self.head_node
        is False we get all requested features. Otherwise this is part of
        initialization on a head node and just used to get the shape of the
        input domain, so we don't need to get any features yet."""
        self.input_handler_kwargs = self.input_handler_kwargs or {}
        self.input_handler_kwargs['file_paths'] = self.file_paths
        self.input_handler_kwargs['features'] = self.features
        time_slice = self.input_handler_kwargs.get('time_slice', slice(None))

        InputHandler = get_input_handler_class(self.input_handler_name)
        input_handler_kwargs = copy.deepcopy(self.input_handler_kwargs)
        features = [] if self.head_node else self.features
        input_handler_kwargs['features'] = features
        input_handler_kwargs['time_slice'] = slice(None)

        return InputHandler(**input_handler_kwargs), time_slice

    def _init_features(self, model):
        """Initialize feature attributes."""
        self.exo_handler_kwargs = self.exo_handler_kwargs or {}
        exo_features = list(self.exo_handler_kwargs)
        features = [f for f in model.lr_features if f not in exo_features]
        return features, exo_features

    def _get_node_chunks(self):
        """Get array of lists such that node_chunks[i] is a list of
        indices for the ith node indexing the chunks that will be sent through
        the generator on the ith node."""
        n_fwp_chunks = self.fwp_slicer.n_chunks
        node_chunks = min(self.max_nodes or np.inf, n_fwp_chunks)
        return np.array_split(np.arange(n_fwp_chunks), node_chunks)

    def _get_fwp_chunk_shape(self):
        """Get fwp_chunk_shape with default shape equal to the input handler
        shape"""
        grid_shape = self.input_handler.grid_shape
        tsteps = len(self.input_handler.time_index[self.time_slice])
        shape_iter = zip(self.fwp_chunk_shape, (*grid_shape, tsteps))
        return tuple(fs or ffs for fs, ffs in shape_iter)

    def preflight(self):
        """Prelight logging and sanity checks"""

        log_dict = {
            'n_nodes': len(self.node_chunks),
            'n_spatial_chunks': self.fwp_slicer.n_spatial_chunks,
            'n_time_chunks': self.fwp_slicer.n_time_chunks,
            'n_total_chunks': self.fwp_slicer.n_chunks,
        }
        logger.info(
            f'Chunk strategy description:\n'
            f'{pprint.pformat(log_dict, indent=2)}'
        )

        out = self.fwp_slicer.get_time_slices()
        self.ti_slices, self.ti_pad_slices = out

        fwp_tsteps = self.fwp_chunk_shape[2] + 2 * self.temporal_pad
        tsteps = len(self.input_handler.time_index[self.time_slice])
        msg = (
            f'Using a padded chunk size ({fwp_tsteps}) larger than the full '
            f'temporal domain ({tsteps}). Should just run without temporal '
            'chunking. '
        )
        if fwp_tsteps > tsteps:
            logger.warning(msg)
            warnings.warn(msg)
        out = self.fwp_slicer.get_spatial_slices()
        self.lr_slices, self.lr_pad_slices, self.hr_slices = out

        if self.bias_correct_kwargs is not None:
            padded_tslice = slice(
                self.ti_pad_slices[0].start, self.ti_pad_slices[-1].stop
            )
            self.input_handler = bias_correct_features(
                features=list(self.bias_correct_kwargs),
                input_handler=self.input_handler,
                time_slice=padded_tslice,
                bc_method=self.bias_correct_method,
                bc_kwargs=self.bias_correct_kwargs,
            )

    def get_chunk_indices(self, chunk_index):
        """Get (spatial, temporal) indices for the given chunk index"""
        return (
            chunk_index % self.fwp_slicer.n_spatial_chunks,
            chunk_index // self.fwp_slicer.n_spatial_chunks,
        )

    def get_hr_lat_lon(self):
        """Get high resolution lat lons"""
        lr_lat_lon = self.input_handler.lat_lon
        shape = tuple(d * self.s_enhance for d in lr_lat_lon.shape[:-1])
        logger.info(
            f'Getting high-resolution grid for full output domain: {shape}'
        )
        return OutputHandler.get_lat_lon(lr_lat_lon, shape)

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
        file_ids = [
            f'{str(i).zfill(6)}_{str(j).zfill(6)}'
            for i in range(self.fwp_slicer.n_time_chunks)
            for j in range(self.fwp_slicer.n_spatial_chunks)
        ]
        out_file_list = [None] * len(file_ids)
        if out_files is not None:
            msg = 'out_pattern must include a {file_id} format key'
            assert '{file_id}' in out_files, msg
            os.makedirs(os.path.dirname(out_files), exist_ok=True)
            out_file_list = [
                out_files.format(file_id=file_id) for file_id in file_ids
            ]
        return out_file_list

    @staticmethod
    def _get_pad_width(window, max_steps, max_pad):
        """
        Parameters
        ----------
        window : slice
            Slice with start and stop of window to pad.
        max_steps : int
            Maximum number of steps available. Padding cannot extend past this
        max_pad : int
            Maximum amount of padding to apply.

        Returns
        -------
        tuple
            Tuple of pad width for the given window.
        """
        start = window.start or 0
        stop = window.stop or max_steps
        start = int(np.maximum(0, (max_pad - start)))
        stop = int(np.maximum(0, max_pad + stop - max_steps))
        return (start, stop)

    def get_pad_width(self, chunk_index):
        """Get padding for the current spatiotemporal chunk

        Returns
        -------
        padding : tuple
            Tuple of tuples with padding width for spatial and temporal
            dimensions. Each tuple includes the start and end of padding for
            that dimension. Ordering is spatial_1, spatial_2, temporal.
        """
        s_chunk_idx, t_chunk_idx = self.get_chunk_indices(chunk_index)
        ti_slice = self.ti_slices[t_chunk_idx]
        lr_slice = self.lr_slices[s_chunk_idx]

        return (
            self._get_pad_width(
                lr_slice[0], self.input_handler.grid_shape[0], self.spatial_pad
            ),
            self._get_pad_width(
                lr_slice[1], self.input_handler.grid_shape[1], self.spatial_pad
            ),
            self._get_pad_width(
                ti_slice, len(self.input_handler.time_index), self.temporal_pad
            ),
        )

    def init_chunk(self, chunk_index=0):
        """Get :class:`FowardPassChunk` instance for the given chunk index.

        This selects the appropriate data from `self.input_handler` and
        `self.exo_data` and returns a structure object (`ForwardPassChunk`)
        with that data and other chunk specific attributes.
        """

        s_chunk_idx, t_chunk_idx = self.get_chunk_indices(chunk_index)

        args_dict = {
            'chunk': chunk_index,
            'temporal_chunk': t_chunk_idx,
            'spatial_chunk': s_chunk_idx,
            'n_node_chunks': self.fwp_slicer.n_chunks,
            'fwp_chunk_shape': self.fwp_chunk_shape,
            'temporal_pad': self.temporal_pad,
            'spatial_pad': self.spatial_pad,
        }
        logger.info(
            'Initializing ForwardPassChunk with: '
            f'{pprint.pformat(args_dict, indent=2)}'
        )

        msg = (
            f'Requested forward pass on chunk_index={chunk_index} > '
            f'n_chunks={self.fwp_slicer.n_chunks}'
        )
        assert chunk_index <= self.fwp_slicer.n_chunks, msg

        hr_slice = self.hr_slices[s_chunk_idx]
        ti_slice = self.ti_slices[t_chunk_idx]
        lr_times = self.input_handler.time_index[ti_slice]
        lr_pad_slice = self.lr_pad_slices[s_chunk_idx]
        ti_pad_slice = self.ti_pad_slices[t_chunk_idx]

        logger.info(f'Getting input data for chunk_index={chunk_index}.')

        exo_data = (
            self.exo_data.get_chunk(
                self.input_handler.shape,
                [lr_pad_slice[0], lr_pad_slice[1], ti_pad_slice],
            )
            if self.exo_data is not None
            else None
        )

        return ForwardPassChunk(
            input_data=self.input_handler.data[
                lr_pad_slice[0], lr_pad_slice[1], ti_pad_slice
            ],
            exo_data=exo_data,
            lr_pad_slice=lr_pad_slice,
            hr_crop_slice=self.fwp_slicer.hr_crop_slices[t_chunk_idx][
                s_chunk_idx
            ],
            hr_lat_lon=self.hr_lat_lon[hr_slice[:2]],
            hr_times=OutputHandler.get_times(
                lr_times, self.t_enhance * len(lr_times)
            ),
            gids=self.gids[hr_slice[:2]],
            out_file=self.out_files[chunk_index],
            pad_width=self.get_pad_width(chunk_index),
            index=chunk_index,
        )

    def load_exo_data(self, model):
        """Extract exogenous data for each exo feature and store data in
        dictionary with key for each exo feature

        Returns
        -------
        exo_data : ExoData
           :class:`ExoData` object composed of multiple
           :class:`SingleExoDataStep` objects. This is the exo data for the
           full spatiotemporal extent.
        """
        data = {}
        exo_data = None
        if self.exo_handler_kwargs:
            for feature in self.exo_features:
                exo_kwargs = copy.deepcopy(self.exo_handler_kwargs[feature])
                exo_kwargs['feature'] = feature
                exo_kwargs['models'] = getattr(model, 'models', [model])
                input_handler_kwargs = exo_kwargs.get(
                    'input_handler_kwargs', {}
                )
                input_handler_kwargs['target'] = self.input_handler.target
                input_handler_kwargs['shape'] = self.input_handler.grid_shape
                exo_kwargs['input_handler_kwargs'] = input_handler_kwargs
                exo_kwargs = get_class_kwargs(ExoDataHandler, exo_kwargs)
                data.update(ExoDataHandler(**exo_kwargs).data)
            exo_data = ExoData(data)
        return exo_data

    def node_finished(self, node_idx):
        """Check if all out files for a given node have been saved"""
        return all(self.chunk_finished(i) for i in self.node_chunks[node_idx])

    def chunk_finished(self, chunk_idx):
        """Check if process for given chunk_index has already been run.
        Considered finished if there is already an output file and incremental
        is False."""

        out_file = self.out_files[chunk_idx]
        check = os.path.exists(out_file) and self.incremental
        if check:
            logger.info(
                f'{out_file} already exists and incremental = True. '
                f'Skipping forward pass for chunk index {chunk_idx}.'
            )
        return check
