""":class:`ForwardPassStrategy` class. This sets up chunks and needed generator
inputs to distribute forward passes."""

import copy
import logging
import os
import pathlib
import pprint
import warnings
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Optional, Tuple, Union

import dask.array as da
import numpy as np
import pandas as pd

from sup3r.bias.utilities import bias_correct_features
from sup3r.pipeline.slicer import ForwardPassSlicer
from sup3r.pipeline.utilities import get_model
from sup3r.postprocessing import OutputHandler
from sup3r.preprocessing import ExoData, ExoDataHandler
from sup3r.preprocessing.names import Dimension
from sup3r.preprocessing.utilities import (
    _parse_time_slice,
    expand_paths,
    get_class_kwargs,
    get_date_range_kwargs,
    get_input_handler_class,
    log_args,
)
from sup3r.utilities.utilities import Timer

logger = logging.getLogger(__name__)


@dataclass
class ForwardPassChunk:
    """Structure storing chunk data and attributes for a specific chunk going
    through the generator."""

    input_data: Union[np.ndarray, da.core.Array]
    exo_data: Dict
    hr_crop_slice: slice
    lr_pad_slice: slice
    hr_lat_lon: Union[np.ndarray, da.core.Array]
    hr_times: pd.DatetimeIndex
    gids: Union[np.ndarray, da.core.Array]
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
        Keyword arguments to send to ``model_class.load(**model_kwargs)`` to
        initialize the GAN. Typically this is just the string path to the
        model directory, but can be multiple models or arguments for more
        complex models.
    fwp_chunk_shape : tuple
        Max shape (spatial_1, spatial_2, temporal) of an unpadded coarse chunk
        to use for a forward pass. The number of nodes that the
        :class:`.ForwardPassStrategy` is set to distribute to is calculated by
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
        the basic spatial / spatiotemporal ``Sup3rGan`` model. This will be
        loaded from ``sup3r.models``
    out_pattern : str
        Output file pattern. Must include {file_id} format key.  Each output
        file will have a unique file_id filled in and the ext determines the
        output type. If pattern is None then data will be returned in an array
        and not saved.
    input_handler_name : str | None
        Class to use for input data. Provide a string name to match an
        rasterizer or handler class in ``sup3r.preprocessing``
    input_handler_kwargs : dict | None
        Any kwargs for initializing the ``input_handler_name`` class.
    exo_handler_kwargs : dict | None
        Dictionary of args to pass to
        :class:`~sup3r.preprocessing.data_handlers.ExoDataHandler` for
        extracting exogenous features for foward passes. This should be
        a nested dictionary with keys for each exogenous feature. The
        dictionaries corresponding to the feature names should include the path
        to exogenous data source and the files used for input to the forward
        passes, at minimum. Can also provide a dictionary of
        ``input_handler_kwargs`` used for the handler which opens the
        exogenous data. e.g.::
            {'topography': {
                'source_file': ...,
                'input_files': ...,
                'input_handler_kwargs': {'target': ..., 'shape': ...}}}
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
        :meth:`~.forward_pass.ForwardPass.run_chunk()` methods concurrently.
    max_nodes : int | None
        Maximum number of nodes to distribute spatiotemporal chunks across. If
        None then a node will be used for each temporal chunk.
    head_node : bool
        Whether initialization is taking place on the head node of a multi node
        job launch. When this is true :class:`.ForwardPassStrategy` is only
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
        self.timer = Timer()

        model = get_model(self.model_class, self.model_kwargs)
        self.s_enhancements = model.s_enhancements
        self.t_enhancements = model.t_enhancements
        self.s_enhance, self.t_enhance = model.s_enhance, model.t_enhance
        self.input_features = model.lr_features
        self.output_features = model.hr_out_features
        self.features, self.exo_features = self._init_features(model)
        self.input_handler = self.timer(self.init_input_handler, log=True)()
        self.time_slice = _parse_time_slice(
            self.input_handler_kwargs.get('time_slice', slice(None))
        )
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
        self.n_chunks = self.fwp_slicer.n_chunks

        if not self.head_node:
            hr_shape = self.hr_lat_lon.shape[:-1]
            self.gids = np.arange(np.prod(hr_shape)).reshape(hr_shape)
            self.exo_data = self.timer(self.load_exo_data, log=True)(model)
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
            'input_shape': self.input_handler.grid_shape,
            'input_time_range': get_date_range_kwargs(
                self.input_handler.time_index[self.time_slice]
            ),
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

        InputHandler = get_input_handler_class(self.input_handler_name)
        input_handler_kwargs = copy.deepcopy(self.input_handler_kwargs)

        input_handler_kwargs['features'] = self.features
        if self.head_node:
            input_handler_kwargs['features'] = []
            input_handler_kwargs['chunks'] = 'auto'

        input_handler_kwargs['time_slice'] = slice(None)

        return InputHandler(**input_handler_kwargs)

    def _init_features(self, model):
        """Initialize feature attributes."""
        self.exo_handler_kwargs = self.exo_handler_kwargs or {}
        exo_features = list(self.exo_handler_kwargs)
        features = [f for f in model.lr_features if f not in exo_features]
        return features, exo_features

    @cached_property
    def node_chunks(self):
        """Get array of lists such that node_chunks[i] is a list of
        indices for the chunks that will be sent through the generator on the
        ith node."""
        node_chunks = min(self.max_nodes or np.inf, self.n_chunks)
        return np.array_split(np.arange(self.n_chunks), node_chunks)

    @property
    def unfinished_chunks(self):
        """List of chunk indices that have not yet been written and are not
        masked."""
        return [
            idx
            for idx in np.arange(self.n_chunks)
            if not self.chunk_skippable(idx, log=False)
        ]

    @property
    def unfinished_node_chunks(self):
        """Get node_chunks lists which only include indices for chunks which
        have not yet been written or are not masked."""
        node_chunks = min(
            self.max_nodes or np.inf, len(self.unfinished_chunks)
        )
        return np.array_split(self.unfinished_chunks, node_chunks)

    def _get_fwp_chunk_shape(self):
        """Get fwp_chunk_shape with default shape equal to the input handler
        shape"""
        grid_shape = self.input_handler.grid_shape
        tsteps = len(self.input_handler.time_index[self.time_slice])
        shape_iter = zip(self.fwp_chunk_shape, (*grid_shape, tsteps))
        return tuple(fs or ffs for fs, ffs in shape_iter)

    def preflight(self):
        """Prelight logging and sanity checks"""

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

        non_masked = self.fwp_slicer.n_spatial_chunks - sum(self.fwp_mask)
        non_masked *= self.fwp_slicer.n_time_chunks
        log_dict = {
            'n_nodes': len(self.node_chunks),
            'n_spatial_chunks': self.fwp_slicer.n_spatial_chunks,
            'n_time_chunks': self.fwp_slicer.n_time_chunks,
            'n_total_chunks': self.fwp_slicer.n_chunks,
            'non_masked_chunks': non_masked,
        }
        logger.info(
            f'Chunk strategy description:\n'
            f'{pprint.pformat(log_dict, indent=2)}'
        )

    def get_chunk_indices(self, chunk_index):
        """Get (spatial, temporal) indices for the given chunk index"""
        return (
            chunk_index % self.fwp_slicer.n_spatial_chunks,
            chunk_index // self.fwp_slicer.n_spatial_chunks,
        )

    @cached_property
    def hr_lat_lon(self):
        """Get high resolution lat lons"""
        lr_lat_lon = self.input_handler.lat_lon
        shape = tuple(d * self.s_enhance for d in lr_lat_lon.shape[:-1])
        logger.info(
            f'Getting high-resolution grid for full output domain: {shape}'
        )
        return OutputHandler.get_lat_lon(lr_lat_lon, shape)

    @cached_property
    def out_files(self):
        """Get list of output file names for each file chunk forward pass."""
        file_ids = [
            f'{str(i).zfill(6)}_{str(j).zfill(6)}'
            for i in range(self.fwp_slicer.n_time_chunks)
            for j in range(self.fwp_slicer.n_spatial_chunks)
        ]
        out_file_list = [None] * len(file_ids)
        if self.out_pattern is not None:
            msg = 'out_pattern must include a {file_id} format key'
            assert '{file_id}' in self.out_pattern, msg
            os.makedirs(os.path.dirname(self.out_pattern), exist_ok=True)
            out_file_list = [
                self.out_pattern.format(file_id=file_id)
                for file_id in file_ids
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

    def prep_chunk_data(self, chunk_index=0):
        """Get low res input data and exo data for given chunk index and bias
        correct low res data if requested.

        Note
        ----
        ``input_data.load()`` is called here to load chunk data into memory
        """

        s_chunk_idx, t_chunk_idx = self.get_chunk_indices(chunk_index)
        lr_pad_slice = self.lr_pad_slices[s_chunk_idx]
        ti_pad_slice = self.ti_pad_slices[t_chunk_idx]
        exo_data = (
            self.timer(self.exo_data.get_chunk, log=True, call_id=chunk_index)(
                [lr_pad_slice[0], lr_pad_slice[1], ti_pad_slice],
            )
            if self.exo_data is not None
            else None
        )

        kwargs = dict(zip(Dimension.dims_2d(), lr_pad_slice))
        kwargs[Dimension.TIME] = ti_pad_slice
        input_data = self.input_handler.isel(**kwargs)
        input_data.load()

        if self.bias_correct_kwargs is not None:
            logger.info(
                f'Bias correcting data for chunk_index={chunk_index}, '
                f'with shape={input_data.shape}'
            )
            input_data = self.timer(
                bias_correct_features, log=True, call_id=chunk_index
            )(
                features=list(self.bias_correct_kwargs),
                input_handler=input_data,
                bc_method=self.bias_correct_method,
                bc_kwargs=self.bias_correct_kwargs,
            )
        return input_data, exo_data

    def init_chunk(self, chunk_index=0):
        """Get :class:`FowardPassChunk` instance for the given chunk index.

        This selects the appropriate data from `self.input_handler` and
        `self.exo_data` and returns a structure object (`ForwardPassChunk`)
        with that data and other chunk specific attributes.
        """

        s_chunk_idx, t_chunk_idx = self.get_chunk_indices(chunk_index)

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

        args_dict = {
            'chunk': chunk_index,
            'temporal_chunk': t_chunk_idx,
            'spatial_chunk': s_chunk_idx,
            'n_node_chunks': self.fwp_slicer.n_chunks,
            'fwp_chunk_shape': self.fwp_chunk_shape,
            'temporal_pad': self.temporal_pad,
            'spatial_pad': self.spatial_pad,
            'lr_pad_slice': lr_pad_slice,
            'ti_pad_slice': ti_pad_slice,
        }
        logger.info(
            'Initializing ForwardPassChunk with: '
            f'{pprint.pformat(args_dict, indent=2)}'
        )

        logger.info(f'Getting input data for chunk_index={chunk_index}.')

        input_data, exo_data = self.timer(
            self.prep_chunk_data, log=True, call_id=chunk_index
        )(chunk_index=chunk_index)

        return ForwardPassChunk(
            input_data=input_data.as_array(),
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
                exo_kwargs['model'] = model
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

    @cached_property
    def fwp_mask(self):
        """Cached spatial mask which returns whether a given spatial chunk
        should be skipped by the forward pass or not. This is used to skip
        running the forward pass for area with just ocean, for example."""

        mask = np.zeros(len(self.lr_pad_slices))
        InputHandler = get_input_handler_class(self.input_handler_name)
        input_handler_kwargs = copy.deepcopy(self.input_handler_kwargs)
        input_handler_kwargs['features'] = 'all'
        handler = InputHandler(**input_handler_kwargs)
        if 'mask' in handler:
            logger.info(
                'Found "mask" in DataHandler. Computing forward pass '
                'chunk mask for %s chunks',
                len(self.lr_pad_slices),
            )
            mask_vals = handler['mask'].values
            for s_chunk_idx, lr_slices in enumerate(self.lr_pad_slices):
                mask_check = mask_vals[lr_slices[0], lr_slices[1]]
                mask[s_chunk_idx] = bool(np.prod(mask_check.flatten()))
        return mask

    def node_finished(self, node_idx):
        """Check if all out files for a given node have been saved"""
        return all(self.chunk_finished(i) for i in self.node_chunks[node_idx])

    def chunk_finished(self, chunk_idx, log=True):
        """Check if process for given chunk_index has already been run.
        Considered finished if there is already an output file and incremental
        is False."""

        out_file = self.out_files[chunk_idx]
        check = (
            out_file is not None
            and os.path.exists(out_file)
            and self.incremental
        )
        if check and log:
            logger.info(
                '%s already exists and incremental = True. Skipping forward '
                'pass for chunk index %s.',
                out_file,
                chunk_idx,
            )
        return check

    def chunk_masked(self, chunk_idx, log=True):
        """Check if the region for this chunk is masked. This is used to skip
        running the forward pass for region with just ocean, for example."""

        s_chunk_idx, _ = self.get_chunk_indices(chunk_idx)
        mask_check = self.fwp_mask[s_chunk_idx]
        if mask_check and log:
            logger.info(
                'Chunk %s has spatial chunk index %s, which corresponds to a '
                'masked spatial region. Skipping forward pass for this chunk.',
                chunk_idx,
                s_chunk_idx,
            )
        return mask_check

    def chunk_skippable(self, chunk_idx, log=True):
        """Check if chunk is already written or masked."""
        return self.chunk_masked(
            chunk_idx=chunk_idx, log=log
        ) or self.chunk_finished(chunk_idx=chunk_idx, log=log)
