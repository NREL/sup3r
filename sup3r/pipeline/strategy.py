""":class:`ForwardPassStrategy` class. This sets up chunks and needed generator
inputs to distribute forward passes."""

import copy
import logging
import os
import pathlib
import pprint
import warnings
from dataclasses import dataclass
from inspect import signature
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sup3r.bias.utilities import bias_correct_feature
from sup3r.pipeline.slicer import ForwardPassSlicer
from sup3r.pipeline.utilities import get_model
from sup3r.postprocessing import (
    OutputHandler,
)
from sup3r.preprocessing import (
    ExoData,
    ExoDataHandler,
)
from sup3r.preprocessing.utilities import (
    expand_paths,
    get_input_handler_class,
    get_source_type,
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
        self.file_exists = self.out_file is not None and os.path.exists(
            self.out_file
        )


@dataclass
class ForwardPassStrategy:
    """Class to prepare data for forward passes through generator.

    TODO: (1) Seems like this could be cleaned up further. Lots of attrs in the
    init.

    A full file list of contiguous times is provided. The corresponding data is
    split into spatiotemporal chunks which can overlap in time and space. These
    chunks are distributed across nodes according to the max nodes input or
    number of temporal chunks. This strategy stores information on these
    chunks, how they overlap, how they are distributed to nodes, and how to
    crop generator output to stich the chunks back togerther.

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
        Max shape (spatial_1, spatial_2, temporal) of an unpadded coarse
        chunk to use for a forward pass. The number of nodes that the
        :class:`ForwardPassStrategy` is set to distribute to is calculated by
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
        sides of the fwp_chunk_shape.
    temporal_pad : int
        Size of temporal overlap between coarse chunks passed to forward
        passes for subsequent temporal stitching. This overlap will pad
        both sides of the fwp_chunk_shape.
    model_class : str
        Name of the sup3r model class for the GAN model to load. The
        default is the basic spatial / spatiotemporal Sup3rGan model. This
        will be loaded from sup3r.models
    out_pattern : str
        Output file pattern. Must include {file_id} format key.  Each output
        file will have a unique file_id filled in and the ext determines the
        output type. If pattern is None then data will be returned
        in an array and not saved.
    input_handler_name : str | None
        Class to use for input data. Provide a string name to match an
        extracter or handler class in `sup3r.preprocessing`
    input_handler_kwargs : dict | None
        Any kwargs for initializing the `input_handler_name` class.
    exo_kwargs : dict | None
        Dictionary of args to pass to :class:`ExoDataHandler` for
        extracting exogenous features for multistep foward pass. This
        should be a nested dictionary with keys for each exogeneous
        feature. The dictionaries corresponding to the feature names
        should include the path to exogenous data source, the resolution
        of the exogenous data, and how the exogenous data should be used
        in the model. e.g. {'topography': {'file_paths': 'path to input
        files', 'source_file': 'path to exo data', 'steps': [..]}.
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
    allowed_const : list | bool
        Tensorflow has a tensor memory limit of 2GB (result of protobuf
        limitation) and when exceeded can return a tensor with a
        constant output. sup3r will raise a ``MemoryError`` in response. If
        your model is allowed to output a constant output, set this to True
        to allow any constant output or a list of allowed possible constant
        outputs. For example, a precipitation model should be allowed to
        output all zeros so set this to ``[0]``. For details on this limit:
        https://github.com/tensorflow/tensorflow/issues/51870
    incremental : bool
        Allow the forward pass iteration to skip spatiotemporal chunks that
        already have an output file (default = True) or iterate through all
        chunks and overwrite any pre-existing outputs (False).
    output_workers : int | None
        Max number of workers to use for writing forward pass output.
    pass_workers : int | None
        Max number of workers to use for performing forward passes on a
        single node. If 1 then all forward passes on chunks distributed to
        a single node will be run serially. pass_workers=2 is the minimum
        number of workers required to run the ForwardPass initialization
        and :meth:`ForwardPass.run_chunk()` methods concurrently.
    max_nodes : int | None
        Maximum number of nodes to distribute spatiotemporal chunks across.
        If None then a node will be used for each temporal chunk.
    """

    file_paths: Union[str, list, pathlib.Path]
    model_kwargs: dict
    fwp_chunk_shape: tuple
    spatial_pad: int
    temporal_pad: int
    model_class: str = 'Sup3rGan'
    out_pattern: Optional[str] = None
    input_handler_name: Optional[str] = None
    input_handler_kwargs: Optional[dict] = None
    exo_kwargs: Optional[dict] = None
    bias_correct_method: Optional[str] = None
    bias_correct_kwargs: Optional[dict] = None
    allowed_const: Optional[Union[list, bool]] = None
    incremental: bool = True
    output_workers: Optional[int] = None
    pass_workers: Optional[int] = None
    max_nodes: Optional[int] = None

    @log_args
    def __post_init__(self):
        self.file_paths = expand_paths(self.file_paths)
        self.exo_kwargs = self.exo_kwargs or {}
        self.input_handler_kwargs = self.input_handler_kwargs or {}
        self.bias_correct_kwargs = self.bias_correct_kwargs or {}
        self.input_type = get_source_type(self.file_paths)
        self.output_type = get_source_type(self.out_pattern)
        model = get_model(self.model_class, self.model_kwargs)
        models = getattr(model, 'models', [model])
        self.s_enhancements = [model.s_enhance for model in models]
        self.t_enhancements = [model.t_enhance for model in models]
        self.s_enhance = np.prod(self.s_enhancements)
        self.t_enhance = np.prod(self.t_enhancements)
        self.input_features = model.lr_features
        self.output_features = model.hr_out_features
        assert len(self.input_features) > 0, 'No input features!'
        assert len(self.output_features) > 0, 'No output features!'
        self.exo_features = (
            [] if not self.exo_kwargs else list(self.exo_kwargs)
        )
        self.features = [
            f for f in self.input_features if f not in self.exo_features
        ]
        self.input_handler_kwargs.update(
            {'file_paths': self.file_paths, 'features': self.features}
        )
        input_handler_kwargs = copy.deepcopy(self.input_handler_kwargs)
        self.time_slice = input_handler_kwargs.pop('time_slice', slice(None))
        InputHandler = get_input_handler_class(self.input_handler_name)
        self.input_handler = InputHandler(**input_handler_kwargs)
        self.exo_data = self.load_exo_data(model)
        self.hr_lat_lon = self.get_hr_lat_lon()
        self.gids = np.arange(np.prod(self.hr_lat_lon.shape[:-1]))
        self.gids = self.gids.reshape(self.hr_lat_lon.shape[:-1])
        self.grid_shape = self.input_handler.lat_lon.shape[:-1]

        self.fwp_slicer = ForwardPassSlicer(
            coarse_shape=self.input_handler.lat_lon.shape[:-1],
            time_steps=len(self.input_handler.time_index),
            time_slice=self.time_slice,
            chunk_shape=self.fwp_chunk_shape,
            s_enhancements=self.s_enhancements,
            t_enhancements=self.t_enhancements,
            spatial_pad=self.spatial_pad,
            temporal_pad=self.temporal_pad,
        )

        self.chunks = self.fwp_slicer.n_chunks
        n_chunks = (
            self.chunks
            if self.max_nodes is None
            else min(self.max_nodes, self.chunks)
        )
        self.node_chunks = np.array_split(np.arange(self.chunks), n_chunks)
        self.nodes = len(self.node_chunks)
        self.out_files = self.get_out_files(out_files=self.out_pattern)
        self.preflight()

    def preflight(self):
        """Prelight logging and sanity checks"""

        log_dict = {
            'n_nodes': self.nodes,
            'n_spatial_chunks': self.fwp_slicer.n_spatial_chunks,
            'n_time_chunks': self.fwp_slicer.n_time_chunks,
            'n_total_chunks': self.chunks,
        }
        logger.info(
            f'Chunk strategy description:\n'
            f'{pprint.pformat(log_dict, indent=2)}'
        )

        out = self.fwp_slicer.get_time_slices()
        self.ti_slices, self.ti_pad_slices = out

        msg = (
            'Using a padded chunk size '
            f'({self.fwp_chunk_shape[2] + 2 * self.temporal_pad}) '
            'larger than the full temporal domain '
            f'({len(self.input_handler.time_index)}). '
            'Should just run without temporal chunking. '
        )
        if self.fwp_chunk_shape[2] + 2 * self.temporal_pad >= len(
            self.input_handler.time_index
        ):
            logger.warning(msg)
            warnings.warn(msg)
        out = self.fwp_slicer.get_spatial_slices()
        self.lr_slices, self.lr_pad_slices, self.hr_slices = out

        if self.bias_correct_kwargs is not None:
            padded_tslice = slice(
                self.ti_pad_slices[0].start, self.ti_pad_slices[-1].stop
            )
            for feat in self.bias_correct_kwargs:
                self.input_handler.data[feat, ..., padded_tslice] = (
                    bias_correct_feature(
                        feat,
                        input_handler=self.input_handler,
                        time_slice=padded_tslice,
                        bc_method=self.bias_correct_method,
                        bc_kwargs=self.bias_correct_kwargs,
                    )
                )

    def get_chunk_indices(self, chunk_index):
        """Get (spatial, temporal) indices for the given chunk index"""
        return (
            chunk_index % self.fwp_slicer.n_spatial_chunks,
            chunk_index // self.fwp_slicer.n_spatial_chunks,
        )

    def get_hr_lat_lon(self):
        """Get high resolution lat lons"""
        logger.info('Getting high-resolution grid for full output domain.')
        lr_lat_lon = self.input_handler.lat_lon
        shape = tuple(d * self.s_enhance for d in lr_lat_lon.shape[:-1])
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
                lr_slice[0], self.grid_shape[0], self.spatial_pad
            ),
            self._get_pad_width(
                lr_slice[1], self.grid_shape[1], self.spatial_pad
            ),
            self._get_pad_width(
                ti_slice, len(self.input_handler.time_index), self.temporal_pad
            ),
        )

    def init_chunk(self, chunk_index=0):
        """Get :class:`FowardPassChunk` instance for the given chunk index."""

        s_chunk_idx, t_chunk_idx = self.get_chunk_indices(chunk_index)

        args_dict = {
            'chunk': chunk_index,
            'temporal_chunk': t_chunk_idx,
            'spatial_chunk': s_chunk_idx,
            'n_node_chunks': self.chunks,
        }
        logger.info(
            'Initializing ForwardPass with: '
            f'{pprint.pformat(args_dict, indent=2)}'
        )

        msg = (
            f'Requested forward pass on chunk_index={chunk_index} > '
            f'n_chunks={self.chunks}'
        )
        assert chunk_index <= self.chunks, msg

        hr_slice = self.hr_slices[s_chunk_idx]
        ti_slice = self.ti_slices[t_chunk_idx]
        lr_times = self.input_handler.time_index[ti_slice]
        lr_pad_slice = self.lr_pad_slices[s_chunk_idx]
        ti_pad_slice = self.ti_pad_slices[t_chunk_idx]

        logger.info(f'Getting input data for chunk_index={chunk_index}.')

        return ForwardPassChunk(
            input_data=self.input_handler.data[
                lr_pad_slice[0], lr_pad_slice[1], ti_pad_slice
            ],
            exo_data=self.get_exo_chunk(
                self.exo_data,
                self.input_handler.data.shape,
                lr_pad_slice,
                ti_pad_slice,
            ),
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

    @classmethod
    def get_exo_chunk(
        cls, exo_data, input_data_shape, lr_pad_slice, ti_pad_slice
    ):
        """Get exo data for the current chunk from the exo data for the full
        extent.

        Parameters
        ----------
        exo_data : ExoData
           :class:`ExoData` object composed of multiple
           :class:`SingleExoDataStep` objects. This includes the exo data for
           the full spatiotemporal extent for each model step.
        input_data_shape : tuple
            Spatiotemporal shape of the full low-resolution extent.
            (lats, lons, time)
        lr_pad_slice : list
            List of spatial slices for the low-resolution input data for the
            current chunk.
        ti_pad_slice : slice
            Temporal slice for the low-resolution input data for the current
            chunk.

        Returns
        -------
        exo_data : ExoData
           :class:`ExoData` object composed of multiple
           :class:`SingleExoDataStep` objects. This is the sliced exo data for
           the current chunk.
        """
        exo_chunk = {}
        if exo_data is not None:
            for feature in exo_data:
                exo_chunk[feature] = {}
                exo_chunk[feature]['steps'] = []
                for step in exo_data[feature]['steps']:
                    chunk_step = {k: step[k] for k in step if k != 'data'}
                    exo_shape = step['data'].shape
                    enhanced_slices = cls._get_enhanced_slices(
                        [lr_pad_slice[0], lr_pad_slice[1], ti_pad_slice],
                        input_data_shape=input_data_shape,
                        exo_data_shape=exo_shape,
                    )
                    chunk_step['data'] = step['data'][
                        enhanced_slices[0],
                        enhanced_slices[1],
                        enhanced_slices[2],
                    ]
                    exo_chunk[feature]['steps'].append(chunk_step)
        return exo_chunk

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
        if self.exo_kwargs:
            for feature in self.exo_features:
                exo_kwargs = copy.deepcopy(self.exo_kwargs[feature])
                exo_kwargs['feature'] = feature
                exo_kwargs['models'] = getattr(model, 'models', [model])
                input_handler_kwargs = exo_kwargs.get(
                    'input_handler_kwargs', {}
                )
                input_handler_kwargs['target'] = self.input_handler.target
                input_handler_kwargs['shape'] = self.input_handler.grid_shape
                exo_kwargs['input_handler_kwargs'] = input_handler_kwargs
                sig = signature(ExoDataHandler)
                exo_kwargs = {
                    k: v for k, v in exo_kwargs.items() if k in sig.parameters
                }
                data.update(ExoDataHandler(**exo_kwargs).data)
            exo_data = ExoData(data)
        return exo_data

    def node_finished(self, node_index):
        """Check if all out files for a given node have been saved

        Parameters
        ----------
        node_index : int
            Index of node to check for completed processes
        """
        return all(
            self._chunk_finished(i) for i in self.node_chunks[node_index]
        )

    def _chunk_finished(self, chunk_index):
        """Check if process for given chunk_index has already been run.

        Parameters
        ----------
        chunk_index : int
            Index of the process chunk to check for completion. Considered
            finished if there is already an output file and incremental is
            False.
        """
        out_file = self.out_files[chunk_index]
        if os.path.exists(out_file) and self.incremental:
            logger.info(
                'Not running chunk index {}, output file ' 'exists: {}'.format(
                    chunk_index, out_file
                )
            )
            return True
        return False
