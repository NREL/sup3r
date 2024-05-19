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

import sup3r.bias.bias_transforms
import sup3r.models
from sup3r.postprocessing import (
    OutputHandler,
)
from sup3r.utilities.execution import DistributedProcess
from sup3r.utilities.utilities import (
    get_chunk_slices,
    get_extracter_class,
    get_source_type,
)

np.random.seed(42)

logger = logging.getLogger(__name__)


class ForwardPassSlicer:
    """Get slices for sending data chunks through generator."""

    def __init__(self,
                 coarse_shape,
                 time_steps,
                 time_slice,
                 chunk_shape,
                 s_enhancements,
                 t_enhancements,
                 spatial_pad,
                 temporal_pad):
        """
        Parameters
        ----------
        coarse_shape : tuple
            Shape of full domain for low res data
        time_steps : int
            Number of time steps for full temporal domain of low res data. This
            is used to construct a dummy_time_index from np.arange(time_steps)
        time_slice : slice
            Slice to use to extract range from time_index
        chunk_shape : tuple
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
        s_enhancements : list
            List of factors by which the Sup3rGan model will enhance the
            spatial dimensions of low resolution data. If there are two 5x
            spatial enhancements, this should be [5, 5] where the total
            enhancement is the product of these factors.
        t_enhancements : list
            List of factor by which the Sup3rGan model will enhance temporal
            dimension of low resolution data
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
        """
        self.grid_shape = coarse_shape
        self.time_steps = time_steps
        self.s_enhancements = s_enhancements
        self.t_enhancements = t_enhancements
        self.s_enhance = np.prod(self.s_enhancements)
        self.t_enhance = np.prod(self.t_enhancements)
        self.dummy_time_index = np.arange(time_steps)
        self.time_slice = time_slice
        self.temporal_pad = temporal_pad
        self.spatial_pad = spatial_pad
        self.chunk_shape = chunk_shape

        self._chunk_lookup = None
        self._s1_lr_slices = None
        self._s2_lr_slices = None
        self._s1_lr_pad_slices = None
        self._s2_lr_pad_slices = None
        self._s_lr_slices = None
        self._s_lr_pad_slices = None
        self._s_lr_crop_slices = None
        self._t_lr_pad_slices = None
        self._t_lr_crop_slices = None
        self._s_hr_slices = None
        self._s_hr_crop_slices = None
        self._t_hr_crop_slices = None
        self._hr_crop_slices = None
        self._gids = None

    def get_spatial_slices(self):
        """Get spatial slices for small data chunks that are passed through
        generator

        Returns
        -------
        s_lr_slices: list
            List of slices for low res data chunks which have not been padded.
            data_handler.data[s_lr_slice] corresponds to an unpadded low res
            input to the model.
        s_lr_pad_slices : list
            List of slices which have been padded so that high res output
            can be stitched together. data_handler.data[s_lr_pad_slice]
            corresponds to a padded low res input to the model.
        s_hr_slices : list
            List of slices for high res data corresponding to the
            lr_slices regions. output_array[s_hr_slice] corresponds to the
            cropped generator output.
        """
        return (self.s_lr_slices, self.s_lr_pad_slices, self.s_hr_slices)

    def get_time_slices(self):
        """Calculate the number of time chunks across the full time index

        Returns
        -------
        t_lr_slices : list
            List of low-res non-padded time index slices. e.g. If
            fwp_chunk_size[2] is 5 then the size of these slices will always
            be 5.
        t_lr_pad_slices : list
            List of low-res padded time index slices. e.g. If fwp_chunk_size[2]
            is 5 the size of these slices will be 15, with exceptions at the
            start and end of the full time index.
        """
        return self.t_lr_slices, self.t_lr_pad_slices

    @property
    def s_lr_slices(self):
        """Get low res spatial slices for small data chunks that are passed
        through generator

        Returns
        -------
        _s_lr_slices : list
            List of spatial slices corresponding to the unpadded spatial region
            going through the generator
        """
        if self._s_lr_slices is None:
            self._s_lr_slices = []
            for _, s1 in enumerate(self.s1_lr_slices):
                for _, s2 in enumerate(self.s2_lr_slices):
                    s_slice = (s1, s2, slice(None), slice(None))
                    self._s_lr_slices.append(s_slice)
        return self._s_lr_slices

    @property
    def s_lr_pad_slices(self):
        """Get low res padded slices for small data chunks that are passed
        through generator

        Returns
        -------
        _s_lr_pad_slices : list
            List of slices which have been padded so that high res output
            can be stitched together. Each entry in this list has a slice for
            each spatial dimension and then slice(None) for temporal and
            feature dimension. This is because the temporal dimension is only
            chunked across nodes and not within a single node.
            data_handler.data[s_lr_pad_slice] gives the padded data volume
            passed through the generator
        """
        if self._s_lr_pad_slices is None:
            self._s_lr_pad_slices = []
            for _, s1 in enumerate(self.s1_lr_pad_slices):
                for _, s2 in enumerate(self.s2_lr_pad_slices):
                    pad_slice = (s1, s2, slice(None), slice(None))
                    self._s_lr_pad_slices.append(pad_slice)

        return self._s_lr_pad_slices

    @property
    def t_lr_pad_slices(self):
        """Get low res temporal padded slices for distributing time chunks
        across nodes. These slices correspond to the time chunks sent to each
        node and are padded according to temporal_pad.

        Returns
        -------
        _t_lr_pad_slices : list
            List of low res temporal slices which have been padded so that high
            res output can be stitched together
        """
        if self._t_lr_pad_slices is None:
            self._t_lr_pad_slices = self.get_padded_slices(
                self.t_lr_slices,
                self.time_steps,
                1,
                self.temporal_pad,
                self.time_slice.step,
            )
        return self._t_lr_pad_slices

    @property
    def t_lr_crop_slices(self):
        """Get low res temporal cropped slices for cropping time index of
        padded input data.

        Returns
        -------
        _t_lr_crop_slices : list
            List of low res temporal slices for cropping padded input data
        """
        if self._t_lr_crop_slices is None:
            self._t_lr_crop_slices = self.get_cropped_slices(
                self.t_lr_slices, self.t_lr_pad_slices, 1)

        return self._t_lr_crop_slices

    @property
    def t_hr_crop_slices(self):
        """Get high res temporal cropped slices for cropping forward pass
        output before stitching together

        Returns
        -------
        _t_hr_crop_slices : list
            List of high res temporal slices for cropping padded generator
            output
        """
        hr_crop_start = None
        hr_crop_stop = None
        if self.temporal_pad > 0:
            hr_crop_start = self.t_enhance * self.temporal_pad
            hr_crop_stop = -hr_crop_start

        if self._t_hr_crop_slices is None:
            # don't use self.get_cropped_slices() here because temporal padding
            # gets weird at beginning and end of timeseries and the temporal
            # axis should always be evenly chunked.
            self._t_hr_crop_slices = [
                slice(hr_crop_start, hr_crop_stop)
                for _ in range(len(self.t_lr_slices))
            ]

        return self._t_hr_crop_slices

    @property
    def s1_hr_slices(self):
        """Get high res spatial slices for first spatial dimension"""
        return self.get_hr_slices(self.s1_lr_slices, self.s_enhance)

    @property
    def s2_hr_slices(self):
        """Get high res spatial slices for second spatial dimension"""
        return self.get_hr_slices(self.s2_lr_slices, self.s_enhance)

    @property
    def s_hr_slices(self):
        """Get high res slices for indexing full generator output array

        Returns
        -------
        _s_hr_slices : list
            List of high res slices. Each entry in this list has a slice for
            each spatial dimension and then slice(None) for temporal and
            feature dimension. This is because the temporal dimension is only
            chunked across nodes and not within a single node. output[hr_slice]
            gives the superresolved domain corresponding to
            data_handler.data[lr_slice]
        """
        if self._s_hr_slices is None:
            self._s_hr_slices = []
            for _, s1 in enumerate(self.s1_hr_slices):
                for _, s2 in enumerate(self.s2_hr_slices):
                    hr_slice = (s1, s2, slice(None), slice(None))
                    self._s_hr_slices.append(hr_slice)
        return self._s_hr_slices

    @property
    def s_lr_crop_slices(self):
        """Get low res cropped slices for cropping input chunk domain

        Returns
        -------
        _s_lr_crop_slices : list
            List of low res cropped slices. Each entry in this list has a
            slice for each spatial dimension and then slice(None) for temporal
            and feature dimension.
        """
        if self._s_lr_crop_slices is None:
            self._s_lr_crop_slices = []
            s1_crop_slices = self.get_cropped_slices(self.s1_lr_slices,
                                                     self.s1_lr_pad_slices,
                                                     1)
            s2_crop_slices = self.get_cropped_slices(self.s2_lr_slices,
                                                     self.s2_lr_pad_slices,
                                                     1)
            for i, _ in enumerate(self.s1_lr_slices):
                for j, _ in enumerate(self.s2_lr_slices):
                    lr_crop_slice = (s1_crop_slices[i],
                                     s2_crop_slices[j],
                                     slice(None),
                                     slice(None),
                                     )
                    self._s_lr_crop_slices.append(lr_crop_slice)
        return self._s_lr_crop_slices

    @property
    def s_hr_crop_slices(self):
        """Get high res cropped slices for cropping generator output

        Returns
        -------
        _s_hr_crop_slices : list
            List of high res cropped slices. Each entry in this list has a
            slice for each spatial dimension and then slice(None) for temporal
            and feature dimension.
        """
        hr_crop_start = None
        hr_crop_stop = None
        if self.spatial_pad > 0:
            hr_crop_start = self.s_enhance * self.spatial_pad
            hr_crop_stop = -hr_crop_start

        if self._s_hr_crop_slices is None:
            self._s_hr_crop_slices = []
            s1_hr_crop_slices = [
                slice(hr_crop_start, hr_crop_stop)
                for _ in range(len(self.s1_lr_slices))
            ]
            s2_hr_crop_slices = [
                slice(hr_crop_start, hr_crop_stop)
                for _ in range(len(self.s2_lr_slices))
            ]

            for _, s1 in enumerate(s1_hr_crop_slices):
                for _, s2 in enumerate(s2_hr_crop_slices):
                    hr_crop_slice = (s1, s2, slice(None), slice(None))
                    self._s_hr_crop_slices.append(hr_crop_slice)
        return self._s_hr_crop_slices

    @property
    def hr_crop_slices(self):
        """Get high res spatiotemporal cropped slices for cropping generator
        output

        Returns
        -------
        _hr_crop_slices : list
            List of high res spatiotemporal cropped slices. Each entry in this
            list has a crop slice for each spatial dimension and temporal
            dimension and then slice(None) for the feature dimension.
            model.generate()[hr_crop_slice] gives the cropped generator output
            corresponding to output_array[hr_slice]
        """
        if self._hr_crop_slices is None:
            self._hr_crop_slices = []
            for t in self.t_hr_crop_slices:
                node_slices = [(s[0], s[1], t, slice(None))
                               for s in self.s_hr_crop_slices]
                self._hr_crop_slices.append(node_slices)
        return self._hr_crop_slices

    @property
    def s1_lr_pad_slices(self):
        """List of low resolution spatial slices with padding for first
        spatial dimension"""
        if self._s1_lr_pad_slices is None:
            self._s1_lr_pad_slices = self.get_padded_slices(
                self.s1_lr_slices,
                self.grid_shape[0],
                1,
                padding=self.spatial_pad,
            )
        return self._s1_lr_pad_slices

    @property
    def s2_lr_pad_slices(self):
        """List of low resolution spatial slices with padding for second
        spatial dimension"""
        if self._s2_lr_pad_slices is None:
            self._s2_lr_pad_slices = self.get_padded_slices(
                self.s2_lr_slices,
                self.grid_shape[1],
                1,
                padding=self.spatial_pad,
            )
        return self._s2_lr_pad_slices

    @property
    def s1_lr_slices(self):
        """List of low resolution spatial slices for first spatial dimension
        considering padding on all sides of the spatial raster."""
        ind = slice(0, self.grid_shape[0])
        slices = get_chunk_slices(self.grid_shape[0],
                                  self.chunk_shape[0],
                                  index_slice=ind)
        return slices

    @property
    def s2_lr_slices(self):
        """List of low resolution spatial slices for second spatial dimension
        considering padding on all sides of the spatial raster."""
        ind = slice(0, self.grid_shape[1])
        slices = get_chunk_slices(self.grid_shape[1],
                                  self.chunk_shape[1],
                                  index_slice=ind)
        return slices

    @property
    def t_lr_slices(self):
        """Low resolution temporal slices"""
        n_tsteps = len(self.dummy_time_index[self.time_slice])
        n_chunks = n_tsteps / self.chunk_shape[2]
        n_chunks = int(np.ceil(n_chunks))
        ti_slices = self.dummy_time_index[self.time_slice]
        ti_slices = np.array_split(ti_slices, n_chunks)
        ti_slices = [
            slice(c[0], c[-1] + 1, self.time_slice.step) for c in ti_slices
        ]
        return ti_slices

    @staticmethod
    def get_hr_slices(slices, enhancement, step=None):
        """Get high resolution slices for temporal or spatial slices

        Parameters
        ----------
        slices : list
            Low resolution slices to be enhanced
        enhancement : int
            Enhancement factor
        step : int | None
            Step size for slices

        Returns
        -------
        hr_slices : list
            High resolution slices
        """
        hr_slices = []
        if step is not None:
            step *= enhancement
        for sli in slices:
            start = sli.start * enhancement
            stop = sli.stop * enhancement
            hr_slices.append(slice(start, stop, step))
        return hr_slices

    @property
    def chunk_lookup(self):
        """Get a 3D array with shape
        (n_spatial_1_chunks, n_spatial_2_chunks, n_temporal_chunks)
        where each value is the chunk index."""
        if self._chunk_lookup is None:
            n_s1 = len(self.s1_lr_slices)
            n_s2 = len(self.s2_lr_slices)
            n_t = self.n_temporal_chunks
            lookup = np.arange(self.n_chunks).reshape((n_t, n_s1, n_s2))
            self._chunk_lookup = np.transpose(lookup, axes=(1, 2, 0))
        return self._chunk_lookup

    @property
    def spatial_chunk_lookup(self):
        """Get a 2D array with shape (n_spatial_1_chunks, n_spatial_2_chunks)
        where each value is the spatial chunk index."""
        n_s1 = len(self.s1_lr_slices)
        n_s2 = len(self.s2_lr_slices)
        return np.arange(self.n_spatial_chunks).reshape((n_s1, n_s2))

    @property
    def n_spatial_chunks(self):
        """Get the number of spatial chunks"""
        return len(self.hr_crop_slices[0])

    @property
    def n_temporal_chunks(self):
        """Get the number of temporal chunks"""
        return len(self.t_hr_crop_slices)

    @property
    def n_chunks(self):
        """Get total number of spatiotemporal chunks"""
        return self.n_spatial_chunks * self.n_temporal_chunks

    @staticmethod
    def get_padded_slices(slices, shape, enhancement, padding, step=None):
        """Get padded slices with the specified padding size, max shape,
        enhancement, and step size

        Parameters
        ----------
        slices : list
            List of low res unpadded slice
        shape : int
            max possible index of a padded slice. e.g. if the slices are
            indexing a dimension with size 10 then a padded slice cannot have
            an index greater than 10.
        enhancement : int
            Enhancement factor. e.g. If these slices are indexing a spatial
            dimension which will be enhanced by 2x then enhancement=2.
        padding : int
            Padding factor. e.g. If these slices are indexing a spatial
            dimension and the spatial_pad is 10 this is 10. It will be
            multiplied by the enhancement factor if the slices are to be used
            to index an enhanced dimension.
        step : int | None
            Step size for slices. e.g. If these slices are indexing a temporal
            dimension and time_slice.step = 3 then step=3.

        Returns
        -------
        list
            Padded slices for temporal or spatial dimensions.
        """
        step = step or 1
        pad = step * padding * enhancement
        pad_slices = []
        for _, s in enumerate(slices):
            start = np.max([0, s.start * enhancement - pad])
            end = np.min([enhancement * shape, s.stop * enhancement + pad])
            pad_slices.append(slice(start, end, step))
        return pad_slices

    @staticmethod
    def get_cropped_slices(unpadded_slices, padded_slices, enhancement):
        """Get cropped slices to cut off padded output

        Parameters
        ----------
        unpadded_slices : list
            List of unpadded slices
        padded_slices : list
            List of padded slices
        enhancement : int
            Enhancement factor for the data to be cropped.

        Returns
        -------
        list
            Cropped slices for temporal or spatial dimensions.
        """
        cropped_slices = []
        for ps, us in zip(padded_slices, unpadded_slices):
            start = us.start
            stop = us.stop
            step = us.step or 1
            if start is not None:
                start = enhancement * (us.start - ps.start) // step
            if stop is not None:
                stop = enhancement * (us.stop - ps.stop) // step
            if start is not None and start <= 0:
                start = None
            if stop is not None and stop >= 0:
                stop = None
            cropped_slices.append(slice(start, stop))
        return cropped_slices


class ForwardPassStrategy(DistributedProcess):
    """Class to prepare data for forward passes through generator.

    A full file list of contiguous times is provided. The corresponding data is
    split into spatiotemporal chunks which can overlap in time and space. These
    chunks are distributed across nodes according to the max nodes input or
    number of temporal chunks. This strategy stores information on these
    chunks, how they overlap, how they are distributed to nodes, and how to
    crop generator output to stich the chunks back togerther.
    """

    def __init__(self,
                 file_paths,
                 model_kwargs,
                 fwp_chunk_shape,
                 spatial_pad,
                 temporal_pad,
                 model_class='Sup3rGan',
                 out_pattern=None,
                 extracter_name=None,
                 extracter_kwargs=None,
                 incremental=True,
                 output_workers=None,
                 pass_workers=None,
                 exo_kwargs=None,
                 bias_correct_method=None,
                 bias_correct_kwargs=None,
                 max_nodes=None,
                 allowed_const=False):
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
        extracter_name : str | None
            :class:`Extracter` class to use for input data. Provide a string
            name to match a class in `sup3r.containers.extracters.`
        extracter_kwargs : dict | None
            Any kwargs for initializing the :class:`Extracter` object.
        incremental : bool
            Allow the forward pass iteration to skip spatiotemporal chunks that
            already have an output file (True, default) or iterate through all
            chunks and overwrite any pre-existing outputs (False).
        output_workers : int | None
            Max number of workers to use for writing forward pass output.
        pass_workers : int | None
            Max number of workers to use for performing forward passes on a
            single node. If 1 then all forward passes on chunks distributed to
            a single node will be run in serial. pass_workers=2 is the minimum
            number of workers required to run the ForwardPass initialization
            and ForwardPass.run_chunk() methods concurrently.
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
        """
        self.extracter_kwargs = extracter_kwargs or {}
        self.file_paths = file_paths
        self.model_kwargs = model_kwargs
        self.fwp_chunk_shape = fwp_chunk_shape
        self.spatial_pad = spatial_pad
        self.temporal_pad = temporal_pad
        self.model_class = model_class
        self.out_pattern = out_pattern
        self.exo_kwargs = exo_kwargs or {}
        self.exo_features = ([]
                             if not self.exo_kwargs else list(self.exo_kwargs))
        self.incremental = incremental
        self.bias_correct_method = bias_correct_method
        self.bias_correct_kwargs = bias_correct_kwargs or {}
        self.allowed_const = allowed_const
        self.out_files = self.get_out_files(out_files=self.out_pattern)
        self.input_type = get_source_type(self.file_paths)
        self.output_type = get_source_type(self.out_pattern)
        self.output_workers = output_workers
        self.pass_workers = pass_workers
        self.model = self.get_model(model_class)
        models = getattr(self.model, 'models', [self.model])
        self.s_enhancements = [model.s_enhance for model in models]
        self.t_enhancements = [model.t_enhance for model in models]
        self.s_enhance = np.prod(self.s_enhancements)
        self.t_enhance = np.prod(self.t_enhancements)
        self.input_features = self.model.lr_features
        self.output_features = self.model.hr_out_features
        assert len(self.input_features) > 0, 'No input features!'
        assert len(self.output_features) > 0, 'No output features!'

        self.features = [
            f for f in self.input_features if f not in self.exo_features
        ]
        self.extracter_kwargs.update(
            {'file_paths': self.file_paths, 'features': self.features}
        )
        self.extracter_class = get_extracter_class(extracter_name)
        self.extracter = self.extracter_class(**self.extracter_kwargs)
        self.lr_lat_lon = self.extracter.lat_lon
        self.grid_shape = self.lr_lat_lon.shape[:-1]
        self.lr_time_index = self.extracter.time_index
        self.hr_lat_lon = self.get_hr_lat_lon()
        self.raw_tsteps = self.get_raw_tsteps()

        self.fwp_slicer = ForwardPassSlicer(self.grid_shape,
                                            self.raw_tsteps,
                                            self.time_slice,
                                            self.fwp_chunk_shape,
                                            self.s_enhancements,
                                            self.t_enhancements,
                                            self.spatial_pad,
                                            self.temporal_pad)

        DistributedProcess.__init__(self,
                                    max_nodes=max_nodes,
                                    max_chunks=self.fwp_slicer.n_chunks,
                                    incremental=self.incremental)

        self.preflight()

    def get_model(self, model_class):
        """Instantiate model after check on class name."""
        model_class = getattr(sup3r.models, model_class, None)
        if isinstance(self.model_kwargs, str):
            self.model_kwargs = {'model_dir': self.model_kwargs}

        if model_class is None:
            msg = ('Could not load requested model class "{}" from '
                   'sup3r.models, Make sure you typed in the model class '
                   'name correctly.'.format(self.model_class))
            logger.error(msg)
            raise KeyError(msg)
        return model_class.load(**self.model_kwargs, verbose=True)

    def preflight(self):
        """Prelight path name formatting and sanity checks"""

        logger.info('Initializing ForwardPassStrategy. '
                    f'Using n_nodes={self.nodes} with '
                    f'n_spatial_chunks={self.fwp_slicer.n_spatial_chunks}, '
                    f'n_temporal_chunks={self.fwp_slicer.n_temporal_chunks}, '
                    f'and n_total_chunks={self.chunks}. '
                    f'{self.chunks / self.nodes:.3f} chunks per node on '
                    'average.')
        logger.info(f'Using max_workers={self.max_workers}, '
                    f'pass_workers={self.pass_workers}, '
                    f'output_workers={self.output_workers}')

        out = self.fwp_slicer.get_time_slices()
        self.ti_slices, self.ti_pad_slices = out

        msg = ('Using a padded chunk size '
               f'({self.fwp_chunk_shape[2] + 2 * self.temporal_pad}) '
               f'larger than the full temporal domain ({self.raw_tsteps}). '
               'Should just run without temporal chunking. ')
        if self.fwp_chunk_shape[2] + 2 * self.temporal_pad >= self.raw_tsteps:
            logger.warning(msg)
            warnings.warn(msg)

        hr_data_shape = (self.extracter.shape[0] * self.s_enhance,
                         self.extracter.shape[1] * self.s_enhance)
        self.gids = np.arange(np.prod(hr_data_shape))
        self.gids = self.gids.reshape(hr_data_shape)

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
        kwargs = copy.deepcopy(self.extracter_kwargs)
        _ = kwargs.pop('time_slice', None)
        return len(self.extracter_class(**kwargs).time_index)

    def get_hr_lat_lon(self):
        """Get high resolution lat lons"""
        logger.info('Getting high-resolution grid for full output domain.')
        lr_lat_lon = self.lr_lat_lon.copy()
        return OutputHandler.get_lat_lon(lr_lat_lon, self.gids.shape)

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
        self._max_nodes = (self._max_nodes if self._max_nodes is not None else
                           self.fwp_slicer.n_temporal_chunks)
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
