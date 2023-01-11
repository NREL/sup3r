# -*- coding: utf-8 -*-
"""
Sup3r forward pass handling module.

@author: bbenton
"""
from concurrent.futures import as_completed
import numpy as np
import logging
import os
import warnings
import copy
from datetime import datetime as dt
import psutil
from inspect import signature

from rex.utilities.fun_utils import get_fun_call_str
from rex.utilities.execution import SpawnProcessPool

import sup3r.models
import sup3r.bias.bias_transforms
from sup3r.preprocessing.data_handling import InputMixIn
from sup3r.preprocessing.exogenous_data_handling import ExogenousDataHandler
from sup3r.postprocessing.file_handling import (OutputHandlerH5,
                                                OutputHandlerNC,
                                                OutputHandler)
from sup3r.utilities.utilities import (get_chunk_slices,
                                       get_source_type,
                                       get_input_handler_class)
from sup3r.utilities.execution import DistributedProcess
from sup3r.utilities import ModuleName
from sup3r.utilities.cli import BaseCLI

np.random.seed(42)

logger = logging.getLogger(__name__)


class ForwardPassSlicer:
    """Get slices for sending data chunks through model."""

    def __init__(self, coarse_shape, time_steps, temporal_slice, chunk_shape,
                 s_enhancements, t_enhancements, spatial_pad, temporal_pad):
        """
        Parameters
        ----------
        coarse_shape : tuple
            Shape of full domain for low res data
        time_steps : int
            Number of time steps for full temporal domain of low res data. This
            is used to construct a dummy_time_index from np.arange(time_steps)
        temporal_slice : slice
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
        exo_s_enhancements : list
            List of spatial enhancement steps specific to the exogenous_data
            inputs. This differs from s_enhancements in that s_enhancements[0]
            will be the spatial enhancement of the first model, but
            exo_s_enhancements[0] may be 1 to signify exo data is required for
            the first non-enhanced spatial input resolution.
        """
        self.grid_shape = coarse_shape
        self.time_steps = time_steps
        self.s_enhancements = s_enhancements
        self.t_enhancements = t_enhancements
        self.s_enhance = np.product(self.s_enhancements)
        self.t_enhance = np.product(self.t_enhancements)
        self.dummy_time_index = np.arange(time_steps)
        self.temporal_slice = temporal_slice
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

    def get_temporal_slices(self):
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
                self.t_lr_slices, self.time_steps, 1,
                self.temporal_pad, self.temporal_slice.step)
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
            self._t_hr_crop_slices = [slice(hr_crop_start, hr_crop_stop)
                                      for _ in range(len(self.t_lr_slices))]

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
                                                     self.s1_lr_pad_slices, 1)
            s2_crop_slices = self.get_cropped_slices(self.s2_lr_slices,
                                                     self.s2_lr_pad_slices, 1)
            for i, _ in enumerate(self.s1_lr_slices):
                for j, _ in enumerate(self.s2_lr_slices):
                    lr_crop_slice = (s1_crop_slices[i], s2_crop_slices[j],
                                     slice(None), slice(None))
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
            s1_hr_crop_slices = [slice(hr_crop_start, hr_crop_stop)
                                 for _ in range(len(self.s1_lr_slices))]
            s2_hr_crop_slices = [slice(hr_crop_start, hr_crop_stop)
                                 for _ in range(len(self.s2_lr_slices))]

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
                node_slices = []
                for s in self.s_hr_crop_slices:
                    node_slices.append((s[0], s[1], t, slice(None)))
                self._hr_crop_slices.append(node_slices)
        return self._hr_crop_slices

    @property
    def s1_lr_pad_slices(self):
        """List of low resolution spatial slices with padding for first
        spatial dimension"""
        if self._s1_lr_pad_slices is None:
            self._s1_lr_pad_slices = self.get_padded_slices(
                self.s1_lr_slices, self.grid_shape[0], 1,
                padding=self.spatial_pad)
        return self._s1_lr_pad_slices

    @property
    def s2_lr_pad_slices(self):
        """List of low resolution spatial slices with padding for second
        spatial dimension"""
        if self._s2_lr_pad_slices is None:
            self._s2_lr_pad_slices = self.get_padded_slices(
                self.s2_lr_slices, self.grid_shape[1], 1,
                padding=self.spatial_pad)
        return self._s2_lr_pad_slices

    @property
    def s1_lr_slices(self):
        """List of low resolution spatial slices for first spatial dimension
        considering padding on all sides of the spatial raster."""
        ind = slice(0, self.grid_shape[0])
        slices = get_chunk_slices(self.grid_shape[0], self.chunk_shape[0],
                                  index_slice=ind)
        return slices

    @property
    def s2_lr_slices(self):
        """List of low resolution spatial slices for second spatial dimension
        considering padding on all sides of the spatial raster."""
        ind = slice(0, self.grid_shape[1])
        slices = get_chunk_slices(self.grid_shape[1], self.chunk_shape[1],
                                  index_slice=ind)
        return slices

    @property
    def t_lr_slices(self):
        """Low resolution temporal slices"""
        n_tsteps = len(self.dummy_time_index[self.temporal_slice])
        n_chunks = n_tsteps / self.chunk_shape[2]
        n_chunks = int(np.ceil(n_chunks))
        ti_slices = self.dummy_time_index[self.temporal_slice]
        ti_slices = np.array_split(ti_slices, n_chunks)
        ti_slices = [slice(c[0], c[-1] + 1, self.temporal_slice.step)
                     for c in ti_slices]
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

        Returns
        -------
        hr_slices : list
            High resolution slices
        """
        hr_slices = []
        if step is not None:
            step = step * enhancement
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
            dimension and temporal_slice.step = 3 then step=3.

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


class ForwardPassStrategy(InputMixIn, DistributedProcess):
    """Class to prepare data for forward passes through generator.

    A full file list of contiguous times is provided. The corresponding data is
    split into spatiotemporal chunks which can overlap in time and space. These
    chunks are distributed across nodes according to the max nodes input or
    number of temporal chunks. This strategy stores information on these
    chunks, how they overlap, how they are distributed to nodes, and how to
    crop generator output to stich the chunks back togerther.
    """

    def __init__(self, file_paths, model_kwargs, fwp_chunk_shape,
                 spatial_pad, temporal_pad,
                 model_class='Sup3rGan',
                 out_pattern=None,
                 input_handler=None,
                 input_handler_kwargs=None,
                 incremental=True,
                 worker_kwargs=None,
                 exo_kwargs=None,
                 bias_correct_method=None,
                 bias_correct_kwargs=None,
                 max_nodes=None):
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
            data handler class to use for input data. Provide a string name to
            match a class in data_handling.py. If None the correct handler will
            be guessed based on file type and time series properties.
        input_handler_kwargs : dict | None
            Any kwargs for initializing the input_handler class
            :class:`sup3r.preprocessing.data_handling.DataHandler`.
        incremental : bool
            Allow the forward pass iteration to skip spatiotemporal chunks that
            already have an output file (True, default) or iterate through all
            chunks and overwrite any pre-existing outputs (False).
        worker_kwargs : dict | None
            Dictionary of worker values. Can include max_workers,
            pass_workers, output_workers, and ti_workers. Each argument needs
            to be an integer or None.

            The value of `max workers` will set the value of all other worker
            args. If max_workers == 1 then all processes will be serialized. If
            max_workers == None then other worker args will use their own
            provided values.

            `output_workers` is the max number of workers to use for writing
            forward pass output. `pass_workers` is the max number of workers to
            use for performing forward passes on a single node. If 1 then all
            forward passes on chunks distributed to a single node will be run
            in serial. pass_workers=2 is the minimum number of workers required
            to run the ForwardPass initialization and ForwardPass.run_chunk()
            methods concurrently. `ti_workers` is the max number of workers
            used to get the full time index. Doing this is parallel can be
            helpful when there are a large number of input files.
        exo_kwargs : dict | None
            Dictionary of args to pass to ExogenousDataHandler for extracting
            exogenous features such as topography for future multistep foward
            pass
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
        """
        self._input_handler_kwargs = input_handler_kwargs or {}
        target = self._input_handler_kwargs.get('target', None)
        grid_shape = self._input_handler_kwargs.get('shape', None)
        raster_file = self._input_handler_kwargs.get('raster_file', None)
        raster_index = self._input_handler_kwargs.get('raster_index', None)
        temporal_slice = self._input_handler_kwargs.get('temporal_slice',
                                                        slice(None, None, 1))
        InputMixIn.__init__(self, target=target,
                            shape=grid_shape,
                            raster_file=raster_file,
                            raster_index=raster_index,
                            temporal_slice=temporal_slice)

        self.file_paths = file_paths
        self.model_kwargs = model_kwargs
        self.fwp_chunk_shape = fwp_chunk_shape
        self.spatial_pad = spatial_pad
        self.temporal_pad = temporal_pad
        self.model_class = model_class
        self.out_pattern = out_pattern
        self.worker_kwargs = worker_kwargs or {}
        self.exo_kwargs = exo_kwargs or {}
        self.incremental = incremental
        self.bias_correct_method = bias_correct_method
        self.bias_correct_kwargs = bias_correct_kwargs or {}
        self._input_handler_class = None
        self._input_handler_name = input_handler
        self._file_ids = None
        self._hr_lat_lon = None
        self._lr_lat_lon = None
        self._init_handler = None
        self._handle_features = None

        self._single_ts_files = self._input_handler_kwargs.get(
            'single_ts_files', None)
        self.cache_pattern = self._input_handler_kwargs.get('cache_pattern',
                                                            None)
        self.max_workers = self.worker_kwargs.get('max_workers', None)
        self.output_workers = self.worker_kwargs.get('output_workers', None)
        self.pass_workers = self.worker_kwargs.get('pass_workers', None)
        self.ti_workers = self.worker_kwargs.get('ti_workers', None)
        self._worker_attrs = ['pass_workers', 'output_workers', 'ti_workers']
        self.cap_worker_args(self.max_workers)

        model_class = getattr(sup3r.models, self.model_class, None)
        if isinstance(self.model_kwargs, str):
            self.model_kwargs = {'model_dir': self.model_kwargs}

        if model_class is None:
            msg = ('Could not load requested model class "{}" from '
                   'sup3r.models, Make sure you typed in the model class '
                   'name correctly.'.format(self.model_class))
            logger.error(msg)
            raise KeyError(msg)

        model = model_class.load(**self.model_kwargs, verbose=True)
        models = getattr(model, 'models', [model])
        self.s_enhancements = [model.s_enhance for model in models]
        self.t_enhancements = [model.t_enhance for model in models]
        self.s_enhance = np.product(self.s_enhancements)
        self.t_enhance = np.product(self.t_enhancements)
        self.output_features = model.output_features

        self.fwp_slicer = ForwardPassSlicer(self.grid_shape,
                                            self.raw_tsteps,
                                            self.temporal_slice,
                                            self.fwp_chunk_shape,
                                            self.s_enhancements,
                                            self.t_enhancements,
                                            self.spatial_pad,
                                            self.temporal_pad)

        DistributedProcess.__init__(self, max_nodes=max_nodes,
                                    max_chunks=self.fwp_slicer.n_chunks,
                                    incremental=self.incremental)

        self.preflight()

    def preflight(self):
        """Prelight path name formatting and sanity checks"""

        logger.info('Initializing ForwardPassStrategy. '
                    f'Using n_nodes={self.nodes} with '
                    f'n_spatial_chunks={self.fwp_slicer.n_spatial_chunks}, '
                    f'n_temporal_chunks={self.fwp_slicer.n_temporal_chunks}, '
                    f'and n_total_chunks={self.chunks}. '
                    f'{self.chunks / self.nodes} chunks per node on average.')
        logger.info(f'Using max_workers={self.max_workers}, '
                    f'pass_workers={self.pass_workers}, '
                    f'output_workers={self.output_workers}')

        out = self.fwp_slicer.get_temporal_slices()
        self.ti_slices, self.ti_pad_slices = out

        msg = ('Using a padded chunk size '
               f'({self.fwp_chunk_shape[2] + 2 * self.temporal_pad}) '
               f'larger than the full temporal domain ({self.raw_tsteps}). '
               'Should just run without temporal chunking. ')
        if (self.fwp_chunk_shape[2] + 2 * self.temporal_pad
                >= self.raw_tsteps):
            logger.warning(msg)
            warnings.warn(msg)

        hr_data_shape = (self.grid_shape[0] * self.s_enhance,
                         self.grid_shape[1] * self.s_enhance)
        self.gids = np.arange(np.product(hr_data_shape))
        self.gids = self.gids.reshape(hr_data_shape)

        out = self.fwp_slicer.get_spatial_slices()
        self.lr_slices, self.lr_pad_slices, self.hr_slices = out

    # pylint: disable=E1102
    @property
    def init_handler(self):
        """Get initial input handler used for extracting handler features and
        low res grid"""
        if self._init_handler is None:
            out = self.input_handler_class(self.file_paths[0], [],
                                           target=self.target,
                                           shape=self.grid_shape,
                                           worker_kwargs=dict(ti_workers=1))
            self._init_handler = out
        return self._init_handler

    @property
    def lr_lat_lon(self):
        """Get low resolution lat lons for input entire grid"""
        if self._lr_lat_lon is None:
            logger.info('Getting low-resolution grid for full input domain.')
            self._lr_lat_lon = self.init_handler.lat_lon
        return self._lr_lat_lon

    @property
    def handle_features(self):
        """Get list of features available in the source data"""
        if self._handle_features is None:
            if self.single_ts_files:
                self._handle_features = self.init_handler.handle_features
            else:
                hf = self.input_handler_class.get_handle_features(
                    self.file_paths)
                self._handle_features = hf
        return self._handle_features

    @property
    def hr_lat_lon(self):
        """Get high resolution lat lons"""
        if self._hr_lat_lon is None:
            logger.info('Getting high-resolution grid for full output domain.')
            lr_lat_lon = self.lr_lat_lon.copy()
            self._hr_lat_lon = OutputHandler.get_lat_lon(lr_lat_lon,
                                                         self.gids.shape)
        return self._hr_lat_lon

    def get_full_domain(self, file_paths):
        """Get target and grid_shape for largest possible domain"""
        return self.input_handler_class.get_full_domain(file_paths)

    def get_lat_lon(self, file_paths, raster_index, invert_lat=False):
        """Get lat/lon grid for requested target and shape"""
        return self.input_handler_class.get_lat_lon(file_paths, raster_index,
                                                    invert_lat=invert_lat)

    def get_time_index(self, file_paths, max_workers=None, **kwargs):
        """Get time index for source data using DataHandler.get_time_index
        method

        Parameters
        ----------
        file_paths : list
            List of file paths for source data

        Returns
        -------
        time_index : ndarray
            Array of time indices for source data
        """
        return self.input_handler_class.get_time_index(file_paths,
                                                       max_workers=max_workers,
                                                       **kwargs)

    @property
    def file_ids(self):
        """Get file id for each output file

        Returns
        -------
        _file_ids : list
            List of file ids for each output file. Will be used to name output
            files of the form filename_{file_id}.ext
        """
        if not self._file_ids:
            self._file_ids = []
            for i in range(self.fwp_slicer.n_temporal_chunks):
                for j in range(self.fwp_slicer.n_spatial_chunks):
                    file_id = f'{str(i).zfill(6)}_{str(j).zfill(6)}'
                    self._file_ids.append(file_id)
        return self._file_ids

    @property
    def out_files(self):
        """Get output file names for forward pass output

        Returns
        -------
        _out_files : list
            List of output files for forward pass output data
        """
        if self._out_files is None:
            self._out_files = self.get_output_file_names(
                out_files=self.out_pattern, file_ids=self.file_ids)
        return self._out_files

    @property
    def input_type(self):
        """Get input data type

        Returns
        -------
        input_type
            e.g. 'nc' or 'h5'
        """
        return get_source_type(self.file_paths)

    @property
    def output_type(self):
        """Get output data type

        Returns
        -------
        output_type
            e.g. 'nc' or 'h5'
        """
        return get_source_type(self.out_pattern)

    @property
    def input_handler_class(self):
        """Get data handler class used to handle input

        Returns
        -------
        _handler_class
            e.g. DataHandlerNC, DataHandlerH5, etc
        """
        if self._input_handler_class is None:
            self._input_handler_class = get_input_handler_class(
                self.file_paths, self._input_handler_name)
        return self._input_handler_class

    @property
    def max_nodes(self):
        """Get the maximum number of nodes that this strategy should distribute
        work to, equal to either the specified max number of nodes or total
        number of temporal chunks"""
        self._max_nodes = (self._max_nodes if self._max_nodes is not None
                           else self.fwp_slicer.n_temporal_chunks)
        return self._max_nodes

    @staticmethod
    def get_output_file_names(out_files, file_ids):
        """Get output file names for each file chunk forward pass

        Parameters
        ----------
        out_files : str
            Output file pattern. Should be of the form
            /<path>/<name>_{file_id}.<ext>. e.g. /tmp/fp_out_{file_id}.h5.
            Each output file will have a unique file_id filled in and the ext
            determines the output type.
        file_ids : list
            List of file ids for each output file. e.g. date range

        Returns
        -------
        list
            List of output file paths
        """
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


class ForwardPass:
    """Class to run forward passes on all chunks provided by the given
    ForwardPassStrategy. The chunks provided by the strategy are all passed
    through the GAN generator to produce high resolution output.
    """

    def __init__(self, strategy, chunk_index=0, node_index=0):
        """Initialize ForwardPass with ForwardPassStrategy. The stragegy
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

        msg = (f'Requested forward pass on chunk_index={chunk_index} > '
               f'n_chunks={strategy.chunks}')
        assert chunk_index <= strategy.chunks, msg

        logger.info(f'Initializing ForwardPass for chunk={chunk_index} '
                    f'(temporal_chunk={self.temporal_chunk_index}, '
                    f'spatial_chunk={self.spatial_chunk_index}). {self.chunks}'
                    f' total chunks for the current node.')

        self.model_kwargs = self.strategy.model_kwargs
        self.model_class = self.strategy.model_class
        model_class = getattr(sup3r.models, self.model_class, None)

        if model_class is None:
            msg = ('Could not load requested model class "{}" from '
                   'sup3r.models, Make sure you typed in the model class '
                   'name correctly.'.format(self.model_class))
            logger.error(msg)
            raise KeyError(msg)

        self.model = model_class.load(**self.model_kwargs, verbose=False)
        self.features = self.model.training_features
        self.output_features = self.model.output_features

        self._file_paths = strategy.file_paths
        self.max_workers = strategy.max_workers
        self.pass_workers = strategy.pass_workers
        self.output_workers = strategy.output_workers
        self.exo_kwargs = strategy.exo_kwargs

        self.exogenous_handler = None
        self.exogenous_data = None
        if self.exo_kwargs:
            exo_features = self.exo_kwargs.get('features', [])
            exo_kwargs = copy.deepcopy(self.exo_kwargs)
            exo_kwargs['target'] = self.target
            exo_kwargs['shape'] = self.shape
            self.features = [f for f in self.features if f not in exo_features]
            self.exogenous_handler = ExogenousDataHandler(**exo_kwargs)
            self.exogenous_data = self.exogenous_handler.data
            shapes = [None if d is None else d.shape
                      for d in self.exogenous_data]
            logger.info('Got exogenous_data of length {} with shapes: {}'
                        .format(len(self.exogenous_data), shapes))

        self.input_handler_class = strategy.input_handler_class

        if strategy.output_type == 'nc':
            self.output_handler_class = OutputHandlerNC
        elif strategy.output_type == 'h5':
            self.output_handler_class = OutputHandlerH5

        input_handler_kwargs = self.update_input_handler_kwargs(strategy)

        logger.info(f'Getting input data for chunk_index={chunk_index}.')
        self.data_handler = self.input_handler_class(**input_handler_kwargs)
        self.data_handler.load_cached_data()
        self.input_data = self.data_handler.data

        self.input_data = self.bias_correct_source_data(self.input_data)

        exo_s_en = self.exo_kwargs.get('s_enhancements', None)
        out = self.pad_source_data(self.input_data, self.pad_width,
                                   self.exogenous_data, exo_s_en)
        self.input_data, self.exogenous_data = out

    def update_input_handler_kwargs(self, strategy):
        """Update the kwargs for the input handler for the current forward pass
        chunk

        Parameters
        ----------
        strategy : ForwardPassStrategy
            ForwardPassStrategy instance with information on data chunks to run
            forward passes on.

        Returns
        -------
        dict
            Updated dictionary of input handler arguments to pass to the
            data handler for the current forward pass chunk
        """
        input_handler_kwargs = copy.deepcopy(strategy._input_handler_kwargs)
        fwp_input_handler_kwargs = dict(
            file_paths=self.file_paths,
            features=self.features,
            target=self.target,
            shape=self.shape,
            temporal_slice=self.temporal_pad_slice,
            raster_file=self.raster_file,
            cache_pattern=self.cache_pattern,
            single_ts_files=self.single_ts_files,
            handle_features=strategy.handle_features,
            val_split=0.0)
        input_handler_kwargs.update(fwp_input_handler_kwargs)
        return input_handler_kwargs

    @property
    def single_ts_files(self):
        """Get whether input files are single time step or not"""
        return self.strategy.single_ts_files

    @property
    def s_enhance(self):
        """Get spatial enhancement factor"""
        return self.strategy.s_enhance

    @property
    def t_enhance(self):
        """Get temporal enhancement factor"""
        return self.strategy.t_enhance

    @property
    def ti_crop_slice(self):
        """Get low-resolution time index crop slice to crop input data time
        index before getting high-resolution time index"""
        return self.strategy.fwp_slicer.t_lr_crop_slices[
            self.temporal_chunk_index]

    @property
    def lr_times(self):
        """Get low-resolution cropped time index to use for getting
        high-resolution time index"""
        return self.data_handler.time_index[self.ti_crop_slice]

    @property
    def hr_lat_lon(self):
        """Get high resolution lat lon for current chunk"""
        return self.strategy.hr_lat_lon[self.hr_slice[0], self.hr_slice[1]]

    @property
    def hr_times(self):
        """Get high resolution times for the current chunk"""
        return self.output_handler_class.get_times(
            self.lr_times, self.t_enhance * len(self.lr_times))

    @property
    def meta(self):
        """Meta data dictionary for the forward pass run (to write to output
        files)."""
        meta_data = {'gan_meta': self.model.meta,
                     'model_kwargs': self.model_kwargs,
                     'model_class': self.model_class,
                     'spatial_enhance': int(self.s_enhance),
                     'temporal_enhance': int(self.t_enhance),
                     'input_files': self.file_paths,
                     'input_features': self.features,
                     'output_features': self.output_features,
                     }
        return meta_data

    @property
    def gids(self):
        """Get gids for the current chunk"""
        return self.strategy.gids[self.hr_slice[0], self.hr_slice[1]]

    @property
    def file_paths(self):
        """Get a list of source filepaths to get data from. This list is
        reduced if there are single timesteps per file."""
        file_paths = self._file_paths
        if self.single_ts_files:
            file_paths = self._file_paths[self.ti_pad_slice]

        return file_paths

    @property
    def temporal_pad_slice(self):
        """Get the low resolution temporal slice including padding."""
        ti_pad_slice = self.ti_pad_slice
        if self.single_ts_files:
            ti_pad_slice = slice(None)
        return ti_pad_slice

    @property
    def lr_padded_slice(self):
        """Get the padded slice argument that can be used to slice the full
        domain source low res data to return just the extent used for the
        current chunk.

        Returns
        -------
        lr_padded_slice : tuple
            Tuple of length four that slices (spatial_1, spatial_2, temporal,
            features) where each tuple entry is a slice object for that axes.
        """
        return self.strategy.lr_pad_slices[self.spatial_chunk_index]

    @property
    def target(self):
        """Get target for current spatial chunk"""
        spatial_slice = self.lr_padded_slice[0], self.lr_padded_slice[1]
        return self.strategy.lr_lat_lon[spatial_slice][-1, 0]

    @property
    def shape(self):
        """Get shape for current spatial chunk"""
        spatial_slice = self.lr_padded_slice[0], self.lr_padded_slice[1]
        return self.strategy.lr_lat_lon[spatial_slice].shape[:-1]

    @property
    def chunks(self):
        """Number of chunks for current node"""
        return len(self.strategy.node_chunks[self.node_index])

    @property
    def spatial_chunk_index(self):
        """Spatial index for the current chunk going through forward pass"""
        return self.chunk_index % self.strategy.fwp_slicer.n_spatial_chunks

    @property
    def temporal_chunk_index(self):
        """Temporal index for the current chunk going through forward pass"""
        return self.chunk_index // self.strategy.fwp_slicer.n_spatial_chunks

    @property
    def out_file(self):
        """Get output file name for the current chunk"""
        return self.strategy.out_files[self.chunk_index]

    @property
    def ti_slice(self):
        """Get ti slice for the current chunk"""
        return self.strategy.ti_slices[self.temporal_chunk_index]

    @property
    def ti_pad_slice(self):
        """Get padded ti slice for the current chunk"""
        return self.strategy.ti_pad_slices[self.temporal_chunk_index]

    @property
    def lr_slice(self):
        """Get lr slice for the current chunk"""
        return self.strategy.lr_slices[self.spatial_chunk_index]

    @property
    def lr_pad_slice(self):
        """Get padded lr slice for the current chunk"""
        return self.strategy.lr_pad_slices[self.spatial_chunk_index]

    @property
    def hr_slice(self):
        """Get hr slice for the current chunk"""
        return self.strategy.hr_slices[self.spatial_chunk_index]

    @property
    def hr_crop_slice(self):
        """Get hr cropping slice for the current chunk"""
        hr_crop_slices = self.strategy.fwp_slicer.hr_crop_slices[
            self.temporal_chunk_index]
        return hr_crop_slices[self.spatial_chunk_index]

    @property
    def lr_crop_slice(self):
        """Get lr cropping slice for the current chunk"""
        lr_crop_slices = self.strategy.fwp_slicer.s_lr_crop_slices
        return lr_crop_slices[self.spatial_chunk_index]

    @property
    def chunk_shape(self):
        """Get shape for the current padded spatiotemporal chunk"""
        return (self.lr_pad_slice[0].stop - self.lr_pad_slice[0].start,
                self.lr_pad_slice[1].stop - self.lr_pad_slice[1].start,
                self.ti_pad_slice.stop - self.ti_pad_slice.start)

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

    @property
    def raster_file(self):
        """Get raster file for the current spatial chunk"""
        raster_file = self.strategy.raster_file
        if raster_file is not None:
            if '{spatial_chunk_index}' not in raster_file:
                raster_file = raster_file.replace(
                    '.txt', '_{spatial_chunk_index}.txt')
            raster_file = raster_file.replace(
                '{spatial_chunk_index}', str(self.spatial_chunk_index))
        return raster_file

    @property
    def pad_width(self):
        """Get padding for the current spatiotemporal chunk

        Returns
        -------
        padding : tuple
            Tuple of tuples with padding width for spatial and temporal
            dimensions. Each tuple includes the start and end of padding for
            that dimension. Ordering is spatial_1, spatial_2, temporal.
        """
        ti_start = self.ti_slice.start or 0
        ti_stop = self.ti_slice.stop or self.strategy.raw_tsteps
        pad_t_start = int(np.maximum(0, (self.strategy.temporal_pad
                                         - ti_start)))
        pad_t_end = int(np.maximum(0, (self.strategy.temporal_pad
                                       + ti_stop - self.strategy.raw_tsteps)))

        s1_start = self.lr_slice[0].start or 0
        s1_stop = self.lr_slice[0].stop or self.strategy.grid_shape[0]
        pad_s1_start = int(np.maximum(0, (self.strategy.spatial_pad
                                          - s1_start)))
        pad_s1_end = int(np.maximum(0, (self.strategy.spatial_pad
                                        + s1_stop
                                        - self.strategy.grid_shape[0])))

        s2_start = self.lr_slice[1].start or 0
        s2_stop = self.lr_slice[1].stop or self.strategy.grid_shape[1]
        pad_s2_start = int(np.maximum(0, (self.strategy.spatial_pad
                                          - s2_start)))
        pad_s2_end = int(np.maximum(0, (self.strategy.spatial_pad
                                        + s2_stop
                                        - self.strategy.grid_shape[1])))
        return ((pad_s1_start, pad_s1_end), (pad_s2_start, pad_s2_end),
                (pad_t_start, pad_t_end))

    @staticmethod
    def pad_source_data(input_data, pad_width, exo_data,
                        exo_s_enhancements, mode='reflect'):
        """Pad the edges of the source data from the data handler.

        Parameters
        ----------
        input_data : np.ndarray
            Source input data from data handler class, shape is:
            (spatial_1, spatial_2, temporal, features)
        spatial_pad : int
            Size of spatial overlap between coarse chunks passed to forward
            passes for subsequent spatial stitching. This overlap will pad both
            sides of the fwp_chunk_shape. Note that the first and last chunks
            in any of the spatial dimension will not be padded.
        pad_width : tuple
            Tuple of tuples with padding width for spatial and temporal
            dimensions. Each tuple includes the start and end of padding for
            that dimension. Ordering is spatial_1, spatial_2, temporal.
        exo_data : None | list
            List of exogenous data arrays for each step of the sup3r resolution
            model. List entries can be None if not exo data is requested for a
            given model step.
        exo_s_enhancements : list
            List of spatial enhancement factors for each step of the sup3r
            resolution model corresponding to the exo_data order.
        mode : str
            Padding mode for np.pad(). Reflect is a good default for the
            convolutional sup3r work.

        Returns
        -------
        out : np.ndarray
            Padded copy of source input data from data handler class, shape is:
            (spatial_1, spatial_2, temporal, features)
        exo_data : list | None
            Padded copy of exo_data input.
        """
        out = np.pad(input_data, (*pad_width, (0, 0)), mode=mode)

        logger.info('Padded input data shape from {} to {} using mode "{}" '
                    'with padding argument: {}'
                    .format(input_data.shape, out.shape, mode, pad_width))

        if exo_data is not None:
            for i, i_exo_data in enumerate(exo_data):
                if i_exo_data is not None:
                    total_s_enhance = exo_s_enhancements[:i + 1]
                    total_s_enhance = [s for s in total_s_enhance
                                       if s is not None]
                    total_s_enhance = np.product(total_s_enhance)
                    exo_pad_width = ((total_s_enhance * pad_width[0][0],
                                      total_s_enhance * pad_width[0][1]),
                                     (total_s_enhance * pad_width[1][0],
                                      total_s_enhance * pad_width[1][1]),
                                     (0, 0))
                    exo_data[i] = np.pad(i_exo_data, exo_pad_width, mode=mode)

        return out, exo_data

    def bias_correct_source_data(self, data):
        """Bias correct data using a method defined by the bias_correct_method
        input to ForwardPassStrategy

        Parameters
        ----------
        data : np.ndarray
            Any source data to be bias corrected, with the feature channel in
            the last axis.

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
                             'using function: {} with kwargs: {}'
                             .format(feature, idf, method, feature_kwargs))

                data[..., idf] = method(data[..., idf], **feature_kwargs)

        return data

    def _prep_exogenous_input(self, chunk_shape):
        """Shape exogenous data according to model type and model steps

        Parameters
        ----------
        chunk_shape : tuple
            Shape of data chunk going through forward pass

        Returns
        -------
        exo_data : list
            List of arrays of exogenous data. If there are 2 spatial
            enhancement steps this is a list of 3 arrays each with the
            appropriate shape based on the enhancement factor
        """
        exo_data = []
        if self.exogenous_data is not None:
            for arr in self.exogenous_data:
                if arr is not None:
                    og_shape = arr.shape
                    arr = np.expand_dims(arr, axis=2)
                    arr = np.repeat(arr, chunk_shape[2], axis=2)

                    target_shape = (arr.shape[0], arr.shape[1], chunk_shape[2],
                                    arr.shape[-1])
                    msg = ('Target shape for exogenous data in forward pass '
                           'chunk was {}, but something went wrong and i '
                           'resized original data shape from {} to {}'
                           .format(target_shape, og_shape, arr.shape))
                    assert arr.shape == target_shape, msg

                exo_data.append(arr)

        return exo_data

    @classmethod
    def _run_generator(cls, data_chunk, hr_crop_slices,
                       model=None, model_kwargs=None, model_class=None,
                       s_enhance=None, t_enhance=None,
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
        exo_data : list | None
            List of arrays of exogenous data for each model step.
            If there are two spatial enhancement steps this is a list of length
            3 with arrays for each intermediate spatial resolution.

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
            msg = ('Forward pass failed on chunk with shape {}.'
                   .format(data_chunk.shape))
            logger.exception(msg)
            raise RuntimeError(msg) from e

        if (s_enhance is not None
                and hi_res.shape[1] != s_enhance * data_chunk.shape[i_lr_s]):
            msg = ('The stated spatial enhancement of {}x did not match '
                   'the low res / high res shapes of {} -> {}'
                   .format(s_enhance, data_chunk.shape, hi_res.shape))
            logger.error(msg)
            raise RuntimeError(msg)

        if (t_enhance is not None
                and hi_res.shape[3] != t_enhance * data_chunk.shape[i_lr_t]):
            msg = ('The stated temporal enhancement of {}x did not match '
                   'the low res / high res shapes of {} -> {}'
                   .format(t_enhance, data_chunk.shape, hi_res.shape))
            logger.error(msg)
            raise RuntimeError(msg)

        return hi_res[0][hr_crop_slices]

    @staticmethod
    def _reshape_data_chunk(model, data_chunk, exo_data):
        """Reshape and transpose data chunk and exogenous data before being
        passed to the sup3r model.

        Parameters
        ----------
        model : Sup3rGan
            Sup3rGan or similar sup3r model
        data_chunk : np.ndarray
            Low resolution data for a single spatiotemporal chunk that is going
            to be passed to the model generate function.
        exo_data : list | None
            Optional exogenous data which can be a list of arrays of exogenous
            inputs to complement data_chunk

        Returns
        -------
        data_chunk : np.ndarray
            Same as input but reshaped to (temporal, spatial_1, spatial_2,
            features) if the model is a spatial-first model or
            (n_obs, spatial_1, spatial_2, temporal, features) if the
            model is spatiotemporal
        exo_data : list | None
            Same reshaping procedure as for data_chunk
        i_lr_t : int
            Axis index for the low-resolution temporal dimension
        i_lr_s : int
            Axis index for the low-resolution spatial_1 dimension
        """

        if exo_data is not None:
            for i, arr in enumerate(exo_data):
                if arr is not None:
                    tp = isinstance(model, sup3r.models.SPATIAL_FIRST_MODELS)
                    tp = tp and (i < len(model.spatial_models))
                    if tp:
                        exo_data[i] = np.transpose(arr, axes=(2, 0, 1, 3))
                    else:
                        exo_data[i] = np.expand_dims(arr, axis=0)

        if isinstance(model, sup3r.models.SPATIAL_FIRST_MODELS):
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
        import_str += 'from reV.pipeline.status import Status;\n'
        import_str += 'from rex import init_logger;\n'
        import_str += ('from sup3r.pipeline.forward_pass '
                       f'import ForwardPassStrategy, {cls.__name__};\n')

        fwps_init_str = get_fun_call_str(ForwardPassStrategy, config)

        node_index = config['node_index']
        log_file = config.get('log_file', None)
        log_level = config.get('log_level', 'INFO')
        log_arg_str = (f'"sup3r", log_level="{log_level}"')
        if log_file is not None:
            log_arg_str += f', log_file="{log_file}"'

        cmd = (f"python -c \'{import_str}\n"
               "t0 = time.time();\n"
               f"logger = init_logger({log_arg_str});\n"
               f"strategy = {fwps_init_str};\n"
               f"{cls.__name__}.run(strategy, {node_index});\n"
               "t_elap = time.time() - t0;\n")

        cmd = BaseCLI.add_status_cmd(config, ModuleName.FORWARD_PASS, cmd)
        cmd += (";\'\n")

        return cmd.replace('\\', '/')

    def _constant_output_check(self, out_data):
        """Check if forward pass output is constant. This can happen when the
        chunk going through the forward pass is too big.

        Parameters
        ----------
        out_data : ndarray
            Forward pass output corresponding to the given chunk index
        """
        for i, f in enumerate(self.output_features):
            msg = (f'All spatiotemporal values are the same for {f} output!')
            if np.all(out_data[0, 0, 0, i] == out_data[..., i]):
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
        """This routine runs forward passes on all spatiotemporal chunks for
        the given node index.

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
        """This routine runs forward passes, on all spatiotemporal chunks for
        the given node index, in serial.

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
            cls._single_proc_run(strategy=strategy, node_index=node_index,
                                 chunk_index=chunk_index)
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
        """This routine runs forward passes, on all spatiotemporal chunks for
        the given node index, with data extraction and forward pass routines in
        parallel.

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
        pool_kws = dict(max_workers=strategy.pass_workers, loggers=['sup3r'])
        with SpawnProcessPool(**pool_kws) as exe:
            now = dt.now()
            for i, chunk_index in enumerate(strategy.node_chunks[node_index]):
                fut = exe.submit(cls._single_proc_run,
                                 strategy=strategy,
                                 node_index=node_index,
                                 chunk_index=chunk_index)
                futures[fut] = {'chunk_index': chunk_index,
                                'start_time': dt.now()}

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
        """This routine runs a forward pass on single spatiotemporal chunk."""

        msg = (f'Running forward pass for chunk_index={self.chunk_index}, '
               f'node_index={self.node_index}, file_paths={self.file_paths}. '
               f'Starting forward pass on chunk_shape={self.chunk_shape} with '
               f'spatial_pad={self.strategy.spatial_pad} and temporal_pad='
               f'{self.strategy.temporal_pad}.')
        logger.info(msg)

        data_chunk = self.input_data
        exo_data = None
        if self.exogenous_data is not None:
            exo_data = self._prep_exogenous_input(data_chunk.shape)

        out_data = self._run_generator(
            data_chunk, hr_crop_slices=self.hr_crop_slice, model=self.model,
            model_kwargs=self.model_kwargs, model_class=self.model_class,
            s_enhance=self.s_enhance, t_enhance=self.t_enhance,
            exo_data=exo_data)

        self._constant_output_check(out_data)

        if self.out_file is not None:
            logger.info(f'Saving forward pass output to {self.out_file}.')
            self.output_handler_class._write_output(
                data=out_data, features=self.model.output_features,
                lat_lon=self.hr_lat_lon, times=self.hr_times,
                out_file=self.out_file, meta_data=self.meta,
                max_workers=self.output_workers, gids=self.gids)
        else:
            return out_data
