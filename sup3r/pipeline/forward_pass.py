# -*- coding: utf-8 -*-
"""
Sup3r forward pass handling module.

@author: bbenton
"""
import json
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
                                                OutputHandlerNC)
from sup3r.utilities.utilities import (get_chunk_slices,
                                       get_source_type,
                                       get_input_handler_class)
from sup3r.utilities import ModuleName

from concurrent.futures import as_completed

np.random.seed(42)

logger = logging.getLogger(__name__)


class ForwardPassSlicer:
    """Get slices for sending data chunks through model."""

    def __init__(self, coarse_shape, time_index, temporal_slice, chunk_shape,
                 s_enhancements, t_enhancements, spatial_pad, temporal_pad):
        """
        Parameters
        ----------
        coarse_shape : tuple
            Shape of full domain for low res data
        time_index : pd.Datetimeindex
            Time index for full temporal domain of low res data
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
            List of factor by which the Sup3rGan model will enhance the spatial
            dimensions of low resolution data
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
        self.s_enhancements = s_enhancements
        self.t_enhancements = t_enhancements
        self.s_enhance = np.product(self.s_enhancements)
        self.t_enhance = np.product(self.t_enhancements)
        self.raw_time_index = time_index
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
            start and end of the raw_time_index.
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
                self.t_lr_slices, len(self.raw_time_index), 1,
                self.temporal_pad, self.temporal_slice.step)
        return self._t_lr_pad_slices

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
        n_tsteps = len(self.raw_time_index[self.temporal_slice])
        n_chunks = n_tsteps / self.chunk_shape[2]
        n_chunks = np.int(np.ceil(n_chunks))
        ti_slices = np.arange(len(self.raw_time_index))[self.temporal_slice]
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


class ForwardPassStrategy(InputMixIn):
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
                 max_workers=None,
                 output_workers=None,
                 exo_kwargs=None,
                 pass_workers=1,
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
        temporal_slice : slice | tuple | list
            Slice defining size of full temporal domain. e.g. If we have 5
            files each with 5 time steps then temporal_slice = slice(None) will
            select all 25 time steps. This can also be a tuple / list with
            length 3 that will be interpreted as slice(*temporal_slice)
        model_class : str
            Name of the sup3r model class for the GAN model to load. The
            default is the basic spatial / spatiotemporal Sup3rGan model. This
            will be loaded from sup3r.models
        target : tuple
            (lat, lon) lower left corner of raster. You should provide
            target+shape or raster_file, or if all three are None the full
            source domain will be used.
        shape : tuple
            (rows, cols) grid size. You should provide target+shape or
            raster_file, or if all three are None the full source domain will
            be used.
        raster_file : str | None
            File for raster_index array for the corresponding target and
            shape. If specified the raster_index will be loaded from the file
            if it exists or written to the file if it does not yet exist.
            If None raster_index will be calculated directly. You should
            provide target+shape or raster_file, or if all three are None the
            full source domain will be used.
        time_chunk_size : int
            Size of chunks to split time dimension into for parallel data
            extraction. If running in serial this can be set to the size
            of the full time index for best performance.
        cache_pattern : str | None
            Pattern for files for saving feature data. e.g.
            file_path_{feature}.pkl Each feature will be saved to a file with
            the feature name replaced in cache_pattern. If not None
            feature arrays will be saved here and not stored in self.data until
            load_cached_data is called. The cache_pattern can also include
            {shape}, {target}, {times} which will help ensure unique cache
            files for complex problems.
        overwrite_cache : bool
            Whether to overwrite cache files storing the computed/extracted
            feature data
        overwrite_ti_cache : bool
            Whether to overwrite time index cache files
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
            Optional kwargs for initializing the input_handler class. For
            example, this could be {'hr_spatial_coarsen': 2} if you wanted to
            artificially coarsen the input data for testing.
        incremental : bool
            Allow the forward pass iteration to skip spatiotemporal chunks that
            already have an output file (True, default) or iterate through all
            chunks and overwrite any pre-existing outputs (False).
        max_workers : int | None
            Providing a value for max workers will be used to set the value of
            extract_workers, compute_workers, output_workers, and load_workers.
            If max_workers == 1 then all processes will be serialized. If None
            extract_workers, compute_workers, load_workers, output_workers will
            use their own provided values.
        extract_workers : int | None
            max number of workers to use for extracting features from source
            data.
        compute_workers : int | None
            max number of workers to use for computing derived features from
            raw features in source data.
        load_workers : int | None
            max number of workers to use for loading cached feature data.
        output_workers : int | None
            max number of workers to use for writing forward pass output.
        pass_workers : int | None
            max number of workers to use for performing forward passes on a
            single node. If 1 then all forward passes on chunks distributed to
            a single node will be run in serial.
        ti_workers : int | None
            max number of workers to use to get full time index. Useful when
            there are many input files each with a single time step. If this is
            greater than one, time indices for input files will be extracted in
            parallel and then concatenated to get the full time index. If input
            files do not all have time indices or if there are few input files
            this should be set to one.
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
        self._i = 0
        self.file_paths = file_paths
        self.model_kwargs = model_kwargs
        self.fwp_chunk_shape = fwp_chunk_shape
        self.spatial_pad = spatial_pad
        self.temporal_pad = temporal_pad
        self.model_class = model_class
        self.out_pattern = out_pattern
        self.max_workers = max_workers
        self.output_workers = output_workers
        self.pass_workers = pass_workers
        self.exo_kwargs = exo_kwargs or {}
        self.incremental = incremental
        self._single_time_step_files = None
        self._input_handler_class = None
        self._input_handler_name = input_handler
        self._max_nodes = max_nodes
        self._input_handler_kwargs = input_handler_kwargs or {}
        self._time_index = None
        self._raw_time_index = None
        self._out_files = None
        self._file_ids = None
        self._time_index_file = None
        self._node_chunks = None
        self.incremental = incremental
        self.bias_correct_method = bias_correct_method
        self.bias_correct_kwargs = bias_correct_kwargs or {}

        self._target = self._input_handler_kwargs.get('target', None)
        self._grid_shape = self._input_handler_kwargs.get('shape', None)
        self.raster_file = self._input_handler_kwargs.get('raster_file', None)
        self.temporal_slice = self._input_handler_kwargs.get('temporal_slice',
                                                             slice(None))
        self.time_chunk_size = self._input_handler_kwargs.get(
            'time_chunk_size', None)
        self.overwrite_cache = self._input_handler_kwargs.get(
            'overwrite_cache', False)
        self.overwrite_ti_cache = self._input_handler_kwargs.get(
            'overwrite_ti_cache', False)
        self.extract_workers = self._input_handler_kwargs.get(
            'extract_workers', None)
        self.compute_workers = self._input_handler_kwargs.get(
            'compute_workers', None)
        self.load_workers = self._input_handler_kwargs.get('load_workers',
                                                           None)
        self.ti_workers = self._input_handler_kwargs.get('ti_workers', None)
        self._cache_pattern = self._input_handler_kwargs.get('cache_pattern',
                                                             None)
        self.cap_worker_args(max_workers)

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

        self.fwp_slicer = ForwardPassSlicer(self.grid_shape,
                                            self.raw_time_index,
                                            self.temporal_slice,
                                            self.fwp_chunk_shape,
                                            self.s_enhancements,
                                            self.t_enhancements,
                                            self.spatial_pad,
                                            self.temporal_pad)

        logger.info('Initializing ForwardPassStrategy for '
                    f'{self.input_file_info}. Using n_nodes={self.nodes} with '
                    f'n_spatial_chunks={self.fwp_slicer.n_spatial_chunks}, '
                    f'n_temporal_chunks={self.fwp_slicer.n_temporal_chunks}, '
                    f'and n_total_chunks={self.chunks}. '
                    f'{self.chunks / self.nodes} chunks per node on average.')
        logger.info(f'Using max_workers={self.max_workers}, '
                    f'extract_workers={self.extract_workers}, '
                    f'compute_workers={self.compute_workers}, '
                    f'pass_workers={self.pass_workers}, '
                    f'load_workers={self.load_workers}, '
                    f'output_workers={self.output_workers}, '
                    f'ti_workers={self.ti_workers}')

        self.preflight()

    @property
    def worker_attrs(self):
        """Get all worker args defined in init"""
        return ['ti_workers', 'compute_workers', 'pass_workers',
                'load_workers', 'output_workers', 'extract_workers']

    def preflight(self):
        """Prelight path name formatting and sanity checks"""
        if self.cache_pattern is not None:
            if '{temporal_chunk_index}' not in self.cache_pattern:
                self.cache_pattern = self.cache_pattern.replace(
                    '.pkl', '_{temporal_chunk_index}.pkl')
            if '{spatial_chunk_index}' not in self.cache_pattern:
                self.cache_pattern = self.cache_pattern.replace(
                    '.pkl', '_{spatial_chunk_index}.pkl')
        if self.raster_file is not None:
            if '{spatial_chunk_index}' not in self.raster_file:
                self.raster_file = self.raster_file.replace(
                    '.txt', '_{spatial_chunk_index}.txt')

        out = self.fwp_slicer.get_temporal_slices()
        self.ti_slices, self.ti_pad_slices = out

        msg = ('Using a padded chunk size '
               f'({self.fwp_chunk_shape[2] + 2 * self.temporal_pad}) '
               'larger than the full temporal domain '
               f'({len(self.raw_time_index)}). Should just run without '
               'temporal chunking. ')
        if (self.fwp_chunk_shape[2] + 2 * self.temporal_pad
                >= len(self.raw_time_index)):
            logger.warning(msg)
            warnings.warn(msg)

        hr_data_shape = (self.grid_shape[0] * self.s_enhance,
                         self.grid_shape[1] * self.s_enhance)
        self.gids = np.arange(np.product(hr_data_shape))
        self.gids = self.gids.reshape(hr_data_shape)

        out = self.fwp_slicer.get_spatial_slices()
        self.lr_slices, self.lr_pad_slices, self.hr_slices = out

        # pylint: disable=E1102
        logger.info('Getting lat/lon for entire forward pass domain.')
        out = self.input_handler_class(self.file_paths[0], [],
                                       target=self.target,
                                       shape=self.grid_shape, ti_workers=1)
        self.lr_lat_lon = out.lat_lon
        self.invert_lat = out.invert_lat
        self.single_time_step_files = self.is_single_ts_files()
        if self.single_time_step_files:
            self.handle_features = out.handle_features
        else:
            hf = self.input_handler_class.get_handle_features(self.file_paths)
            self.handle_features = hf

    def get_full_domain(self, file_paths):
        """Get target and grid_shape for largest possible domain"""
        return self.input_handler_class.get_full_domain(file_paths)

    def get_time_index(self, file_paths, max_workers=None):
        """Get time index for source data

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
                                                       max_workers=max_workers)

    def is_single_ts_files(self):
        """Check if there is a file for each time step, in which case we can
        send a subset of files to the data handler according to ti_pad_slice"""

        if self._single_time_step_files is None:
            t_steps = self.input_handler_class.get_time_index(
                self.file_paths[:1], max_workers=1)
            check = (len(self._file_paths) == len(self.raw_time_index)
                     and t_steps is not None and len(t_steps) == 1)
            self._single_time_step_files = check
        return self._single_time_step_files

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
                    check = (self.out_pattern is not None
                             and '{times}' in self.out_pattern)
                    if check:
                        ti = self.raw_time_index[self.ti_slices[i]]
                        start = str(ti[0]).strip('+').strip('-').strip(':')
                        start = ''.join(start.split(' '))
                        end = str(ti[-1]).strip('+').strip('-').strip(':')
                        end = ''.join(end.split(' '))
                        file_id = f'{start}-{end}_{str(j).zfill(6)}'
                        self._file_ids.append(file_id)
                    else:
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

    def __len__(self):
        """Get the number of nodes that this strategy is distributing to"""
        return self.fwp_slicer.n_chunks

    @property
    def max_nodes(self):
        """Get the maximum number of nodes that this strategy should distribute
        work to, equal to either the specified max number of nodes or total
        number of temporal chunks"""
        nodes = (self._max_nodes if self._max_nodes is not None
                 else self.fwp_slicer.n_temporal_chunks)
        nodes = np.min((nodes, self.chunks))
        return nodes

    @property
    def nodes(self):
        """Get the number of nodes that this strategy should distribute work
        to, equal to either the specified max number of nodes or total number
        of temporal chunks"""
        return len(self.node_chunks)

    @property
    def chunks(self):
        """Get the number of spatiotemporal chunks going through forward pass,
        calculated as the source time index divided by the temporal part of the
        fwp_chunk_shape times the number of spatial chunks"""
        return self.fwp_slicer.n_chunks

    @property
    def node_chunks(self):
        """Get chunked list of spatiotemporal chunk indices that will be
        used to distribute sets of spatiotemporal chunks across nodes. For
        example, if we want to distribute 10 spatiotemporal chunks across 2
        nodes this will return [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]. So the first
        node will be used to run forward passes on the first 5 spatiotemporal
        chunks and the second node will be used for the last 5"""
        if self._node_chunks is None:
            n_chunks = np.min((self.max_nodes, self.chunks))
            self._node_chunks = np.array_split(np.arange(self.chunks),
                                               n_chunks)
        return self._node_chunks

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
        self.meta_data = self.model.meta

        self._file_paths = strategy.file_paths
        self.max_workers = strategy.max_workers
        self.pass_workers = strategy.pass_workers
        self.output_workers = strategy.output_workers
        self.exo_kwargs = strategy.exo_kwargs
        self.single_time_step_files = strategy.single_time_step_files

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

        n_tsteps = len(self.strategy.raw_time_index[self.ti_slice])

        self.hr_data_shape = (self.strategy.s_enhance * self.data_shape[0],
                              self.strategy.s_enhance * self.data_shape[1],
                              self.strategy.t_enhance * n_tsteps,
                              len(self.output_features))

        self.input_handler_class = strategy.input_handler_class

        if strategy.output_type == 'nc':
            self.output_handler_class = OutputHandlerNC
        elif strategy.output_type == 'h5':
            self.output_handler_class = OutputHandlerH5

        input_handler_kwargs = copy.deepcopy(strategy._input_handler_kwargs)
        fwp_input_handler_kwargs = dict(
            file_paths=self.file_paths,
            features=self.features,
            target=self.target,
            shape=self.shape,
            temporal_slice=self.temporal_pad_slice,
            raster_file=self.raster_file,
            cache_pattern=self.cache_pattern,
            time_chunk_size=self.strategy.time_chunk_size,
            overwrite_cache=self.strategy.overwrite_cache,
            max_workers=self.max_workers,
            extract_workers=strategy.extract_workers,
            compute_workers=strategy.compute_workers,
            load_workers=strategy.load_workers,
            ti_workers=strategy.ti_workers,
            handle_features=self.strategy.handle_features,
            val_split=0.0)
        input_handler_kwargs.update(fwp_input_handler_kwargs)
        self.data_handler = self.input_handler_class(**input_handler_kwargs)
        self.data_handler.load_cached_data()
        self.input_data = self.data_handler.data

        self.input_data = self.bias_correct_source_data(self.input_data)

        exo_s_en = self.exo_kwargs.get('s_enhancements', None)
        pad_width = self.get_padding()
        out = self.pad_source_data(self.input_data, pad_width,
                                   self.exogenous_data, exo_s_en)
        self.input_data, self.exogenous_data = out

    @property
    def file_paths(self):
        """Get a list of source filepaths to get data from. This list is
        reduced if there are single timesteps per file."""
        file_paths = self._file_paths
        if self.single_time_step_files:
            file_paths = self._file_paths[self.ti_pad_slice]

        return file_paths

    @property
    def temporal_pad_slice(self):
        """Get the low resolution temporal slice including padding."""
        ti_pad_slice = self.ti_pad_slice
        if self.single_time_step_files:
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
    def data_shape(self):
        """Get data shape for the current padded temporal chunk"""
        return (*self.strategy.grid_shape,
                len(self.strategy.raw_time_index[self.ti_pad_slice]))

    @property
    def chunk_shape(self):
        """Get shape for the current padded spatiotemporal chunk"""
        return (self.lr_pad_slice[0].stop - self.lr_pad_slice[0].start,
                self.lr_pad_slice[1].stop - self.lr_pad_slice[1].start,
                self.data_shape[2])

    @property
    def cache_pattern(self):
        """Get cache pattern for the current chunk"""
        cache_pattern = self.strategy.cache_pattern
        if cache_pattern is not None:
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
            raster_file = raster_file.replace(
                '{spatial_chunk_index}', str(self.spatial_chunk_index))
        return raster_file

    def get_padding(self):
        """Get padding for the current spatiotemporal chunk

        Returns
        -------
        padding : tuple
            Tuple of tuples with padding width for spatial and temporal
            dimensions. Each tuple includes the start and end of padding for
            that dimension. Ordering is spatial_1, spatial_2, temporal.
        """
        ti_start = self.ti_slice.start or 0
        ti_stop = self.ti_slice.stop or len(self.strategy.raw_time_index)
        pad_t_start = int(np.maximum(0, (self.strategy.temporal_pad
                                         - ti_start)))
        pad_t_end = int(np.maximum(0, (self.strategy.temporal_pad
                                       + ti_stop
                                       - len(self.strategy.raw_time_index))))

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
    def forward_pass_chunk(cls, data_chunk, hr_crop_slices,
                           model=None, model_kwargs=None, model_class=None,
                           s_enhance=None, t_enhance=None,
                           exo_data=None):
        """Run forward pass on smallest data chunk. Each chunk has a maximum
        shape given by self.strategy.fwp_chunk_shape.

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

        for i, arr in enumerate(exo_data):
            if arr is not None:
                check = isinstance(model, sup3r.models.SPATIAL_FIRST_MODELS)
                check = check and (i < len(model.spatial_models))
                if check:
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

        job_name = config.get('job_name', None)
        if job_name is not None:
            status_dir = config.get('status_dir', None)
            status_file_arg_str = f'"{status_dir}", '
            status_file_arg_str += f'module="{ModuleName.FORWARD_PASS}", '
            status_file_arg_str += f'job_name="{job_name}", '
            status_file_arg_str += 'attrs=job_attrs'

            cmd += ('job_attrs = {};\n'.format(json.dumps(config)
                                               .replace("null", "None")
                                               .replace("false", "False")
                                               .replace("true", "True")))
            cmd += 'job_attrs.update({"job_status": "successful"});\n'
            cmd += 'job_attrs.update({"time": t_elap});\n'
            cmd += f'Status.make_job_file({status_file_arg_str})'

        cmd += (";\'\n")

        return cmd.replace('\\', '/')

    def _run_single_fwd_pass(self):
        """Run forward pass for current chunk index

        Returns
        -------
        out_data : ndarray
            Forward pass output corresponding to the given chunk index
        """
        data_chunk = self.input_data
        exo_data = []

        if self.exogenous_data is not None:
            exo_data = self._prep_exogenous_input(data_chunk.shape)

        out_data = self.forward_pass_chunk(
            data_chunk, hr_crop_slices=self.hr_crop_slice, model=self.model,
            model_kwargs=self.model_kwargs, model_class=self.model_class,
            s_enhance=self.strategy.s_enhance,
            t_enhance=self.strategy.t_enhance,
            exo_data=exo_data)
        return out_data

    @classmethod
    def incremental_check_run(cls, strategy, node_index, chunk_index):
        """Run forward pass on chunk with incremental check

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
        """
        out_file = strategy.out_files[chunk_index]
        if os.path.exists(out_file) and strategy.incremental:
            logger.info('Not running chunk index {}, output file '
                        'exists: {}'.format(chunk_index, out_file))
        else:
            try:
                fwp = cls(strategy, chunk_index, node_index)
                logger.info(f'Running forward pass for '
                            f'chunk_index={chunk_index}, '
                            f'node_index={node_index}, '
                            f'file_paths={fwp.file_paths}')
                fwp.run_chunk()
            except Exception as e:
                msg = ('Sup3r ForwardPass chunk failed!')
                logger.exception(msg)
                raise RuntimeError(msg) from e

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
        start = dt.now()
        if strategy.pass_workers == 1:
            logger.debug(f'Running forward passes on node {node_index} in '
                         'serial.')
            for i, chunk_index in enumerate(strategy.node_chunks[node_index]):
                now = dt.now()
                cls.incremental_check_run(strategy, node_index, chunk_index)
                mem = psutil.virtual_memory()
                logger.info('Finished forward pass on chunk_index='
                            f'{chunk_index} in {dt.now() - now}. {i + 1} of '
                            f'{len(strategy.node_chunks[node_index])} '
                            'complete. Current memory usage is '
                            f'{mem.used / 1e9:.3f} GB out of '
                            f'{mem.total / 1e9:.3f} GB total.')

        else:
            logger.debug(f'Running forward passes on node {node_index} in '
                         'parallel with pass_workers='
                         f'{strategy.pass_workers}.')
            with SpawnProcessPool(max_workers=strategy.pass_workers) as exe:
                futures = {}
                now = dt.now()
                for chunk_index in strategy.node_chunks[node_index]:
                    future = exe.submit(cls.incremental_check_run,
                                        strategy=strategy,
                                        node_index=node_index,
                                        chunk_index=chunk_index)
                    futures[future] = chunk_index

                logger.info(f'Started {len(futures)} forward passes '
                            f'in {dt.now() - now}.')

                for i, future in enumerate(as_completed(futures)):
                    try:
                        future.result()
                        mem = psutil.virtual_memory()
                        logger.info('Finished forward pass on chunk_index='
                                    f'{futures[future]}. {i + 1} of '
                                    f'{len(futures)} complete. '
                                    'Current memory usage is '
                                    f'{mem.used / 1e9:.3f} GB out of '
                                    f'{mem.total / 1e9:.3f} GB total.')
                    except Exception as e:
                        msg = ('Error running forward pass on chunk '
                               f'{futures[future]}.')
                        logger.exception(msg)
                        raise RuntimeError(msg) from e
        logger.info('Finished forward passes on '
                    f'{len(strategy.node_chunks[node_index])} chunks in '
                    f'{dt.now() - start}')

    def run_chunk(self):
        """This routine runs a forward pass on single spatiotemporal chunk.
        """
        msg = (f'Starting forward pass on data shape {self.chunk_shape} with '
               f'spatial_pad of {self.strategy.spatial_pad} and temporal_pad '
               f'of {self.strategy.temporal_pad}.')
        logger.info(msg)

        out_data = self._run_single_fwd_pass()

        lr_times = self.strategy.raw_time_index[self.ti_slice]
        gids = self.strategy.gids[self.hr_slice[0], self.hr_slice[1]]
        lr_lat_lon = self.strategy.lr_lat_lon[self.lr_slice[0],
                                              self.lr_slice[1]]

        if self.out_file is not None:
            logger.info(f'Saving forward pass output to {self.out_file}.')
            self.output_handler_class.write_output(
                data=out_data,
                features=self.model.output_features,
                low_res_lat_lon=lr_lat_lon,
                low_res_times=lr_times,
                out_file=self.out_file, meta_data=self.meta_data,
                max_workers=self.output_workers,
                gids=gids)
        else:
            return out_data
