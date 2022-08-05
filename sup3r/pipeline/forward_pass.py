# -*- coding: utf-8 -*-
"""
Sup3r forward pass handling module.

@author: bbenton
"""
import json
import psutil
import numpy as np
import logging
from concurrent.futures import as_completed, ThreadPoolExecutor
from datetime import datetime as dt
import os
import warnings

from rex.utilities.fun_utils import get_fun_call_str
from rex import log_mem

import sup3r.models
from sup3r.preprocessing.data_handling import InputMixIn
from sup3r.preprocessing.exogenous_data_handling import ExogenousDataHandler
from sup3r.postprocessing.file_handling import (OutputHandlerH5,
                                                OutputHandlerNC)
from sup3r.utilities.utilities import (get_chunk_slices,
                                       get_source_type,
                                       get_input_handler_class,
                                       estimate_max_workers)
from sup3r.utilities import ModuleName

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

        self._s_lr_slices = None
        self._s_lr_pad_slices = None
        self._t_lr_pad_slices = None
        self._s_hr_slices = None
        self._s_hr_crop_slices = None
        self._t_hr_crop_slices = None
        self._s_exo_slices = None

    def get_spatial_slices(self):
        """Get spatial slices for small data chunks that are passed through
        generator

        Returns
        -------
        lr_slices: list
            List of slices for low res data chunks which have not been padded
        padded_slices : list
            List of slices which have been padded so that high res output
            can be stitched together
        hr_slices : list
            List of slices for high res data corresponding to the
            lr_slices regions
        cropped_slices : list
            List of slices used for cropping generator output
            when forward passes are performed on overlapping chunks
        """
        return (self.s_lr_slices, self.s_lr_pad_slices, self.s_hr_slices,
                self.s_hr_crop_slices)

    def get_temporal_slices(self):
        """Calculate the number of time chunks across the full time index

        Returns
        -------
        t_slices : list
            List of low-res non-padded time index slices. e.g. If
            fwp_chunk_size[2] is 5 then the size of these slices will always
            be 5.
        t_pad_slices : list
            List of low-res padded time index slices. e.g. If fwp_chunk_size[2]
            is 5 the size of these slices will be 15, with exceptions at the
            start and end of the raw_time_index.
        t_crop_slices : list
            List of cropped slices for stitching high res output. e.g. If we
            have a ti_slice with size 5, ti_pad_slice size of 15, and t_enhance
            of 4 then a padded high res slice will have size 60 and a non
            padded high res slice will have size 20.  In order to correctly
            stitch the high res output we have to crop the padded part of these
            high res slices. Thus if a high res slice is slice(40, 100) the
            cropped high res slice would be slice(20, -20). It would not be
            slice(60, 80) because these slices are used to crop chunks of size
            60, with valid time indices in range(0, 60).
        """
        return self.t_lr_slices, self.t_lr_pad_slices, self.t_hr_crop_slices

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
            can be stitched together
        """
        if self._s_lr_pad_slices is None:
            self._s_lr_pad_slices = []
            s1_pad_slices = self.get_padded_slices(self.s1_lr_slices,
                                                   self.grid_shape[0],
                                                   1, self.spatial_pad)
            s2_pad_slices = self.get_padded_slices(self.s2_lr_slices,
                                                   self.grid_shape[1],
                                                   1, self.spatial_pad)
            for i, _ in enumerate(self.s1_lr_slices):
                for j, _ in enumerate(self.s2_lr_slices):
                    pad_slice = (s1_pad_slices[i], s2_pad_slices[j],
                                 slice(None), slice(None))
                    self._s_lr_pad_slices.append(pad_slice)
        return self._s_lr_pad_slices

    @property
    def s_exo_slices(self):
        """Get padded slices for each spatial enhancement step to use for
        indexing exogenous data

        Returns
        -------
        _s_exo_slices : list
            List of lists of slices which have been padded for indexing
            exogenous data for the appropriate spatial enhancement step.  e.g.
            If there are two spatial enhancement steps this will be a list of
            length=3. _s_exo_slices[0] will be a list of padded slices for the
            model input resolution, _s_exo_slices[1] will be a list of slices
            for the resolution after the first spatial enhancement step, and
            _s_exo_slices[2] will be a list of slices for the second spatial
            enhancement step.
        """
        if self._s_exo_slices is None:
            self._s_exo_slices = [self.s_lr_pad_slices]
            for s, _ in enumerate(self.s_enhancements):
                s_enhance = np.product(self.s_enhancements[:s + 1])
                exo_slices = []
                s1_pad_slices = self.get_padded_slices(self.s1_lr_slices,
                                                       self.grid_shape[0],
                                                       s_enhance,
                                                       self.spatial_pad)
                s2_pad_slices = self.get_padded_slices(self.s2_lr_slices,
                                                       self.grid_shape[1],
                                                       s_enhance,
                                                       self.spatial_pad)
                for i, _ in enumerate(self.s1_lr_slices):
                    for j, _ in enumerate(self.s2_lr_slices):
                        pad_slice = (s1_pad_slices[i], s2_pad_slices[j],
                                     slice(None), slice(None))
                        exo_slices.append(pad_slice)
                self._s_exo_slices.append(exo_slices)
        return np.array(self._s_exo_slices)

    @property
    def t_lr_pad_slices(self):
        """Get low res temporal padded slices for distributing time chunks
        across nodes

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
        if self._t_hr_crop_slices is None:
            self._t_hr_crop_slices = self.get_cropped_slices(
                self.t_lr_slices, self.t_lr_pad_slices, self.t_enhance)
        return self._t_hr_crop_slices

    @property
    def s_hr_slices(self):
        """Get high res slices for indexing full generator output array

        Returns
        -------
        _s_hr_slices : list
            List of high res slices
        """
        if self._s_hr_slices is None:
            self._s_hr_slices = []
            s1_slices = self.get_hr_slices(self.s1_lr_slices, self.s_enhance)
            s2_slices = self.get_hr_slices(self.s2_lr_slices, self.s_enhance)
            for i, _ in enumerate(self.s1_lr_slices):
                for j, _ in enumerate(self.s2_lr_slices):
                    hr_slice = (s1_slices[i], s2_slices[j],
                                slice(None), slice(None))
                    self._s_hr_slices.append(hr_slice)
        return self._s_hr_slices

    @property
    def s_hr_crop_slices(self):
        """Get high res cropped slices for cropping generator output

        Returns
        -------
        _s_hr_crop_slices : list
            List of high res cropped slices
        """
        if self._s_hr_crop_slices is None:
            self._s_hr_crop_slices = []
            s1_crop_slices = self.get_cropped_slices(self.s1_lr_slices,
                                                     self.s1_lr_pad_slices,
                                                     self.s_enhance)
            s2_crop_slices = self.get_cropped_slices(self.s2_lr_slices,
                                                     self.s2_lr_pad_slices,
                                                     self.s_enhance)
            for i, _ in enumerate(self.s1_lr_slices):
                for j, _ in enumerate(self.s2_lr_slices):
                    hr_crop_slice = (s1_crop_slices[i], s2_crop_slices[j],
                                     slice(None), slice(None))
                    self._s_hr_crop_slices.append(hr_crop_slice)
        return self._s_hr_crop_slices

    @property
    def s1_lr_pad_slices(self):
        """Low resolution spatial slices with padding for first spatial
        dimension"""
        return self.get_padded_slices(self.s1_lr_slices,
                                      shape=self.grid_shape[0],
                                      enhancement=1,
                                      padding=self.spatial_pad)

    @property
    def s2_lr_pad_slices(self):
        """Low resolution spatial slices with padding for second spatial
        dimension"""
        return self.get_padded_slices(self.s2_lr_slices,
                                      shape=self.grid_shape[1],
                                      enhancement=1,
                                      padding=self.spatial_pad)

    @property
    def s1_lr_slices(self):
        """Low resolution spatial slices for first spatial dimension"""
        return get_chunk_slices(self.grid_shape[0], self.chunk_shape[0])

    @property
    def s2_lr_slices(self):
        """Low resolution spatial slices for second spatial dimension"""
        return get_chunk_slices(self.grid_shape[1], self.chunk_shape[1])

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
        for s in slices:
            start = s.start * enhancement
            stop = s.stop * enhancement
            hr_slices.append(slice(start, stop, step))
        return hr_slices

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
        for ps, s in zip(padded_slices, unpadded_slices):
            start = s.start
            stop = s.stop
            step = s.step or 1
            if start is not None:
                start = enhancement * (s.start - ps.start) // step
            if stop is not None:
                stop = enhancement * (s.stop - ps.stop) // step
            if start is not None and start <= 0:
                start = None
            if stop is not None and stop >= 0:
                stop = None
            cropped_slices.append(slice(start, stop))
        return cropped_slices


class ForwardPassStrategy(InputMixIn):
    """Class to prepare data for forward passes through generator.

    A full file list of contiguous times is provided.  This file list is split
    up into chunks each of which is passed to a different node. These chunks
    can overlap in time. The file list chunk is further split up into temporal
    and spatial chunks which can also overlap. This handler stores information
    on these chunks, how they overlap, and how to crop generator output to
    stich the chunks back togerther.
    """

    def __init__(self, file_paths, model_args, s_enhance, t_enhance,
                 fwp_chunk_shape, spatial_pad, temporal_pad,
                 temporal_slice=slice(None),
                 model_class='Sup3rGan',
                 target=None, shape=None,
                 raster_file=None,
                 time_chunk_size=None,
                 cache_pattern=None,
                 out_pattern=None,
                 overwrite_cache=False,
                 input_handler=None,
                 spatial_coarsen=None,
                 max_workers=None,
                 pass_workers=None,
                 extract_workers=None,
                 compute_workers=None,
                 load_workers=None,
                 output_workers=None,
                 exo_kwargs=None):
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
        model_args : str | list
            Positional arguments to send to `model_class.load(*model_args)` to
            initialize the GAN. Typically this is just the string path to the
            model directory, but can be multiple models or arguments for more
            complex models.
        s_enhance : int
            Factor by which the Sup3rGan model will enhance the spatial
            dimensions of low resolution data
        t_enhance : int
            Factor by which the Sup3rGan model will enhance temporal dimension
            of low resolution data
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
        spatial_coarsen : int | None
            Optional input to coarsen the low-resolution spatial field from the
            file_paths input. This can be used if (for example) you have 2km
            validation data, you can coarsen it with the same factor as
            s_enhance to do a validation study.
        max_workers : int | None
            Providing a value for max workers will be used to set the value of
            pass_workers, extract_workers, compute_workers, output_workers, and
            load_workers.  If max_workers == 1 then all processes will be
            serialized. If None extract_workers, compute_workers, load_workers,
            output_workers, and pass_workers will use their own provided
            values.
        pass_workers : int | None
            max number of workers to use for forward passes. If max_workers ==
            1 then processes will be serialized. If None max_workers will be
            estimated based on memory limits.
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
        exo_kwargs : dict | None
            Dictionary of args to pass to ExogenousDataHandler for extracting
            exogenous features such as topography for future multistep foward
            pass
        """
        if max_workers is not None:
            extract_workers = compute_workers = max_workers
            load_workers = pass_workers = output_workers = max_workers

        self.exo_kwargs = exo_kwargs
        if self.exo_kwargs is not None:
            self.s_enhancements = self.exo_kwargs.get('s_enhancements',
                                                      [s_enhance])
            self.t_enhancements = self.exo_kwargs.get('t_enhancements',
                                                      [t_enhance])
        else:
            self.s_enhancements = [s_enhance]
            self.t_enhancements = [t_enhance]

        self.s_enhance = np.product(self.s_enhancements)
        self.t_enhance = np.product(self.t_enhancements)
        self._i = 0
        self.file_paths = file_paths
        self.model_args = model_args
        self.fwp_chunk_shape = fwp_chunk_shape
        self.spatial_pad = spatial_pad
        self.temporal_pad = temporal_pad
        self.model_class = model_class
        self.out_pattern = out_pattern
        self.raster_file = raster_file
        self.temporal_slice = temporal_slice
        self.time_chunk_size = time_chunk_size
        self.overwrite_cache = overwrite_cache
        self.pass_workers = pass_workers
        self.max_workers = max_workers
        self.extract_workers = extract_workers
        self.compute_workers = compute_workers
        self.load_workers = load_workers
        self.output_workers = output_workers
        self._cache_pattern = cache_pattern
        self._input_handler_name = input_handler
        self._spatial_coarsen = spatial_coarsen
        self._grid_shape = shape
        self._target = target
        self._time_index = None
        self._raw_time_index = None
        self._out_files = None
        self._file_ids = None

        self.fwp_slicer = ForwardPassSlicer(self.grid_shape,
                                            self.raw_time_index,
                                            self.temporal_slice,
                                            self.fwp_chunk_shape,
                                            self.s_enhancements,
                                            self.t_enhancements,
                                            self.spatial_pad,
                                            self.temporal_pad)

        if isinstance(self.model_args, str):
            self.model_args = [self.model_args]

        logger.info('Initializing ForwardPassStrategy for '
                    f'{self.input_file_info}')

        if self.cache_pattern is not None:
            if '{node_index}' not in self.cache_pattern:
                self.cache_pattern = self.cache_pattern.replace(
                    '.pkl', '_{node_index}.pkl')

        out = self.fwp_slicer.get_temporal_slices()
        self.ti_slices, self.ti_pad_slices, self.ti_hr_crop_slices = out

        msg = (f'Using a larger temporal_pad {temporal_pad} than '
               f'temporal_chunk_size {fwp_chunk_shape[2]}.')
        if temporal_pad > fwp_chunk_shape[2]:
            logger.warning(msg)
            warnings.warn(msg)

        msg = (f'Using a larger spatial_pad {spatial_pad} than '
               f'spatial_chunk_size {fwp_chunk_shape[:2]}.')
        if any(spatial_pad > sc for sc in fwp_chunk_shape[:2]):
            logger.warning(msg)
            warnings.warn(msg)

        msg = ('Using a padded chunk size '
               f'({fwp_chunk_shape[2] + 2 * temporal_pad}) '
               'larger than the full temporal domain '
               f'({len(self.raw_time_index)}). Should just run without '
               'temporal chunking. ')
        if (fwp_chunk_shape[2] + 2 * temporal_pad
                >= len(self.raw_time_index)):
            logger.warning(msg)
            warnings.warn(msg)

    def get_full_domain(self, file_paths):
        """Get target and grid_shape for largest possible domain"""
        return self.input_handler_class.get_full_domain(file_paths)

    def get_time_index(self, file_paths):
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
        return self.input_handler_class.get_time_index(file_paths)

    @property
    def file_ids(self):
        """Get file id for each output file

        Returns
        -------
        _file_ids : list
            List of file ids for each output file. Will be used to name output
            files of the form filename_{file_id}.ext
        """
        if self._file_ids is None:
            n_chunks = len(self.time_index)
            n_chunks /= self.fwp_chunk_shape[2]
            n_chunks = np.int(np.ceil(n_chunks))
            self._file_ids = []
            for i in range(n_chunks):
                check = (self.out_pattern is not None
                         and '{times}' in self.out_pattern)
                if check:
                    ti = self.raw_time_index[self.ti_slices[i]]
                    start = str(ti[0]).strip('+').strip('-').strip(':')
                    start = ''.join(start.split(' '))
                    end = str(ti[-1]).strip('+').strip('-').strip(':')
                    end = ''.join(end.split(' '))
                    self._file_ids.append(f'{start}-{end}')
                else:
                    self._file_ids.append(str(i).zfill(5))
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
        hclass = get_input_handler_class(self.file_paths,
                                         self._input_handler_name)
        return hclass

    def get_node_kwargs(self, node_index):
        """Get node specific variables given an associated index

        Parameters
        ----------
        node_index : int
            Index to select node specific variables. This index selects the
            corresponding file set, cropped_file_slice, padded_file_slice,
            and sets of padded/overlapping/cropped spatial slices for spatial
            chunks

        Returns
        -------
        kwargs : dict
            Dictionary containing the node specific variables
        """

        if node_index >= len(self.file_ids):
            msg = (f'Index is out of bounds. There are {len(self.file_ids)} '
                   f'file chunks and the index requested was {node_index}.')
            raise ValueError(msg)

        out_file = self.out_files[node_index]
        ti_pad_slice = self.ti_pad_slices[node_index]
        ti_slice = self.ti_slices[node_index]
        ti_hr_crop_slice = self.ti_hr_crop_slices[node_index]
        data_shape = (self.grid_shape[0], self.grid_shape[1],
                      len(self.raw_time_index[ti_pad_slice]))
        cache_pattern = (
            None if self.cache_pattern is None
            else self.cache_pattern.replace('{node_index}', str(node_index)))

        out = self.fwp_slicer.get_spatial_slices()
        lr_slices, lr_pad_slices = out[:2]
        hr_slices, hr_crop_slices = out[2:]

        chunk_shape = (lr_pad_slices[0][0].stop - lr_pad_slices[0][0].start,
                       lr_pad_slices[0][1].stop - lr_pad_slices[0][1].start,
                       data_shape[2])

        kwargs = dict(file_paths=self.file_paths,
                      out_file=out_file,
                      ti_pad_slice=ti_pad_slice,
                      ti_slice=ti_slice,
                      ti_hr_crop_slice=ti_hr_crop_slice,
                      lr_slices=lr_slices,
                      lr_pad_slices=lr_pad_slices,
                      hr_slices=hr_slices,
                      hr_crop_slices=hr_crop_slices,
                      data_shape=data_shape,
                      chunk_shape=chunk_shape,
                      cache_pattern=cache_pattern,
                      node_index=node_index,
                      max_workers=self.max_workers,
                      pass_workers=self.pass_workers,
                      extract_workers=self.extract_workers,
                      compute_workers=self.compute_workers,
                      load_workers=self.load_workers,
                      output_workers=self.output_workers,
                      exo_kwargs=self.exo_kwargs,
                      exo_slices=self.fwp_slicer.s_exo_slices)

        return kwargs

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        """Iterate over all file chunks and select node specific variables

        Returns
        -------
        kwargs : dict
            Dictionary storing kwargs for ForwardPass initialization from
            config

        Raises
        ------
        StopIteration
            Stops iteration after reaching last file chunk
        """

        if self._i < len(self.ti_slices):
            kwargs = self.get_node_kwargs(self._i)
            self._i += 1
            return kwargs
        else:
            raise StopIteration

    @property
    def nodes(self):
        """Get the number of nodes that this strategy should distribute work
        to, calculated as the source time index divided by the temporal part of
        the fwp_chunk_shape"""
        return len(self.ti_slices)

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
                out_files = out_files[:-1] + '_{file_id}' + out_files[-1]
            dirname = os.path.dirname(out_files)
            if not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
            for file_id in file_ids:
                out_file = out_files.format(file_id=file_id)
                out_file_list.append(out_file)
        else:
            out_file_list = [None] * len(file_ids)
        return out_file_list


class ForwardPass:
    """Class to run forward passes on all chunks provided by the given
    ForwardPassStrategy. The chunks provided by the strategy are all passed
    through the GAN generator to produce high resolution output.
    """

    def __init__(self, strategy, node_index=0):
        """Initialize ForwardPass with ForwardPassStrategy. The stragegy
        provides the data chunks to run forward passes on

        Parameters
        ----------
        strategy : ForwardPassStrategy
            ForwardPassStrategy instance with information on data chunks to run
            forward passes on.
        node_index : int
            Index used to select subset of full file list on which to run
            forward passes on a single node.
        """

        logger.info(f'Initializing ForwardPass for node={node_index}')
        self.strategy = strategy
        self.model_args = self.strategy.model_args
        self.model_class = self.strategy.model_class
        model_class = getattr(sup3r.models, self.model_class, None)

        if model_class is None:
            msg = ('Could not load requested model class "{}" from '
                   'sup3r.models, Make sure you typed in the model class '
                   'name correctly.'.format(self.model_class))
            logger.error(msg)
            raise KeyError(msg)

        self.model = model_class.load(*self.model_args, verbose=True)
        self.features = self.model.training_features
        self.output_features = self.model.output_features
        self.meta_data = self.model.meta
        self.node_index = node_index

        kwargs = strategy.get_node_kwargs(node_index)

        self.file_paths = kwargs['file_paths']
        self.out_file = kwargs['out_file']
        self.ti_slice = kwargs['ti_slice']
        self.ti_pad_slice = kwargs['ti_pad_slice']
        self.ti_hr_crop_slice = kwargs['ti_hr_crop_slice']
        self.lr_slices = kwargs['lr_slices']
        self.lr_pad_slices = kwargs['lr_pad_slices']
        self.hr_slices = kwargs['hr_slices']
        self.hr_crop_slices = kwargs['hr_crop_slices']
        self.data_shape = kwargs['data_shape']
        self.chunk_shape = kwargs['chunk_shape']
        self.cache_pattern = kwargs['cache_pattern']
        self.max_workers = kwargs['max_workers']
        self.pass_workers = kwargs['pass_workers']
        self.extract_workers = kwargs['extract_workers']
        self.compute_workers = kwargs['compute_workers']
        self.load_workers = kwargs['load_workers']
        self.output_workers = kwargs['output_workers']
        self.exo_kwargs = kwargs['exo_kwargs']
        self.exo_slices = kwargs['exo_slices']

        self.exo_features = self.exo_kwargs.get('features', [])
        self.features = [f for f in self.features
                         if f not in self.exo_features]

        fwp_out_shape = (*self.data_shape, len(self.output_features))
        fwp_out_mem = self.strategy.s_enhance**2 * self.strategy.t_enhance
        fwp_out_mem *= 4 * np.product(fwp_out_shape)
        mem = psutil.virtual_memory()
        msg = (f'Full size ({fwp_out_shape}) of forward pass output '
               f'({fwp_out_mem / 1e9} GB) is too large to hold in memory. '
               'Run with smaller fwp_chunk_shape[2] or spatial extent.')
        if mem.total < fwp_out_mem:
            logger.warning(msg)
            log_mem(logger)

        self.input_handler_class = strategy.input_handler_class

        if strategy.output_type == 'nc':
            self.output_handler_class = OutputHandlerNC
        elif strategy.output_type == 'h5':
            self.output_handler_class = OutputHandlerH5

        self.data_handler = self.input_handler_class(
            self.file_paths, self.features, target=self.strategy.target,
            shape=self.strategy.grid_shape, temporal_slice=self.ti_pad_slice,
            raster_file=self.strategy.raster_file,
            cache_pattern=self.cache_pattern,
            time_chunk_size=self.strategy.time_chunk_size,
            overwrite_cache=self.strategy.overwrite_cache,
            val_split=0.0,
            hr_spatial_coarsen=self.strategy._spatial_coarsen,
            max_workers=self.max_workers,
            extract_workers=self.extract_workers,
            compute_workers=self.compute_workers,
            load_workers=self.load_workers)

        self.data_handler.load_cached_data()

        n_tsteps = len(self.data_handler.raw_time_index[self.ti_slice])
        self.hr_data_shape = (self.strategy.s_enhance * self.data_shape[0],
                              self.strategy.s_enhance * self.data_shape[1],
                              self.strategy.t_enhance * n_tsteps,
                              len(self.output_features))
        self.data = np.zeros(self.hr_data_shape, dtype=np.float32)

        if self.exo_kwargs is not None:
            self.exo_handler = ExogenousDataHandler(**self.exo_kwargs)
            self.exogenous_data = self.exo_handler.data
        else:
            self.exogenous_data = None

    @property
    def pass_workers(self):
        """Get estimate for max pass workers based on memory usage"""
        proc_mem = 8 * np.product(self.strategy.fwp_chunk_shape)
        proc_mem *= self.strategy.s_enhance**2 * self.strategy.t_enhance
        n_procs = len(self.hr_slices)
        max_workers = estimate_max_workers(self._pass_workers,
                                           proc_mem, n_procs)
        return max_workers

    @pass_workers.setter
    def pass_workers(self, pass_workers):
        """Update pass workers value"""
        self._pass_workers = pass_workers

    def _prep_exogenous_input(self, chunk_shape, exogenous_slices, model):
        """Shape exogenous data according to model type and model steps

        Parameters
        ----------
        chunk_shape : tuple
            Shape of data chunk going through forward pass
        exogenous_slices : list
            List of padded slices to index exogenous data for each model step.
            If there are two spatial enhancement steps this is a list of length
            3 with padded slices for each intermediate spatial resolution.
        model : MultiStepGan
            Model instance used to determine number of spatial enhancement
            steps

        Returns
        -------
        exo_data : list
            List of arrays of exogenous data. If there are 2 spatial
            enhancement steps this is a list of 3 arrays each with the
            appropriate shape based on the enhancement factor
        """
        if self.exogenous_data is not None:
            exo_shape = (*self.exogenous_data[0].shape[:2],
                         chunk_shape[2],
                         self.exogenous_data[0].shape[-1])
            exo_data = [np.resize(arr, exo_shape)[tuple(s)] for arr, s
                        in zip(self.exogenous_data, exogenous_slices)]
        else:
            exo_data = []
        if isinstance(model, sup3r.models.SpatialThenTemporalGan):
            for i, arr in enumerate(exo_data):
                if i < len(model.spatial_models):
                    exo_data[i] = np.transpose(arr, axes=(2, 0, 1, 3))
                else:
                    exo_data[i] = np.expand_dims(arr, axis=0)
        else:
            exo_data = [np.expand_dims(arr, axis=0) for arr in exo_data]
        return exo_data

    def forward_pass_chunk(self, lr_slices, hr_slices, hr_crop_slices,
                           model=None, model_args=None, model_class=None,
                           s_enhance=None, t_enhance=None,
                           exogenous_slices=None):
        """Run forward pass on smallest data chunk. Each chunk has a maximum
        shape given by self.strategy.fwp_chunk_shape.

        Parameters
        ----------
        lr_slices : list
            List of slices for extracting padded region from low res source
            data.
        hr_slices : list
            List of high res slices used for placing correct forward pass
            output into final data array.
        hr_crop_slices : list
            List of slices for extracting cropped region of interest from
            output. Output can include an extra overlapping boundary to
            reduce chunking error. The cropping cuts off this padded region
            before stitching chunks.
        model : Sup3rGan
            A loaded Sup3rGan model (any model imported from sup3r.models).
            You need to provide either model or (model_args and model_class)
        model_args : str | list
            Positional arguments to send to `model_class.load(*model_args)` to
            initialize the GAN. Typically this is just the string path to the
            model directory, but can be multiple models or arguments for more
            complex models.
            You need to provide either model or (model_args and model_class)
        model_class : str
            Name of the sup3r model class for the GAN model to load. The
            default is the basic spatial / spatiotemporal Sup3rGan model. This
            will be loaded from sup3r.models
            You need to provide either model or (model_args and model_class)
        model_path : str
            Path to file for Sup3rGan used to generate high resolution
            data
        exogenous_slices : list
            List of padded slices to index exogenous data for each model step.
            If there are two spatial enhancement steps this is a list of length
            3 with padded slices for each intermediate spatial resolution.

        Returns
        -------
        ndarray
            High resolution data generated by GAN
        """

        if model is None:
            msg = 'If model not provided, model_args and model_class must be'
            assert model_args is not None, msg
            assert model_class is not None, msg
            model_class = getattr(sup3r.models, model_class)
            model = model_class.load(*model_args, verbose=False)

        data_chunk = self.data_handler.data[lr_slices]
        exo_data = self._prep_exogenous_input(data_chunk.shape,
                                              exogenous_slices, model)
        if isinstance(model, sup3r.models.SpatialThenTemporalGan):
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
            msg = ('Forward pass failed on chunk with low-res slices {} '
                   'and high-res slices {}.'.format(lr_slices, hr_slices))
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

        data = hi_res[0][hr_crop_slices][..., self.ti_hr_crop_slice, :]
        self.data[hr_slices] = data

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

        fwp_arg_str = f'strategy, node_index={node_index}'

        log_file = config.get('log_file', None)
        log_level = config.get('log_level', 'INFO')
        log_arg_str = (f'"sup3r", log_level="{log_level}"')
        if log_file is not None:
            log_arg_str += f', log_file="{log_file}"'

        cmd = (f"python -c \'{import_str}\n"
               "t0 = time.time();\n"
               f"logger = init_logger({log_arg_str});\n"
               f"strategy = {fwps_init_str};\n"
               f"fwp = {cls.__name__}({fwp_arg_str});\n"
               "fwp.run();\n"
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

    def _run_serial(self):
        """Run forward passes in serial"""
        logger.info('Starting serial iteration through forward pass chunks.')
        zip_iter = zip(self.hr_slices, self.lr_pad_slices, self.hr_crop_slices)
        for i, (sh, slp, shc) in enumerate(zip_iter):
            self.forward_pass_chunk(
                lr_slices=slp, hr_slices=sh, hr_crop_slices=shc,
                model=self.model, model_args=self.model_args,
                model_class=self.model_class,
                s_enhance=self.strategy.s_enhance,
                t_enhance=self.strategy.t_enhance,
                exogenous_slices=self.exo_slices[:, i])

            logger.debug('Coarse data chunks being passed to model '
                         'with shape {} which is slice {} of full shape {}.'
                         .format(self.data_handler.data[slp].shape, slp,
                                 self.data_handler.data.shape))

            mem = psutil.virtual_memory()
            logger.info(f'{i+1} out of {len(self.hr_slices)} '
                        'forward pass chunks completed. '
                        'Memory utilization is '
                        f'{mem.used / 1e9:.3f} GB out of '
                        f'{mem.total / 1e9:.3f} GB '
                        f'total ({100*mem.used / mem.total:.1f}% used)')

    def _run_parallel(self, max_workers=None):
        """Run forward passes in parallel"""
        futures = {}
        now = dt.now()
        logger.info('Starting thread pool with {} workers'
                    .format(max_workers))
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            for i, (sh, slp, shc) in enumerate(zip(
                    self.hr_slices, self.lr_pad_slices,
                    self.hr_crop_slices)):
                future = exe.submit(
                    self.forward_pass_chunk, lr_slices=slp, hr_slices=sh,
                    hr_crop_slices=shc, model_args=self.model_args,
                    model_class=self.model_class,
                    s_enhance=self.strategy.s_enhance,
                    t_enhance=self.strategy.t_enhance,
                    exogenous_slices=self.exo_slices[:, i])
                futures[future] = i

                logger.debug('Coarse data chunks being passed to model '
                             'with shape {} which is slice {} '
                             'of full shape {}.'
                             .format(self.data_handler.data[slp].shape, slp,
                                     self.data_handler.data.shape))

            logger.info('Started forward pass for '
                        f'{len(self.hr_slices)} chunks in '
                        f'{dt.now() - now}.')

            for i, future in enumerate(as_completed(futures)):
                try:
                    future.result()
                except Exception as e:
                    msg = ('Error doing forward pass for chunk '
                           f'{futures[future]}.')
                    logger.exception(msg)
                    raise RuntimeError(msg) from e

                mem = psutil.virtual_memory()
                logger.info(f'{i+1} out of {len(futures)} '
                            'forward pass chunks completed. '
                            'Memory utilization is '
                            f'{mem.used / 1e9:.3f} GB out of '
                            f'{mem.total / 1e9:.3f} GB '
                            f'total ({100*mem.used/mem.total:.1f}% used)')

    def run(self):
        """ForwardPass is initialized with a file_slice_index. This index
        selects a file subset from the full file list in ForwardPassStrategy.
        This routine runs forward passes on all data chunks associated with
        this file subset.
        """
        msg = (f'Starting forward passes on data shape {self.data_shape}. '
               f'Using {len(self.lr_slices)} chunks each with shape of '
               f'{self.chunk_shape}, spatial_pad of '
               f'{self.strategy.spatial_pad} and temporal_pad of '
               f'{self.strategy.temporal_pad}.')
        logger.info(msg)
        max_workers = self.pass_workers
        if max_workers == 1:
            self._run_serial()
        else:
            self._run_parallel(max_workers)

        logger.info('All forward passes are complete.')

        if self.out_file is not None:
            logger.info(f'Saving forward pass output to {self.out_file}.')
            self.output_handler_class.write_output(
                data=self.data, features=self.data_handler.output_features,
                low_res_lat_lon=self.data_handler.lat_lon,
                low_res_times=self.data_handler.raw_time_index[self.ti_slice],
                out_file=self.out_file, meta_data=self.meta_data,
                max_workers=self.output_workers)
        else:
            return self.data
