# -*- coding: utf-8 -*-
"""
Sup3r forward pass handling module.

@author: bbenton
"""
import psutil
import numpy as np
import logging
from concurrent.futures import as_completed
from datetime import datetime as dt
import os
import warnings

from rex.utilities.execution import SpawnProcessPool
from rex.utilities.fun_utils import get_fun_call_str

import sup3r.models
import sup3r.preprocessing.data_handling
from sup3r.preprocessing.data_handling import (DataHandlerH5,
                                               DataHandlerNC,
                                               DataHandlerNCforCC,
                                               InputMixIn)
from sup3r.postprocessing.file_handling import (OutputHandlerH5,
                                                OutputHandlerNC)
from sup3r.utilities.utilities import (get_chunk_slices,
                                       get_source_type,
                                       is_time_series,
                                       estimate_max_workers)
from sup3r.utilities import ModuleName

np.random.seed(42)

logger = logging.getLogger(__name__)


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
                 fwp_chunk_shape, spatial_overlap, temporal_overlap,
                 model_class='Sup3rGan',
                 target=None, shape=None,
                 temporal_slice=slice(None),
                 raster_file=None,
                 time_chunk_size=None,
                 cache_pattern=None,
                 out_pattern=None,
                 overwrite_cache=False,
                 handler_class=None,
                 max_workers=None,
                 pass_workers=None,
                 extract_workers=None,
                 compute_workers=None,
                 load_workers=None,
                 output_workers=None):
        """Use these inputs to initialize data handlers on different nodes and
        to define the size of the data chunks that will be passed through the
        generator.

        Parameters
        ----------
        file_paths : list | str
            A list of files to extract raster data from. Each file must have
            the same number of timesteps. Can also pass a string with a
            unix-style file path which will be passed through glob.glob
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
            If temporal_overlap / spatial_overlap are non zero the chunk sent
            to the generator can be bigger than this shape. If running in
            serial set this equal to the shape of the full spatiotemporal data
            volume for best performance.
        spatial_overlap : int
            Size of spatial overlap between coarse chunks passed to forward
            passes for subsequent spatial stitching. This overlap will pad both
            sides of the fwp_chunk_shape. Note that the first and last chunks
            in any of the spatial dimension will not be padded.
        temporal_overlap : int
            Size of temporal overlap between coarse chunks passed to forward
            passes for subsequent temporal stitching. This overlap will pad
            both sides of the fwp_chunk_shape. Note that the first and last
            chunks in the temporal dimension will not be padded.
        model_class : str
            Name of the sup3r model class for the GAN model to load. The
            default is the basic spatial / spatiotemporal Sup3rGan model. This
            will be loaded from sup3r.models
        target : tuple
            (lat, lon) lower left corner of raster. Either need target+shape or
            raster_file.
        shape : tuple
            (rows, cols) grid size. Either need target+shape or raster_file.
        temporal_slice : slice
            Slice defining size of full temporal domain. e.g. If we have 5
            files each with 5 time steps then temporal_slice = slice(None) will
            select all 25 time steps.
        raster_file : str | None
            File for raster_index array for the corresponding target and
            shape. If specified the raster_index will be loaded from the file
            if it exists or written to the file if it does not yet exist.
            If None raster_index will be calculated directly. Either need
            target+shape or raster_file.
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
            determines the output type. If None then data will be returned in
            an array and not saved.
        output_type : str
            Either nc (netcdf) or h5. This selects the extension for output
            files and determines which output handler to use when writing the
            results of the forward passes
        handler_class : str
            data handler class to use for input data. Provide a string name to
            match a class in data_handling.py. If None the correct handler will
            be guessed based on file type and time series properties.
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
        """
        if max_workers is not None:
            extract_workers = compute_workers = max_workers
            load_workers = pass_workers = output_workers = max_workers

        self._i = 0
        self.file_paths = file_paths
        self.model_args = model_args
        self.t_enhance = t_enhance
        self.s_enhance = s_enhance
        self.fwp_chunk_shape = fwp_chunk_shape
        self.spatial_overlap = spatial_overlap
        self.temporal_overlap = temporal_overlap
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
        self._handler_class = handler_class
        self._grid_shape = shape
        self._target = target
        self._time_index = None
        self._raw_time_index = None
        self._out_files = None
        self._file_ids = None

        if isinstance(self.model_args, str):
            self.model_args = [self.model_args]

        logger.info('Initializing ForwardPassStrategy for '
                    f'{self.input_file_info}')

        if self.cache_pattern is not None:
            if '{node_index}' not in self.cache_pattern:
                self.cache_pattern = self.cache_pattern.replace(
                    '.pkl', '_{node_index}.pkl')

        out = self.get_time_slices(fwp_chunk_shape, temporal_overlap)
        self.ti_slices, self.ti_pad_slices, self.ti_hr_crop_slices = out

        msg = (f'Using a larger temporal_overlap {temporal_overlap} than '
               f'temporal_chunk_size {fwp_chunk_shape[2]}.')
        if temporal_overlap > fwp_chunk_shape[2]:
            logger.warning(msg)
            warnings.warn(msg)

        msg = (f'Using a larger spatial_overlap {spatial_overlap} than '
               f'spatial_chunk_size {fwp_chunk_shape[:2]}.')
        if any(spatial_overlap > sc for sc in fwp_chunk_shape[:2]):
            logger.warning(msg)
            warnings.warn(msg)

        msg = ('Using a padded chunk size '
               f'{fwp_chunk_shape[2] + 2 * temporal_overlap} '
               'larger than the full temporal domain '
               f'{len(self.raw_time_index)}. Should just run without temporal '
               'chunking. ')
        if (fwp_chunk_shape[2] + 2 * temporal_overlap
                >= len(self.raw_time_index)):
            logger.warning(msg)
            warnings.warn(msg)

    def get_full_domain(self, file_paths):
        """Get target and grid_shape for largest possible domain"""
        return self.data_handler_class.get_full_domain(file_paths)

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
        return self.data_handler_class.get_time_index(file_paths)

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
            self._file_ids = [str(fid).zfill(5) for fid in range(n_chunks)]
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
    def data_handler_class(self):
        """Get data handler class used to handle input

        Returns
        -------
        _handler_class
            e.g. DataHandlerNC, DataHandlerH5, etc
        """
        if self._handler_class is None:
            time_series_files = is_time_series(self.file_paths)
            if self.input_type == 'nc':
                self._handler_class = DataHandlerNC
                if not time_series_files:
                    self._handler_class = DataHandlerNCforCC
            elif self.input_type == 'h5':
                self._handler_class = DataHandlerH5
        elif isinstance(self._handler_class, str):
            out = getattr(sup3r.preprocessing.data_handling,
                          self._handler_class, None)
            self._handler_class = out
            if out is None:
                msg = ('Could not find requested data handler class '
                       f'"{self._handler_class}" in '
                       'sup3r.preprocessing.data_handling.')
                logger.error(msg)
                raise KeyError(msg)
        return self._handler_class

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
                      len(self.time_index[ti_pad_slice]))
        cache_pattern = (
            None if self.cache_pattern is None
            else self.cache_pattern.replace('{node_index}', str(node_index)))

        out = self.get_spatial_slices(data_shape=data_shape)
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
                      output_workers=self.output_workers)

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

    def get_time_slices(self, fwp_chunk_size, time_overlap):
        """Calculate the number of time chunks across the full time index

        Parameters
        ----------
        fwp_chunk_size : tuple
            Shape of data chunks passed to generator
        time_overlap : int
            Size of temporal overlap between time chunks

        Returns
        -------
        ti_chunks : list
            List of time index slices
        ti_pad_chunks : list
            List of padded time index slices
        ti_hr_crop_chunks : list
            List of cropped chunks for stitching high res output
        """
        n_chunks = len(self.time_index)
        n_chunks /= fwp_chunk_size[2]
        n_chunks = np.int(np.ceil(n_chunks))
        ti_chunks = np.arange(len(self.time_index))
        ti_chunks = np.array_split(ti_chunks, n_chunks)
        ti_pad_chunks = []
        for i, chunk in enumerate(ti_chunks):
            if len(ti_chunks) > 1:
                if i == 0:
                    tmp = np.concatenate([chunk, ti_chunks[1][:time_overlap]])
                elif i == len(ti_chunks) - 1:
                    tmp = np.concatenate([ti_chunks[-2][-time_overlap:],
                                          chunk])
                else:
                    tmp = np.concatenate([ti_chunks[i - 1][-time_overlap:],
                                          chunk,
                                          ti_chunks[i + 1][:time_overlap]])
            else:
                tmp = chunk
            ti_pad_chunks.append(tmp)

        ti_chunks = [slice(chunk[0], chunk[-1] + 1) for chunk in ti_chunks]
        ti_pad_chunks = [slice(chunk[0], chunk[-1] + 1)
                         for chunk in ti_pad_chunks]
        ti_hr_crop_chunks = self.get_ti_hr_crop_slices(ti_chunks,
                                                       ti_pad_chunks)
        return ti_chunks, ti_pad_chunks, ti_hr_crop_chunks

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
            if '{file_id}' not in out_files:
                tmp = out_files.split('.')
                out_files = ''.join(tmp[:-1]) + '{file_id}' + tmp[-1]
            dirname = os.path.dirname(out_files)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            for file_id in file_ids:
                out_file = out_files.format(file_id=file_id)
                out_file_list.append(out_file)
        else:
            out_file_list = [None] * len(file_ids)
        return out_file_list

    def get_spatial_slices(self, data_shape=None):
        """
        Get slices for small data chunks that are passed through generator

        Parameters
        ----------

        data_shape : slice
            Size of data volume corresponding to the spatial and temporal
            extent of files in file_paths.

        Returns
        -------
        lr_slices: list
            List of slices for low res data chunks which have not been padded
        lr_pad_slices : list
            List of slices which have been padded so that high res output
            can be stitched together
        hr_slices : list
            List of slices for high res data corresponding to the
            lr_slices regions
        hr_crop_slices : list
            List of slices used for cropping generator output
            when forward passes are performed on overlapping chunks
        """

        s1_slices = get_chunk_slices(data_shape[0], self.fwp_chunk_shape[0])
        s2_slices = get_chunk_slices(data_shape[1], self.fwp_chunk_shape[1])
        t_slices = [slice(None)]

        lr_pad_slices = []
        lr_slices = []
        hr_slices = []
        hr_crop_slices = []

        for s1 in s1_slices:
            for s2 in s2_slices:
                for t in t_slices:

                    lr_slices.append((s1, s2, t, slice(None)))

                    hr_slice = self.get_hr_slices([s1, s2, t])
                    hr_slices.append(tuple(hr_slice + [slice(None)]))

                    p_slice = self.get_padded_slices([s1, s2, t],
                                                     ends=list(data_shape))
                    lr_pad_slices.append(tuple(p_slice + [slice(None)]))

                    hrc_slice = self.get_hr_cropped_slices([s1, s2, t],
                                                           p_slice)
                    hr_crop_slices.append(tuple(hrc_slice + [slice(None)]))

        return (lr_slices, lr_pad_slices, hr_slices, hr_crop_slices)

    def get_padded_slices(self, slices, ends):
        """Pad slices for data chunk overlap

        Parameters
        ----------
        slices : list
            List of unpadded slices for data chunk
            (spatial_1, spatial_2, temporal)
        ends : list
            List of max indices for spatial and temporal domains
            (spatial_1, spatial_2, temporal)

        Returns
        -------
        list
            List of padded slices
            (spatial_1, spatial_2, temporal)
        """

        pad_slices = []
        for i, s in enumerate(slices):
            start = s.start
            stop = s.stop
            if start is not None:
                if i < 2:
                    start = max(start - self.spatial_overlap, 0)
                else:
                    start = max(start - self.temporal_overlap, 0)
            if stop is not None:
                if i < 2:
                    stop = min(stop + self.spatial_overlap, ends[i])
                else:
                    stop = min(stop + self.temporal_overlap, ends[i])
            pad_slices.append(slice(start, stop))
        return pad_slices

    def get_hr_slices(self, slices):
        """Get high res slices from low res slices

        Parameters
        ----------
        slices : list
            List of low res slices

        Returns
        -------
        list
            List of high res slices
            (spatial_1, spatial_2, temporal)
        """

        hr_slices = []
        for i, s in enumerate(slices):
            start = s.start
            stop = s.stop
            if start is not None:
                if i < 2:
                    start *= self.s_enhance
                else:
                    start *= self.t_enhance
            if stop is not None:
                if i < 2:
                    stop *= self.s_enhance
                else:
                    stop *= self.t_enhance
            hr_slices.append(slice(start, stop))

        return hr_slices

    def get_ti_hr_crop_slices(self, ti_slices, ti_pad_slices):
        """Get cropped temporal slices for stitching

        Parameters
        ----------
        ti_slices : list
            List of unpadded slices for time chunks
            (temporal)
        ti_pad_slices : list
            List of padded slices for time chunks
            (temporal)

        Returns
        -------
        list
            List of cropped slices
            (temporal)
        """

        cropped_slices = []
        for _, (ps, s) in enumerate(zip(ti_pad_slices, ti_slices)):
            start = s.start
            stop = s.stop
            if start is not None:
                start = self.t_enhance * (s.start - ps.start)
            if stop is not None:
                stop = self.t_enhance * (s.stop - ps.stop)

            if start is not None and start <= 0:
                start = None
            if stop is not None and stop >= 0:
                stop = None

            cropped_slices.append(slice(start, stop))
        return cropped_slices

    def get_hr_cropped_slices(self, lr_slices, lr_pad_slices):
        """Get cropped spatial and temporal slices for stitching

        Parameters
        ----------
        lr_slices : list
            List of unpadded slices for data chunk
            (spatial_1, spatial_2, temporal)
        lr_pad_slices : list
            List of padded slices for data chunk
            (spatial_1, spatial_2, temporal)

        Returns
        -------
        list
            List of cropped slices
            (spatial_1, spatial_2, temporal)
        """

        cropped_slices = []
        for i, (ps, s) in enumerate(zip(lr_pad_slices, lr_slices)):
            start = s.start
            stop = s.stop
            if start is not None:
                if i < 2:
                    start = self.s_enhance * (s.start - ps.start)
                else:
                    start = self.t_enhance * (s.start - ps.start)
            if stop is not None:
                if i < 2:
                    stop = self.s_enhance * (s.stop - ps.stop)
                else:
                    stop = self.t_enhance * (s.stop - ps.stop)

            if start is not None and start <= 0:
                start = None
            if stop is not None and stop >= 0:
                stop = None

            cropped_slices.append(slice(start, stop))
        return cropped_slices


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

        self.model = model_class.load(*self.model_args)

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

        self.data_handler_class = strategy.data_handler_class

        if strategy.output_type == 'nc':
            self.output_handler_class = OutputHandlerNC
        elif strategy.output_type == 'h5':
            self.output_handler_class = OutputHandlerH5

        self.data_handler = self.data_handler_class(
            self.file_paths, self.features, target=self.strategy.target,
            shape=self.strategy.grid_shape, temporal_slice=self.ti_pad_slice,
            raster_file=self.strategy.raster_file,
            cache_pattern=self.cache_pattern,
            time_chunk_size=self.strategy.time_chunk_size,
            overwrite_cache=self.strategy.overwrite_cache,
            val_split=0.0,
            max_workers=self.max_workers,
            extract_workers=self.extract_workers,
            compute_workers=self.compute_workers,
            load_workers=self.load_workers)

        self.data_handler.load_cached_data()

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

    @staticmethod
    def forward_pass_chunk(data_chunk, crop_slices,
                           model=None, model_args=None, model_class=None,
                           s_enhance=None, t_enhance=None):
        """Run forward pass on smallest data chunk. Each chunk has a maximum
        shape given by self.strategy.fwp_chunk_shape.

        Parameters
        ----------
        data_chunk : ndarray
            Data chunk to run through model generator
            (spatial_1, spatial_2, temporal, features)
        crop_slices : list
            List of slices for extracting cropped region of interest from
            output. Output can include an extra overlapping boundary to
            facilitate stitching of chunks
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

        if isinstance(model, sup3r.models.SpatialThenTemporalGan):
            i_lr_t = 0
            i_lr_s = 1
            data_chunk = np.transpose(data_chunk, axes=(2, 0, 1, 3))
        else:
            i_lr_t = 3
            i_lr_s = 1
            data_chunk = np.expand_dims(data_chunk, axis=0)

        hi_res = model.generate(data_chunk)

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

        return hi_res[0][crop_slices]

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
            import_str += 'import os; os.environ[\"CUDA_VISIBLE_DEVICES\"] = '
            import_str += '\"-1\"; '
        import_str += 'from sup3r.pipeline.forward_pass '
        import_str += f'import ForwardPassStrategy, {cls.__name__}; '
        import_str += 'from rex import init_logger'

        fwps_init_str = get_fun_call_str(ForwardPassStrategy, config)

        node_index = config['node_index']
        fwp_arg_str = f'strategy, node_index={node_index}'
        log_file = config.get('log_file', None)
        log_level = config.get('log_level', 'INFO')
        log_arg_str = '\"sup3r\", '
        log_arg_str += f'log_file=\"{log_file}\", '
        log_arg_str += f'log_level=\"{log_level}\"'

        job_name = config.get('job_name', None)
        status_dir = config.get('status_dir', None)
        status_file_arg_str = f'\"{status_dir}\", '
        status_file_arg_str += f'module=\"{ModuleName.FORWARD_PASS}\", '
        status_file_arg_str += f'job_name=\"{job_name}\", '
        status_file_arg_str += 'attrs={\"job_status\": \"successful\"}'

        cmd = (f"python -c \'{import_str};\n"
               f"logger = init_logger({log_arg_str});\n"
               f"strategy = {fwps_init_str};\n"
               f"fwp = {cls.__name__}({fwp_arg_str});\n"
               "fwp.run()")

        if job_name is not None:
            cmd += (";\nfrom reV.pipeline.status import Status;\n"
                    f"Status.make_job_file({status_file_arg_str})")
        cmd += (";\'\n")

        return cmd.replace('\\', '/')

    def _run_serial(self, data):
        """Run forward passes in serial

        Parameters
        ----------
        data : ndarray
            Array to fill with forward pass output

        Returns
        -------
        data : ndarray
            Array filled with forward pass output
        """
        logger.info('Starting serial iteration through forward pass chunks.')
        zip_iter = zip(self.hr_slices, self.lr_pad_slices, self.hr_crop_slices)
        for i, (sh, slp, shc) in enumerate(zip_iter):
            data[sh] = ForwardPass.forward_pass_chunk(
                self.data_handler.data[slp], crop_slices=shc,
                model=self.model,
                model_args=self.model_args,
                model_class=self.model_class,
                s_enhance=self.strategy.s_enhance,
                t_enhance=self.strategy.t_enhance)

            logger.debug('Coarse data chunks being passed to model '
                         'with shape {} which is slice {} of full shape {}.'
                         .format(self.data_handler.data[slp].shape, slp,
                                 self.data_handler.data.shape))

            interval = np.int(np.ceil(len(self.hr_slices) / 10))
            if interval > 0 and i % interval == 0:
                mem = psutil.virtual_memory()
                logger.info(f'{i+1} out of {len(self.hr_slices)} '
                            'forward pass chunks completed. '
                            'Memory utilization is '
                            f'{mem.used:.3f} GB out of {mem.total:.3f} GB '
                            f'total ({100*mem.used/mem.total:.1f}% used)')

        return data

    def _run_parallel(self, data, max_workers=None):
        """Run forward passes in parallel

        Parameters
        ----------
        data : ndarray
            Array to fill with forward pass output
        max_workers : int | None
            Number of workers to use for parallel forward passes

        Returns
        -------
        data : ndarray
            Array filled with forward pass output
        """
        futures = {}
        now = dt.now()
        logger.info('Starting process pool with {} workers'
                    .format(max_workers))
        with SpawnProcessPool(max_workers=max_workers) as exe:
            zip_iter = zip(self.hr_slices, self.lr_pad_slices,
                           self.hr_crop_slices)
            for i, (sh, slp, shc) in enumerate(zip_iter):

                future = exe.submit(ForwardPass.forward_pass_chunk,
                                    data_chunk=self.data_handler.data[slp],
                                    crop_slices=shc,
                                    model_args=self.model_args,
                                    model_class=self.model_class,
                                    s_enhance=self.strategy.s_enhance,
                                    t_enhance=self.strategy.t_enhance)

                meta = {'s_high': sh, 'idx': i}
                futures[future] = meta

                logger.debug('Coarse data chunks being passed to model '
                             'with shape {} which is slice {} '
                             'of full shape {}.'
                             .format(self.data_handler.data[slp].shape, slp,
                                     self.data_handler.data.shape))

            logger.info('Started forward pass for '
                        f'{len(self.hr_slices)} chunks in '
                        f'{dt.now() - now}.')

            interval = np.int(np.ceil(len(futures) / 10))
            for i, future in enumerate(as_completed(futures)):
                slices = futures[future]
                data[slices['s_high']] = future.result()
                if interval > 0 and i % interval == 0:
                    mem = psutil.virtual_memory()
                    logger.info(f'{i+1} out of {len(futures)} '
                                'forward pass chunks completed. '
                                'Memory utilization is '
                                f'{mem.used:.3f} GB out of {mem.total:.3f} GB '
                                f'total ({100*mem.used/mem.total:.1f}% used)')
            return data

    def run(self):
        """ForwardPass is initialized with a file_slice_index. This index
        selects a file subset from the full file list in ForwardPassStrategy.
        This routine runs forward passes on all data chunks associated with
        this file subset.
        """
        msg = (f'Starting forward passes on data shape {self.data_shape}. '
               f'Using {len(self.lr_slices)} chunks each with shape of '
               f'{self.chunk_shape}, spatial_overlap of '
               f'{self.strategy.spatial_overlap} and temporal_overlap of '
               f'{self.strategy.temporal_overlap}.')
        logger.info(msg)
        max_workers = self.pass_workers
        data = np.zeros((self.strategy.s_enhance * self.data_shape[0],
                         self.strategy.s_enhance * self.data_shape[1],
                         self.strategy.t_enhance * self.data_shape[2],
                         len(self.output_features)),
                        dtype=np.float32)

        if max_workers == 1:
            data = self._run_serial(data)
        else:
            data = self._run_parallel(data, max_workers)

        logger.info('All forward passes are complete.')
        data = data[:, :, self.ti_hr_crop_slice, :]
        if self.out_file is not None:
            logger.info(f'Saving forward pass output to {self.out_file}.')
            self.output_handler_class.write_output(
                data=data, features=self.data_handler.output_features,
                low_res_lat_lon=self.data_handler.lat_lon,
                low_res_times=self.strategy.time_index[self.ti_slice],
                out_file=self.out_file, meta_data=self.meta_data,
                max_workers=self.output_workers)
        else:
            return data
