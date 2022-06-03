# -*- coding: utf-8 -*-
"""
Sup3r forward pass handling module.

@author: bbenton
"""
import numpy as np
import logging
from concurrent.futures import as_completed
from datetime import datetime as dt
import os
import warnings
import glob


from rex.utilities.execution import SpawnProcessPool
from rex.utilities.fun_utils import get_fun_call_str

from sup3r.preprocessing.data_handling import DataHandlerNC
from sup3r.postprocessing.file_handling import OutputHandlerNC, OutputHandlerH5
from sup3r.utilities.utilities import (get_wrf_date_range,
                                       get_file_t_steps,
                                       get_chunk_slices,
                                       get_source_type)
from sup3r.models import Sup3rGan

np.random.seed(42)

logger = logging.getLogger(__name__)


class ForwardPassStrategy:
    """Class to prepare data for forward passes through generator.

    A full file list of contiguous times is provided.  This file list is split
    up into chunks each of which is passed to a different node. These chunks
    can overlap in time. The file list chunk is further split up into temporal
    and spatial chunks which can also overlap. This handler stores information
    on these chunks, how they overlap, and how to crop generator output to
    stich the chunks back togerther.
    """

    DATA_HANDLER = DataHandlerNC

    def __init__(self, file_paths,
                 target=None, shape=None,
                 temporal_slice=slice(None),
                 forward_pass_chunk_shape=(100, 100, 100),
                 raster_file=None,
                 s_enhance=3,
                 t_enhance=4,
                 extract_workers=None,
                 compute_workers=None,
                 max_pass_workers=None,
                 temporal_extract_chunk_size=100,
                 cache_file_prefix=None,
                 out_file_prefix=None,
                 overwrite_cache=False,
                 spatial_overlap=15,
                 temporal_overlap=15,
                 output_type='nc'):
        """Use these inputs to initialize data handlers on different nodes and
        to define the size of the data chunks that will be passed through the
        generator.

        Parameters
        ----------
        file_paths : list | str
            A list of files to extract raster data from. Each file must have
            the same number of timesteps. Can also pass a string with a
            unix-style file path which will be passed through glob.glob
        target : tuple
            (lat, lon) lower left corner of raster. Either need target+shape or
            raster_file.
        shape : tuple
            (rows, cols) grid size. Either need target+shape or raster_file.
        temporal_slice : slice
            Slice defining size of full temporal domain. e.g. If we have 5
            files each with 5 time steps then temporal_slice = slice(None) will
            select all 25 time steps.
        forward_pass_chunk_shape : tuple
            Max shape (spatial_1, spatial_2, temporal) of an unpadded chunk to
            use for a forward pass. The number of nodes that the
            ForwardPassStrategy is set to distribute to is calculated by
            dividing up the total time index from all file_paths by the
            temporal part of this chunk shape. Each node will then be
            parallelized accross parallel processes by the spatial chunk shape.
            If temporal_overlap / spatial_overlap are non zero the chunk sent
            to the generator can be bigger than this shape. If running in
            serial set this equal to the shape of the full spatiotemporal data
            volume for best performance.
        raster_file : str | None
            File for raster_index array for the corresponding target and
            shape. If specified the raster_index will be loaded from the file
            if it exists or written to the file if it does not yet exist.
            If None raster_index will be calculated directly. Either need
            target+shape or raster_file.
        s_enhance : int
            Factor by which to enhance spatial dimensions of low resolution
            data
        t_enhance : int
            Factor by which to enhance temporal dimension of low resolution
            data
        compute_workers : int | None
            max number of workers to use for computing features. If
            compute_workers == 1 then extraction will be serialized.
        extract_workers : int | None
            max number of workers to use for data extraction. If
            extract_workers == 1 then extraction will be serialized.
        max_pass_workers : int | None
            Max number of workers to use for forward passes on each node. If
            max_pass_workers == 1 then forward passes on chunks will be
            serialized.
        temporal_extract_chunk_size : int
            Size of chunks to split time dimension into for parallel data
            extraction. If running in serial this can be set to the size
            of the full time index for best performance.
        cache_file_prefix : str
            Prefix of path to cached feature data files
        overwrite_cache : bool
            Whether to overwrite cache files storing the computed/extracted
            feature data
        out_file_prefix : str
            Prefix of path to save forward pass output files. e.g. If this is
            /tmp/output then each output file will have path
            /tmp/output_<file_id>.pkl. If None then data will be returned in an
            array and not saved.
        spatial_overlap : int
            Size of spatial overlap between chunks passed to forward passes
            for subsequent spatial stitching
        temporal_overlap : int
            Size of temporal overlap between chunks passed to forward passes
            for subsequent temporal stitching
        output_type : str
            Either nc (netcdf) or h5. This selects the extension for output
            files and determines which output handler to use when writing the
            results of the forward passes
        """

        if isinstance(file_paths, str):
            file_paths = glob.glob(file_paths)
        self.file_paths = sorted(file_paths)
        self.output_type = output_type
        self.input_type = get_source_type(file_paths)
        self._i = 0
        self.file_t_steps = get_file_t_steps(self.file_paths)
        self.raster_file = raster_file
        self.target = target
        self.shape = shape
        self.forward_pass_chunk_shape = forward_pass_chunk_shape
        self.temporal_slice = temporal_slice
        self.max_pass_workers = max_pass_workers
        self.extract_workers = extract_workers
        self.compute_workers = compute_workers
        self.cache_file_prefix = cache_file_prefix
        self.out_file_prefix = out_file_prefix
        self.temporal_extract_chunk_size = temporal_extract_chunk_size
        self.overwrite_cache = overwrite_cache
        self.t_enhance = t_enhance
        self.s_enhance = s_enhance
        self.spatial_overlap = spatial_overlap
        self.temporal_overlap = temporal_overlap
        self.file_overlap = int(np.ceil(temporal_overlap / self.file_t_steps))
        self.fp_chunk_size = int(np.ceil(forward_pass_chunk_shape[2]
                                         / self.file_t_steps))

        out = self.get_file_slices(self.file_paths)
        self.file_slices, self.padded_file_slices = out[:2]
        self.cropped_file_slices, self.temporal_slices = out[2:]

        self.file_ids = self.get_file_ids(
            file_paths=file_paths, file_slices=self.file_slices)
        self.out_files = self.get_output_file_names(
            out_file_prefix=out_file_prefix, file_ids=self.file_ids,
            file_type=self.output_type)

        msg = (f'Using a larger temporal_overlap {temporal_overlap} than '
               f'temporal_chunk_size {forward_pass_chunk_shape[2]}.')
        if temporal_overlap > forward_pass_chunk_shape[2]:
            logger.warning(msg)
            warnings.warn(msg)

        msg = (f'Using a larger spatial_overlap {spatial_overlap} than '
               f'spatial_chunk_size {forward_pass_chunk_shape[:2]}.')
        if any(spatial_overlap > sc for sc in forward_pass_chunk_shape[:2]):
            logger.warning(msg)
            warnings.warn(msg)

        msg = ('Using a padded chunk size '
               f'{forward_pass_chunk_shape[2] + 2 * temporal_overlap} '
               'larger than the full temporal domain '
               f'{self.file_t_steps * len(file_paths)}. Should just run '
               'without temporal chunking. ')
        if (forward_pass_chunk_shape[2] + 2 * temporal_overlap
                >= self.file_t_steps * len(file_paths)):
            logger.warning(msg)
            warnings.warn(msg)

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

        file_paths = self.file_paths[self.padded_file_slices[node_index]]
        cropped_file_slice = self.cropped_file_slices[node_index]
        out_file = self.out_files[node_index]
        temporal_slice = self.temporal_slices[node_index]
        ts_indices = np.arange(len(file_paths) * self.file_t_steps)
        data_shape = (self.shape[0], self.shape[1],
                      len(ts_indices[temporal_slice]))
        cache_file_prefix = (None if self.cache_file_prefix is None
                             else f'{self.cache_file_prefix}_{node_index}')

        out = self.get_chunk_slices(data_shape=data_shape)
        lr_slices, lr_pad_slices, hr_slices, hr_crop_slices = out

        chunk_shape = (lr_slices[0][0].stop - lr_slices[0][0].start,
                       lr_slices[0][1].stop - lr_slices[0][1].start,
                       data_shape[2])

        kwargs = dict(file_paths=file_paths,
                      cropped_file_slice=cropped_file_slice,
                      out_file=out_file,
                      temporal_slice=temporal_slice,
                      lr_slices=lr_slices,
                      lr_pad_slices=lr_pad_slices,
                      hr_slices=hr_slices,
                      hr_crop_slices=hr_crop_slices,
                      data_shape=data_shape,
                      chunk_shape=chunk_shape,
                      cache_file_prefix=cache_file_prefix,
                      node_index=node_index)

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

        if self._i < len(self.file_slices):
            kwargs = self.get_node_kwargs(self._i)
            self._i += 1
            return kwargs
        else:
            raise StopIteration

    @property
    def nodes(self):
        """Get the number of nodes that this strategy should distribute work
        to, calculated as the source time index divided by the temporal part of
        the forward_pass_chunk_shape"""
        return len(self.file_slices)

    @classmethod
    def file_info_logging(cls, file_paths):
        """More concise file info about data files

        Parameters
        ----------
        file_paths : list
            List of file paths

        Returns
        -------
        str
            message to append to log output that does not include a huge info
            dump of file paths
        """

        return cls.DATA_HANDLER.file_info_logging(file_paths)

    @staticmethod
    def get_output_file_names(out_file_prefix, file_ids, file_type='nc'):
        """Get output file names for each file chunk forward pass

        Parameters
        ----------
        out_file_prefix : str
            Prefix of output file names
        file_ids : list
            List of file ids for each output file. e.g. date range
        file_type : str
            Either netcdf or h5. This selects the extension for output files
            and determines which output handler to use when writing the results
            of the forward passes

        Returns
        -------
        list
            List of output file names
        """

        if file_type.lower() == 'nc' or file_type.lower() == 'netcdf':
            ext = '.nc'
        elif file_type.lower() == 'h5':
            ext = '.h5'

        out_files = []
        if out_file_prefix is not None:
            dirname = os.path.dirname(out_file_prefix)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            for i, file_id in enumerate(file_ids):
                out_file = f'{out_file_prefix}_{file_id}_{i}_{ext}'
                out_files.append(out_file)
        else:
            out_files = [None] * len(file_ids)
        return out_files

    def get_combined_output_file_name(self, out_file_prefix, file_paths):
        """Get combined output file name. Use same file name format for
        chunked output files.

        Parameters
        ----------
        out_file_prefix : str
            Prefix of output file names
        file_paths : list
            A list of files to extract raster data from

        Returns
        -------
        str
            Combined output file name
        """
        file_id = self.get_file_ids(file_paths, slice(None))
        outfile = self.get_output_file_names(out_file_prefix, file_id)
        return outfile[0]

    @staticmethod
    def get_file_ids(file_paths, file_slices):
        """Get file ids for naming logs, cache_files, and output files

        Parameters
        ----------
        file_paths : list
            A list of files to extract raster data from
        file_slices : list
            List of slices specifying file chunks sent to different nodes

        Returns
        -------
        list
            List of file_ids for naming corresponding logs, outputs, and cache
        """
        file_ids = []
        for chunk in file_slices:
            start, end = get_wrf_date_range(file_paths[chunk])
            file_ids.append(f'{start}_{end}')
        return file_ids

    def get_file_slices(self, file_paths):
        """
        Get slices for the provided file list. These sets of slices are used to
        specify which files are passed to each node for data extraction and to
        account for temporal overlap of files.

        Parameters
        ----------
        file_paths : list
            A list of files to extract raster data from

        Returns
        -------
        file_slices : list
            List of file slices
        padded_file_slices : list
            List of file slices including specified file overlap
        cropped_file_slices : list
            List of temporal slices used to crop the overlap associated with
            the file slice padding
        temporal_slices : list
            List of slices used to specify requested temporal extent for file
            set passed to data handler.
        """

        cropped_f_slices = []
        padded_f_slices = []

        file_slices = get_chunk_slices(len(file_paths), self.fp_chunk_size)

        for f in file_slices:
            start = max(f.start - self.file_overlap, 0)
            stop = min(f.stop + self.file_overlap, len(self.file_paths))
            padded_f_slices.append(slice(start, stop))

        for f, fp in zip(file_slices, padded_f_slices):
            start = self.file_t_steps * self.t_enhance * (f.start - fp.start)
            if start <= 0:
                start = None

            stop = self.file_t_steps * self.t_enhance * (f.stop - fp.stop)
            if stop >= 0:
                stop = None
            cropped_f_slices.append(slice(start, stop))

        temporal_slices = [slice(None, None, self.temporal_slice.step)]
        temporal_slices = temporal_slices * len(file_slices)
        temporal_slices[0] = slice(self.temporal_slice.start, None,
                                   self.temporal_slice.step)
        stop = self.temporal_slice.stop
        if stop is not None and stop > 0:
            stop = stop % self.file_t_steps
        temporal_slices[-1] = slice(None, stop, self.temporal_slice.step)

        return file_slices, padded_f_slices, cropped_f_slices, temporal_slices

    def get_chunk_slices(self, data_shape=None):
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

        s1_slices = get_chunk_slices(data_shape[0],
                                     self.forward_pass_chunk_shape[0])
        s2_slices = get_chunk_slices(data_shape[1],
                                     self.forward_pass_chunk_shape[1])
        if self.file_t_steps < self.forward_pass_chunk_shape[2]:
            t_slices = [slice(None)]
        else:
            t_slices = get_chunk_slices(data_shape[2],
                                        self.forward_pass_chunk_shape[2])

        lr_pad_slices = []
        lr_slices = []
        hr_slices = []
        hr_crop_slices = []

        for s1 in s1_slices:
            for s2 in s2_slices:
                for t in t_slices:

                    lr_slices.append(tuple([s1, s2, t, slice(None)]))

                    hr_slice = self.get_hr_slices([s1, s2, t])
                    hr_slices.append(tuple(hr_slice + [slice(None)]))

                    p_slice = self.get_padded_slices([s1, s2, t],
                                                     ends=list(data_shape))
                    lr_pad_slices.append(tuple(p_slice + [slice(None)]))

                    c_slice = self.get_cropped_slices([s1, s2, t], p_slice)
                    hr_crop_slices.append(tuple(c_slice + [slice(None)]))

        return lr_slices, lr_pad_slices, hr_slices, hr_crop_slices

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

    def get_cropped_slices(self, lr_slices, lr_pad_slices):
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

    DATA_HANDLER = DataHandlerNC
    OUTPUT_HANDLER = OutputHandlerNC

    def __init__(self, strategy, model_path, node_index=0):
        """Initialize ForwardPass with ForwardPassStrategy. The stragegy
        provides the data chunks to run forward passes on

        Parameters
        ----------
        strategy : ForwardPassStrategy
            ForwardPassStrategy instance with information on data chunks to run
            forward passes on.
        model_path : str
            Path to Sup3rGan used to generate high resolution data
        node_index : int
            Index used to select subset of full file list on which to run
            forward passes on a single node.
        """
        self.strategy = strategy
        self.model_path = model_path
        self.features = Sup3rGan.load(model_path).training_features
        self.meta_data = Sup3rGan.load(model_path).meta
        self.node_index = node_index

        kwargs = strategy.get_node_kwargs(node_index)

        self.file_paths = kwargs['file_paths']
        self.cropped_file_slice = kwargs['cropped_file_slice']
        self.out_file = kwargs['out_file']
        self.temporal_slice = kwargs['temporal_slice']
        self.lr_slices = kwargs['lr_slices']
        self.lr_pad_slices = kwargs['lr_pad_slices']
        self.hr_slices = kwargs['hr_slices']
        self.hr_crop_slices = kwargs['hr_crop_slices']
        self.data_shape = kwargs['data_shape']
        self.chunk_shape = kwargs['chunk_shape']
        self.cache_file_prefix = kwargs['cache_file_prefix']

        self.data_handler = self.DATA_HANDLER(
            self.file_paths, self.features, target=self.strategy.target,
            shape=self.strategy.shape, temporal_slice=self.temporal_slice,
            raster_file=self.strategy.raster_file,
            extract_workers=self.strategy.extract_workers,
            compute_workers=self.strategy.compute_workers,
            cache_file_prefix=self.cache_file_prefix,
            time_chunk_size=self.strategy.temporal_extract_chunk_size,
            overwrite_cache=self.strategy.overwrite_cache,
            val_split=0.0)

        if self.cache_file_prefix is not None:
            self.data_handler.load_cached_data()

    @staticmethod
    def forward_pass_chunk(data_chunk, crop_slices, model_path):
        """Run forward pass on smallest data chunk. Each chunk has a maximum
        shape given by self.strategy.forward_pass_chunk_shape.

        Parameters
        ----------
        data_chunk : ndarray
            Data chunk to run through model generator
            (spatial_1, spatial_2, temporal, features)
        crop_slices : list
            List of slices for extracting cropped region of interest from
            output. Output can include an extra overlapping boundary to
            facilitate stitching of chunks
        model_path : str
            Path to file for Sup3rGan used to generate high resolution
            data

        Returns
        -------
        ndarray
            High resolution data generated by GAN
        """

        model = Sup3rGan.load(model_path)
        data_chunk = np.expand_dims(data_chunk, axis=0)

        hi_res = model.generate(data_chunk)

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

        import_str = ('from sup3r.pipeline.forward_pass '
                      f'import ForwardPassStrategy, {cls.__name__}; '
                      'from rex import init_logger')

        fps_init_str = get_fun_call_str(ForwardPassStrategy, config)

        model_path = config['model_path']
        node_index = config['node_index']
        fwp_arg_str = f'strategy, \"{model_path}\", node_index={node_index}'
        log_file = config.get('log_file', None)
        log_level = config.get('log_level', 'INFO')
        log_arg_str = '\"sup3r\", '
        log_arg_str += f'log_file=\"{log_file}\", '
        log_arg_str += f'log_level=\"{log_level}\"'

        cmd = (f"python -c \'{import_str};\n"
               f"logger = init_logger({log_arg_str});\n"
               f"strategy = {fps_init_str};\n"
               f"fwp = {cls.__name__}({fwp_arg_str});\n"
               "fwp.run()\'\n")

        return cmd

    def run(self):
        """ForwardPass is initialized with a file_slice_index. This index
        selects a file subset from the full file list in ForwardPassStrategy.
        This routine runs forward passes on all data chunks associated with
        this file subset.
        """
        logger.info(
            f'Starting forward passes on data shape {self.data_shape}. Using '
            f'{len(self.lr_slices)} chunks each with shape of '
            f'{self.chunk_shape}, spatial_overlap of '
            f'{self.strategy.spatial_overlap} and temporal_overlap of '
            f'{self.strategy.temporal_overlap}')

        data = np.zeros(
            (self.strategy.s_enhance * self.data_shape[0],
             self.strategy.s_enhance * self.data_shape[1],
             self.strategy.t_enhance * self.data_shape[2], 2),
            dtype=np.float32)

        if self.strategy.max_pass_workers == 1:
            for s_high, s_low_pad, s_high_crop in zip(self.hr_slices,
                                                      self.lr_pad_slices,
                                                      self.hr_crop_slices):

                data_chunk = self.data_handler.data[s_low_pad]
                data[s_high] = ForwardPass.forward_pass_chunk(
                    data_chunk, crop_slices=s_high_crop,
                    model_path=self.model_path)
        else:
            futures = {}
            now = dt.now()
            with SpawnProcessPool(
                    max_workers=self.strategy.max_pass_workers) as exe:
                for s_high, s_low_pad, s_high_crop in zip(self.hr_slices,
                                                          self.lr_pad_slices,
                                                          self.hr_crop_slices):

                    data_chunk = self.data_handler.data[s_low_pad]
                    future = exe.submit(ForwardPass.forward_pass_chunk,
                                        data_chunk=data_chunk,
                                        crop_slices=s_high_crop,
                                        model_path=self.model_path)
                    meta = {'s_high': s_high}
                    futures[future] = meta

                logger.info(f'Started forward pass for {len(self.hr_slices)} '
                            f'chunks in {dt.now() - now}.')

                for i, future in enumerate(as_completed(futures)):
                    slices = futures[future]
                    data[slices['s_high']] = future.result()
                    if i % (len(futures) // 10 + 1) == 0:
                        logger.debug(f'{i+1} out of {len(futures)} forward '
                                     'passes completed.')

        logger.info('All forward passes are complete.')
        data = data[:, :, self.cropped_file_slice, :]

        if self.out_file is not None:
            logger.info(f'Saving forward pass output to {self.out_file}.')
            self.OUTPUT_HANDLER.write_output(
                data, self.data_handler.output_features,
                self.data_handler.lat_lon,
                self.data_handler.time_index,
                self.data_handler.time_description,
                self.out_file, self.meta_data)
        else:
            return data


class ForwardPassToH5(ForwardPass):
    """ForwardPass subclass with H5 output"""

    OUTPUT_HANDLER = OutputHandlerH5
