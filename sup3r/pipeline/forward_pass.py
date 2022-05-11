# -*- coding: utf-8 -*-
"""
Sup3r forward pass handling module.

@author: bbenton
"""

import numpy as np
import logging
from concurrent.futures import as_completed
from datetime import datetime as dt
import pickle
import os

from rex.utilities.execution import SpawnProcessPool

from sup3r.preprocessing.data_handling import DataHandlerNC
from sup3r.utilities.utilities import get_wrf_date_range, get_file_t_steps
from sup3r.models.spatiotemporal import SpatioTemporalGan

np.random.seed(42)

logger = logging.getLogger(__name__)


class ForwardPass:
    """Class to handle parallel forward passes through the generator with
    multiple data files.

    A full file list of contiguous times is provided.  This file list is split
    up into chunks each of which is passed to a different node. These chunks
    can overlap in time.

    On each node the file list chunk is further split up into temporal and
    spatial chunks which can also overlap. Each of these chunks is passed
    through the GAN generator to produce high resolution output. These high
    resolution chunks are then stitched back together, accounting for possible
    overlap. The stitched output is saved or returned in an array.
    """

    def __init__(self, file_paths,
                 model_path,
                 target=None, shape=None,
                 temporal_slice=slice(None),
                 forward_pass_chunk_shape=(100, 100, 100),
                 raster_file=None,
                 s_enhance=3,
                 t_enhance=4,
                 max_extract_workers=None,
                 max_compute_workers=None,
                 max_pass_workers=None,
                 temporal_extract_chunk_size=100,
                 cache_file_prefix=None,
                 out_file_prefix=None,
                 overwrite_cache=False,
                 spatial_overlap=15,
                 temporal_overlap=15):

        """Use these inputs to initialize data handlers on different nodes and
        to define the size of the data chunks that will be passed through the
        generator.

        Parameters
        ----------
        file_paths : list
            A list of NETCDF files to extract raster data from. Each file must
            have the same number of timesteps.
        model_path : str
            Path to SpatioTemporalGan used to generate high resolution data
        target : tuple
            (lat, lon) lower left corner of raster. Either need target+shape or
            raster_file.
        shape : tuple
            (rows, cols) grid size. Either need target+shape or raster_file.
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
        temporal_slice : slice
            Slice defining size of full temporal domain. e.g. If the shape
            argument is (100, 100) and temporal_slice is slice(0, 101, 1) then
            the full spatiotemporal data volume will be (100, 100, 100).
        temporal_extract_chunk_size : int
            Size of chunks to split time dimension into for parallel data
            extraction. If running in serial this can be set to the size
            of the full time index for best performance.
        forward_pass_chunk_shape : tuple
            Max shape of a chunk to pass through the generator. If running
            in serial set this equal to the shape of the full data volume for
            best performance.
        max_compute_workers : int | None
            max number of workers to use for computing features.
            If max_compute_workers == 1 then extraction will be serialized.
        max_extract_workers : int | None
            max number of workers to use for data extraction.
            If max_extract_workers == 1 then extraction will be serialized.
        max_pass_workers : int | None
            Max number of workers to use for forward passes on each node.
            If max_pass_workers == 1 then forward passes on chunks will be
            serialized.
        overwrite_cache : bool
            Whether to overwrite cache files
        cache_file_prefix : str
            Prefix of path to cached feature data files
        out_file_prefix : str
            Prefix of path to forward pass output files. If None then data
            will be returned in an array and not saved.
        spatial_overlap : int
            Size of spatial overlap between chunks passed to forward passes
            for subsequent spatial stitching
        temporal_overlap : int
            Size of temporal overlap between chunks passed to forward passes
            for subsequent temporal stitching
        """

        self.file_t_steps = get_file_t_steps(file_paths)
        self.file_paths = sorted(file_paths)
        self.features = SpatioTemporalGan.load(model_path).training_features
        self.raster_file = raster_file
        self.target = target
        self.shape = shape
        self.spatial_chunk_size = forward_pass_chunk_shape[:2]
        self.temporal_pass_chunk_size = forward_pass_chunk_shape[-1]
        self.temporal_slice = temporal_slice
        self.max_pass_workers = max_pass_workers
        self.max_extract_workers = max_extract_workers
        self.max_compute_workers = max_compute_workers
        self.model_path = model_path
        self.cache_file_prefix = cache_file_prefix
        self.out_file_prefix = out_file_prefix
        self.temporal_extract_chunk_size = temporal_extract_chunk_size
        self.overwrite_cache = overwrite_cache
        self.t_enhance = t_enhance
        self.s_enhance = s_enhance
        self.spatial_overlap = spatial_overlap
        self.temporal_overlap = temporal_overlap
        self.file_overlap = int(np.ceil(temporal_overlap / self.file_t_steps))
        self.file_path_chunk_size = int(np.ceil(self.temporal_pass_chunk_size
                                                / self.file_t_steps))
        out = self.get_file_slices(
            self.file_paths, file_path_chunk_size=self.file_path_chunk_size,
            file_overlap=self.file_overlap, file_t_steps=self.file_t_steps,
            temporal_slice=self.temporal_slice, t_enhance=t_enhance)

        self.file_chunks, self.padded_file_chunks = out[:2]
        self.file_crop_slices, self.temporal_slices = out[2:]

        self.file_ids = self.get_file_ids(
            file_paths=file_paths, file_chunks=self.file_chunks)
        self.out_files = self.get_output_file_names(
            out_file_prefix=out_file_prefix, file_ids=self.file_ids)

        msg = (f'Using a larger temporal_overlap {temporal_overlap} than '
               f'temporal_chunk_size {self.temporal_pass_chunk_size}.')
        if temporal_overlap > self.temporal_pass_chunk_size:
            logger.warning(msg)

        msg = (f'Using a larger spatial_overlap {spatial_overlap} than '
               f'spatial_chunk_size {self.spatial_chunk_size}.')
        if any(spatial_overlap > sc for sc in self.spatial_chunk_size):
            logger.warning(msg)

        msg = ('Using a padded chunk size '
               f'{self.temporal_pass_chunk_size + 2 * temporal_overlap} '
               'larger than the full temporal domain '
               f'{self.file_t_steps * len(file_paths)}. Should just run '
               'without temporal chunking. ')
        if (self.temporal_pass_chunk_size + 2 * temporal_overlap
                >= self.file_t_steps * len(file_paths)):
            logger.warning(msg)

    @classmethod
    def file_info_logging(cls, file_path):
        """More concise file info about NETCDF files

        Parameters
        ----------
        file_path : list
            List of file paths

        Returns
        -------
        str
            message to append to log output that does not include a huge info
            dump of file paths
        """

        return DataHandlerNC.file_info_logging(file_path)

    @classmethod
    def get_output_file_names(cls, out_file_prefix, file_ids):
        """Get output file names for each file chunk forward pass

        Parameters
        ----------
        out_file_prefix : str
            Prefix of output file names
        file_ids : list
            List of file ids for each output file. e.g. date range

        Returns
        -------
        list
            List of output file names
        """
        out_files = []
        if out_file_prefix is not None:
            dirname = os.path.dirname(out_file_prefix)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            for file_id in file_ids:
                out_files.append(
                    f'{out_file_prefix}_{file_id}.pkl')
        return out_files

    @classmethod
    def get_file_ids(cls, file_paths, file_chunks):
        """Get file ids for naming logs, cache_files, and output files

        Parameters
        ----------
        file_paths : list
            A list of NETCDF files to extract raster data from
        file_chunks : list
            List of slices specifying file chunks sent to different nodes

        Returns
        -------
        list
            List of file_ids for naming corresponding logs, outputs, and cache
        """
        file_ids = []
        for chunk in file_chunks:
            start, end = get_wrf_date_range(file_paths[chunk])
            file_ids.append(f'{start}_{end}')
        return file_ids

    @classmethod
    def get_file_slices(cls, file_paths, file_path_chunk_size,
                        file_overlap, file_t_steps, temporal_slice,
                        t_enhance=4):
        """
        Get slices for the provided file list. These sets of slices are used to
        specify which files are passed to each node for data extraction and to
        account for temporal overlap of files.

        Parameters
        ----------
        file_paths : list
            A list of NETCDF files to extract raster data from
        file_path_chunk_size : int
            Number of file paths to pass to each node for subsequent data
            extraction and forward pass
        file_overlap : int
            Number of files to pad file chunks with so that we have temporal
            overlap between file sets passed to each node
        file_t_steps : int
            Number of time steps in a single file.
        temporal_slice : slice
            Slice describing full temporal domain across all files

        Returns
        -------
        file_slices : list
            List of file slices
        padded_file_slices : list
            List of file slices including specified file overlap
        file_crop_slices : list
            List of temporal slices used to crop the overlap associated with
            the file slice padding
        temporal_slices : list
            List of slices used to specify requested temporal extent for file
            set passed to data handler.
        t_enhance : int
            Factor by which to enhance the temporal resolution
        """

        file_crop_slices = []
        padded_file_slices = []

        n_chunks = int(np.ceil(len(file_paths) / file_path_chunk_size))
        file_chunks = np.array_split(np.arange(len(file_paths)), n_chunks)
        file_slices = [slice(f[0], f[-1] + 1) for f in file_chunks]

        for f in file_chunks:
            start = max(f[0] - file_overlap, 0)
            stop = min(f[-1] + file_overlap + 1, len(file_paths))
            padded_file_slices.append(slice(start, stop))

        for fs, fs_p in zip(file_slices, padded_file_slices):
            start = file_t_steps * t_enhance * (fs.start - fs_p.start)
            if start <= 0:
                start = None

            stop = -file_t_steps * t_enhance * (fs_p.stop - fs.stop)
            if stop >= 0:
                stop = None
            file_crop_slices.append(slice(start, stop))

        temporal_slices = [slice(None, None, temporal_slice.step)] * n_chunks
        temporal_slices[0] = slice(temporal_slice.start, None,
                                   temporal_slice.step)
        stop = temporal_slice.stop
        if stop is not None and stop > 0:
            stop = stop % file_t_steps
        temporal_slices[-1] = slice(None, stop, temporal_slice.step)

        return file_slices, padded_file_slices, file_crop_slices, \
            temporal_slices

    @classmethod
    def get_chunk_slices(cls, data_shape=None,
                         spatial_chunk_size=(10, 10),
                         temporal_chunk_size=24,
                         s_enhance=3,
                         t_enhance=4,
                         spatial_overlap=15,
                         temporal_overlap=15,
                         file_t_steps=1):
        """
        Get slices for small data chunks that are passed through generator

        Parameters
        ----------

        data_shape : slice
            Size of data volume corresponding to the spatial and temporal
            extent of files in file_paths.
        spatial_chunk_size : tuple
            Max size of a spatial domain to pass through generator during
            subprocesses. The full spatial domain (shape) will be chunked into
            pieces with max spatial extent equal to spatial_chunk_size
        temporal_chunk_size : int
            Max size of a temporal chunk to pass through the generator. e.g.
            If spatial_chunk_size is (10, 10) and temporal chunk size is 10
            then the spatiotemporal shape of each data chunk passed through
            the generator will be a max shape of (10, 10, 10).
        s_enhance : int
            Factor by which to enhance spatial dimensions of low resolution
            data
        t_enhance : int
            Factor by which to enhance temporal dimension of low resolution
            data
        spatial_overlap : int
            Size of spatial overlap between chunks passed to forward passes
            for subsequent spatial stitching
        temporal_overlap : int
            Size of temporal overlap between chunks passed to forward passes
            for subsequent temporal stitching

        Returns
        -------

        low_res_pad_slices : list
            List of slices which have been padded so that high res output
            can be stitched together
        high_res_slices : list
            List of slices for high res data corresponding to the
            low_res_slices regions
        high_res_crop_slices : list
            List of slices used for cropping generator output
            when forward passes are performed on overlapping chunks
        """

        n_chunks = int(np.ceil(data_shape[0] / spatial_chunk_size[0]))
        s1_slices = np.array_split(np.arange(data_shape[0]), n_chunks)
        s1_slices = [slice(s1[0], s1[-1] + 1) for s1 in s1_slices]

        n_chunks = int(np.ceil(data_shape[1] / spatial_chunk_size[1]))
        s2_slices = np.array_split(np.arange(data_shape[1]), n_chunks)
        s2_slices = [slice(s2[0], s2[-1] + 1) for s2 in s2_slices]

        if file_t_steps < temporal_chunk_size:
            t_slices = [slice(None)]
        else:
            n_chunks = int(np.ceil(data_shape[2] / temporal_chunk_size))
            t_slices = np.array_split(np.arange(data_shape[2]), n_chunks)
            t_slices = [slice(t[0], t[-1] + 1) for t in t_slices]

        low_res_pad_slices = []
        low_res_slices = []
        high_res_slices = []
        high_res_crop_slices = []

        for s1 in s1_slices:
            for s2 in s2_slices:
                for t in t_slices:

                    low_res_slices.append(
                        tuple([s1, s2, t, slice(None)]))

                    s1_h, s2_h, t_h = cls.get_high_res_slices(
                        [s1, s2, t], s_enhance=s_enhance, t_enhance=t_enhance)
                    high_res_slices.append(
                        tuple([s1_h, s2_h, t_h, slice(None)]))

                    s1_p, s2_p, t_p = cls.get_padded_slices(
                        [s1, s2, t], ends=list(data_shape),
                        spatial_overlap=spatial_overlap,
                        temporal_overlap=temporal_overlap)
                    low_res_pad_slices.append(
                        tuple([s1_p, s2_p, t_p, slice(None)]))

                    s1_c, s2_c, t_c = cls.get_cropped_slices(
                        [s1, s2, t], [s1_p, s2_p, t_p],
                        s_enhance=s_enhance, t_enhance=t_enhance)
                    high_res_crop_slices.append(
                        tuple([s1_c, s2_c, t_c, slice(None)]))

        return low_res_slices, low_res_pad_slices, high_res_slices, \
            high_res_crop_slices

    def forward_pass_file_chunk(self, file_paths, temporal_slice=slice(None),
                                crop_slice=slice(None), out_file=None):
        """
        Routine to run forward pass on all data chunks associated with the
        files in file_paths

        Parameters
        ----------

        file_paths : list
            A list of NETCDF files to extract raster data from
        out_file : str | None
            File to store forward pass output. If None data will be returned in
            an array instead of saved.
        temporal_slice : slice
            Slice used to select temporal extent from files
        crop_slice : slice
            Slice to crop temporal output if there is temporal overlap between
            file chunks passed to each node.
        """

        handler = DataHandlerNC(
            file_paths, self.features, target=self.target, shape=self.shape,
            temporal_slice=temporal_slice, raster_file=self.raster_file,
            max_extract_workers=self.max_extract_workers,
            max_compute_workers=self.max_compute_workers,
            cache_file_prefix=self.cache_file_prefix,
            time_chunk_size=self.temporal_extract_chunk_size,
            overwrite_cache=self.overwrite_cache,
            val_split=0.0)
        handler.load_cached_data()

        data_shape = (self.shape[0], self.shape[1], len(handler.time_index))

        out = self.get_chunk_slices(
            data_shape=data_shape, spatial_chunk_size=self.spatial_chunk_size,
            temporal_chunk_size=self.temporal_pass_chunk_size,
            s_enhance=self.s_enhance, t_enhance=self.t_enhance,
            spatial_overlap=self.spatial_overlap,
            temporal_overlap=self.temporal_overlap,
            file_t_steps=get_file_t_steps(file_paths))

        low_res_slices, low_res_pad_slices = out[:2]
        high_res_slices, high_res_crop_slices = out[2:]

        chunk_shape = (low_res_slices[0][0].stop - low_res_slices[0][0].start,
                       low_res_slices[0][1].stop - low_res_slices[0][1].start,
                       data_shape[2])
        logger.info(
            f'Starting forward passes on data shape {data_shape}. Using '
            f'{len(low_res_slices)} chunks each with shape of {chunk_shape}, '
            f'spatial_overlap of {self.spatial_overlap} and temporal_overlap '
            f'of {self.temporal_overlap}')

        data = np.zeros(
            (self.s_enhance * data_shape[0], self.s_enhance * data_shape[1],
             self.t_enhance * data_shape[2], 2), dtype=np.float32)

        if self.max_pass_workers == 1:
            for s_high, s_low_pad, s_high_crop in zip(high_res_slices,
                                                      low_res_pad_slices,
                                                      high_res_crop_slices):

                data_chunk = handler.data[s_low_pad]
                data[s_high] = ForwardPass.forward_pass_chunk(
                    data_chunk, crop_slices=s_high_crop,
                    model_path=self.model_path)
        else:
            futures = {}
            now = dt.now()
            with SpawnProcessPool(max_workers=self.max_pass_workers) as exe:
                for s_high, s_low_pad, s_high_crop in zip(
                        high_res_slices, low_res_pad_slices,
                        high_res_crop_slices):

                    data_chunk = handler.data[s_low_pad]
                    future = exe.submit(ForwardPass.forward_pass_chunk,
                                        data_chunk=data_chunk,
                                        crop_slices=s_high_crop,
                                        model_path=self.model_path)
                    meta = {'s_high': s_high}
                    futures[future] = meta

                logger.info(
                    f'Started forward pass for {len(high_res_slices)} chunks '
                    f'in {dt.now() - now}.')

                for i, future in enumerate(as_completed(futures)):
                    slices = futures[future]
                    data[slices['s_high']] = future.result()
                    if i % (len(futures) // 10 + 1) == 0:
                        logger.debug(
                            f'{i+1} out of {len(futures)} forward passes '
                            'completed.')

        logger.info('All forward passes are complete.')

        data = data[:, :, crop_slice, :]

        if out_file is not None:
            with open(out_file, 'wb') as fh:
                logger.info(f'Saving forward pass output to {out_file}.')
                pickle.dump(data, fh)

        else:
            return data

    @staticmethod
    def forward_pass_chunk(data_chunk, crop_slices, model_path):
        """Run forward pass on smallest data chunk

        Parameters
        ----------
        data_chunk : ndarray
            Data chunk to run through model generator
            (spatial_1, spatial_2, temporal, features)
        model_path : str
            Path to file for SpatioTemporalGan used to generate high resolution
            data
        crop_slices : list
            List of slices for extracting cropped region of interest from
            output. Output can include an extra overlapping boundary to
            facilitate stitching of chunks

        Returns
        -------
        ndarray
            High resolution data generated by GAN
        """

        model = SpatioTemporalGan.load(model_path)
        data_chunk = np.expand_dims(data_chunk, axis=0)

        hi_res = model.generate_hires(data_chunk)

        return hi_res[0][crop_slices]

    @staticmethod
    def get_padded_slices(slices, ends, spatial_overlap=15,
                          temporal_overlap=15):
        """Pad slices for data chunk overlap

        Parameters
        ----------
        slices : list
            List of unpadded slices for data chunk
            (spatial_1, spatial_2, temporal)
        ends : list
            List of max indices for spatial and temporal domains
            (spatial_1, spatial_2, temporal)
        spatial_overlap : int
            Size of spatial overlap between chunks passed to forward passes for
            subsequent spatial stitching
        temporal_overlap : int
            Size of temporal overlap between chunks passed to forward passes

        Returns
        -------
        s1 : slice
            spatial_1 padded slice
        s2 : slice
            spatial_2 padded slice
        t : slice
            temporal padded slice

        """

        s1 = slice(max(slices[0].start - spatial_overlap, 0),
                   min(slices[0].stop + spatial_overlap, ends[0]))
        s2 = slice(max(slices[1].start - spatial_overlap, 0),
                   min(slices[1].stop + spatial_overlap, ends[1]))
        t_start = slices[2].start
        if t_start is not None:
            t_start = max(t_start - temporal_overlap, 0)
        t_end = slices[2].stop
        if t_end is not None:
            t_end = max(t_end + temporal_overlap, ends[2])
        t = slice(t_start, t_end)

        return s1, s2, t

    @staticmethod
    def get_high_res_slices(slices, s_enhance=3, t_enhance=4):
        """Get high res slices from low res slices

        Parameters
        ----------
        slices : list
            List of low res slices
        s_enhance : int
            Factor by which to enhance spatial dimensions of low resolution
            data
        t_enhance : int
            Factor by which to enhance temporal dimension of low resolution

        Returns
        -------
        s1 : slice
            spatial_1 high res slice
        s2 : slice
            spatial_2 high res slice
        t : slice
            temporal high res slice
        """

        s1 = slice(slices[0].start * s_enhance, slices[0].stop * s_enhance)
        s2 = slice(slices[1].start * s_enhance, slices[1].stop * s_enhance)
        t_start = slices[2].start
        if t_start is not None:
            t_start *= t_enhance
        t_end = slices[2].stop
        if t_end is not None:
            t_end *= t_enhance
        t = slice(t_start, t_end)

        return s1, s2, t

    @staticmethod
    def get_cropped_slices(low_res_slices, low_res_pad_slices, s_enhance=3,
                           t_enhance=4):
        """Get cropped spatial and temporal slices for stitching

        Parameters
        ----------
        low_res_slices : list
            List of unpadded slices for data chunk
            (spatial_1, spatial_2, temporal)
        low_res_pad_slices : list
            List of padded slices for data chunk
            (spatial_1, spatial_2, temporal)
        s_enhance : int
            Factor by which to enhance spatial dimensions of low resolution
            data
        t_enhance : int
            Factor by which to enhance temporal dimension of low resolution

        Returns
        -------
        s1 : slice
            spatial_1 high res cropped slice
        s2 : slice
            spatial_2 high res cropped slice
        t : slice
            temporal high res cropped slice
        """

        if low_res_pad_slices[0].start < low_res_slices[0].start:
            s1_start = s_enhance
            s1_start *= (low_res_slices[0].start - low_res_pad_slices[0].start)
        else:
            s1_start = None

        if low_res_pad_slices[0].stop > low_res_slices[0].stop:
            s1_stop = -s_enhance
            s1_stop *= (low_res_pad_slices[0].stop - low_res_slices[0].stop)
        else:
            s1_stop = None

        if low_res_pad_slices[1].start < low_res_slices[1].start:
            s2_start = s_enhance
            s2_start *= (low_res_slices[1].start - low_res_pad_slices[1].start)
        else:
            s2_start = None

        if low_res_pad_slices[1].stop > low_res_slices[1].stop:
            s2_stop = -s_enhance
            s2_stop *= (low_res_pad_slices[1].stop - low_res_slices[1].stop)
        else:
            s2_stop = None

        if (low_res_pad_slices[2].start is not None
                and low_res_pad_slices[2].start < low_res_slices[2].start):
            t_start = t_enhance
            t_start *= (low_res_slices[2].start - low_res_pad_slices[2].start)
        else:
            t_start = None

        if (low_res_pad_slices[2].stop is not None
                and low_res_pad_slices[2].stop > low_res_slices[2].stop):
            t_stop = -s_enhance
            t_stop *= (low_res_pad_slices[2].stop - low_res_slices[2].stop)
        else:
            t_stop = None

        s1 = slice(s1_start, s1_stop)
        s2 = slice(s2_start, s2_stop)
        t = slice(t_start, t_stop)

        return s1, s2, t

    @classmethod
    def combine_out_files(cls, file_paths, temporal_chunk_size,
                          out_file_prefix, fp_out=None):
        """Combine the output of each file_set passed to a different node

        Parameters
        ----------
        file_paths : list
            A list of NETCDF files to extract raster data from. Each file must
            have the same number of timesteps.
        out_file_prefix : str
            Prefix of path to forward pass output files. If None then data will
            be returned in an array and not saved.
        fp_out : str, optional
            Combined output file name, by default None

        Returns
        -------
        ndarray
            Array of combined output
            (spatial_1, spatial_2, temporal, 2)
        """

        logger.info('Combining forward pass output for '
                    f'{cls.file_info_logging(file_paths)}')

        file_t_steps = get_file_t_steps(file_paths)

        file_path_chunk_size = int(np.ceil(temporal_chunk_size / file_t_steps))
        n_chunks = int(np.ceil(len(file_paths) / file_path_chunk_size))
        file_chunks = np.array_split(np.arange(len(file_paths)), n_chunks)
        file_slices = [slice(f[0], f[-1] + 1) for f in file_chunks]

        file_ids = cls.get_file_ids(file_paths, file_slices)
        out_files = cls.get_output_file_names(out_file_prefix, file_ids)

        out = []
        for fp in out_files:
            with open(fp, 'rb') as fh:
                out.append(pickle.load(fh))

        out = np.concatenate(out, axis=2)

        if fp_out is not None:
            logger.info(f'Saving combined output: {fp_out}')
            with open(fp_out, 'wb') as fh:
                pickle.dump(out, fh)
        else:
            return out
