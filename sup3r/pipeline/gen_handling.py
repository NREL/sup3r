# -*- coding: utf-8 -*-
"""
Sup3r forward pass handling module.
"""

import numpy as np
import logging
from concurrent.futures import as_completed
from datetime import datetime as dt
import pickle
import copy
import os

from rex.utilities.execution import SpawnProcessPool
from rex.utilities.hpc import SLURM

from sup3r.preprocessing.data_handling import DataHandlerNC
from sup3r.models.models import SpatioTemporalGan
from sup3r import __version__

np.random.seed(42)

logger = logging.getLogger(__name__)


class ForwardPassHandler:
    """Class to handle parallel forward
    passes through the generator with multiple
    data files.
    """

    def __init__(self, file_paths, file_path_chunk_size,
                 features, model_path,
                 target=None, shape=None,
                 temporal_shape=slice(None, None, 1),
                 temporal_chunk_size=100,
                 spatial_chunk_size=(100, 100),
                 raster_file=None,
                 s_enhance=3,
                 t_enhance=4,
                 max_extract_workers=None,
                 max_compute_workers=None,
                 max_pass_workers=None,
                 time_chunk_size=100,
                 cache_file_prefix=None,
                 out_file_prefix=None,
                 overwrite_cache=True,
                 spatial_overlap=15,
                 temporal_overlap=15):

        """Use these inputs to initialize data handlers
        on different nodes and to define the size of
        the data chunks that will be passed through the
        generator.

        Parameters
        ----------
        file_paths : list
            A list of NETCDF files to extract raster data from
        file_path_chunk_size : int
            Number of file paths to pass to each node for subsequent
            data extraction and forward pass
        features : list
            list of features to extract
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
        spatial_chunk_size : tuple
            Max size of a spatial domain to pass through generator during
            subprocesses. The full spatial domain (shape) will be chunked into
            pieces with max spatial extent equal to spatial_chunk_size. If
            running in serial set equal to the full spatial domain for best
            performance.
        temporal_shape : slice
            Slice defining size of full temporal domain. e.g. If shape is
            (100, 100) and temporal_shape is slice(0, 101, 1) then the full
            spatiotemporal data volume will be (100, 100, 100).
        temporal_chunk_size : int
            Max size of a temporal chunk to pass through the generator. e.g.
            If spatial_chunk_size is (10, 10) and temporal chunk size is 10
            then the spatiotemporal shape of each data chunk passed through
            the generator will be a max shape of (10, 10, 10). If running
            in serial set this equal to the length of the full time index
            for best performance.
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

        n_chunks = int(np.ceil(len(file_paths) / file_path_chunk_size))
        file_chunks = np.array_split(np.arange(len(file_paths)), n_chunks)
        self.file_paths = sorted(file_paths)
        self.file_chunks = [slice(f[0], f[-1] + 1) for f in file_chunks]
        self.features = features
        self.raster_file = raster_file
        self.target = target
        self.shape = shape
        self.spatial_chunk_size = spatial_chunk_size
        self.temporal_chunk_size = temporal_chunk_size
        self.temporal_shape = temporal_shape
        self.max_pass_workers = max_pass_workers
        self.max_extract_workers = max_extract_workers
        self.max_compute_workers = max_compute_workers
        self.model_path = model_path
        self.cache_file_prefix = cache_file_prefix
        self.out_file_prefix = out_file_prefix
        self.time_chunk_size = time_chunk_size
        self.overwrite_cache = overwrite_cache
        self.t_enhance = t_enhance
        self.s_enhance = s_enhance
        self.spatial_overlap = spatial_overlap
        self.temporal_overlap = temporal_overlap

        if self.out_file_prefix is not None:
            dirname = os.path.dirname(self.out_file_prefix)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

    def run(self, **kwargs):
        """Run forward pass eagle jobs
        """

        slurm_manager = SLURM()

        default_kwargs = {"alloc": 'seasiawind',
                          "memory": 83,
                          "walltime": 40,
                          "basename": 'sup3r',
                          "feature": '--qos=high'}

        user_input = copy.deepcopy(default_kwargs)
        user_input.update(kwargs)
        stdout_path = user_input.get('stdout', './')

        for chunk in self.file_chunks:
            file_id_range = [str(chunk.start), str(chunk.stop - 1)]
            cache_file_prefix = self.cache_file_prefix
            cache_file_prefix += f'_{"-".join(file_id_range)}'
            out_file = self.out_file_prefix
            out_file += f'fwd_pass_output_{"-".join(file_id_range)}.pkl'
            kwargs = {'file_paths': self.file_paths[chunk],
                      'model_path': self.model_path,
                      'features': self.features,
                      'target': self.target,
                      'shape': self.shape,
                      'temporal_shape': self.temporal_shape,
                      'spatial_chunk_size': self.spatial_chunk_size,
                      'raster_file': self.raster_file,
                      'max_extract_workers': self.max_extract_workers,
                      'max_compute_workers': self.max_compute_workers,
                      'time_chunk_size': self.time_chunk_size,
                      'cache_file_prefix': self.cache_file_prefix,
                      'max_pass_workers': self.max_pass_workers,
                      's_enhance': self.s_enhance,
                      't_enhance': self.t_enhance,
                      'out_file': out_file,
                      'overwrite_cache': self.overwrite_cache,
                      'spatial_overlap': self.spatial_overlap,
                      'temporal_overlap': self.temporal_overlap}

            node_name = f'{user_input["basename"]}_'
            node_name += f'fwd_pass_{"-".join(file_id_range)}'

            cmd = ("python -c \"from sup3r.pipeline.gen_handling "
                   "import ForwardPassHandler;"
                   f"ForwardPassHandler.kick_off_node(**{kwargs})\"")
            out = slurm_manager.sbatch(
                cmd, alloc=user_input["alloc"],
                memory=user_input["memory"],
                walltime=user_input["walltime"],
                feature=user_input["feature"],
                name=node_name,
                stdout_path=stdout_path)[0]

            print(f'\ncmd:\n{cmd}\n')

            if out:
                msg = (f'Kicked off job "{node_name}" '
                       f'(SLURM jobid #{out}) on '
                       f'Eagle with {user_input}')
            else:
                msg = (f'Was unable to kick off job '
                       f'"{node_name}". Please see the '
                       'stdout error messages')
            print(msg)

    @classmethod
    def get_chunk_slices(cls,
                         data_shape=None,
                         temporal_chunk_size=24,
                         spatial_chunk_size=(10, 10),
                         s_enhance=3, t_enhance=4,
                         spatial_overlap=15,
                         temporal_overlap=15):
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

        s1_slices = np.arange(0, data_shape[0])
        n_chunks = int(np.ceil(len(s1_slices) / spatial_chunk_size[0]))
        s1_slices = np.array_split(s1_slices, n_chunks)
        s1_slices = [slice(s1[0], s1[-1] + 1) for s1 in s1_slices]

        s2_slices = np.arange(0, data_shape[1])
        n_chunks = int(np.ceil(len(s2_slices) / spatial_chunk_size[1]))
        s2_slices = np.array_split(s2_slices, n_chunks)
        s2_slices = [slice(s2[0], s2[-1] + 1) for s2 in s2_slices]

        temporal_chunk_size = np.min(
            [temporal_chunk_size, data_shape[2]])
        temporal_slices = np.arange(0, data_shape[2])
        n_chunks = int(np.ceil(len(temporal_slices) / temporal_chunk_size))
        temporal_slices = np.array_split(temporal_slices, n_chunks)
        temporal_slices = [slice(t[0], t[-1] + 1) for t in temporal_slices]

        low_res_pad_slices = []
        high_res_slices = []
        high_res_crop_slices = []

        for s1 in s1_slices:
            for s2 in s2_slices:
                for t in temporal_slices:

                    s1_h, s2_h, t_h = cls.high_res_slices(
                        [s1, s2, t],
                        s_enhance=s_enhance,
                        t_enhance=t_enhance)
                    high_res_slices.append([s1_h, s2_h, t_h])

                    s1_p, s2_p, t_p = cls.pad_slices(
                        [s1, s2, t],
                        ends=list(data_shape),
                        spatial_overlap=spatial_overlap,
                        temporal_overlap=temporal_overlap)
                    low_res_pad_slices.append([s1_p, s2_p, t_p])

                    s1_c, s2_c, t_c = cls.cropped_slices(
                        [s1, s2, t], [s1_p, s2_p, t_p],
                        s_enhance=s_enhance, t_enhance=t_enhance)
                    high_res_crop_slices.append([s1_c, s2_c, t_c])

                    # pad spatial and temporal slices
                    # for chunk overlap

        return low_res_pad_slices, \
            high_res_slices, high_res_crop_slices

    @classmethod
    def kick_off_node(cls, file_paths=None, model_path=None,
                      features=None,
                      target=None, shape=None,
                      temporal_shape=slice(None, None, 1),
                      temporal_pass_chunk_size=24,
                      spatial_chunk_size=(10, 10),
                      raster_file=None,
                      max_extract_workers=None,
                      max_compute_workers=None,
                      temporal_extract_chunk_size=100,
                      cache_file_prefix=None,
                      max_pass_workers=None,
                      s_enhance=3, t_enhance=4,
                      out_file=None,
                      overwrite_cache=True,
                      spatial_overlap=15,
                      temporal_overlap=15):
        """
        Routine to run forward pass on all data chunks associated with the
        files in file_paths

        Parameters
        ----------

        file_paths : list
            A list of NETCDF files to extract raster data from
        model_path : str
            Path to SpatioTemporalGan used to generate high resolution data
        features : list
            list of features to extract
        data_shape : slice
            Size of data volume corresponding to the spatial and temporal
            extent of files in file_paths.
        spatial_chunk_size : tuple
            Max size of a spatial domain to pass through generator during
            subprocesses. The full spatial domain (shape) will be chunked into
            pieces with max spatial extent equal to spatial_chunk_size
        temporal_pass_chunk_size : int
            Max size of a temporal chunk to pass through the generator. e.g.
            If spatial_chunk_size is (10, 10) and temporal chunk size is 10
            then the spatiotemporal shape of each data chunk passed through
            the generator will be a max shape of (10, 10, 10).
        temporal_extract_chunk_size : int
            Size of chunks to split time dimension into for parallel data
            extraction. If running in serial this can be set to the size
            of the full time index for best performance.
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
        s_enhance : int
            Factor by which to enhance spatial dimensions of low resolution
            data
        t_enhance : int
            Factor by which to enhance temporal dimension of low resolution
            data
        out_file : str | None
            File to store forward pass output. If None data will be returned in
            an array instead of saved.
        overwrite_cache : bool
            Whether to overwrite cache files
        spatial_overlap : int
            Size of spatial overlap between chunks passed to forward passes
            for subsequent spatial stitching
        temporal_overlap : int
            Size of temporal overlap between chunks passed to forward passes
            for subsequent temporal stitching
        """

        handler = DataHandlerNC(file_paths, features,
                                target=target, shape=shape,
                                time_shape=temporal_shape,
                                raster_file=raster_file,
                                max_extract_workers=max_extract_workers,
                                max_compute_workers=max_compute_workers,
                                cache_file_prefix=cache_file_prefix,
                                time_chunk_size=temporal_extract_chunk_size,
                                overwrite_cache=overwrite_cache,
                                val_split=0.0)
        handler.load_cached_data()

        data_shape = (shape[0], shape[1], len(handler.time_index))

        low_res_pad_slices, high_res_slices, \
            high_res_crop_slices = cls.get_chunk_slices(
                data_shape=data_shape,
                temporal_chunk_size=temporal_pass_chunk_size,
                spatial_chunk_size=spatial_chunk_size,
                s_enhance=s_enhance, t_enhance=t_enhance,
                spatial_overlap=spatial_overlap,
                temporal_overlap=temporal_overlap)

        data = np.zeros(
            (s_enhance * data_shape[0], s_enhance * data_shape[1],
             t_enhance * data_shape[2], 2),
            dtype=np.float32)

        if max_pass_workers == 1:
            for s_high, s_low_pad, s_high_crop in zip(
                    high_res_slices, low_res_pad_slices,
                    high_res_crop_slices):

                data_chunk = handler.data[tuple(s_low_pad + [slice(None)])]
                data[tuple(s_high + [slice(None)])] = cls.forward_pass_chunk(
                    data_chunk, crop_slices=s_high_crop,
                    model_path=model_path)
        else:
            futures = {}
            now = dt.now()
            with SpawnProcessPool(max_workers=max_pass_workers) as exe:
                for s_high, s_low_pad, s_high_crop in zip(
                        high_res_slices, low_res_pad_slices,
                        high_res_crop_slices):

                    data_chunk = handler.data[
                        tuple(s_low_pad + [slice(None)])]
                    future = exe.submit(
                        cls.forward_pass_chunk,
                        data_chunk=data_chunk,
                        crop_slices=s_high_crop,
                        model_path=model_path)
                    meta = {'s_high': s_high}
                    futures[future] = meta

                logger.info(
                    f'Started forward pass for {len(high_res_slices)} '
                    f'chunks in {dt.now() - now}.')

                for i, future in enumerate(as_completed(futures)):
                    slices = futures[future]
                    data[tuple(slices['s_high']
                               + [slice(None)])] = future.result()
                    if i % (len(futures) // 10 + 1) == 0:
                        logger.debug(
                            f'{i+1} out of {len(futures)} forward passes '
                            'completed.')

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
            Path to file for SpatioTemporalGan used to
            generate high resolution data
        crop_slices : list
            List of slices for extracting cropped region
            of interest from output. Output can include
            an extra overlapping boundary to facilitate
            stitching of chunks

        Returns
        -------
        ndarray
            High resolution data generated by GAN
        """

        model = SpatioTemporalGan.load(model_path)
        data_chunk = np.expand_dims(data_chunk, axis=0)
        high_res = model.generate(data_chunk)

        return high_res[0][tuple(crop_slices + [slice(None)])]

    @staticmethod
    def pad_slices(slices, ends, spatial_overlap=15, temporal_overlap=15):
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
            Size of spatial overlap between chunks passed to forward passes
            for subsequent spatial stitching
        temporal_overlap : int
            Size of temporal overlap between chunks passed to forward passes
            for subsequent temporal stitching

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

        t = slice(max(slices[2].start - temporal_overlap, 0),
                  min(slices[2].stop + temporal_overlap, ends[2]))
        return s1, s2, t

    @staticmethod
    def high_res_slices(slices, s_enhance=3, t_enhance=4):
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
            data

        Returns
        -------
        s1 : slice
            spatial_1 high res slice
        s2 : slice
            spatial_2 high res slice
        t : slice
            temporal high res slice
        """

        s1 = slice(slices[0].start * s_enhance,
                   slices[0].stop * s_enhance)
        s2 = slice(slices[1].start * s_enhance,
                   slices[1].stop * s_enhance)
        t = slice(slices[2].start * t_enhance,
                  slices[2].stop * t_enhance)
        return s1, s2, t

    @staticmethod
    def cropped_slices(low_res_slices, low_res_pad_slices,
                       s_enhance=3,
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
            data

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

        if low_res_pad_slices[2].start < low_res_slices[2].start:
            t_start = t_enhance
            t_start *= (low_res_slices[2].start - low_res_pad_slices[2].start)
        else:
            t_start = None

        if low_res_pad_slices[2].stop > low_res_slices[2].stop:
            t_stop = -t_enhance
            t_stop *= (low_res_pad_slices[2].stop - low_res_slices[2].stop)
        else:
            t_stop = None

        s1 = slice(s1_start, s1_stop)
        s2 = slice(s2_start, s2_stop)
        t = slice(t_start, t_stop)

        return s1, s2, t
