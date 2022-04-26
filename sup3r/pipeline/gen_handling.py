# -*- coding: utf-8 -*-
"""
Sup3r forward pass handling module.
"""

import numpy as np
import logging
from concurrent.futures import as_completed
from datetime import datetime as dt

from rex.utilities.execution import SpawnProcessPool

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
                 features, model_path, target=None, shape=None,
                 raster_file=None,
                 spatial_chunk_size=(10, 10),
                 temporal_chunk_size=24,
                 temporal_shape=slice(None, None, 1),
                 max_workers=None):
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
        spatial_chunk_size : tuple
            Max size of a spatial domain to pass through generator during
            subprocesses. The full spatial domain (shape) will be chunked into
            pieces with max spatial extent equal to spatial_chunk_size
        temporal_shape : slice
            Slice defining size of full temporal domain. e.g. If shape is
            (100, 100) and temporal_shape is slice(0, 101, 1) then the full
            spatiotemporal data volume will be (100, 100, 100).
        temporal_chunk_size : int
            Max size of a temporal chunk to pass through the generator. e.g.
            If spatial_chunk_size is (10, 10) and temporal chunk size is 10
            then the spatiotemporal shape of each data chunk passed through
            the generator will be a max shape of (10, 10, 10).
        max_workers : int | None
            Max number of workers to use during calls to SpawnProcessPool on
            each node.
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
        self.max_workers = max_workers
        self.model_path = model_path

    @classmethod
    def kick_off_node(cls, file_paths, file_chunk, model_path, features,
                      spatial_1_shape=slice(0, 100),
                      spatial_2_shape=slice(0, 100),
                      spatial_chunk_size=(10, 10),
                      temporal_chunk_size=24,
                      temporal_shape=slice(0, 100, 1),
                      max_workers=None):
        """
        Routine to run forward pass on all data chunks associated with the
        files in file_paths

        Parameters
        ----------

        file_paths : list
            A list of NETCDF files to extract raster data from
        file_chunk : int
            Slice indicating which set of files to extract. e.g. if file_chunk
            equals slice(0, 4) then the first four files will be extract on
            this node.
        model_path : str
            Path to SpatioTemporalGan used to generate high resolution data
        features : list
            list of features to extract
        spatial_1_shape : slice
            Slice specifying spatial extent of spatial_1 for full domain.
        spatial_2_shape : slice
            Slice specifying spatial extent of spatial_2 for full domain.
        spatial_chunk_size : tuple
            Max size of a spatial domain to pass through generator during
            subprocesses. The full spatial domain (shape) will be chunked into
            pieces with max spatial extent equal to spatial_chunk_size
        temporal_shape : slice
            Slice defining size of full temporal domain. e.g. If shape is
            (100, 100) and temporal_shape is slice(0, 101, 1) then the full
            spatiotemporal data volume will be (100, 100, 100).
        temporal_chunk_size : int
            Max size of a temporal chunk to pass through the generator. e.g.
            If spatial_chunk_size is (10, 10) and temporal chunk size is 10
            then the spatiotemporal shape of each data chunk passed through
            the generator will be a max shape of (10, 10, 10).
        max_workers : int | None
            Max number of workers to use during calls to SpawnProcessPool on
            each node.
        """

        s1_slices = np.arange(
            spatial_1_shape.start, spatial_1_shape.stop)
        n_chunks = int(np.ceil(len(s1_slices) / spatial_chunk_size[0]))
        s1_slices = np.array_split(s1_slices, n_chunks)
        s1_slices = [slice(s1[0], s1[-1] + 1) for s1 in s1_slices]

        s2_slices = np.arange(
            spatial_2_shape.start, spatial_2_shape.stop)
        n_chunks = int(np.ceil(len(s2_slices) / spatial_chunk_size[1]))
        s2_slices = np.array_split(s2_slices, n_chunks)
        s2_slices = [slice(s2[0], s2[-1] + 1) for s2 in s2_slices]

        temporal_slices = np.arange(
            temporal_shape.start, temporal_shape.stop, temporal_shape.step)
        n_chunks = int(np.ceil(len(temporal_slices) / temporal_chunk_size))
        temporal_slices = np.array_split(temporal_slices, n_chunks)
        temporal_slices = [slice(t[0], t[-1] + 1) for t in temporal_slices]

        spatiotemporal_slices = []

        data = np.zeros(((spatial_1_shape.stop - spatial_1_shape.start + 1),
                         (spatial_2_shape.stop - spatial_2_shape.start + 1),
                         (temporal_shape.stop - temporal_shape.start + 1), 2),
                        dtype=np.float32)

        for s1 in s1_slices:
            for s2 in s2_slices:
                for t in temporal_slices:
                    spatiotemporal_slices.append([s1, s2, t])

        if max_workers == 1:
            for slices in spatiotemporal_slices:
                s1 = slices[0]
                s2 = slices[1]
                t = slices[2]
                data[s1, s2, t, :] = cls.forward_pass_chunk(
                    file_paths=file_paths[file_chunk], features=features,
                    model_path=model_path, spatial_slice=[s1, s2])
            return data

        futures = {}
        now = dt.now()
        with SpawnProcessPool(max_workers=max_workers) as exe:
            for slices in spatiotemporal_slices:
                future = exe.submit(cls.forward_pass_chunk,
                                    file_paths=file_paths[file_chunk],
                                    features=features,
                                    model_path=model_path,
                                    spatial_slice=[slices[0], slices[1]])
                meta = {'slices': slices}
                futures[future] = meta

            logger.info(
                f'Started forward pass for {len(spatiotemporal_slices)} '
                f'chunks in {dt.now() - now}.')

            for i, future in enumerate(as_completed(futures)):
                slices = futures[future]
                s1 = slices[0]
                s2 = slices[1]
                t = slices[2]
                data[s1, s2, t, :] = future.result()

                if i % (len(futures) // 10 + 1) == 0:
                    logger.debug(f'{i+1} out of {len(futures)} forward passes '
                                 'completed.')

        return data

    @staticmethod
    def forward_pass_chunk(file_paths, features, model_path,
                           spatial_slice, temporal_slice):

        """Run forward pass on smallest data chunk

        Parameters
        ----------
        file_paths : list
            A list of NETCDF files to extract raster data from
        features : list
            list of features to extract
        model : SpatioTemporalGan
            SpatioTemporalGan used to generate high resolution data
        spatial_slice : list
            List of slices specifying spatial extent to extract from data
        temporal_slice : slice
            Slice specifying temporal extent to extract from data


        Returns
        -------
        ndarray
            High resolution data generated by GAN
        """

        data = DataHandlerNC.extract_data(
            file_paths, spatial_slice, features,
            time_shape=temporal_slice)

        model = SpatioTemporalGan(model_path)
        high_res = model.generate(np.expand_dims(data, axis=0))

        return high_res[0]
