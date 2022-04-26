# -*- coding: utf-8 -*-
"""
Sup3r forward pass handling module.
"""

import numpy as np

from sup3r.preprocessing.data_handling import DataHandlerNC


class ForwardPassHandler:
    """Class to handle parallel forward
    passes through the generator with multiple
    data files.
    """

    def __init__(self, file_paths, file_path_chunk_size,
                 features, model, target=None, shape=None,
                 raster_file=None, s_enhance=3,
                 t_enhance=4, spatial_chunk_size=(10, 10),
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
        model : SpatioTemporalGan
            SpatioTemporalGan used to generate high resolution data
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
        t_enhance : int
            Factor by which to enhance temporal dimension
            of the low resolution data
        s_enhance : int
            Factor by which to enhance spatial dimensions
            of the low resolution data
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
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        self.spatial_chunk_size = spatial_chunk_size
        self.temporal_chunk_size = temporal_chunk_size
        self.temporal_shape = temporal_shape
        self.max_workers = max_workers
        self.model = model

    def kick_off_node(self, file_paths, file_chunk, features, raster_file=None,
                      target=None, shape=None, s_enhance=3, t_enhance=4,
                      spatial_chunk_size=(10, 10), temporal_chunk_size=24,
                      temporal_shape=slice(None, None, 1),
                      max_workers=None):
        """
        file_paths : list
            A list of NETCDF files to extract raster data from
        file_chunk : int
            Slice indicating which set of files to extract. e.g. if file_chunk
            equals slice(0, 4) then the first four files will be extract on
            this node.
        features : list
            list of features to extract
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
        t_enhance : int
            Factor by which to enhance temporal dimension
            of the low resolution data
        s_enhance : int
            Factor by which to enhance spatial dimensions
            of the low resolution data
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

        raise NotImplementedError

    @staticmethod
    def forward_pass_chunk(file_paths, features, model,
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
            file_paths, spatial_slice, features, time_pruning=1)
        data = data[tuple(spatial_slice + [temporal_slice] + [slice(None)])]

        high_res = model.generate(np.expand_dims(data, axis=0))

        return high_res[0]




