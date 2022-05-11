# -*- coding: utf-8 -*-
"""Entry point for sup3r pipeline execution

@author: bbenton
"""

import copy
import os
import logging
import sys
import tensorflow as tf
import sklearn
import pandas as pd
import numpy as np
import pprint

from rex.utilities.hpc import SLURM
from rex import init_logger
from rex.utilities.loggers import create_dirs

from sup3r.pipeline.forward_pass import ForwardPass
from sup3r import __version__


logger = logging.getLogger(__name__)


class SUP3R:
    """Entry point for sup3r pipeline execution"""

    def __init__(self, out_dir, make_out_dirs=True):
        """
        Parameters
        ----------
        out_dir : str
            Project directory.
        year : int | str
            Processing year.
        make_out_dirs : bool
            Flag to make output directories for logs, output, cache
        """

        self._out_dir = out_dir
        self._log_dir = os.path.join(out_dir, 'logs/')
        self._output_dir = os.path.join(out_dir, 'output/')
        self._cache_dir = os.path.join(out_dir, 'cache/')
        self._std_out_dir = os.path.join(out_dir, 'stdout/')

        if make_out_dirs:
            self.make_out_dirs()

    def make_out_dirs(self):
        """Ensure that all output directories exist"""
        all_dirs = [self._out_dir, self._log_dir, self._cache_dir,
                    self._output_dir, self._std_out_dir]
        for d in all_dirs:
            create_dirs(d)

    def _init_loggers(self, loggers=None, log_file='sup3r.log',
                      log_level='DEBUG', log_version=True,
                      use_log_dir=True):
        """Initialize sup3r loggers.

        Parameters
        ----------
        loggers : None | list | tuple
            List of logger names to initialize. None defaults to all NSRDB
            loggers.
        log_file : str
            Log file name. Will be placed in the sup3r out dir.
        log_level : str | None
            Logging level (DEBUG, INFO). If None, no logging will be
            initialized.
        date : None | datetime
            Optional date to put in the log file name.
        use_log_dir : bool
            Flag to use the class log directory (self._log_dir = ./logs/)
        """

        if log_level in ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'):

            if loggers is None:
                loggers = ('sup3r.sup3r', 'sup3r.pipeline.forward_pass',
                           'sup3r.preprocessing.data_handling')

            if log_file is not None and use_log_dir:
                log_file = os.path.join(self._log_dir, log_file)
                create_dirs(os.path.dirname(log_file))

            for name in loggers:
                init_logger(name, log_level=log_level, log_file=log_file)

        if log_version:
            self._log_version()

    @staticmethod
    def _parse_versions(version_record=None):
        """Parse version record if not provided by init.
        Parameters
        ----------
        version_record : dict | None
            Optional record of import package versions. None (default) will
            save active environment versions. A dictionary will be interpreted
            as versions from a loaded model and will be saved as an attribute.
        Returns
        -------
        version_record : dict
            A record of important versions that this model was built with.
        """
        active_versions = {'sup3r': __version__,
                           'tensorflow': tf.__version__,
                           'sklearn': sklearn.__version__,
                           'pandas': pd.__version__,
                           'numpy': np.__version__,
                           'python': sys.version,
                           }
        logger.info('Active python environment versions: \n{}'
                    .format(pprint.pformat(active_versions, indent=4)))

        if version_record is None:
            version_record = active_versions

        return version_record

    @staticmethod
    def _log_version():
        """Check SUP3R and python version and 64-bit and print to logger."""

        logger.info('Active python environment versions: \n{}'
                    .format(pprint.pformat(SUP3R._parse_versions(), indent=4)))

        is_64bits = sys.maxsize > 2 ** 32
        if is_64bits:
            logger.info(
                f'Running on 64-bit python, sys.maxsize: {sys.maxsize}')
        else:
            logger.warning(
                f'Running 32-bit python, sys.maxsize: {sys.maxsize}')

    @classmethod
    def run_forward_passes_single_node(cls, kwargs):
        """Run forward passes on single node

        Parameters
        ----------
        kwargs : dict
            This dict accepts the following keys:

            file_paths : list
                A list of NETCDF files to extract raster data from. Each file
                must have the same number of timesteps.
            model_path : str
                Path to SpatioTemporalGan used to generate high resolution data
            target : tuple
                (lat, lon) lower left corner of raster. Either need
                target+shape or raster_file.
            shape : tuple
                (rows, cols) grid size. Either need target+shape or
                raster_file.
            raster_file : str | None
                File for raster_index array for the corresponding target and
                shape. If specified the raster_index will be loaded from the
                file if it exists or written to the file if it does not yet
                exist.  If None raster_index will be calculated directly.
                Either need target+shape or raster_file.
            s_enhance : int
                Factor by which to enhance spatial dimensions of low resolution
                data
            t_enhance : int
                Factor by which to enhance temporal dimension of low resolution
                data
            temporal_slice : slice
                Slice defining size of full temporal domain. e.g. If shape is
                (100, 100) and temporal_slice is slice(0, 101, 1) then the full
                spatiotemporal data volume will be (100, 100, 100).
            temporal_extract_chunk_size : int
                Size of chunks to split time dimension into for parallel data
                extraction. If running in serial this can be set to the size
                of the full time index for best performance.
            forward_pass_chunk_shape : tuple
                Max shape of a chunk to pass through the generator. If running
                in serial set this equal to the shape of the full data volume
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
                Size of temporal overlap between chunks passed to forward
                passes for subsequent temporal stitching
        """

        out_dir = kwargs.pop('out_dir', './')
        sup3r = cls(out_dir)
        base_log_file = os.path.join(sup3r._log_dir, 'sup3r_fwd_pass.log')
        kwargs['cache_file_prefix'] = os.path.join(sup3r._cache_dir, 'cache')
        kwargs['out_file_prefix'] = os.path.join(sup3r._output_dir, 'output')

        handler = ForwardPass(**kwargs)
        logger.info(
            f'Running forward passes for {len(handler.file_ids)} file chunks')

        for chunk, chunk_crop, temporal_slice, out_file, file_ids in zip(
                handler.padded_file_chunks, handler.file_crop_slices,
                handler.temporal_slices, handler.out_files, handler.file_ids):

            log_file = base_log_file.replace('.log', f'_{file_ids}.log')

            sup3r._init_loggers(log_file=log_file)

            cache_file_prefix = handler.cache_file_prefix
            cache_file_prefix += f'_{file_ids}'
            kwargs = {'file_paths': handler.file_paths[chunk],
                      'out_file': out_file,
                      'temporal_slice': temporal_slice,
                      'crop_slice': chunk_crop}

            logger.info(
                'Running forward passes for '
                f'{handler.file_info_logging(handler.file_paths[chunk])} ')

            ForwardPass.forward_pass_file_chunk(**kwargs)

    @classmethod
    def run_forward_passes(cls, kwargs, eagle_args):
        """Run forward pass eagle jobs

        Parameters
        ----------
        kwargs : dict
            This dict accepts the following keys:

            file_paths : list
                A list of NETCDF files to extract raster data from. Each file
                must have the same number of timesteps.
            model_path : str
                Path to SpatioTemporalGan used to generate high resolution data
            target : tuple
                (lat, lon) lower left corner of raster. Either need
                target+shape or raster_file.
            shape : tuple
                (rows, cols) grid size. Either need target+shape or
                raster_file.
            raster_file : str | None
                File for raster_index array for the corresponding target and
                shape. If specified the raster_index will be loaded from the
                file if it exists or written to the file if it does not yet
                exist.  If None raster_index will be calculated directly.
                Either need target+shape or raster_file.
            s_enhance : int
                Factor by which to enhance spatial dimensions of low resolution
                data
            t_enhance : int
                Factor by which to enhance temporal dimension of low resolution
                data
            temporal_slice : slice
                Slice defining size of full temporal domain. e.g. If shape is
                (100, 100) and temporal_slice is slice(0, 101, 1) then the full
                spatiotemporal data volume will be (100, 100, 100).
            temporal_extract_chunk_size : int
                Size of chunks to split time dimension into for parallel data
                extraction. If running in serial this can be set to the size
                of the full time index for best performance.
            forward_pass_chunk_shape : tuple
                Max shape of a chunk to pass through the generator. If running
                in serial set this equal to the shape of the full data volume
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
                Size of temporal overlap between chunks passed to forward
                passes for subsequent temporal stitching

        eagle_args : dict
            Default inputs:
                {"alloc": 'seasiawind',
                 "memory": 83,
                 "walltime": 1,
                 "feature": '--qos=high'}
        """

        slurm_manager = SLURM()

        default_kwargs = {"alloc": 'seasiawind',
                          "memory": 83,
                          "walltime": 1,
                          "feature": '--qos=high'}

        user_input = copy.deepcopy(default_kwargs)
        user_input.update(eagle_args)

        out_dir = kwargs.pop('out_dir', './')
        sup3r = cls(out_dir)
        stdout_path = sup3r._std_out_dir
        base_log_file = os.path.join(sup3r._log_dir, 'sup3r_fwd_pass.log')
        kwargs['cache_file_prefix'] = os.path.join(sup3r._cache_dir, 'cache')
        kwargs['out_file_prefix'] = os.path.join(sup3r._output_dir, 'output')

        handler = ForwardPass(**kwargs)
        logger.info(
            f'Running forward passes for {len(handler.file_ids)} file chunks')

        for chunk, chunk_crop, temporal_slice, out_file, file_ids in zip(
                handler.padded_file_chunks, handler.file_crop_slices,
                handler.temporal_slices, handler.out_files,
                handler.file_ids):

            log_file = base_log_file.replace('.log', f'_{file_ids}.log')

            sup3r._init_loggers(log_file=log_file)

            user_input.update({'log_file': log_file})

            cache_file_prefix = handler.cache_file_prefix
            cache_file_prefix += f'_{file_ids}'
            kwargs = {'file_paths': handler.file_paths[chunk],
                      'out_file': out_file,
                      'temporal_slice': temporal_slice,
                      'crop_slice': chunk_crop}

            node_name += f'sup3r_fwd_pass_{file_ids}'

            cmd = (
                "python -c \"from sup3r.pipeline.gen_handling "
                "import ForwardPassHandler;"
                f"ForwardPassHandler.forward_pass_file_chunk(**{kwargs})\"")

            logger.info(
                'Running forward passes for '
                f'{handler.file_info_logging(handler.file_paths[chunk])} ')
            out = slurm_manager.sbatch(cmd, alloc=user_input["alloc"],
                                       memory=user_input["memory"],
                                       walltime=user_input["walltime"],
                                       feature=user_input["feature"],
                                       name=node_name,
                                       stdout_path=stdout_path)[0]

            print(f'\ncmd:\n{cmd}\n')

            if out:
                msg = (f'Kicked off job "{node_name}" (SLURM jobid #{out}) on '
                       f'Eagle with {user_input}')
            else:
                msg = (f'Was unable to kick off job "{node_name}". Please see '
                       'the stdout error messages')
            print(msg)

    @classmethod
    def combine_node_output(cls, kwargs, eagle_args):
        """Combine foward pass output from all nodes.

        kwargs: dict
            Required inputs:
                file_paths, temporal_chunk_size, fp_out
        eagle_args : dict
            Default inputs:
                {"alloc": 'seasiawind',
                 "memory": 83,
                 "walltime": 1,
                 "basename": 'sup3r',
                 "feature": '--qos=high',
                 "stdout": './'}
        """

        slurm_manager = SLURM()

        default_kwargs = {"alloc": 'seasiawind',
                          "memory": 83,
                          "walltime": 1,
                          "basename": 'sup3r',
                          "feature": '--qos=high',
                          "stdout": './'}

        user_input = copy.deepcopy(default_kwargs)
        user_input.update(eagle_args)

        out_dir = kwargs.pop('out_dir', './')
        sup3r = cls(out_dir)
        stdout_path = sup3r._std_out_dir
        log_file = os.path.join(sup3r._log_dir, 'sup3r_fwd_pass.log')
        kwargs['out_file_prefix'] = os.path.join(sup3r._output_dir, 'output')

        sup3r._init_loggers(log_file=log_file)

        node_name = f'{user_input["basename"]}_combine_node_output'

        cmd = ("python -c \"from sup3r.pipeline.gen_handling "
               "import ForwardPassHandler;"
               f"ForwardPassHandler.combine_output_files(**{kwargs})\"")

        out = slurm_manager.sbatch(
            cmd, alloc=user_input["alloc"],
            memory=user_input["memory"],
            walltime=user_input["walltime"],
            feature=user_input["feature"],
            name=node_name,
            stdout_path=stdout_path)[0]

        print(f'\ncmd:\n{cmd}\n')

        if out:
            msg = (f'Kicked off job "{node_name}" (SLURM jobid #{out}) on '
                   f'Eagle with {user_input}')
        else:
            msg = (f'Was unable to kick off job "{node_name}". Please see the '
                   'stdout error messages')
        print(msg)
