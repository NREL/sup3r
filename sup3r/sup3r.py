# -*- coding: utf-8 -*-
"""Entry point for sup3r pipeline execution

@author: bbenton
"""

import copy
import os
import logging
import sys

from rex.utilities.hpc import SLURM
from rex import init_logger
from rex.utilities.loggers import create_dirs

from sup3r.pipeline.gen_handling import ForwardPassHandler
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
                loggers = ('sup3r.sup3r', 'sup3r.forward_pass',
                           'sup3r.combine_nodes')

            if log_file is not None and use_log_dir:
                log_file = os.path.join(self._log_dir, log_file)
                create_dirs(os.path.dirname(log_file))

            for name in loggers:
                init_logger(name, log_level=log_level, log_file=log_file)

        if log_version:
            self._log_version()

    @staticmethod
    def _log_version():
        """Check SUP3R and python version and 64-bit and print to logger."""

        logger.info(f'Running SUP3R version: {__version__}')
        logger.info(f'Running python version: {sys.version_info}')

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

        kwargs: dict
            Required inputs:
                file_paths, features, model_path
            Default inputs:
                target=None, shape=None,
                temporal_shape=slice(None, None, 1),
                temporal_pass_chunk_size=100,
                spatial_chunk_size=(100, 100),
                raster_file=None,
                s_enhance=3,
                t_enhance=4,
                max_extract_workers=None,
                max_compute_workers=None,
                max_pass_workers=None,
                temporal_extract_chunk_size=100,
                overwrite_cache=True,
                spatial_overlap=15,
                temporal_overlap=15,
                out_dir='./'
        """

        out_dir = kwargs.pop('out_dir', './')
        sup3r = cls(out_dir)
        base_log_file = os.path.join(sup3r._log_dir, 'sup3r_fwd_pass.log')
        kwargs['cache_file_prefix'] = os.path.join(sup3r._cache_dir, 'cache')
        kwargs['out_file_prefix'] = os.path.join(sup3r._output_dir, 'output')

        handler = ForwardPassHandler(**kwargs)
        logger.info(
            f'Running forward passes for {len(handler.file_ids)} file chunks')

        for chunk, chunk_crop, time_shape, out_file, file_ids in zip(
                handler.padded_file_chunks,
                handler.file_crop_slices,
                handler.time_shapes, handler.out_files,
                handler.file_ids):

            log_file = base_log_file.replace('.log', f'_{file_ids}.log')

            sup3r._init_loggers(log_file=log_file)

            cache_file_prefix = handler.cache_file_prefix
            cache_file_prefix += f'_{file_ids}'
            kwargs = {'file_paths': handler.file_paths[chunk],
                      'model_path': handler.model_path,
                      'features': handler.features,
                      'target': handler.target,
                      'shape': handler.shape,
                      'temporal_shape': time_shape,
                      'spatial_chunk_size': handler.spatial_chunk_size,
                      'raster_file': handler.raster_file,
                      'max_extract_workers': handler.max_extract_workers,
                      'max_compute_workers': handler.max_compute_workers,
                      'temporal_extract_chunk_size':
                          handler.temporal_extract_chunk_size,
                      'temporal_pass_chunk_size':
                          handler.temporal_pass_chunk_size,
                      'cache_file_prefix': cache_file_prefix,
                      'max_pass_workers': handler.max_pass_workers,
                      's_enhance': handler.s_enhance,
                      't_enhance': handler.t_enhance,
                      'out_file': out_file,
                      'overwrite_cache': handler.overwrite_cache,
                      'spatial_overlap': handler.spatial_overlap,
                      'temporal_overlap': handler.temporal_overlap,
                      'crop_slice': chunk_crop}

            logger.info(
                'Running forward passes '
                f'{handler.file_info_logging(handler.file_paths[chunk])} ')
            ForwardPassHandler.forward_pass_file_chunk(**kwargs)

    @classmethod
    def run_forward_passes(cls, kwargs, eagle_args):
        """Run forward pass eagle jobs

        kwargs: dict
            Required inputs:
                file_paths, features, model_path
            Default inputs:
                target=None, shape=None,
                temporal_shape=slice(None, None, 1),
                temporal_pass_chunk_size=100,
                spatial_chunk_size=(100, 100),
                raster_file=None,
                s_enhance=3,
                t_enhance=4,
                max_extract_workers=None,
                max_compute_workers=None,
                max_pass_workers=None,
                temporal_extract_chunk_size=100,
                overwrite_cache=True,
                spatial_overlap=15,
                temporal_overlap=15,
        eagle_args : dict
            Default inputs:
                {"alloc": 'seasiawind',
                 "memory": 83,
                 "walltime": 1,
                 "feature": '--qos=high',
                 "stdout": './'}

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

        handler = ForwardPassHandler(**kwargs)
        logger.info(
            f'Running forward passes for {len(handler.file_ids)} file chunks')

        for chunk, chunk_crop, time_shape, out_file, file_ids in zip(
                handler.padded_file_chunks,
                handler.file_crop_slices,
                handler.time_shapes, handler.out_files,
                handler.file_ids):

            log_file = base_log_file.replace('.log', f'_{file_ids}.log')

            sup3r._init_loggers(log_file=log_file)

            user_input.update({'log_file': log_file})

            cache_file_prefix = handler.cache_file_prefix
            cache_file_prefix += f'_{file_ids}'
            kwargs = {'file_paths': handler.file_paths[chunk],
                      'model_path': handler.model_path,
                      'features': handler.features,
                      'target': handler.target,
                      'shape': handler.shape,
                      'temporal_shape': time_shape,
                      'spatial_chunk_size': handler.spatial_chunk_size,
                      'raster_file': handler.raster_file,
                      'max_extract_workers': handler.max_extract_workers,
                      'max_compute_workers': handler.max_compute_workers,
                      'temporal_extract_chunk_size':
                          handler.temporal_extract_chunk_size,
                      'temporal_pass_chunk_size':
                          handler.temporal_pass_chunk_size,
                      'cache_file_prefix': cache_file_prefix,
                      'max_pass_workers': handler.max_pass_workers,
                      's_enhance': handler.s_enhance,
                      't_enhance': handler.t_enhance,
                      'out_file': out_file,
                      'overwrite_cache': handler.overwrite_cache,
                      'spatial_overlap': handler.spatial_overlap,
                      'temporal_overlap': handler.temporal_overlap,
                      'crop_slice': chunk_crop}

            node_name += f'sup3r_fwd_pass_{file_ids}'

            cmd = (
                "python -c \"from sup3r.pipeline.gen_handling "
                "import ForwardPassHandler;"
                f"ForwardPassHandler.forward_pass_file_chunk(**{kwargs})\"")

            logger.info(
                'Running forward passes '
                f'{handler.file_info_logging(handler.file_paths[chunk])} ')
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
            msg = (f'Kicked off job "{node_name}" '
                   f'(SLURM jobid #{out}) on '
                   f'Eagle with {user_input}')
        else:
            msg = (f'Was unable to kick off job '
                   f'"{node_name}". Please see the '
                   'stdout error messages')
        print(msg)
