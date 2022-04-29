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

from sup3r.pipeline.gen_handling import ForwardPassHandler
from sup3r import __version__

logger = logging.getLogger(__name__)


class SUP3R:
    """Entry point for sup3r pipeline execution"""

    @staticmethod
    def _log_version():
        """Check NSRDB and python version and 64-bit and print to logger."""

        logger.info('Running NSRDB version: {}'.format(__version__))
        logger.info('Running python version: {}'.format(sys.version_info))

        is_64bits = sys.maxsize > 2 ** 32
        if is_64bits:
            logger.info('Running on 64-bit python, sys.maxsize: {}'
                        .format(sys.maxsize))
        else:
            logger.warning('Running 32-bit python, sys.maxsize: {}'
                           .format(sys.maxsize))

    @staticmethod
    def run_forward_passes(kwargs):
        """Run forward pass eagle jobs

        kwargs: dict
            file_paths, file_path_chunk_size,
            features, model_path,
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
            cache_file_prefix=None,
            out_file_prefix=None,
            overwrite_cache=True,
            spatial_overlap=15,
            temporal_overlap=15
        """

        slurm_manager = SLURM()

        default_kwargs = {"alloc": 'seasiawind',
                          "memory": 83,
                          "walltime": 1,
                          "basename": 'sup3r',
                          "feature": '--qos=high',
                          "stdout": './',
                          "log_file": 'sup3r_fwd_pass.log'}

        user_input = copy.deepcopy(default_kwargs)

        for k in default_kwargs:
            if k in kwargs:
                user_input[k] = kwargs.pop(k)

        stdout_path = user_input.get('stdout')
        log_file = user_input.get('log_file')
        log_file = os.path.join(stdout_path, log_file)

        init_logger(__name__, log_level='DEBUG', log_file=log_file)
        init_logger('sup3r', log_level='DEBUG', log_file=log_file)

        handler = ForwardPassHandler(**kwargs)

        for chunk, chunk_crop, time_shape, out_file, file_ids in zip(
                handler.padded_file_chunks,
                handler.file_crop_slices,
                handler.time_shapes, handler.out_files,
                handler.file_ids):

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
                      'crop_slice': chunk_crop,
                      'log_file': log_file}

            node_name = f'{user_input["basename"]}_'
            node_name += f'fwd_pass_{file_ids}'

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
