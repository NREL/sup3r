# -*- coding: utf-8 -*-
"""
Sup3r data pipeline architecture.
"""
import logging
import os
import json

from reV.pipeline.pipeline import Pipeline
from rex.utilities.loggers import init_logger, create_dirs

from sup3r.pipeline.config import Sup3rPipelineConfig
from sup3r.utilities import ModuleName
from sup3r.models.base import Sup3rGan
from sup3r.postprocessing.file_handling import OutputHandlerH5

logger = logging.getLogger(__name__)


class Sup3rPipeline(Pipeline):
    """NSRDB pipeline execution framework."""

    CMD_BASE = 'python -m sup3r.cli -c {fp_config} {command}'
    COMMANDS = ModuleName.all_names()
    RETURN_CODES = {0: 'successful',
                    1: 'running',
                    2: 'failed',
                    3: 'complete'}

    def __init__(self, pipeline, monitor=True, verbose=False):
        """
        Parameters
        ----------
        pipeline : str | dict
            Pipeline config file path or dictionary.
        monitor : bool
            Flag to perform continuous monitoring of the pipeline.
        verbose : bool
            Flag to submit pipeline steps with -v flag for debug logging
        """
        self.monitor = monitor
        self.verbose = verbose
        self._config = Sup3rPipelineConfig(pipeline)
        self._run_list = self._config.pipeline
        self._init_status()

        # init logger for pipeline module if requested in input config
        if 'logging' in self._config:
            init_logger('sup3r.pipeline', **self._config.logging)
            init_logger('reV.pipeline', **self._config.logging)

    @classmethod
    def init_pass_collect(cls, out_dir, file_paths, model_path,
                          fwp_kwargs=None, dc_kwargs=None):
        """Generate config files for forward pass and collection

        Parameters
        ----------
        out_dir : str
            Parent directory for pipeline run.
        file_paths : str | list
            A single source h5 wind file to extract raster data from or a list
            of netcdf files with identical grid. The string can be a unix-style
            file path which will be passed through glob.glob
        model_path : str
            Path to gan model for forward pass
        fwp_kwargs : dict
            Dictionary of keyword args passed to the ForwardPassStrategy class.
        dc_kwargs : dict
            Dictionary of keyword args passed to the Collection.collect()
            method.
        """
        fwp_kwargs = fwp_kwargs or {}
        dc_kwargs = dc_kwargs or {}
        logger.info('Generating config files for forward pass and data '
                    'collection')
        log_dir = os.path.join(out_dir, 'logs/')
        output_dir = os.path.join(out_dir, 'output/')
        cache_dir = os.path.join(out_dir, 'cache/')
        std_out_dir = os.path.join(out_dir, 'stdout/')
        all_dirs = [out_dir, log_dir, cache_dir, output_dir, std_out_dir]
        for d in all_dirs:
            create_dirs(d)
        out_pattern = os.path.join(output_dir, 'fwp_out_{file_id}.h5')
        cache_pattern = os.path.join(cache_dir, 'cache_{feature}.pkl')
        log_pattern = os.path.join(log_dir, 'log_{node_index}.log')
        model_params = Sup3rGan.load_saved_params(model_path,
                                                  verbose=False)['meta']
        features = model_params['output_features']
        features = OutputHandlerH5.get_renamed_features(features)
        fwp_config = {'file_paths': file_paths,
                      'model_args': model_path,
                      'out_pattern': out_pattern,
                      'cache_pattern': cache_pattern,
                      'log_pattern': log_pattern}
        fwp_config.update(fwp_kwargs)
        fwp_config_file = os.path.join(out_dir, 'config_fwp.json')
        with open(fwp_config_file, 'w') as f:
            json.dump(fwp_config, f, sort_keys=True, indent=4)
            logger.info(f'Saved forward-pass config file: {fwp_config_file}.')

        collect_file = os.path.join(output_dir, 'out_collection.h5')
        log_file = os.path.join(log_dir, 'collect.log')
        input_files = os.path.join(output_dir, 'fwp_out_*.h5')
        col_config = {'file_paths': input_files,
                      'out_file': collect_file,
                      'features': features,
                      'log_file': log_file}
        col_config.update(dc_kwargs)
        col_config_file = os.path.join(out_dir, 'config_collect.json')
        with open(col_config_file, 'w') as f:
            json.dump(col_config, f, sort_keys=True, indent=4)
            logger.info(f'Saved data-collect config file: {col_config_file}.')

        pipe_config = {'logging': {'log_level': 'DEBUG'},
                       'pipeline': [{'forward-pass': fwp_config_file},
                                    {'data-collect': col_config_file}]}
        pipeline_file = os.path.join(out_dir, 'config_pipeline.json')
        with open(pipeline_file, 'w') as f:
            json.dump(pipe_config, f, sort_keys=True, indent=4)
            logger.info(f'Saved pipeline config file: {pipeline_file}.')

        script_file = os.path.join(out_dir, 'run.sh')
        with open(script_file, 'w') as f:
            cmd = 'python -m sup3r.cli -c config_pipeline.json pipeline'
            f.write(cmd)
            logger.info(f'Saved script file: {script_file}.')
