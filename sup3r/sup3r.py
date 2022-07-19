# -*- coding: utf-8 -*-
"""Entry point for sup3r pipeline execution

@author: bbenton
"""

import os
import logging
import sys
import pprint
import json

from rex import init_logger
from rex.utilities.loggers import create_dirs

from sup3r.utilities import VERSION_RECORD
from sup3r.models.base import Sup3rGan
from sup3r.postprocessing.file_handling import OutputHandlerH5


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
                           'sup3r.postprocessing.collection',
                           'sup3r.preprocessing.data_handling')

            if log_file is not None and use_log_dir:
                log_file = os.path.join(self._log_dir, log_file)
                create_dirs(os.path.dirname(log_file))

            for name in loggers:
                init_logger(name, log_level=log_level, log_file=log_file)

        if log_version:
            is_64bits = sys.maxsize > 2 ** 32
            if is_64bits:
                logger.info(
                    f'Running on 64-bit python, sys.maxsize: {sys.maxsize}')
            else:
                logger.warning(
                    f'Running 32-bit python, sys.maxsize: {sys.maxsize}')

            logger.info('Active python environment versions: \n{}'
                        .format(pprint.pformat(VERSION_RECORD, indent=2)))

    @classmethod
    def init_pass_collect(cls, out_dir, file_paths, model_path, **kwargs):
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
        """
        logger.info('Generating config files for forward pass and data '
                    'collection')
        obj = cls(out_dir)
        out_pattern = os.path.join(obj._output_dir, 'fp_out_{file_id}.h5')
        cache_pattern = os.path.join(obj._cache_dir, 'cache_{feature}.pkl')
        log_pattern = os.path.join(obj._log_dir, 'log_{node_index}.log')
        model_params = Sup3rGan.load_saved_params(model_path,
                                                  verbose=False)['meta']
        features = model_params['output_features']
        features = OutputHandlerH5.get_renamed_features(features)
        fp_config = {'file_paths': file_paths,
                     'model_path': model_path,
                     'out_pattern': out_pattern,
                     'cache_pattern': cache_pattern,
                     'log_pattern': log_pattern,
                     'overwrite_cache': True}
        fp_kwargs = {k: v for k, v in kwargs.items() if k != 'file_paths'}
        fp_config.update(fp_kwargs)
        fp_config_file = os.path.join(out_dir, 'fp_config.json')
        with open(fp_config_file, 'w') as f:
            json.dump(fp_config, f)
            logger.info(f'Saved forward-pass config file: {fp_config_file}.')

        collect_file = os.path.join(obj._output_dir, 'out_collection.h5')
        log_file = os.path.join(obj._log_dir, 'collect.log')
        input_files = os.path.join(obj._output_dir, 'fp_out_*.h5')
        col_config = {'file_paths': input_files,
                      'out_file': collect_file,
                      'features': features,
                      'log_file': log_file}
        col_kwargs = {k: v for k, v in kwargs.items() if k != 'file_paths'}
        col_config.update(col_kwargs)
        col_config_file = os.path.join(out_dir, 'collect_config.json')
        with open(col_config_file, 'w') as f:
            json.dump(col_config, f)
            logger.info(f'Saved data-collect config file: {col_config_file}.')

        pipe_config = {'logging': {'log_level': 'DEBUG'},
                       'pipeline': [{'forward-pass': fp_config_file},
                                    {'data-collect': col_config_file}]}
        pipeline_file = os.path.join(out_dir, 'pipeline_config.json')
        with open(pipeline_file, 'w') as f:
            json.dump(pipe_config, f)
            logger.info(f'Saved pipeline config file: {pipeline_file}.')

        script_file = os.path.join(out_dir, 'run.sh')
        with open(script_file, 'w') as f:
            cmd = 'python -m sup3r.cli -c pipeline_config.json pipeline'
            f.write(cmd)
            logger.info(f'Saved script file: {script_file}.')
