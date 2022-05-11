# -*- coding: utf-8 -*-
"""Entry point for sup3r pipeline execution

@author: bbenton
"""

import os
import logging
import sys
import tensorflow as tf
import sklearn
import pandas as pd
import numpy as np
import pprint

from rex import init_logger
from rex.utilities.loggers import create_dirs

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
