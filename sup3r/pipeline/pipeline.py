# -*- coding: utf-8 -*-
"""
Sup3r data pipeline architecture.
"""
import logging

from reV.pipeline.pipeline import Pipeline
from rex.utilities.loggers import init_logger

from sup3r.pipeline.config import Sup3rPipelineConfig
from sup3r.utilities import ModuleName

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
