# -*- coding: utf-8 -*-
"""
Sup3r pipeline config

Created on January 25 2022

@author: bnb32
"""
from reV.config.base_analysis_config import AnalysisConfig
from reV.config.pipeline import PipelineConfig
from reV.utilities.exceptions import ConfigError


class Sup3rPipelineConfig(PipelineConfig):
    """Sup3r pipeline configuration based on reV pipeline"""

    # pylint: disable=W0201
    def _parse_dirout(self):
        """Parse pipeline steps for common dirout and unique job names."""

        dirouts = []
        names = []
        for di in self.pipeline:
            for f_config in di.values():
                config = AnalysisConfig(f_config, check_keys=False,
                                        run_preflight=False)
                dirouts.append(config.dirout)

                if 'name' in config:
                    names.append(config.name)

        if len(set(dirouts)) != 1:
            raise ConfigError('Pipeline steps must have a common output '
                              'directory but received {} different '
                              'directories.'.format(len(set(dirouts))))
        else:
            self._dirout = dirouts[0]

        if len(set(names)) != len(names):
            raise ConfigError('Pipeline steps must have a unique job names '
                              'directory but received {} duplicate names.'
                              .format(len(names) - len(set(names))))
