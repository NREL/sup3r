# -*- coding: utf-8 -*-
"""
Sup3r pipeline config

Created on January 25 2022

@author: bnb32
"""
from reV.config.base_config import BaseConfig as RevBaseConfig
from reV.config.base_analysis_config import AnalysisConfig
from reV.config.pipeline import PipelineConfig
from reV.utilities.exceptions import ConfigError

from sup3r import SUP3R_DIR, TEST_DATA_DIR, CONFIG_DIR


class BaseConfig(RevBaseConfig):
    """Base class for configuration frameworks."""

    REQUIREMENTS = ()
    """Required keys for config"""

    STR_REP = {'SUP3R_DIR': SUP3R_DIR,
               'CONFIG_DIR': CONFIG_DIR,
               'TEST_DATA_DIR': TEST_DATA_DIR}
    """Mapping of config inputs (keys) to desired replacements (values) in
    addition to relative file paths as demarcated by ./ and ../"""

    def __init__(self, config, check_keys=False, perform_str_rep=True):
        """
        Parameters
        ----------
        config : str | dict
            File path to config json (str), serialized json object (str),
            or dictionary with pre-extracted config.
        check_keys : bool, optional
            Flag to check config keys against Class properties, by default
            False because unlike reV, sup3r configs are not based on config
            class properties.
        perform_str_rep : bool
            Flag to perform string replacement for REVDIR, TESTDATADIR, and ./
        """
        super().__init__(config, check_keys=check_keys,
                         perform_str_rep=perform_str_rep)


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
