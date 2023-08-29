# -*- coding: utf-8 -*-
"""
Sup3r pipeline config

Created on January 25 2022

@author: bnb32
"""
import os
from typing import ClassVar

from reV.config.base_analysis_config import AnalysisConfig
from reV.config.base_config import BaseConfig as RevBaseConfig
from reV.utilities.exceptions import ConfigError, PipelineError

from sup3r import CONFIG_DIR, SUP3R_DIR, TEST_DATA_DIR


class BaseConfig(RevBaseConfig):
    """Base class for configuration frameworks."""

    REQUIREMENTS = ()
    """Required keys for config"""

    STR_REP: ClassVar[dict] = {
        'SUP3R_DIR': SUP3R_DIR,
        'CONFIG_DIR': CONFIG_DIR,
        'TEST_DATA_DIR': TEST_DATA_DIR,
    }
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
        super().__init__(
            config, check_keys=check_keys, perform_str_rep=perform_str_rep
        )


class Sup3rPipelineConfig(AnalysisConfig):
    """Sup3r pipeline configuration based on reV pipeline"""

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str | dict
            File path to config json (str), serialized json object (str),
            or dictionary with pre-extracted config.
        """

        super().__init__(config, run_preflight=False)
        self._check_pipeline()
        self._parse_dirout()
        self._check_dirout_status()

    def _check_pipeline(self):
        """Check pipeline steps input. ConfigError if bad input."""

        if 'pipeline' not in self:
            raise ConfigError(
                'Could not find required key "pipeline" in the '
                'pipeline config.'
            )

        if not isinstance(self.pipeline, list):
            raise ConfigError(
                'Config arg "pipeline" must be a list of '
                '(command, f_config) pairs, but received "{}".'.format(
                    type(self.pipeline)
                )
            )

        for di in self.pipeline:
            for f_config in di.values():
                if not os.path.exists(f_config):
                    raise ConfigError(
                        'Pipeline step depends on non-existent '
                        'file: {}'.format(f_config)
                    )

    def _check_dirout_status(self):
        """Check unique status file in dirout."""

        if os.path.exists(self.dirout):
            for fname in os.listdir(self.dirout):
                if fname.endswith(
                    '_status.json'
                ) and fname != '{}_status.json'.format(self.name):
                    msg = (
                        'Cannot run pipeline "{}" in directory '
                        '{}. Another pipeline appears to have '
                        'been run here with status json: {}'.format(
                            self.name, self.dirout, fname
                        )
                    )
                    raise PipelineError(msg)

    @property
    def pipeline(self):
        """Get the pipeline steps.

        Returns
        -------
        pipeline : list
            reV pipeline run steps. Should be a list of (command, config)
            pairs.
        """

        return self['pipeline']

    @property
    def logging(self):
        """Get logging kwargs for the pipeline.

        Returns
        -------
        dict
        """
        return self.get('logging', {"log_file": None, "log_level": "INFO"})

    @property
    def hardware(self):
        """Get argument specifying which hardware the pipeline is being run on.

        Defaults to "eagle" (most common use of the reV pipeline)

        Returns
        -------
        hardware : str
            Name of hardware that this pipeline is being run on.
            Defaults to "eagle".
        """
        return self.get('hardware', 'eagle')

    @property
    def status_file(self):
        """Get status file path.

        Returns
        -------
        _status_file : str
            reV status file path.
        """
        if self.dirout is None:
            raise ConfigError('Pipeline has not yet been initialized.')

        return os.path.join(self.dirout, '{}_status.json'.format(self.name))

    # pylint: disable=W0201
    def _parse_dirout(self):
        """Parse pipeline steps for common dirout and unique job names."""

        dirouts = []
        names = []
        for di in self.pipeline:
            for f_config in di.values():
                config = AnalysisConfig(
                    f_config, check_keys=False, run_preflight=False
                )
                dirouts.append(config.dirout)

                if 'name' in config:
                    names.append(config.name)

        if len(set(dirouts)) != 1:
            raise ConfigError(
                'Pipeline steps must have a common output '
                'directory but received {} different '
                'directories.'.format(len(set(dirouts)))
            )
        else:
            self._dirout = dirouts[0]

        if len(set(names)) != len(names):
            raise ConfigError(
                'Pipeline steps must have a unique job names '
                'directory but received {} duplicate names.'.format(
                    len(names) - len(set(names))
                )
            )
