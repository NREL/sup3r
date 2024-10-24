"""H5/NETCDF file collection."""

import glob
import logging
import re
from abc import ABC, abstractmethod

from rex.utilities.fun_utils import get_fun_call_str

from sup3r.postprocessing.writers.base import OutputMixin
from sup3r.utilities import ModuleName
from sup3r.utilities.cli import BaseCLI

logger = logging.getLogger(__name__)


class BaseCollector(OutputMixin, ABC):
    """Base collector class for H5/NETCDF collection"""

    def __init__(self, file_paths):
        """Parameters
        ----------
        file_paths : list | str
            Explicit list of str file paths that will be sorted and collected
            or a single string with unix-style /search/patt*ern.<ext>. Files
            should have non-overlapping time_index and spatial domains.
        """
        if not isinstance(file_paths, list):
            file_paths = glob.glob(file_paths)
        self.file_paths = file_paths
        self.flist = sorted(file_paths)
        self.data = None
        self.file_attrs = {}
        msg = (
            'File names must end with two zero padded integers, denoting '
            'the spatial chunk index and the temporal chunk index '
            'respectively. e.g. sup3r_chunk_000000_000000.h5'
        )

        assert all(self.get_chunk_indices(file) for file in self.flist), msg

    @staticmethod
    def get_chunk_indices(file):
        """Get spatial and temporal chunk indices from the given file name.

        Returns
        -------
        temporal_chunk_index : str
            Zero padded integer for the temporal chunk index
        spatial_chunk_index : str
            Zero padded integer for the spatial chunk index
        """
        return re.match(r'.*_([0-9]+)_([0-9]+).*\w+$', file).groups()

    @classmethod
    @abstractmethod
    def collect(cls, *args, **kwargs):
        """Collect data files from a dir to one output file."""

    @classmethod
    def get_node_cmd(cls, config):
        """Get a CLI call to collect data.

        Parameters
        ----------
        config : dict
            sup3r collection config with all necessary args and kwargs to
            run data collection.
        """
        import_str = (
            'from sup3r.postprocessing.collectors '
            f'import {cls.__name__};\n'
            'from rex import init_logger;\n'
            'import time;\n'
            'from gaps import Status'
        )

        dc_fun_str = get_fun_call_str(cls.collect, config)

        log_file = config.get('log_file', None)
        log_level = config.get('log_level', 'INFO')
        log_arg_str = f'"sup3r", log_level="{log_level}"'
        if log_file is not None:
            log_arg_str += f', log_file="{log_file}"'

        cmd = (
            f"python -c '{import_str};\n"
            't0 = time.time();\n'
            f'logger = init_logger({log_arg_str});\n'
            f'{dc_fun_str};\n'
            't_elap = time.time() - t0;\n'
        )

        pipeline_step = config.get('pipeline_step') or ModuleName.DATA_COLLECT
        cmd = BaseCLI.add_status_cmd(config, pipeline_step, cmd)
        cmd += ";'\n"

        return cmd.replace('\\', '/')
