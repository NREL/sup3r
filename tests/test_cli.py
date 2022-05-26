# -*- coding: utf-8 -*-
"""pytests for sup3r cli"""
import json
import os
import tempfile
import pytest

from click.testing import CliRunner

from sup3r.pipeline.forward_pass_cli import from_config

from sup3r import TEST_DATA_DIR


input_files = [
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_00_00_00'),
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_01_00_00'),
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_00_00_00'),
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_01_00_00'),
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_00_00_00'),
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_01_00_00'),
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_00_00_00'),
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_01_00_00')]
INPUT_FILES = sorted(input_files)
TARGET_COORD = (39.01, -105.15)


@pytest.fixture(scope='module')
def runner():
    """Cli runner helper utility."""
    return CliRunner()


def test_fwd_pass_cli(runner):
    """Test cli call to run forward pass"""
    with tempfile.TemporaryDirectory() as td:
        config = {'file_paths': INPUT_FILES,
                  'target': TARGET_COORD,
                  'model_path': os.path.join(td, 'model'),
                  'shape': (20, 20),
                  'forward_pass_chunk_shape': (20, 20, 12),
                  'temporal_extract_chunk_size': 10,
                  'out_file_prefix': os.path.join(td, 'out'),
                  's_enhance': 3,
                  't_enhance': 4,
                  'max_extract_workers': None,
                  'spatial_overlap': 5,
                  'temporal_overlap': 5,
                  'max_pass_workers': None,
                  'overwrite_cache': False,
                  'cache_file_prefix': os.path.join(td, 'cache_')}

        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as fh:
            json.dump(config, fh)

        result = runner.invoke(from_config, ['-c', config_path, True])

        if result.exit_code != 0:
            import traceback
            msg = ('Failed with error {}'
                   .format(traceback.print_exception(*result.exc_info)))
            raise RuntimeError(msg)
