# -*- coding: utf-8 -*-
"""pytests for sup3r cli"""
import json
import os
import tempfile
import pytest

from click.testing import CliRunner

from sup3r.pipeline.forward_pass_cli import from_config as fp_main
from sup3r.pipeline.data_extract_cli import from_config as dh_main
from sup3r.models.base import Sup3rGan
from sup3r.preprocessing.data_handling import DataHandlerH5
from sup3r.preprocessing.batch_handling import BatchHandler

from sup3r import TEST_DATA_DIR, CONFIG_DIR

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
FEATURES = ['U_100m', 'V_100m', 'BVF_squared_200m']
FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')


@pytest.fixture(scope='module')
def runner():
    """Cli runner helper utility."""
    return CliRunner()


def test_fwd_pass_cli(runner):
    """Test cli call to run forward pass"""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)

    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            shape=(20, 20),
                            sample_shape=(18, 18, 24),
                            temporal_slice=slice(None, None, 1),
                            val_split=0.005,
                            max_extract_workers=1,
                            max_compute_workers=1)

    batch_handler = BatchHandler([handler], batch_size=4,
                                 s_enhance=3,
                                 t_enhance=4,
                                 n_batches=4)

    with tempfile.TemporaryDirectory() as td:
        model.train(batch_handler, n_epoch=1,
                    weight_gen_advers=0.0,
                    train_gen=True, train_disc=False,
                    checkpoint_int=2,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        config = {'file_paths': INPUT_FILES,
                  'target': (19, -125),
                  'model_path': out_dir,
                  'shape': (8, 8),
                  'forward_pass_chunk_shape': (4, 4, 6),
                  'temporal_extract_chunk_size': 10,
                  'out_file_prefix': os.path.join(td, 'out'),
                  's_enhance': 3,
                  't_enhance': 4,
                  'max_extract_workers': None,
                  'spatial_overlap': 5,
                  'temporal_overlap': 5,
                  'max_pass_workers': None,
                  'overwrite_cache': False,
                  'cache_file_prefix': os.path.join(td, 'cache'),
                  'execution_control': {
                      "nodes": 1,
                      "option": "local"}}

        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as fh:
            json.dump(config, fh)

        result = runner.invoke(fp_main, ['-c', config_path, '-v'])

        if result.exit_code != 0:
            import traceback
            msg = ('Failed with error {}'
                   .format(traceback.print_exception(*result.exc_info)))
            raise RuntimeError(msg)


def test_data_extract_cli(runner):
    """Test cli call to run data extraction"""
    with tempfile.TemporaryDirectory() as td:
        config = {'file_path': FP_WTK,
                  'target': TARGET_COORD,
                  'features': FEATURES,
                  'shape': (20, 20),
                  'cache_file_prefix': os.path.join(td, 'cache'),
                  'sample_shape': (20, 20, 12),
                  'val_split': 0.05,
                  'handler_class': 'DataHandlerH5'}

        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as fh:
            json.dump(config, fh)

        result = runner.invoke(dh_main, ['-c', config_path, '-v'])

        if result.exit_code != 0:
            import traceback
            msg = ('Failed with error {}'
                   .format(traceback.print_exception(*result.exc_info)))
            raise RuntimeError(msg)
