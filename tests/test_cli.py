# -*- coding: utf-8 -*-
"""pytests for sup3r cli"""
import json
import os
import tempfile
import pytest
import glob
import shutil
from netCDF4 import Dataset

from click.testing import CliRunner

from sup3r.pipeline.forward_pass_cli import from_config as fp_main
from sup3r.preprocessing.data_extract_cli import from_config as dh_main
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
FEATURES = ['U_100m', 'V_100m', 'BVF2_200m']
FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')


def make_fake_nc_files(td):
    """Make dummy nc files with increasing times"""
    fake_dates = [f'2014-10-01_0{i}_00_00' for i in range(8)]
    fake_times = list(range(8))

    fake_files = [os.path.join(td, f'input_{date}') for date in fake_dates]
    for i, f in enumerate(INPUT_FILES):
        shutil.copy(f, fake_files[i])
        with Dataset(fake_files[i], 'r+') as dset:
            dset['XTIME'][:] = fake_times[i]
    return fake_files


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
                            extract_workers=1,
                            compute_workers=1)

    batch_handler = BatchHandler([handler], batch_size=4,
                                 s_enhance=3,
                                 t_enhance=4,
                                 n_batches=4)

    with tempfile.TemporaryDirectory() as td:

        input_files = make_fake_nc_files(td)

        model.train(batch_handler, n_epoch=1,
                    weight_gen_advers=0.0,
                    train_gen=True, train_disc=False,
                    checkpoint_int=2,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        fp_chunk_shape = (4, 4, 6)
        n_nodes = len(input_files) // fp_chunk_shape[2] + 1
        cache_prefix = os.path.join(td, 'cache')
        out_files = os.path.join(td, 'out_{file_id}.nc')
        log_prefix = os.path.join(td, 'log')
        config = {'file_paths': input_files,
                  'target': (19.3, -123.5),
                  'model_path': out_dir,
                  'out_pattern': out_files,
                  'cache_file_prefix': cache_prefix,
                  'log_file_prefix': log_prefix,
                  'shape': (8, 8),
                  'forward_pass_chunk_shape': fp_chunk_shape,
                  'temporal_extract_chunk_size': 10,
                  's_enhance': 3,
                  't_enhance': 4,
                  'extract_workers': None,
                  'spatial_overlap': 5,
                  'temporal_overlap': 5,
                  'max_pass_workers': None,
                  'overwrite_cache': False,
                  'execution_control': {
                      "nodes": 1,
                      "option": "local"}}

        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as fh:
            json.dump(config, fh)

        result = runner.invoke(fp_main, ['-c', config_path, '-v'])
        assert len(glob.glob(f'{cache_prefix}*')) == len(FEATURES * n_nodes)
        assert len(glob.glob(f'{log_prefix}*')) == n_nodes
        assert len(glob.glob(os.path.join(td, 'out_*.nc'))) == n_nodes

        if result.exit_code != 0:
            import traceback
            msg = ('Failed with error {}'
                   .format(traceback.print_exception(*result.exc_info)))
            raise RuntimeError(msg)


def test_data_extract_cli(runner):
    """Test cli call to run data extraction"""
    with tempfile.TemporaryDirectory() as td:
        cache_prefix = os.path.join(td, 'cache')
        log_file = os.path.join(td, 'log.log')
        config = {'file_paths': FP_WTK,
                  'target': TARGET_COORD,
                  'features': FEATURES,
                  'shape': (20, 20),
                  'sample_shape': (20, 20, 12),
                  'cache_file_prefix': cache_prefix,
                  'log_file': log_file,
                  'val_split': 0.05,
                  'handler_class': 'DataHandlerH5'}

        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as fh:
            json.dump(config, fh)

        result = runner.invoke(dh_main, ['-c', config_path, '-v'])

        assert len(glob.glob(f'{cache_prefix}*')) == len(FEATURES)
        assert len(glob.glob(f'{log_file}')) == 1

        if result.exit_code != 0:
            import traceback
            msg = ('Failed with error {}'
                   .format(traceback.print_exception(*result.exc_info)))
            raise RuntimeError(msg)
