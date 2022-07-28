# -*- coding: utf-8 -*-
"""pytests for sup3r cli"""
import json
import os
import tempfile
import pytest
import glob
from rex import ResourceX
import numpy as np

from click.testing import CliRunner

from sup3r.pipeline.forward_pass_cli import from_config as fwp_main
from sup3r.preprocessing.data_extract_cli import from_config as dh_main
from sup3r.postprocessing.data_collect_cli import from_config as dc_main
from sup3r.models.base import Sup3rGan
from sup3r.utilities.test_utils import make_fake_nc_files, make_fake_h5_chunks

from sup3r import TEST_DATA_DIR, CONFIG_DIR

INPUT_FILE = os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_00_00_00')
TARGET_COORD = (39.01, -105.15)
FEATURES = ['U_100m', 'V_100m', 'BVF2_200m']
FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
fwp_chunk_shape = (4, 4, 4)
s_enhance = 3
t_enhance = 4
target = (19.3, -123.5)
shape = (8, 8)


@pytest.fixture(scope='module')
def runner():
    """Cli runner helper utility."""
    return CliRunner()


def test_data_collection_cli(runner):
    """Test cli call to data collection on forward pass output"""

    with tempfile.TemporaryDirectory() as td:
        fp_out = os.path.join(td, 'out_combined.h5')
        out = make_fake_h5_chunks(td)
        (out_files, data, ws_true, wd_true, features, slices_lr,
            slices_hr, low_res_lat_lon, low_res_times) = out

        features = ['windspeed_100m', 'winddirection_100m']
        config = {'file_paths': out_files,
                  'out_file': fp_out,
                  'features': features,
                  'log_file': os.path.join(td, 'log.log'),
                  'execution_control': {
                      "option": "local"}}

        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as fh:
            json.dump(config, fh)

        result = runner.invoke(dc_main, ['-c', config_path, '-v'])

        if result.exit_code != 0:
            import traceback
            msg = ('Failed with error {}'
                   .format(traceback.print_exception(*result.exc_info)))
            raise RuntimeError(msg)

        assert os.path.exists(fp_out)
        with ResourceX(fp_out) as fh:
            assert all(f in fh for f in features)
            full_ti = fh.time_index
            combined_ti = []
            for i, f in enumerate(out_files):
                with ResourceX(f) as fh_i:
                    if i == 0:
                        ws = fh_i['windspeed_100m']
                        wd = fh_i['winddirection_100m']
                    else:
                        ws = np.concatenate([ws, fh_i['windspeed_100m']],
                                            axis=0)
                        wd = np.concatenate([wd, fh_i['winddirection_100m']],
                                            axis=0)
                    combined_ti += list(fh_i.time_index)
            assert len(full_ti) == len(combined_ti)
            assert np.allclose(ws, fh['windspeed_100m'])
            assert np.allclose(wd, fh['winddirection_100m'])


def test_fwd_pass_cli(runner):
    """Test cli call to run forward pass"""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 8, 8, 4, len(FEATURES))))
    model.meta['training_features'] = FEATURES
    model.meta['output_features'] = FEATURES[:2]

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        fwp_chunk_shape = (4, 4, 6)
        n_nodes = len(input_files) // fwp_chunk_shape[2] + 1
        cache_pattern = os.path.join(td, 'cache')
        out_files = os.path.join(td, 'out_{file_id}.nc')
        log_prefix = os.path.join(td, 'log.log')
        config = {'file_paths': input_files,
                  'target': (19.3, -123.5),
                  'model_args': out_dir,
                  'out_pattern': out_files,
                  'cache_pattern': cache_pattern,
                  'log_pattern': log_prefix,
                  'shape': (8, 8),
                  'fwp_chunk_shape': fwp_chunk_shape,
                  'time_chunk_size': 10,
                  's_enhance': 3,
                  't_enhance': 4,
                  'max_workers': 1,
                  'spatial_pad': 5,
                  'temporal_pad': 5,
                  'overwrite_cache': False,
                  'execution_control': {
                      "option": "local"}}

        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as fh:
            json.dump(config, fh)

        result = runner.invoke(fwp_main, ['-c', config_path, '-v'])

        if result.exit_code != 0:
            import traceback
            msg = ('Failed with error {}'
                   .format(traceback.print_exception(*result.exc_info)))
            raise RuntimeError(msg)

        assert len(glob.glob(f'{td}/cache*')) == len(FEATURES * n_nodes)
        assert len(glob.glob(f'{td}/log*')) == n_nodes
        assert len(glob.glob(f'{td}/out*')) == n_nodes


def test_data_extract_cli(runner):
    """Test cli call to run data extraction"""
    with tempfile.TemporaryDirectory() as td:
        cache_pattern = os.path.join(td, 'cache')
        log_file = os.path.join(td, 'log.log')
        config = {'file_paths': FP_WTK,
                  'target': TARGET_COORD,
                  'features': FEATURES,
                  'shape': (20, 20),
                  'sample_shape': (20, 20, 12),
                  'cache_pattern': cache_pattern,
                  'log_file': log_file,
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

        assert len(glob.glob(f'{cache_pattern}*')) == len(FEATURES)
        assert len(glob.glob(f'{log_file}')) == 1
