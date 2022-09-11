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

from sup3r.pipeline.pipeline_cli import from_config as pipe_main
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


def test_pipeline_fwp_collect(runner):
    """Test pipeline with forward pass and data collection"""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 8, 8, 4, len(FEATURES))))
    model.meta['training_features'] = FEATURES
    model.meta['output_features'] = FEATURES[:2]
    model.meta['s_enhance'] = 3
    model.meta['t_enhance'] = 4

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)
        fp_out = os.path.join(td, 'fwp_combined.h5')
        fwp_chunk_shape = (4, 4, 6)
        n_nodes = len(input_files) // fwp_chunk_shape[2] + 1
        n_nodes *= shape[0] // fwp_chunk_shape[0]
        n_nodes *= shape[1] // fwp_chunk_shape[1]
        cache_pattern = os.path.join(td, 'cache')
        out_files = os.path.join(td, 'out_{file_id}.h5')
        log_prefix = os.path.join(td, 'log.log')
        fwp_config = {'file_paths': input_files,
                      'target': (19.3, -123.5),
                      'model_args': out_dir,
                      'out_pattern': out_files,
                      'cache_pattern': cache_pattern,
                      'log_pattern': log_prefix,
                      'shape': shape,
                      'fwp_chunk_shape': fwp_chunk_shape,
                      'time_chunk_size': 10,
                      'max_workers': 1,
                      'spatial_pad': 5,
                      'temporal_pad': 5,
                      'overwrite_cache': True,
                      'execution_control': {
                          "option": "local"}}

        features = ['windspeed_100m', 'winddirection_100m']
        out_files = os.path.join(td, 'out_*.h5')
        dc_config = {'file_paths': out_files,
                     'out_file': fp_out,
                     'features': features,
                     'log_file': os.path.join(td, 'log.log'),
                     'execution_control': {
                         "option": "local"}}

        fwp_config_path = os.path.join(td, 'config_fwp.json')
        dc_config_path = os.path.join(td, 'config_dc.json')
        pipe_config_path = os.path.join(td, 'config_pipe.json')
        pipe_flog = os.path.join(td, 'pipeline.log')

        pipe_config = {"logging": {"log_level": "DEBUG",
                                   "log_file": pipe_flog},
                       "pipeline": [{"forward-pass": fwp_config_path},
                                    {"data-collect": dc_config_path}]}

        with open(fwp_config_path, 'w') as fh:
            json.dump(fwp_config, fh)
        with open(dc_config_path, 'w') as fh:
            json.dump(dc_config, fh)
        with open(pipe_config_path, 'w') as fh:
            json.dump(pipe_config, fh)

        result = runner.invoke(pipe_main, ['-c', pipe_config_path,
                                           '-v', '--monitor'])
        if result.exit_code != 0:
            import traceback
            msg = ('Failed with error {}'
                   .format(traceback.print_exception(*result.exc_info)))
            raise RuntimeError(msg)

        assert os.path.exists(fp_out)
        with ResourceX(fp_out) as fh:
            assert all(f in fh for f in features)
            full_ti = fh.time_index
            full_gids = fh.meta['gid']
            combined_ti = []
            for _, f in enumerate(glob.glob(out_files)):
                with ResourceX(f) as fh_i:
                    fi_ti = fh_i.time_index
                    fi_gids = fh_i.meta['gid']
                    assert all(gid in full_gids for gid in fi_gids)
                    s_indices = np.where(full_gids.isin(fi_gids))[0]
                    t_indices = np.where(full_ti.isin(fi_ti))[0]
                    t_indices = slice(t_indices[0], t_indices[-1] + 1)
                    chunk = fh['windspeed_100m'][t_indices, s_indices]
                    assert np.allclose(chunk, fh_i['windspeed_100m'])
                    chunk = fh['winddirection_100m'][t_indices, s_indices]
                    assert np.allclose(chunk, fh_i['winddirection_100m'])
                    combined_ti += list(fh_i.time_index)
            assert len(full_ti) == len(set(combined_ti))


def test_data_collection_cli(runner):
    """Test cli call to data collection on forward pass output"""

    with tempfile.TemporaryDirectory() as td:
        fp_out = os.path.join(td, 'out_combined.h5')
        out = make_fake_h5_chunks(td)
        (out_files, _, _, _, features, _, _, _, _) = out

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
    model.meta['s_enhance'] = 3
    model.meta['t_enhance'] = 4

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        fwp_chunk_shape = (4, 4, 6)
        t_chunks = len(input_files) // fwp_chunk_shape[2] + 1
        n_nodes = t_chunks * shape[0] // fwp_chunk_shape[0]
        n_nodes *= shape[1] // fwp_chunk_shape[1]
        cache_pattern = os.path.join(td, 'cache')
        out_files = os.path.join(td, 'out_{file_id}.nc')
        log_prefix = os.path.join(td, 'log.log')
        config = {'file_paths': input_files,
                  'target': (19.3, -123.5),
                  'model_args': out_dir,
                  'out_pattern': out_files,
                  'cache_pattern': cache_pattern,
                  'log_pattern': log_prefix,
                  'shape': shape,
                  'fwp_chunk_shape': fwp_chunk_shape,
                  'time_chunk_size': 10,
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

        # include time index cache file
        n_cache_files = 1 + ((len(FEATURES) + 1) * t_chunks)
        assert len(glob.glob(f'{td}/cache*')) == n_cache_files
        assert len(glob.glob(f'{td}/*.log')) == n_nodes
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

        assert len(glob.glob(f'{cache_pattern}*')) == len(FEATURES) + 1
        assert len(glob.glob(f'{log_file}')) == 1


def test_pipeline_fwp_qa(runner):
    """Test the sup3r pipeline with Forward Pass and QA modules
    via pipeline cli"""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 8, 8, 4, len(FEATURES))))
    model.meta['training_features'] = FEATURES
    model.meta['output_features'] = FEATURES[:2]
    model.meta['s_enhance'] = 3
    model.meta['t_enhance'] = 4

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        fwp_config = {'file_paths': input_files,
                      'target': (19.3, -123.5),
                      'model_args': out_dir,
                      'out_pattern': os.path.join(td, 'out_{file_id}.h5'),
                      'log_pattern': os.path.join(td, 'fwp_log.log'),
                      'log_level': 'DEBUG',
                      'shape': (8, 8),
                      'fwp_chunk_shape': (100, 100, 100),
                      'time_chunk_size': 10,
                      'max_workers': 1,
                      'spatial_pad': 5,
                      'temporal_pad': 5,
                      'overwrite_cache': False,
                      'execution_control': {
                          "option": "local"}}

        qa_config = {'source_file_paths': input_files,
                     'out_file_path': os.path.join(td, 'out_000000_000000.h5'),
                     'qa_fp': os.path.join(td, 'qa.h5'),
                     's_enhance': 3,
                     't_enhance': 4,
                     'temporal_coarsening_method': 'subsample',
                     'target': (19.3, -123.5),
                     'shape': (8, 8),
                     'max_workers': 1,
                     'execution_control': {
                         "option": "local"}}

        fwp_config_path = os.path.join(td, 'config_fwp.json')
        qa_config_path = os.path.join(td, 'config_qa.json')
        pipe_config_path = os.path.join(td, 'config_pipe.json')

        pipe_flog = os.path.join(td, 'pipeline.log')
        pipe_config = {"logging": {"log_level": "DEBUG",
                                   "log_file": pipe_flog},
                       "pipeline": [{"forward-pass": fwp_config_path},
                                    {"qa": qa_config_path}]}

        with open(fwp_config_path, 'w') as fh:
            json.dump(fwp_config, fh)
        with open(qa_config_path, 'w') as fh:
            json.dump(qa_config, fh)
        with open(pipe_config_path, 'w') as fh:
            json.dump(pipe_config, fh)

        result = runner.invoke(pipe_main, ['-c', pipe_config_path,
                                           '-v', '--monitor'])
        if result.exit_code != 0:
            import traceback
            msg = ('Failed with error {}'
                   .format(traceback.print_exception(*result.exc_info)))
            raise RuntimeError(msg)

        assert len(glob.glob(f'{td}/fwp_log*.log')) == 1
        assert len(glob.glob(f'{td}/out*.h5')) == 1
        assert len(glob.glob(f'{td}/qa.h5')) == 1
        assert len(glob.glob(f'{td}/*_status.json')) == 1

        status_fp = glob.glob(f'{td}/*_status.json')[0]
        with open(status_fp, 'r') as f:
            status = json.load(f)

        assert len(status) == 2
        assert len(status['forward-pass']) == 2
        fwp_status = status['forward-pass']
        del fwp_status['pipeline_index']
        fwp_status = list(fwp_status.values())[0]
        assert fwp_status['job_status'] == 'successful'
        assert fwp_status['time'] > 0

        assert len(status['qa']) == 2
        qa_status = status['qa']
        del qa_status['pipeline_index']
        qa_status = list(qa_status.values())[0]
        assert qa_status['job_status'] == 'successful'
        assert qa_status['time'] > 0
