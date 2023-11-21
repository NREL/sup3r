"""Sup3r pipeline tests"""
import os
import glob
import json
import shutil
import tempfile

import click
import numpy as np
from rex import ResourceX
from rex.utilities.loggers import LOGGERS
from gaps import Pipeline

from sup3r import CONFIG_DIR, TEST_DATA_DIR
from sup3r.models.base import Sup3rGan
from sup3r.utilities.pytest import make_fake_nc_files

INPUT_FILE = os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_00_00_00')
FEATURES = ['U_100m', 'V_100m', 'BVF2_200m']


def test_fwp_pipeline():
    """Test sup3r pipeline"""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 8, 8, 4, len(FEATURES))))
    input_resolution = {'spatial': '12km', 'temporal': '60min'}
    model.meta['input_resolution'] = input_resolution
    assert model.input_resolution == input_resolution
    assert model.output_resolution == {'spatial': '4km', 'temporal': '15min'}
    _ = model.generate(np.ones((4, 8, 8, 4, len(FEATURES))))
    model.meta['lr_features'] = FEATURES
    model.meta['hr_out_features'] = FEATURES[:2]
    model.meta['hr_exo_features'] = FEATURES[2:]
    assert model.s_enhance == 3
    assert model.t_enhance == 4

    test_context = click.Context(click.Command("pipeline"), obj={})
    with tempfile.TemporaryDirectory() as td, test_context as ctx:
        ctx.obj["NAME"] = "test"
        ctx.obj["VERBOSE"] = False

        input_files = make_fake_nc_files(td, INPUT_FILE, 20)
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        fp_chunk_shape = (4, 4, 3)
        shape = (8, 8)
        target = (19.3, -123.5)
        n_tsteps = 10
        t_slice = slice(5, 5 + n_tsteps)
        cache_pattern = os.path.join(td, 'cache')
        out_files = os.path.join(td, 'fp_out_{file_id}.h5')
        log_prefix = os.path.join(td, 'log')
        t_enhance = 4

        input_handler_kwargs = dict(target=target, shape=shape,
                                    overwrite_cache=True,
                                    time_chunk_size=10,
                                    worker_kwargs=dict(max_workers=1),
                                    temporal_slice=[t_slice.start,
                                                    t_slice.stop])
        config = {'worker_kwargs': {'max_workers': 1},
                  'file_paths': input_files,
                  'model_kwargs': {'model_dir': out_dir},
                  'out_pattern': out_files,
                  'cache_pattern': cache_pattern,
                  'log_pattern': log_prefix,
                  'fwp_chunk_shape': fp_chunk_shape,
                  'input_handler_kwargs': input_handler_kwargs,
                  'spatial_pad': 2,
                  'temporal_pad': 2,
                  'overwrite_cache': True,
                  'execution_control': {
                      "nodes": 1,
                      "option": "local"},
                  'max_nodes': 1}

        fp_config_path = os.path.join(td, 'fp_config.json')
        with open(fp_config_path, 'w') as fh:
            json.dump(config, fh)

        out_files = os.path.join(td, 'fp_out_*.h5')
        features = ['windspeed_100m', 'winddirection_100m']
        fp_out = os.path.join(td, 'out_combined.h5')
        config = {'max_workers': 1,
                  'file_paths': out_files,
                  'out_file': fp_out,
                  'features': features,
                  'log_file': os.path.join(td, 'log.log'),
                  'execution_control': {
                      "option": "local"}}

        collect_config_path = os.path.join(td, 'collect_config.json')
        with open(collect_config_path, 'w') as fh:
            json.dump(config, fh)

        fpipeline = os.path.join(TEST_DATA_DIR, 'pipeline',
                                 'config_pipeline.json')
        tmp_fpipeline = os.path.join(td, 'config_pipeline.json')
        shutil.copy(fpipeline, tmp_fpipeline)

        Pipeline.run(tmp_fpipeline, monitor=True)

        assert os.path.exists(fp_out)
        with ResourceX(fp_out) as f:
            assert len(f.time_index) == t_enhance * n_tsteps

        status_fps = glob.glob(f'{td}/.gaps/*status*.json')
        assert len(status_fps) == 1
        status_file = status_fps[0]
        with open(status_file, 'r') as fh:
            status = json.load(fh)
            assert all(s in status for s in ('forward-pass', 'data-collect'))
            assert all(s not in str(status)
                       for s in ('fail', 'pending', 'submitted'))
            assert 'successful' in str(status)


def test_multiple_fwp_pipeline():
    """Test sup3r pipeline with multiple fwp steps"""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 8, 8, 4, len(FEATURES))))
    input_resolution = {'spatial': '12km', 'temporal': '60min'}
    model.meta['input_resolution'] = input_resolution
    assert model.input_resolution == input_resolution
    assert model.output_resolution == {'spatial': '4km', 'temporal': '15min'}
    _ = model.generate(np.ones((4, 8, 8, 4, len(FEATURES))))
    model.meta['lr_features'] = FEATURES
    model.meta['hr_out_features'] = FEATURES[:2]
    model.meta['hr_exo_features'] = FEATURES[2:]
    assert model.s_enhance == 3
    assert model.t_enhance == 4

    test_context = click.Context(click.Command("pipeline"), obj={})
    with tempfile.TemporaryDirectory() as td, test_context as ctx:
        ctx.obj["NAME"] = "test"
        ctx.obj["VERBOSE"] = False

        input_files = make_fake_nc_files(td, INPUT_FILE, 20)
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        fp_chunk_shape = (4, 4, 3)
        shape = (8, 8)
        target = (19.3, -123.5)
        n_tsteps = 10
        t_slice = slice(5, 5 + n_tsteps)
        t_enhance = 4

        input_handler_kwargs = dict(target=target, shape=shape,
                                    overwrite_cache=True,
                                    time_chunk_size=10,
                                    worker_kwargs=dict(max_workers=1),
                                    temporal_slice=[t_slice.start,
                                                    t_slice.stop])

        sub_dir_1 = os.path.join(td, 'dir1')
        os.mkdir(sub_dir_1)
        cache_pattern = os.path.join(sub_dir_1, 'cache')
        log_prefix = os.path.join(td, 'log1')
        out_files = os.path.join(sub_dir_1, 'fp_out_{file_id}.h5')
        config = {'worker_kwargs': {'max_workers': 1},
                  'file_paths': input_files,
                  'model_kwargs': {'model_dir': out_dir},
                  'out_pattern': out_files,
                  'cache_pattern': cache_pattern,
                  'log_level': "DEBUG",
                  'log_pattern': log_prefix,
                  'fwp_chunk_shape': fp_chunk_shape,
                  'input_handler_kwargs': input_handler_kwargs,
                  'spatial_pad': 2,
                  'temporal_pad': 2,
                  'overwrite_cache': True,
                  'execution_control': {
                      "nodes": 1,
                      "option": "local"},
                  'max_nodes': 1}

        fp_config_path_1 = os.path.join(td, 'fp_config1.json')
        with open(fp_config_path_1, 'w') as fh:
            json.dump(config, fh)

        sub_dir_2 = os.path.join(td, 'dir2')
        os.mkdir(sub_dir_2)
        cache_pattern = os.path.join(sub_dir_2, 'cache')
        log_prefix = os.path.join(td, 'log2')
        out_files = os.path.join(sub_dir_2, 'fp_out_{file_id}.h5')
        config = {'worker_kwargs': {'max_workers': 1},
                  'file_paths': input_files,
                  'model_kwargs': {'model_dir': out_dir},
                  'out_pattern': out_files,
                  'cache_pattern': cache_pattern,
                  'log_level': "DEBUG",
                  'log_pattern': log_prefix,
                  'fwp_chunk_shape': fp_chunk_shape,
                  'input_handler_kwargs': input_handler_kwargs,
                  'spatial_pad': 2,
                  'temporal_pad': 2,
                  'overwrite_cache': True,
                  'execution_control': {
                      "nodes": 1,
                      "option": "local"},
                  'max_nodes': 1}

        fp_config_path_2 = os.path.join(td, 'fp_config2.json')
        with open(fp_config_path_2, 'w') as fh:
            json.dump(config, fh)

        out_files_1 = os.path.join(sub_dir_1, 'fp_out_*.h5')
        features = ['windspeed_100m', 'winddirection_100m']
        fp_out_1 = os.path.join(sub_dir_1, 'out_combined.h5')
        config = {'max_workers': 1,
                  'file_paths': out_files_1,
                  'out_file': fp_out_1,
                  'features': features,
                  'log_file': os.path.join(td, 'log.log'),
                  'execution_control': {"option": "local"}}

        collect_config_path_1 = os.path.join(td, 'collect_config1.json')
        with open(collect_config_path_1, 'w') as fh:
            json.dump(config, fh)

        out_files_2 = os.path.join(sub_dir_2, 'fp_out_*.h5')
        fp_out_2 = os.path.join(sub_dir_2, 'out_combined.h5')
        config = {'max_workers': 1,
                  'file_paths': out_files_2,
                  'out_file': fp_out_2,
                  'features': features,
                  'log_file': os.path.join(td, 'log2.log'),
                  'execution_control': {"option": "local"}}

        collect_config_path_2 = os.path.join(td, 'collect_config2.json')
        with open(collect_config_path_2, 'w') as fh:
            json.dump(config, fh)

        pipe_config = {"logging": {"log_file": None, "log_level": "INFO"},
                       "pipeline": [{'fp1': fp_config_path_1,
                                     'command': 'forward-pass'},
                                    {'fp2': fp_config_path_2,
                                     'command': 'forward-pass'},
                                    {'data-collect': collect_config_path_1},
                                    {'collect2': collect_config_path_2,
                                     'command': 'data-collect'}]}

        tmp_fpipeline = os.path.join(td, 'config_pipeline.json')
        with open(tmp_fpipeline, 'w') as fh:
            json.dump(pipe_config, fh)

        Pipeline.run(tmp_fpipeline, monitor=True)

        for fp_out in [fp_out_1, fp_out_2]:
            assert os.path.exists(fp_out)
            with ResourceX(fp_out) as f:
                assert len(f.time_index) == t_enhance * n_tsteps

        status_fps = glob.glob(f'{td}/.gaps/*status*.json')
        assert len(status_fps) == 1
        status_file = status_fps[0]
        with open(status_file, 'r') as fh:
            status = json.load(fh)
            expected_names = {'fp1', 'fp2', 'data-collect', 'collect2'}
            assert all(s in status for s in expected_names)
            assert all(s not in str(status)
                       for s in ('fail', 'pending', 'submitted'))
            assert 'successful' in str(status)

        LOGGERS.clear()
