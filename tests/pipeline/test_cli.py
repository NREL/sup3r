# -*- coding: utf-8 -*-
"""pytests for sup3r cli"""
import glob
import json
import os
import tempfile

import numpy as np
import pytest
from click.testing import CliRunner
from rex import ResourceX, init_logger

from sup3r import CONFIG_DIR, TEST_DATA_DIR
from sup3r.models.base import Sup3rGan
from sup3r.pipeline.forward_pass_cli import from_config as fwp_main
from sup3r.pipeline.pipeline_cli import from_config as pipe_main
from sup3r.postprocessing.data_collect_cli import from_config as dc_main
from sup3r.preprocessing.data_extract_cli import from_config as dh_main
from sup3r.qa.visual_qa_cli import from_config as vqa_main
from sup3r.utilities.pytest import make_fake_h5_chunks, make_fake_nc_files
from sup3r.utilities.utilities import correct_path

INPUT_FILE = os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_00_00_00')
FEATURES = ['U_100m', 'V_100m', 'BVF2_200m']
FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
fwp_chunk_shape = (4, 4, 6)
shape = (8, 8)


@pytest.fixture(scope='module')
def runner():
    """Cli runner helper utility."""
    return CliRunner()


def test_pipeline_fwp_collect(runner, log=False):
    """Test pipeline with forward pass and data collection"""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 8, 8, 4, len(FEATURES))))
    model.meta['lr_features'] = FEATURES
    model.meta['hr_out_features'] = FEATURES[:2]
    model.meta['s_enhance'] = 3
    model.meta['t_enhance'] = 4

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)
        fp_out = os.path.join(td, 'fwp_combined.h5')
        n_nodes = len(input_files) // fwp_chunk_shape[2] + 1
        n_nodes *= shape[0] // fwp_chunk_shape[0]
        n_nodes *= shape[1] // fwp_chunk_shape[1]
        out_files = os.path.join(td, 'out_{file_id}.h5')
        fwp_config = {'input_handler_kwargs': {
            'worker_kwargs': {'max_workers': 1},
            'target': (19.3, -123.5),
            'shape': shape},
            'file_paths': input_files,
            'model_kwargs': {'model_dir': out_dir},
            'out_pattern': out_files,
            'fwp_chunk_shape': fwp_chunk_shape,
            'worker_kwargs': {'max_workers': 1},
            'spatial_pad': 1,
            'temporal_pad': 1,
            'execution_control': {
                "option": "local"}}

        features = ['windspeed_100m', 'winddirection_100m']
        out_files = os.path.join(td, 'out_*.h5')
        dc_config = {'file_paths': out_files,
                     'out_file': fp_out,
                     'features': features,
                     'execution_control': {
                         "option": "local"}}

        fwp_config_path = os.path.join(td, 'config_fwp.json')
        dc_config_path = os.path.join(td, 'config_dc.json')
        pipe_config_path = os.path.join(td, 'config_pipe.json')

        pipe_config = {"pipeline": [{"forward-pass":
                                     correct_path(fwp_config_path)},
                                    {"data-collect":
                                     correct_path(dc_config_path)}]}

        with open(fwp_config_path, 'w') as fh:
            json.dump(fwp_config, fh)
        with open(dc_config_path, 'w') as fh:
            json.dump(dc_config, fh)
        with open(pipe_config_path, 'w') as fh:
            json.dump(pipe_config, fh)

        result = runner.invoke(pipe_main, ['-c', pipe_config_path, '-v',
                                           '--monitor'])
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
        (out_files, data, ws_true, wd_true, features, _,
         t_slices_hr, _, s_slices_hr, _, low_res_times) = out

        features = ['windspeed_100m', 'winddirection_100m']
        config = {'worker_kwargs': {'max_workers': 1},
                  'file_paths': out_files,
                  'out_file': fp_out,
                  'features': features,
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
            full_ti = fh.time_index
            combined_ti = []
            for _, f in enumerate(out_files):
                tmp = f.replace('.h5', '').split('_')
                t_idx = int(tmp[-3])
                s1_idx = int(tmp[-2])
                s2_idx = int(tmp[-1])
                t_hr = t_slices_hr[t_idx]
                s1_hr = s_slices_hr[s1_idx]
                s2_hr = s_slices_hr[s2_idx]
                with ResourceX(f) as fh_i:
                    if s1_idx == s2_idx == 0:
                        combined_ti += list(fh_i.time_index)

                    ws_i = np.transpose(data[s1_hr, s2_hr, t_hr, 0],
                                        axes=(2, 0, 1))
                    wd_i = np.transpose(data[s1_hr, s2_hr, t_hr, 1],
                                        axes=(2, 0, 1))
                    ws_i = ws_i.reshape(48, 625)
                    wd_i = wd_i.reshape(48, 625)
                    assert np.allclose(ws_i, fh_i['windspeed_100m'], atol=0.01)
                    assert np.allclose(wd_i, fh_i['winddirection_100m'],
                                       atol=0.1)

                    for k, v in fh_i.global_attrs.items():
                        assert k in fh.global_attrs, k
                        assert fh.global_attrs[k] == v, k

            assert len(full_ti) == len(combined_ti)
            assert len(full_ti) == 2 * len(low_res_times)
            wd_true = np.transpose(wd_true[..., 0], axes=(2, 0, 1))
            ws_true = np.transpose(ws_true[..., 0], axes=(2, 0, 1))
            wd_true = wd_true.reshape(96, 2500)
            ws_true = ws_true.reshape(96, 2500)
            assert np.allclose(ws_true, fh['windspeed_100m'], atol=0.01)
            assert np.allclose(wd_true, fh['winddirection_100m'], atol=0.1)


def test_fwd_pass_cli(runner, log=False):
    """Test cli call to run forward pass"""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 8, 8, 4, len(FEATURES))))
    model.meta['lr_features'] = FEATURES
    model.meta['hr_out_features'] = FEATURES[:2]
    assert model.s_enhance == 3
    assert model.t_enhance == 4

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)
        t_chunks = len(input_files) // fwp_chunk_shape[2] + 1
        n_chunks = t_chunks * shape[0] // fwp_chunk_shape[0]
        n_chunks = n_chunks * shape[1] // fwp_chunk_shape[1]
        out_files = os.path.join(td, 'out_{file_id}.nc')
        cache_pattern = os.path.join(td, 'cache')
        log_prefix = os.path.join(td, 'log.log')
        input_handler_kwargs = {'target': (19.3, -123.5),
                                'shape': shape,
                                'worker_kwargs': {'max_workers': 1},
                                'cache_pattern': cache_pattern}
        config = {'file_paths': input_files,
                  'model_kwargs': {'model_dir': out_dir},
                  'out_pattern': out_files,
                  'log_pattern': log_prefix,
                  'input_handler_kwargs': input_handler_kwargs,
                  'fwp_chunk_shape': fwp_chunk_shape,
                  'worker_kwargs': {'max_workers': 1},
                  'spatial_pad': 1,
                  'temporal_pad': 1,
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
        n_cache_files = 1 + t_chunks + (len(FEATURES) * n_chunks)
        assert len(glob.glob(f'{td}/cache*')) == n_cache_files
        assert len(glob.glob(f'{td}/*.log')) == t_chunks
        assert len(glob.glob(f'{td}/out*')) == n_chunks


def test_data_extract_cli(runner):
    """Test cli call to run data extraction"""
    with tempfile.TemporaryDirectory() as td:
        cache_pattern = os.path.join(td, 'cache')
        log_file = os.path.join(td, 'log.log')
        config = {'file_paths': FP_WTK,
                  'target': (39.01, -105.15),
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


def test_pipeline_fwp_qa(runner, log=False):
    """Test the sup3r pipeline with Forward Pass and QA modules
    via pipeline cli"""

    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    input_resolution = {'spatial': '12km', 'temporal': '60min'}
    model.meta['input_resolution'] = input_resolution
    assert model.input_resolution == input_resolution
    assert model.output_resolution == {'spatial': '4km', 'temporal': '15min'}
    _ = model.generate(np.ones((4, 8, 8, 4, len(FEATURES))))
    model.meta['lr_features'] = FEATURES
    model.meta['hr_out_features'] = FEATURES[:2]
    assert model.s_enhance == 3
    assert model.t_enhance == 4

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        fwp_config = {'file_paths': input_files,
                      'model_kwargs': {'model_dir': out_dir},
                      'out_pattern': os.path.join(td, 'out_{file_id}.h5'),
                      'log_pattern': os.path.join(td, 'fwp_log.log'),
                      'log_level': 'DEBUG',
                      'input_handler_kwargs': {'target': (19.3, -123.5),
                                               'shape': (8, 8),
                                               'overwrite_cache': False},
                      'fwp_chunk_shape': (100, 100, 100),
                      'max_workers': 1,
                      'spatial_pad': 5,
                      'temporal_pad': 5,
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
                       "pipeline": [{"forward-pass":
                                     correct_path(fwp_config_path)},
                                    {"qa": correct_path(qa_config_path)}]}

        with open(fwp_config_path, 'w') as fh:
            json.dump(fwp_config, fh)
        with open(qa_config_path, 'w') as fh:
            json.dump(qa_config, fh)
        with open(pipe_config_path, 'w') as fh:
            json.dump(pipe_config, fh)

        result = runner.invoke(pipe_main, ['-c', pipe_config_path, '-v',
                                           '--monitor'])
        if result.exit_code != 0:
            import traceback
            msg = ('Failed with error {}'
                   .format(traceback.print_exception(*result.exc_info)))
            raise RuntimeError(msg)

        assert len(glob.glob(f'{td}/fwp_log*.log')) == 1
        assert len(glob.glob(f'{td}/out*.h5')) == 1
        assert len(glob.glob(f'{td}/qa.h5')) == 1
        status_fps = glob.glob(f'{td}/.gaps/*status*.json')
        assert len(status_fps) == 1
        status_fp = status_fps[0]
        with open(status_fp) as f:
            status = json.load(f)

        fwp_status = status['forward-pass']
        del fwp_status['pipeline_index']
        fwp_status = next(iter(fwp_status.values()))
        assert fwp_status['job_status'] == 'successful'
        assert fwp_status['time'] > 0

        assert len(status['qa']) == 2
        qa_status = status['qa']
        del qa_status['pipeline_index']
        qa_status = next(iter(qa_status.values()))
        assert qa_status['job_status'] == 'successful'
        assert qa_status['time'] > 0


def test_visual_qa(runner, log=False):
    """Make sure visual qa module creates the right number of plots"""

    if log:
        init_logger('sup3r', log_level='DEBUG')

    time_step = 500
    plot_features = ['windspeed_100m', 'winddirection_100m']
    with ResourceX(FP_WTK) as res:
        time_index = res.time_index

    n_files = len(time_index[::time_step]) * len(plot_features)

    with tempfile.TemporaryDirectory() as td:
        out_pattern = os.path.join(td, 'plot_{feature}_{index}.png')

        config = {'file_paths': FP_WTK,
                  'features': plot_features,
                  'out_pattern': out_pattern,
                  'time_step': time_step,
                  'spatial_slice': [0, 100, 10],
                  'max_workers': 1}

        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as fh:
            json.dump(config, fh)

        result = runner.invoke(vqa_main, ['-c', config_path, '-v'])

        if result.exit_code != 0:
            import traceback
            msg = ('Failed with error {}'
                   .format(traceback.print_exception(*result.exc_info)))
            raise RuntimeError(msg)

        n_out_files = len(glob.glob(out_pattern.format(feature='*',
                                                       index='*')))
        assert n_out_files == n_files
