"""Sup3r pipeline tests"""
import tempfile
import os
import json
import shutil
import numpy as np
import glob

from rex import ResourceX

from sup3r.pipeline.pipeline import Sup3rPipeline as Pipeline
from sup3r.models.base import Sup3rGan
from sup3r.utilities.pytest import make_fake_nc_files
from sup3r import TEST_DATA_DIR, CONFIG_DIR

INPUT_FILE = os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_00_00_00')
FEATURES = ['U_100m', 'V_100m', 'BVF2_200m']


def test_fwp_pipeline():
    """Test sup3r pipeline"""

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

        status_file = glob.glob(os.path.join(td, '*_status.json'))[0]
        with open(status_file, 'r') as fh:
            status = json.load(fh)
            assert all(s in status for s in ('forward-pass', 'data-collect'))
            assert all(s not in str(status)
                       for s in ('fail', 'pending', 'submitted'))
            assert 'successful' in str(status)