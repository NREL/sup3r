"""Sup3r pipeline tests"""
import tempfile
import os
import json
import shutil
import numpy as np
import glob

from sup3r.pipeline.pipeline import Sup3rPipeline as Pipeline
from sup3r.models.base import Sup3rGan
from sup3r.utilities.test_utils import make_fake_nc_files
from sup3r import TEST_DATA_DIR, CONFIG_DIR

INPUT_FILE = os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_00_00_00')
FEATURES = ['U_100m', 'V_100m', 'BVF2_200m']


def test_pipeline():
    """Test sup3r pipeline"""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 8, 8, 4, len(FEATURES))))
    model.meta['training_features'] = FEATURES

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        fp_chunk_shape = (4, 4, 3)
        cache_prefix = os.path.join(td, 'cache')
        out_files = os.path.join(td, 'fp_out_{file_id}.h5')
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
                  'spatial_overlap': 2,
                  'temporal_overlap': 2,
                  'pass_workers': None,
                  'overwrite_cache': True,
                  'execution_control': {
                      "nodes": 1,
                      "option": "local"}}

        fp_config_path = os.path.join(td, 'fp_config.json')
        with open(fp_config_path, 'w') as fh:
            json.dump(config, fh)

        out_files = os.path.join(td, 'fp_out_*.h5')
        features = ['windspeed_100m', 'winddirection_100m']
        fp_out = os.path.join(td, 'out_combined.h5')
        config = {'file_paths': out_files,
                  'f_out': fp_out,
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

        status_file = glob.glob(os.path.join(td, '*_status.json'))
        with open(status_file, 'r') as fh:
            status = json.load(fh)
            assert all(s not in str(status)
                       for s in ('fail', 'pending', 'submitted'))
            assert 'successful' in str(status)
