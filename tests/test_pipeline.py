"""Sup3r pipeline tests"""
import tempfile
import os
import json
import shutil
import numpy as np

from netCDF4 import Dataset

from sup3r.pipeline.pipeline import Sup3rPipeline as Pipeline
from sup3r.models.base import Sup3rGan
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
forward_pass_chunk_shape = (4, 4, 4)
s_enhance = 3
t_enhance = 4
target = (19.3, -123.5)
shape = (8, 8)


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


def test_pipeline():
    """Test sup3r pipeline"""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 8, 8, 4, len(FEATURES))))
    model.meta['training_features'] = FEATURES

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td)
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        fp_chunk_shape = (4, 4, 3)
        n_nodes = len(input_files) // fp_chunk_shape[2] + 1
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
