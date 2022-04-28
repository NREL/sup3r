# -*- coding: utf-8 -*-
"""pytests for data handling"""

import os
import tempfile

from sup3r import TEST_DATA_DIR, CONFIG_DIR
from sup3r.preprocessing.data_handling import DataHandlerH5
from sup3r.preprocessing.batch_handling import BatchHandler
from sup3r.pipeline.gen_handling import ForwardPassHandler
from sup3r.models.models import SpatioTemporalGan


FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
TARGET_COORD = (39.01, -105.15)
FEATURES = ['U_100m', 'V_100m', 'BVF_squared_200m']

input_file = os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_00_00_00')
input_files = [
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_00_00_00'),
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_01_00_00'),
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_00_00_00'),
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_01_00_00'),
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_00_00_00'),
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_01_00_00'),
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_00_00_00'),
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_01_00_00')]
target = (19, -125)
targets = target
shape = (8, 8)
spatial_chunk_size = (8, 8)
spatial_sample_shape = (8, 8)
raster_file = os.path.join(tempfile.gettempdir(), 'tmp_raster_nc.txt')
time_shape = slice(None, None, 1)
temporal_sample_shape = 6
list_chunk_size = 10
temporal_chunk_size = 10


def test_fwd_pass_handler():
    """Test forward pass handler. Make sure it is
    returning the correct data shape"""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x.json')
    fp_disc_s = os.path.join(CONFIG_DIR, 'spatiotemporal/disc_space.json')
    fp_disc_t = os.path.join(CONFIG_DIR, 'spatiotemporal/disc_time.json')

    SpatioTemporalGan.seed()
    model = SpatioTemporalGan(fp_gen, fp_disc_s, fp_disc_t,
                              learning_rate=1e-4)

    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            shape=(20, 20),
                            temporal_sample_shape=24,
                            spatial_sample_shape=(18, 18),
                            time_shape=slice(None, None, 1),
                            val_split=0.005,
                            max_extract_workers=1,
                            max_compute_workers=1)

    batch_handler = BatchHandler([handler], batch_size=4,
                                 s_enhance=3, t_enhance=4,
                                 n_batches=4)

    with tempfile.TemporaryDirectory() as td:
        model.train(batch_handler, n_epoch=1,
                    weight_gen_advers_s=0.0, weight_gen_advers_t=0.0,
                    train_gen=True, train_disc_s=False, train_disc_t=False,
                    checkpoint_int=2,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        cache_file_prefix = os.path.join(td, 'cache')
        print(cache_file_prefix)
        pass_handler = ForwardPassHandler(
            input_files, list_chunk_size, FEATURES,
            raster_file=raster_file, model_path=out_dir,
            target=target, shape=shape, temporal_shape=time_shape,
            temporal_chunk_size=temporal_chunk_size,
            spatial_chunk_size=spatial_chunk_size,
            cache_file_prefix=cache_file_prefix,
            s_enhance=3, t_enhance=4)

        data = pass_handler.kick_off_node(
            input_files, model_path=out_dir, features=FEATURES,
            target=target, shape=shape,
            temporal_shape=time_shape,
            temporal_chunk_size=temporal_chunk_size,
            spatial_chunk_size=spatial_chunk_size,
            raster_file=raster_file,
            cache_file_prefix=cache_file_prefix,
            out_file=None,
            overwrite_cache=True)

        assert data.shape == (pass_handler.s_enhance * shape[0],
                              pass_handler.s_enhance * shape[1],
                              pass_handler.t_enhance * len(input_files),
                              2)
