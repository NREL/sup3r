"""Output method tests"""
import numpy as np
import os
import tempfile
import shutil
from netCDF4 import Dataset

from sup3r.postprocessing.file_handling import OutputHandlerNC, OutputHandlerH5
from sup3r.postprocessing.collection import Collector
from sup3r.utilities.utilities import invert_uv, transform_rotate_wind
from sup3r.preprocessing.data_handling import DataHandlerH5
from sup3r.preprocessing.batch_handling import BatchHandler
from sup3r.pipeline.forward_pass import ForwardPass, ForwardPassStrategy
from sup3r.models.base import Sup3rGan
from sup3r import TEST_DATA_DIR, CONFIG_DIR

from rex import ResourceX

FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
TARGET_COORD = (39.01, -105.15)
FEATURES = ['U_100m', 'V_100m']
target = (19.3, -123.5)
shape = (8, 8)
sample_shape = (8, 8, 6)
temporal_slice = slice(None, None, 1)
list_chunk_size = 10
forward_pass_chunk_shape = (4, 4, 4)
s_enhance = 3
t_enhance = 4

INPUT_FILES = [
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_00_00_00'),
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_01_00_00'),
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_01_00_00'),
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_01_00_00'),
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_01_00_00'),
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_01_00_00'),
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_01_00_00'),
    os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_01_00_00')]


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


def test_get_lat_lon():
    """Check that regridding works correctly"""
    low_res_lats = np.array([[1, 1, 1], [0, 0, 0]])
    low_res_lons = np.array([[-120, -100, -80], [-120, -100, -80]])
    lat_lon = np.concatenate([np.expand_dims(low_res_lats, axis=-1),
                              np.expand_dims(low_res_lons, axis=-1)], axis=-1)
    shape = (4, 6)
    new_lat_lon = OutputHandlerNC.get_lat_lon(lat_lon, shape)

    new_lats = np.round(new_lat_lon[..., 0], 2)
    new_lons = np.round(new_lat_lon[..., 1], 2)

    assert np.allclose(new_lats[0, :], 1.25)
    assert np.allclose(new_lats[-1, :], -0.25)

    assert np.allclose(new_lons[:, 0], -125)
    assert np.allclose(new_lons[:, -1], -75)


def test_invert_uv():
    """Make sure inverse uv transform returns inputs"""
    lats = np.array([[1, 1, 1], [0, 0, 0]])
    lons = np.array([[-120, -100, -80], [-120, -100, -80]])
    lat_lon = np.concatenate([np.expand_dims(lats, axis=-1),
                              np.expand_dims(lons, axis=-1)], axis=-1)
    windspeed = np.random.rand(lat_lon.shape[0], lat_lon.shape[1], 5)
    winddirection = 360 * np.random.rand(lat_lon.shape[0], lat_lon.shape[1], 5)

    u, v = transform_rotate_wind(np.array(windspeed, dtype=np.float32),
                                 np.array(winddirection, dtype=np.float32),
                                 lat_lon)

    ws, wd = invert_uv(u, v, lat_lon)

    assert np.allclose(windspeed, ws)
    assert np.allclose(winddirection, wd)


def test_invert_uv_inplace():
    """Make sure inverse uv transform in output handler returns same as direct
    transform"""

    lats = np.array([[1, 1, 1], [0, 0, 0]])
    lons = np.array([[-120, -100, -80], [-120, -100, -80]])
    lat_lon = np.concatenate([np.expand_dims(lats, axis=-1),
                              np.expand_dims(lons, axis=-1)], axis=-1)
    u = np.random.rand(lat_lon.shape[0], lat_lon.shape[1], 5)
    v = np.random.rand(lat_lon.shape[0], lat_lon.shape[1], 5)

    data = np.concatenate([np.expand_dims(u, axis=-1),
                           np.expand_dims(v, axis=-1)], axis=-1)

    OutputHandlerH5.invert_uv_features(data, ['U_100m', 'V_100m'], lat_lon)

    ws, wd = invert_uv(u, v, lat_lon)

    assert np.allclose(data[..., 0], ws)
    assert np.allclose(data[..., 1], wd)


def test_forward_pass_collection():
    """Test forward pass collection."""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)

    # only use wind features since model output only gives 2 features
    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            shape=(20, 20),
                            sample_shape=(18, 18, 24),
                            temporal_slice=slice(None, None, 1),
                            val_split=0.005,
                            extract_workers=1,
                            compute_workers=1)

    batch_handler = BatchHandler([handler], batch_size=4,
                                 s_enhance=s_enhance,
                                 t_enhance=t_enhance,
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

        cache_file_prefix = os.path.join(td, 'cache')
        out_files = os.path.join(td, 'out_{file_id}.h5')
        handler = ForwardPassStrategy(
            input_files, target=target, shape=shape,
            temporal_slice=temporal_slice,
            cache_file_prefix=cache_file_prefix,
            forward_pass_chunk_shape=forward_pass_chunk_shape,
            overwrite_cache=True, out_pattern=out_files, pass_workers=1)

        for i in range(handler.nodes):
            forward_pass = ForwardPass(handler, model_path=out_dir,
                                       node_index=i)
            forward_pass.run()

        fp_out = os.path.join(td, 'out_combined.h5')
        Collector.collect(handler.out_files, fp_out,
                          features=['windspeed_100m', 'winddirection_100m'])

        with ResourceX(fp_out) as fh:
            full_ti = fh.time_index
            combined_ti = []
            for i, f in enumerate(handler.out_files):
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
