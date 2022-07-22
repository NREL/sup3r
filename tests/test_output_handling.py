"""Output method tests"""
import json
import numpy as np
import os
import tensorflow as tf
import tempfile

from sup3r import __version__
from sup3r.postprocessing.file_handling import OutputHandlerNC, OutputHandlerH5
from sup3r.postprocessing.collection import Collector
from sup3r.utilities.utilities import invert_uv, transform_rotate_wind
from sup3r.utilities.test_utils import make_fake_h5_chunks

from rex import ResourceX


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


def test_h5_out_and_collect():
    """Test h5 file output writing and collection with dummy data"""

    with tempfile.TemporaryDirectory() as td:
        fp_out = os.path.join(td, 'out_combined.h5')
        out = make_fake_h5_chunks(td)
        (out_files, data, ws_true, wd_true, features, slices_lr,
            slices_hr, low_res_lat_lon, low_res_times) = out

        Collector.collect(out_files, fp_out, features=features)

        with ResourceX(fp_out) as fh:
            full_ti = fh.time_index
            combined_ti = []
            for i, f in enumerate(out_files):
                slice_hr = slices_hr[i]
                with ResourceX(f) as fh_i:
                    combined_ti += list(fh_i.time_index)

                    ws_i = np.transpose(data[..., slice_hr, 0], axes=(2, 0, 1))
                    wd_i = np.transpose(data[..., slice_hr, 1], axes=(2, 0, 1))
                    ws_i = ws_i.reshape(48, 2500)
                    wd_i = wd_i.reshape(48, 2500)
                    assert np.allclose(ws_i, fh_i['windspeed_100m'], atol=0.01)
                    assert np.allclose(wd_i, fh_i['winddirection_100m'],
                                       atol=0.1)

                    if i == 0:
                        ws = fh_i['windspeed_100m']
                        wd = fh_i['winddirection_100m']
                    else:
                        ws = np.concatenate([ws, fh_i['windspeed_100m']],
                                            axis=0)
                        wd = np.concatenate([wd, fh_i['winddirection_100m']],
                                            axis=0)

                    for k, v in fh_i.global_attrs.items():
                        assert k in fh.global_attrs, k
                        assert fh.global_attrs[k] == v, k

            assert len(full_ti) == len(combined_ti)
            assert len(full_ti) == 2 * len(low_res_times)
            assert np.allclose(ws, fh['windspeed_100m'])
            assert np.allclose(wd, fh['winddirection_100m'])
            wd_true = np.transpose(wd_true[..., 0], axes=(2, 0, 1))
            ws_true = np.transpose(ws_true[..., 0], axes=(2, 0, 1))
            wd_true = wd_true.reshape(96, 2500)
            ws_true = ws_true.reshape(96, 2500)
            assert np.allclose(ws_true, fh['windspeed_100m'], atol=0.01)
            assert np.allclose(wd_true, fh['winddirection_100m'], atol=0.1)

            assert fh.global_attrs['package'] == 'sup3r'
            assert fh.global_attrs['version'] == __version__
            assert 'full_version_record' in fh.global_attrs
            version_record = json.loads(fh.global_attrs['full_version_record'])
            assert version_record['tensorflow'] == tf.__version__
            assert 'gan_meta' in fh.global_attrs
            gan_meta = json.loads(fh.global_attrs['gan_meta'])
            assert isinstance(gan_meta, dict)
            assert gan_meta['foo'] == 'bar'
