"""Output method tests"""
import json
import os
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf
from rex import ResourceX, init_logger

from sup3r import __version__
from sup3r.postprocessing.collection import CollectorH5
from sup3r.postprocessing.file_handling import OutputHandlerH5, OutputHandlerNC
from sup3r.utilities.pytest import make_fake_h5_chunks
from sup3r.utilities.utilities import invert_uv, transform_rotate_wind


def test_get_lat_lon():
    """Check that regridding works correctly"""
    low_res_lats = np.array([[1, 1, 1], [0, 0, 0]])
    low_res_lons = np.array([[-120, -100, -80], [-120, -100, -80]])
    lat_lon = np.concatenate(
        [
            np.expand_dims(low_res_lats, axis=-1),
            np.expand_dims(low_res_lons, axis=-1),
        ],
        axis=-1,
    )
    shape = (4, 6)
    new_lat_lon = OutputHandlerNC.get_lat_lon(lat_lon, shape)

    new_lats = np.round(new_lat_lon[..., 0], 2)
    new_lons = np.round(new_lat_lon[..., 1], 2)

    assert np.allclose(new_lats[0, :], 1.25)
    assert np.allclose(new_lats[-1, :], -0.25)

    assert np.allclose(new_lons[:, 0], -125)
    assert np.allclose(new_lons[:, -1], -75)

    lat_lon = lat_lon[::-1]
    new_lat_lon = OutputHandlerNC.get_lat_lon(lat_lon, shape)

    new_lats = np.round(new_lat_lon[..., 0], 2)
    new_lons = np.round(new_lat_lon[..., 1], 2)

    assert np.allclose(new_lats[0, :], -0.25)
    assert np.allclose(new_lats[-1, :], 1.25)

    assert np.allclose(new_lons[:, 0], -125)
    assert np.allclose(new_lons[:, -1], -75)


def test_invert_uv():
    """Make sure inverse uv transform returns inputs"""
    lats = np.array([[1, 1, 1], [0, 0, 0]])
    lons = np.array([[-120, -100, -80], [-120, -100, -80]])
    lat_lon = np.concatenate(
        [np.expand_dims(lats, axis=-1), np.expand_dims(lons, axis=-1)], axis=-1
    )
    windspeed = np.random.rand(lat_lon.shape[0], lat_lon.shape[1], 5)
    winddirection = 360 * np.random.rand(lat_lon.shape[0], lat_lon.shape[1], 5)

    u, v = transform_rotate_wind(
        np.array(windspeed, dtype=np.float32),
        np.array(winddirection, dtype=np.float32),
        lat_lon,
    )

    ws, wd = invert_uv(u, v, lat_lon)

    assert np.allclose(windspeed, ws)
    assert np.allclose(winddirection, wd)

    lat_lon = lat_lon[::-1]
    u, v = transform_rotate_wind(
        np.array(windspeed, dtype=np.float32),
        np.array(winddirection, dtype=np.float32),
        lat_lon,
    )

    ws, wd = invert_uv(u, v, lat_lon)

    assert np.allclose(windspeed, ws)
    assert np.allclose(winddirection, wd)


def test_invert_uv_inplace():
    """Make sure inverse uv transform in output handler returns same as direct
    transform"""

    lats = np.array([[1, 1, 1], [0, 0, 0]])
    lons = np.array([[-120, -100, -80], [-120, -100, -80]])
    lat_lon = np.concatenate(
        [np.expand_dims(lats, axis=-1), np.expand_dims(lons, axis=-1)], axis=-1
    )
    u = np.random.rand(lat_lon.shape[0], lat_lon.shape[1], 5)
    v = np.random.rand(lat_lon.shape[0], lat_lon.shape[1], 5)

    data = np.concatenate(
        [np.expand_dims(u, axis=-1), np.expand_dims(v, axis=-1)], axis=-1
    )
    OutputHandlerH5.invert_uv_features(data, ['U_100m', 'V_100m'], lat_lon)

    ws, wd = invert_uv(u, v, lat_lon)

    assert np.allclose(data[..., 0], ws)
    assert np.allclose(data[..., 1], wd)

    lat_lon = lat_lon[::-1]
    data = np.concatenate(
        [np.expand_dims(u, axis=-1), np.expand_dims(v, axis=-1)], axis=-1
    )
    OutputHandlerH5.invert_uv_features(data, ['U_100m', 'V_100m'], lat_lon)

    ws, wd = invert_uv(u, v, lat_lon)

    assert np.allclose(data[..., 0], ws)
    assert np.allclose(data[..., 1], wd)


def test_h5_out_and_collect():
    """Test h5 file output writing and collection with dummy data"""

    with tempfile.TemporaryDirectory() as td:
        fp_out = os.path.join(td, 'out_combined.h5')

        out = make_fake_h5_chunks(td)
        (
            out_files,
            data,
            ws_true,
            wd_true,
            features,
            _,
            t_slices_hr,
            _,
            s_slices_hr,
            _,
            low_res_times,
        ) = out

        CollectorH5.collect(out_files, fp_out, features=features)
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

                    ws_i = np.transpose(
                        data[s1_hr, s2_hr, t_hr, 0], axes=(2, 0, 1)
                    )
                    wd_i = np.transpose(
                        data[s1_hr, s2_hr, t_hr, 1], axes=(2, 0, 1)
                    )
                    ws_i = ws_i.reshape(48, 625)
                    wd_i = wd_i.reshape(48, 625)
                    assert np.allclose(ws_i, fh_i['windspeed_100m'], atol=0.01)
                    assert np.allclose(
                        wd_i, fh_i['winddirection_100m'], atol=0.1
                    )

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

            assert fh.global_attrs['package'] == 'sup3r'
            assert fh.global_attrs['version'] == __version__
            assert 'full_version_record' in fh.global_attrs
            version_record = json.loads(fh.global_attrs['full_version_record'])
            assert version_record['tensorflow'] == tf.__version__
            assert 'foo' in fh.global_attrs
            gan_meta = fh.global_attrs['foo']
            assert isinstance(gan_meta, str)
            assert gan_meta == 'bar'


def test_h5_collect_mask(log=False):
    """Test h5 file collection with mask meta"""

    if log:
        init_logger('sup3r', log_level='DEBUG')

    with tempfile.TemporaryDirectory() as td:
        fp_out = os.path.join(td, 'out_combined.h5')
        fp_out_mask = os.path.join(td, 'out_combined_masked.h5')
        mask_file = os.path.join(td, 'mask.csv')

        out = make_fake_h5_chunks(td)
        (out_files, data, _, _, features, _, _, _, _, _, _) = out

        CollectorH5.collect(out_files, fp_out, features=features)
        indices = np.arange(np.prod(data.shape[:2]))
        indices = indices[slice(-len(indices) // 2, None)]
        removed = []
        for _ in range(10):
            removed.append(np.random.choice(indices))
        mask_slice = [i for i in indices if i not in removed]
        with ResourceX(fp_out) as fh:
            mask_meta = fh.meta
            mask_meta = mask_meta.iloc[mask_slice].reset_index(drop=True)
            mask_meta['gid'][:] = np.arange(len(mask_meta))
            mask_meta.to_csv(mask_file, index=False)

        CollectorH5.collect(
            out_files,
            fp_out_mask,
            features=features,
            target_final_meta_file=mask_file,
            max_workers=1,
            join_times=False,
        )
        with ResourceX(fp_out_mask) as fh:
            mask_meta = pd.read_csv(mask_file, dtype=np.float32)
            assert np.array_equal(mask_meta['gid'], fh.meta.index.values)
            assert np.array_equal(mask_meta['longitude'], fh.meta['longitude'])
            assert np.array_equal(mask_meta['latitude'], fh.meta['latitude'])

            with ResourceX(fp_out) as fh_o:
                assert np.array_equal(
                    fh_o['windspeed_100m', :, mask_slice], fh['windspeed_100m']
                )
