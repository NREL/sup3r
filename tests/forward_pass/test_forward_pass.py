# -*- coding: utf-8 -*-
"""pytests for data handling"""
import json
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytest
import tensorflow as tf
import xarray as xr
from rex import ResourceX, init_logger

from sup3r import CONFIG_DIR, TEST_DATA_DIR, __version__
from sup3r.models import LinearInterp, Sup3rGan, WindGan
from sup3r.pipeline.forward_pass import ForwardPass, ForwardPassStrategy
from sup3r.preprocessing.data_handling import DataHandlerNC
from sup3r.utilities.pytest import (make_fake_nc_files,
                                    make_fake_multi_time_nc_files)

FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
TARGET_COORD = (39.01, -105.15)
FEATURES = ['U_100m', 'V_100m', 'BVF2_200m']
INPUT_FILE = os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_00_00_00')
target = (19.3, -123.5)
shape = (8, 8)
sample_shape = (8, 8, 6)
temporal_slice = slice(None, None, 1)
list_chunk_size = 10
fwp_chunk_shape = (4, 4, 150)
s_enhance = 3
t_enhance = 4


def test_fwp_nc_cc(log=False):
    """Test forward pass handler output for netcdf write with cc data."""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)

    input_files = [os.path.join(TEST_DATA_DIR, 'ua_test.nc'),
                   os.path.join(TEST_DATA_DIR, 'va_test.nc'),
                   os.path.join(TEST_DATA_DIR, 'orog_test.nc'),
                   os.path.join(TEST_DATA_DIR, 'zg_test.nc')]
    features = ['U_100m', 'V_100m']
    target = (13.67, 125.0)
    _ = model.generate(np.ones((4, 10, 10, 6, len(features))))
    model.meta['training_features'] = features
    model.meta['output_features'] = features
    model.meta['s_enhance'] = 3
    model.meta['t_enhance'] = 4
    with tempfile.TemporaryDirectory() as td:
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        cache_pattern = os.path.join(td, 'cache')
        out_files = os.path.join(td, 'out_{file_id}.nc')
        # 1st forward pass
        max_workers = 1
        input_handler_kwargs = dict(
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            cache_pattern=cache_pattern,
            overwrite_cache=True,
            worker_kwargs=dict(max_workers=max_workers))
        handler = ForwardPassStrategy(
            input_files, model_kwargs={'model_dir': out_dir},
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=1, temporal_pad=1,
            input_handler_kwargs=input_handler_kwargs, out_pattern=out_files,
            worker_kwargs=dict(max_workers=max_workers),
            input_handler='DataHandlerNCforCC')
        forward_pass = ForwardPass(handler)
        assert forward_pass.output_workers == max_workers
        assert forward_pass.data_handler.compute_workers == max_workers
        assert forward_pass.data_handler.load_workers == max_workers
        assert forward_pass.data_handler.norm_workers == max_workers
        assert forward_pass.data_handler.extract_workers == max_workers
        forward_pass.run(handler, node_index=0)

        with xr.open_dataset(handler.out_files[0]) as fh:
            assert fh[FEATURES[0]].shape == (
                t_enhance * len(handler.time_index),
                s_enhance * fwp_chunk_shape[0],
                s_enhance * fwp_chunk_shape[1])
            assert fh[FEATURES[1]].shape == (
                t_enhance * len(handler.time_index),
                s_enhance * fwp_chunk_shape[0],
                s_enhance * fwp_chunk_shape[1])


def test_fwp_single_ts_vs_multi_ts_input_files():
    """Test forward pass handler output for spatial only model."""

    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 10, 10, len(FEATURES))))
    model.meta['training_features'] = FEATURES
    model.meta['output_features'] = ['U_100m', 'V_100m']
    model.meta['s_enhance'] = 2
    model.meta['t_enhance'] = 1
    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        out_dir = os.path.join(td, 's_gan')
        model.save(out_dir)

        cache_pattern = os.path.join(td, 'cache')
        out_files = os.path.join(td, 'out_{file_id}_single_ts.nc')

        max_workers = 1
        input_handler_kwargs = dict(
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            worker_kwargs=dict(max_workers=max_workers),
            cache_pattern=cache_pattern,
            overwrite_cache=True)
        single_ts_handler = ForwardPassStrategy(
            input_files, model_kwargs={'model_dir': out_dir},
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=1, temporal_pad=1,
            input_handler_kwargs=input_handler_kwargs, out_pattern=out_files,
            worker_kwargs=dict(max_workers=max_workers))
        single_ts_forward_pass = ForwardPass(single_ts_handler)
        single_ts_forward_pass.run(single_ts_handler, node_index=0)

        input_files = make_fake_multi_time_nc_files(td, INPUT_FILE, 8, 2)

        cache_pattern = os.path.join(td, 'cache')
        out_files = os.path.join(td, 'out_{file_id}_multi_ts.nc')

        max_workers = 1
        input_handler_kwargs = dict(
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            worker_kwargs=dict(max_workers=max_workers),
            cache_pattern=cache_pattern,
            overwrite_cache=True)
        multi_ts_handler = ForwardPassStrategy(
            input_files, model_kwargs={'model_dir': out_dir},
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=1, temporal_pad=1,
            input_handler_kwargs=input_handler_kwargs, out_pattern=out_files,
            worker_kwargs=dict(max_workers=max_workers))
        multi_ts_forward_pass = ForwardPass(multi_ts_handler)
        multi_ts_forward_pass.run(multi_ts_handler, node_index=0)

        kwargs = {'combine': 'nested', 'concat_dim': 'Time'}
        with xr.open_mfdataset(single_ts_handler.out_files,
                               **kwargs) as single_ts:
            with xr.open_mfdataset(multi_ts_handler.out_files,
                                   **kwargs) as multi_ts:
                for feat in model.meta['output_features']:
                    assert np.array_equal(single_ts[feat].values,
                                          multi_ts[feat].values)


def test_fwp_spatial_only():
    """Test forward pass handler output for spatial only model."""

    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 10, 10, len(FEATURES))))
    model.meta['training_features'] = FEATURES
    model.meta['output_features'] = ['U_100m', 'V_100m']
    model.meta['s_enhance'] = 2
    model.meta['t_enhance'] = 1
    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        out_dir = os.path.join(td, 's_gan')
        model.save(out_dir)

        cache_pattern = os.path.join(td, 'cache')
        out_files = os.path.join(td, 'out_{file_id}.nc')

        max_workers = 1
        input_handler_kwargs = dict(
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            worker_kwargs=dict(max_workers=max_workers),
            cache_pattern=cache_pattern,
            overwrite_cache=True)
        handler = ForwardPassStrategy(
            input_files, model_kwargs={'model_dir': out_dir},
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=1, temporal_pad=1,
            input_handler_kwargs=input_handler_kwargs, out_pattern=out_files,
            worker_kwargs=dict(max_workers=max_workers))
        forward_pass = ForwardPass(handler)
        assert forward_pass.output_workers == max_workers
        assert forward_pass.data_handler.compute_workers == max_workers
        assert forward_pass.data_handler.load_workers == max_workers
        assert forward_pass.data_handler.norm_workers == max_workers
        assert forward_pass.data_handler.extract_workers == max_workers
        forward_pass.run(handler, node_index=0)

        with xr.open_dataset(handler.out_files[0]) as fh:
            assert fh[FEATURES[0]].shape == (
                len(handler.time_index),
                2 * fwp_chunk_shape[0],
                2 * fwp_chunk_shape[1])
            assert fh[FEATURES[1]].shape == (
                len(handler.time_index),
                2 * fwp_chunk_shape[0],
                2 * fwp_chunk_shape[1])


def test_fwp_nc():
    """Test forward pass handler output for netcdf write."""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 10, 10, 6, len(FEATURES))))
    model.meta['training_features'] = FEATURES
    model.meta['output_features'] = ['U_100m', 'V_100m']
    model.meta['s_enhance'] = 3
    model.meta['t_enhance'] = 4
    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        cache_pattern = os.path.join(td, 'cache')
        out_files = os.path.join(td, 'out_{file_id}.nc')

        max_workers = 1
        input_handler_kwargs = dict(
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            worker_kwargs=dict(max_workers=max_workers),
            cache_pattern=cache_pattern,
            overwrite_cache=True)
        handler = ForwardPassStrategy(
            input_files, model_kwargs={'model_dir': out_dir},
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=1, temporal_pad=1,
            input_handler_kwargs=input_handler_kwargs, out_pattern=out_files,
            worker_kwargs=dict(max_workers=max_workers))
        forward_pass = ForwardPass(handler)
        assert forward_pass.output_workers == max_workers
        assert forward_pass.data_handler.compute_workers == max_workers
        assert forward_pass.data_handler.load_workers == max_workers
        assert forward_pass.data_handler.norm_workers == max_workers
        assert forward_pass.data_handler.extract_workers == max_workers
        forward_pass.run(handler, node_index=0)

        with xr.open_dataset(handler.out_files[0]) as fh:
            assert fh[FEATURES[0]].shape == (
                t_enhance * len(handler.time_index),
                s_enhance * fwp_chunk_shape[0],
                s_enhance * fwp_chunk_shape[1])
            assert fh[FEATURES[1]].shape == (
                t_enhance * len(handler.time_index),
                s_enhance * fwp_chunk_shape[0],
                s_enhance * fwp_chunk_shape[1])


def test_fwp_temporal_slice():
    """Test forward pass handler output to h5 file. Includes temporal
    slicing."""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 10, 10, 6, 2)))
    model.meta['training_features'] = ['U_100m', 'V_100m']
    model.meta['output_features'] = ['U_100m', 'V_100m']
    model.meta['s_enhance'] = 3
    model.meta['t_enhance'] = 4
    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 20)
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        cache_pattern = os.path.join(td, 'cache')
        out_files = os.path.join(td, 'out_{file_id}.h5')

        max_workers = 1
        temporal_slice = slice(5, 17, 3)
        raw_time_index = np.arange(20)
        n_tsteps = len(raw_time_index[temporal_slice])
        input_handler_kwargs = dict(
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            worker_kwargs=dict(max_workers=max_workers),
            cache_pattern=cache_pattern,
            overwrite_cache=True)
        handler = ForwardPassStrategy(
            input_files, model_kwargs={'model_dir': out_dir},
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=1, temporal_pad=1,
            input_handler_kwargs=input_handler_kwargs, out_pattern=out_files,
            worker_kwargs=dict(max_workers=max_workers))
        forward_pass = ForwardPass(handler)
        assert forward_pass.output_workers == max_workers
        assert forward_pass.data_handler.compute_workers == max_workers
        assert forward_pass.data_handler.load_workers == max_workers
        assert forward_pass.data_handler.norm_workers == max_workers
        assert forward_pass.data_handler.extract_workers == max_workers
        forward_pass.run(handler, node_index=0)

        with ResourceX(handler.out_files[0]) as fh:
            assert fh.shape == (
                t_enhance * n_tsteps,
                s_enhance**2 * fwp_chunk_shape[0] * fwp_chunk_shape[1])
            assert all(f in fh.attrs for f in ('windspeed_100m',
                                               'winddirection_100m'))

            assert fh.global_attrs['package'] == 'sup3r'
            assert fh.global_attrs['version'] == __version__
            assert 'full_version_record' in fh.global_attrs
            version_record = json.loads(fh.global_attrs['full_version_record'])
            assert version_record['tensorflow'] == tf.__version__
            assert 'gan_meta' in fh.global_attrs
            gan_meta = json.loads(fh.global_attrs['gan_meta'])
            assert isinstance(gan_meta, dict)
            assert gan_meta['training_features'] == ['U_100m', 'V_100m']


def test_fwp_handler():
    """Test forward pass handler. Make sure it is
    returning the correct data shape"""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    model.meta['training_features'] = FEATURES
    model.meta['output_features'] = FEATURES[:-1]
    model.meta['s_enhance'] = s_enhance
    model.meta['t_enhance'] = t_enhance
    _ = model.generate(np.ones((4, 10, 10, 12, 3)))

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        max_workers = 1
        cache_pattern = os.path.join(td, 'cache')
        input_handler_kwargs = dict(
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            worker_kwargs=dict(max_workers=max_workers),
            cache_pattern=cache_pattern,
            overwrite_cache=True)
        handler = ForwardPassStrategy(
            input_files, model_kwargs={'model_dir': out_dir},
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=1, temporal_pad=1,
            input_handler_kwargs=input_handler_kwargs,
            worker_kwargs=dict(max_workers=max_workers))
        forward_pass = ForwardPass(handler)
        assert forward_pass.data_handler.compute_workers == max_workers
        assert forward_pass.data_handler.load_workers == max_workers
        assert forward_pass.data_handler.norm_workers == max_workers
        assert forward_pass.data_handler.extract_workers == max_workers
        data = forward_pass.run_chunk()

        assert data.shape == (s_enhance * fwp_chunk_shape[0],
                              s_enhance * fwp_chunk_shape[1],
                              t_enhance * len(input_files), 2)


def test_fwp_chunking(log=False, plot=False):
    """Test forward pass spatialtemporal chunking. Make sure chunking agrees
    closely with non chunking forward pass.
    """

    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    model.meta['training_features'] = FEATURES
    model.meta['output_features'] = FEATURES[:-1]
    model.meta['s_enhance'] = s_enhance
    model.meta['t_enhance'] = t_enhance
    _ = model.generate(np.ones((4, 10, 10, 12, 3)))

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        out_dir = os.path.join(td, 'test_1')
        model.save(out_dir)
        spatial_pad = 20
        temporal_pad = 20
        cache_pattern = os.path.join(td, 'cache')
        fwp_shape = (4, 4, len(input_files) // 2)
        handler = ForwardPassStrategy(
            input_files, model_kwargs={'model_dir': out_dir},
            fwp_chunk_shape=fwp_shape,
            worker_kwargs=dict(max_workers=1),
            spatial_pad=spatial_pad, temporal_pad=temporal_pad,
            input_handler_kwargs=dict(target=target, shape=shape,
                                      temporal_slice=temporal_slice,
                                      cache_pattern=cache_pattern,
                                      overwrite_cache=True,
                                      worker_kwargs=dict(max_workers=1)))
        data_chunked = np.zeros((shape[0] * s_enhance, shape[1] * s_enhance,
                                 len(input_files) * t_enhance,
                                 len(model.output_features)))
        handlerNC = DataHandlerNC(input_files, FEATURES, target=target,
                                  val_split=0.0, shape=shape,
                                  worker_kwargs=dict(ti_workers=1))
        pad_width = ((spatial_pad, spatial_pad), (spatial_pad, spatial_pad),
                     (temporal_pad, temporal_pad), (0, 0))
        hr_crop = (slice(s_enhance * spatial_pad, -s_enhance * spatial_pad),
                   slice(s_enhance * spatial_pad, -s_enhance * spatial_pad),
                   slice(t_enhance * temporal_pad, -t_enhance * temporal_pad),
                   slice(None))
        input_data = np.pad(handlerNC.data, pad_width=pad_width,
                            mode='reflect')
        data_nochunk = model.generate(
            np.expand_dims(input_data, axis=0))[0][hr_crop]
        for i in range(handler.chunks):
            fwp = ForwardPass(handler, chunk_index=i)
            out = fwp.run_chunk()
            t_hr_slice = slice(fwp.ti_slice.start * t_enhance,
                               fwp.ti_slice.stop * t_enhance)
            data_chunked[fwp.hr_slice][..., t_hr_slice, :] = out

        err = (data_chunked - data_nochunk)
        err /= data_nochunk
        if plot:
            for ifeature in range(data_nochunk.shape[-1]):
                fig = plt.figure(figsize=(15, 5))
                ax1 = fig.add_subplot(131)
                ax2 = fig.add_subplot(132)
                ax3 = fig.add_subplot(133)
                vmin = np.min(data_nochunk)
                vmax = np.max(data_nochunk)
                nc = ax1.imshow(data_nochunk[..., 0, ifeature], vmin=vmin,
                                vmax=vmax)
                ch = ax2.imshow(data_chunked[..., 0, ifeature], vmin=vmin,
                                vmax=vmax)
                diff = ax3.imshow(err[..., 0, ifeature])
                ax1.set_title('Non chunked output')
                ax2.set_title('Chunked output')
                ax3.set_title('Difference')
                fig.colorbar(nc, ax=ax1, shrink=0.6,
                             label=f'{model.output_features[ifeature]}')
                fig.colorbar(ch, ax=ax2, shrink=0.6,
                             label=f'{model.output_features[ifeature]}')
                fig.colorbar(diff, ax=ax3, shrink=0.6, label='Difference')
                plt.savefig(f'./chunk_vs_nochunk_{ifeature}.png')
                plt.close()

        assert np.mean(np.abs(err.flatten())) < 0.01


def test_fwp_nochunking():
    """Test forward pass without chunking. Make sure using a single chunk
    (a.k.a nochunking) matches direct forward pass of full dataset.
    """

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    model.meta['training_features'] = FEATURES
    model.meta['output_features'] = FEATURES[:-1]
    model.meta['s_enhance'] = s_enhance
    model.meta['t_enhance'] = t_enhance
    _ = model.generate(np.ones((4, 10, 10, 12, 3)))

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        cache_pattern = os.path.join(td, 'cache')
        input_handler_kwargs = dict(
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            worker_kwargs=dict(max_workers=1),
            cache_pattern=cache_pattern,
            overwrite_cache=True)
        handler = ForwardPassStrategy(
            input_files, model_kwargs={'model_dir': out_dir},
            fwp_chunk_shape=(shape[0], shape[1], list_chunk_size),
            spatial_pad=0, temporal_pad=0,
            input_handler_kwargs=input_handler_kwargs,
            worker_kwargs=dict(max_workers=1))
        forward_pass = ForwardPass(handler)
        data_chunked = forward_pass.run_chunk()

        handlerNC = DataHandlerNC(input_files, FEATURES,
                                  target=target, shape=shape,
                                  temporal_slice=temporal_slice,
                                  cache_pattern=None,
                                  time_chunk_size=100,
                                  overwrite_cache=True,
                                  val_split=0.0,
                                  worker_kwargs=dict(max_workers=1))

        data_nochunk = model.generate(
            np.expand_dims(handlerNC.data, axis=0))[0]

        assert np.array_equal(data_chunked, data_nochunk)


def test_fwp_multi_step_model_topo_exoskip(log=False):
    """Test the forward pass with a multi step model class using exogenous data
    for the first two steps and not the last"""

    if log:
        init_logger('sup3r', log_level='DEBUG')

    Sup3rGan.seed()
    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    s1_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    s1_model.meta['training_features'] = ['U_100m', 'V_100m', 'topography']
    s1_model.meta['output_features'] = ['U_100m', 'V_100m']
    s1_model.meta['s_enhance'] = 2
    s1_model.meta['t_enhance'] = 1
    _ = s1_model.generate(np.ones((4, 10, 10, 3)))

    s2_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    s2_model.meta['training_features'] = ['U_100m', 'V_100m', 'topography']
    s2_model.meta['output_features'] = ['U_100m', 'V_100m']
    s2_model.meta['s_enhance'] = 2
    s2_model.meta['t_enhance'] = 1
    _ = s2_model.generate(np.ones((4, 10, 10, 3)))

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    st_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    st_model.meta['training_features'] = ['U_100m', 'V_100m', 'topography']
    st_model.meta['output_features'] = ['U_100m', 'V_100m']
    st_model.meta['s_enhance'] = 3
    st_model.meta['t_enhance'] = 4
    _ = st_model.generate(np.ones((4, 10, 10, 6, 2)))

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)

        st_out_dir = os.path.join(td, 'st_gan')
        s1_out_dir = os.path.join(td, 's1_gan')
        s2_out_dir = os.path.join(td, 's2_gan')
        st_model.save(st_out_dir)
        s1_model.save(s1_out_dir)
        s2_model.save(s2_out_dir)

        max_workers = 1
        fwp_chunk_shape = (4, 4, 8)
        s_enhance = 12
        t_enhance = 4

        exo_kwargs = {'file_paths': input_files,
                      'features': ['topography'],
                      'source_file': FP_WTK,
                      'target': target,
                      'shape': shape,
                      's_enhancements': [1, 2, 2],
                      'agg_factors': [2, 4, 16],
                      'exo_steps': [0, 1]
                      }

        model_kwargs = {'spatial_model_dirs': [s1_out_dir, s2_out_dir],
                        'temporal_model_dirs': st_out_dir}

        out_files = os.path.join(td, 'out_{file_id}.h5')
        input_handler_kwargs = dict(
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            worker_kwargs=dict(max_workers=max_workers),
            overwrite_cache=True)
        handler = ForwardPassStrategy(
            input_files, model_kwargs=model_kwargs,
            model_class='SpatialThenTemporalGan',
            fwp_chunk_shape=fwp_chunk_shape,
            input_handler_kwargs=input_handler_kwargs,
            spatial_pad=0, temporal_pad=0,
            out_pattern=out_files,
            worker_kwargs=dict(max_workers=max_workers),
            exo_kwargs=exo_kwargs,
            max_nodes=1)

        forward_pass = ForwardPass(handler)

        assert forward_pass.output_workers == max_workers
        assert forward_pass.pass_workers == max_workers
        assert forward_pass.max_workers == max_workers
        assert forward_pass.data_handler.max_workers == max_workers
        assert forward_pass.data_handler.compute_workers == max_workers
        assert forward_pass.data_handler.load_workers == max_workers
        assert forward_pass.data_handler.norm_workers == max_workers
        assert forward_pass.data_handler.extract_workers == max_workers

        forward_pass.run(handler, node_index=0)

        with ResourceX(handler.out_files[0]) as fh:
            assert fh.shape == (
                t_enhance * len(input_files),
                s_enhance**2 * fwp_chunk_shape[0] * fwp_chunk_shape[1])
            assert all(f in fh.attrs for f in ('windspeed_100m',
                                               'winddirection_100m'))

            assert fh.global_attrs['package'] == 'sup3r'
            assert fh.global_attrs['version'] == __version__
            assert 'full_version_record' in fh.global_attrs
            version_record = json.loads(fh.global_attrs['full_version_record'])
            assert version_record['tensorflow'] == tf.__version__
            assert 'gan_meta' in fh.global_attrs
            gan_meta = json.loads(fh.global_attrs['gan_meta'])
            assert len(gan_meta) == 3  # three step model
            assert gan_meta[0]['training_features'] == ['U_100m', 'V_100m',
                                                        'topography']


def test_fwp_multi_step_spatial_model_topo_noskip():
    """Test the forward pass with a multi step spatial only model class using
    exogenous data for all model steps"""
    Sup3rGan.seed()
    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    s1_model = WindGan(fp_gen, fp_disc, learning_rate=1e-4)
    s1_model.meta['training_features'] = ['U_100m', 'V_100m', 'topography']
    s1_model.meta['output_features'] = ['U_100m', 'V_100m']
    s1_model.meta['s_enhance'] = 2
    s1_model.meta['t_enhance'] = 1
    _ = s1_model.generate(np.ones((4, 10, 10, 3)))

    s2_model = WindGan(fp_gen, fp_disc, learning_rate=1e-4)
    s2_model.meta['training_features'] = ['U_100m', 'V_100m', 'topography']
    s2_model.meta['output_features'] = ['U_100m', 'V_100m']
    s2_model.meta['s_enhance'] = 2
    s2_model.meta['t_enhance'] = 1
    _ = s2_model.generate(np.ones((4, 10, 10, 3)))

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)

        s1_out_dir = os.path.join(td, 's1_gan')
        s2_out_dir = os.path.join(td, 's2_gan')
        s1_model.save(s1_out_dir)
        s2_model.save(s2_out_dir)

        max_workers = 1
        fwp_chunk_shape = (4, 4, 8)
        s_enhancements = [2, 2, 1]
        s_enhance = np.product(s_enhancements)

        exo_kwargs = {'file_paths': input_files,
                      'features': ['topography'],
                      'source_file': FP_WTK,
                      'target': target,
                      'shape': shape,
                      's_enhancements': [1, 2, 2],
                      'agg_factors': [12, 4, 2]
                      }

        model_kwargs = {'model_dirs': [s1_out_dir, s2_out_dir]}

        out_files = os.path.join(td, 'out_{file_id}.h5')
        input_handler_kwargs = dict(
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            worker_kwargs=dict(max_workers=max_workers),
            overwrite_cache=True)
        handler = ForwardPassStrategy(
            input_files, model_kwargs=model_kwargs,
            model_class='MultiStepGan',
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=1, temporal_pad=1,
            input_handler_kwargs=input_handler_kwargs,
            out_pattern=out_files,
            worker_kwargs=dict(max_workers=max_workers),
            exo_kwargs=exo_kwargs,
            max_nodes=1)

        forward_pass = ForwardPass(handler)
        forward_pass.run(handler, node_index=0)

        with ResourceX(handler.out_files[0]) as fh:
            assert fh.shape == (
                len(input_files),
                s_enhance**2 * fwp_chunk_shape[0] * fwp_chunk_shape[1])
            assert all(f in fh.attrs for f in ('windspeed_100m',
                                               'winddirection_100m'))

            assert fh.global_attrs['package'] == 'sup3r'
            assert fh.global_attrs['version'] == __version__
            assert 'full_version_record' in fh.global_attrs
            version_record = json.loads(fh.global_attrs['full_version_record'])
            assert version_record['tensorflow'] == tf.__version__
            assert 'gan_meta' in fh.global_attrs
            gan_meta = json.loads(fh.global_attrs['gan_meta'])
            assert len(gan_meta) == 2  # two step model
            assert gan_meta[0]['training_features'] == ['U_100m', 'V_100m',
                                                        'topography']


def test_fwp_multi_step_model_topo_noskip():
    """Test the forward pass with a multi step model class using exogenous data
    for all model steps"""
    Sup3rGan.seed()
    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    s1_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    s1_model.meta['training_features'] = ['U_100m', 'V_100m', 'topography']
    s1_model.meta['output_features'] = ['U_100m', 'V_100m']
    s1_model.meta['s_enhance'] = 2
    s1_model.meta['t_enhance'] = 1
    _ = s1_model.generate(np.ones((4, 10, 10, 3)))

    s2_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    s2_model.meta['training_features'] = ['U_100m', 'V_100m', 'topography']
    s2_model.meta['output_features'] = ['U_100m', 'V_100m']
    s2_model.meta['s_enhance'] = 2
    s2_model.meta['t_enhance'] = 1
    _ = s2_model.generate(np.ones((4, 10, 10, 3)))

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    st_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    st_model.meta['training_features'] = ['U_100m', 'V_100m', 'topography']
    st_model.meta['output_features'] = ['U_100m', 'V_100m']
    st_model.meta['s_enhance'] = 3
    st_model.meta['t_enhance'] = 4
    _ = st_model.generate(np.ones((4, 10, 10, 6, 3)))

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)

        st_out_dir = os.path.join(td, 'st_gan')
        s1_out_dir = os.path.join(td, 's1_gan')
        s2_out_dir = os.path.join(td, 's2_gan')
        st_model.save(st_out_dir)
        s1_model.save(s1_out_dir)
        s2_model.save(s2_out_dir)

        max_workers = 1
        fwp_chunk_shape = (4, 4, 8)
        s_enhancements = [2, 2, 3]
        s_enhance = np.product(s_enhancements)
        t_enhance = 4

        exo_kwargs = {'file_paths': input_files,
                      'features': ['topography'],
                      'source_file': FP_WTK,
                      'target': target,
                      'shape': shape,
                      's_enhancements': [1, 2, 2],
                      'agg_factors': [2, 4, 12]
                      }

        model_kwargs = {'spatial_model_dirs': [s1_out_dir, s2_out_dir],
                        'temporal_model_dirs': st_out_dir}

        out_files = os.path.join(td, 'out_{file_id}.h5')
        input_handler_kwargs = dict(
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            worker_kwargs=dict(max_workers=max_workers),
            overwrite_cache=True)
        handler = ForwardPassStrategy(
            input_files, model_kwargs=model_kwargs,
            model_class='SpatialThenTemporalGan',
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=1, temporal_pad=1,
            input_handler_kwargs=input_handler_kwargs,
            out_pattern=out_files,
            worker_kwargs=dict(max_workers=max_workers),
            exo_kwargs=exo_kwargs,
            max_nodes=1)

        forward_pass = ForwardPass(handler)

        assert forward_pass.output_workers == max_workers
        assert forward_pass.data_handler.compute_workers == max_workers
        assert forward_pass.data_handler.load_workers == max_workers
        assert forward_pass.data_handler.norm_workers == max_workers
        assert forward_pass.data_handler.extract_workers == max_workers

        forward_pass.run(handler, node_index=0)

        with ResourceX(handler.out_files[0]) as fh:
            assert fh.shape == (
                t_enhance * len(input_files),
                s_enhance**2 * fwp_chunk_shape[0] * fwp_chunk_shape[1])
            assert all(f in fh.attrs for f in ('windspeed_100m',
                                               'winddirection_100m'))

            assert fh.global_attrs['package'] == 'sup3r'
            assert fh.global_attrs['version'] == __version__
            assert 'full_version_record' in fh.global_attrs
            version_record = json.loads(fh.global_attrs['full_version_record'])
            assert version_record['tensorflow'] == tf.__version__
            assert 'gan_meta' in fh.global_attrs
            gan_meta = json.loads(fh.global_attrs['gan_meta'])
            assert len(gan_meta) == 3  # three step model
            assert gan_meta[0]['training_features'] == ['U_100m', 'V_100m',
                                                        'topography']


def test_fwp_multi_step_model():
    """Test the forward pass with a multi step model class"""
    Sup3rGan.seed()
    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    s_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    s_model.meta['training_features'] = ['U_100m', 'V_100m']
    s_model.meta['output_features'] = ['U_100m', 'V_100m']
    s_model.meta['s_enhance'] = 2
    s_model.meta['t_enhance'] = 1
    _ = s_model.generate(np.ones((4, 10, 10, 2)))

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    st_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    st_model.meta['training_features'] = ['U_100m', 'V_100m']
    st_model.meta['output_features'] = ['U_100m', 'V_100m']
    st_model.meta['s_enhance'] = 3
    st_model.meta['t_enhance'] = 4
    _ = st_model.generate(np.ones((4, 10, 10, 6, 2)))

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)

        st_out_dir = os.path.join(td, 'st_gan')
        s_out_dir = os.path.join(td, 's_gan')
        st_model.save(st_out_dir)
        s_model.save(s_out_dir)

        out_files = os.path.join(td, 'out_{file_id}.h5')

        max_workers = 1
        fwp_chunk_shape = (4, 4, 8)
        s_enhance = 6
        t_enhance = 4

        model_kwargs = {'spatial_model_dirs': s_out_dir,
                        'temporal_model_dirs': st_out_dir}

        input_handler_kwargs = dict(
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            worker_kwargs=dict(max_workers=max_workers),
            overwrite_cache=True)
        handler = ForwardPassStrategy(
            input_files, model_kwargs=model_kwargs,
            model_class='SpatialThenTemporalGan',
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=0, temporal_pad=0,
            input_handler_kwargs=input_handler_kwargs,
            out_pattern=out_files,
            worker_kwargs=dict(max_workers=max_workers),
            max_nodes=1)

        forward_pass = ForwardPass(handler)
        ones = np.ones((fwp_chunk_shape[2], fwp_chunk_shape[0],
                        fwp_chunk_shape[1], 2))
        out = forward_pass.model.generate(ones)
        assert out.shape == (1, 24, 24, 32, 2)

        assert forward_pass.output_workers == max_workers
        assert forward_pass.data_handler.compute_workers == max_workers
        assert forward_pass.data_handler.load_workers == max_workers
        assert forward_pass.data_handler.norm_workers == max_workers
        assert forward_pass.data_handler.extract_workers == max_workers

        forward_pass.run(handler, node_index=0)

        with ResourceX(handler.out_files[0]) as fh:
            assert fh.shape == (
                t_enhance * len(input_files),
                s_enhance**2 * fwp_chunk_shape[0] * fwp_chunk_shape[1])
            assert all(f in fh.attrs for f in ('windspeed_100m',
                                               'winddirection_100m'))

            assert fh.global_attrs['package'] == 'sup3r'
            assert fh.global_attrs['version'] == __version__
            assert 'full_version_record' in fh.global_attrs
            version_record = json.loads(fh.global_attrs['full_version_record'])
            assert version_record['tensorflow'] == tf.__version__
            assert 'gan_meta' in fh.global_attrs
            gan_meta = json.loads(fh.global_attrs['gan_meta'])
            assert len(gan_meta) == 2  # two step model
            assert gan_meta[0]['training_features'] == ['U_100m', 'V_100m']


def test_slicing_no_pad(log=False):
    """Test the slicing of input data via the ForwardPassStrategy +
    ForwardPassSlicer vs. the actual source data. Does not include any
    reflected padding at the edges."""

    if log:
        init_logger('sup3r', log_level='DEBUG')

    Sup3rGan.seed()
    s_enhance = 3
    t_enhance = 4
    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    st_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    features = ['U_100m', 'V_100m']
    st_model.meta['training_features'] = features
    st_model.meta['output_features'] = features
    st_model.meta['s_enhance'] = s_enhance
    st_model.meta['t_enhance'] = t_enhance
    _ = st_model.generate(np.ones((4, 10, 10, 6, 2)))

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        out_files = os.path.join(td, 'out_{file_id}.h5')
        st_out_dir = os.path.join(td, 'st_gan')
        st_model.save(st_out_dir)

        handler = DataHandlerNC(input_files, features,
                                target=target, shape=shape,
                                sample_shape=(1, 1, 1),
                                val_split=0.0,
                                worker_kwargs=dict(max_workers=1))

        input_handler_kwargs = dict(
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            worker_kwargs=dict(max_workers=1),
            overwrite_cache=True)
        strategy = ForwardPassStrategy(
            input_files, model_kwargs={'model_dir': st_out_dir},
            model_class='Sup3rGan',
            fwp_chunk_shape=(3, 2, 4),
            spatial_pad=0, temporal_pad=0,
            input_handler_kwargs=input_handler_kwargs,
            out_pattern=out_files,
            worker_kwargs=dict(max_workers=1),
            max_nodes=1)

        for ichunk in range(strategy.chunks):
            forward_pass = ForwardPass(strategy, chunk_index=ichunk)
            s_slices = strategy.lr_pad_slices[forward_pass.spatial_chunk_index]
            lr_data_slice = (s_slices[0], s_slices[1],
                             forward_pass.ti_pad_slice,
                             slice(None))

            truth = handler.data[lr_data_slice]
            assert np.allclose(forward_pass.input_data, truth)


def test_slicing_pad(log=False):
    """Test the slicing of input data via the ForwardPassStrategy +
    ForwardPassSlicer vs. the actual source data. Includes reflected padding
    at the edges."""

    if log:
        init_logger('sup3r', log_level='DEBUG')

    Sup3rGan.seed()
    s_enhance = 3
    t_enhance = 4
    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    st_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    features = ['U_100m', 'V_100m']
    st_model.meta['training_features'] = features
    st_model.meta['output_features'] = features
    st_model.meta['s_enhance'] = s_enhance
    st_model.meta['t_enhance'] = t_enhance
    _ = st_model.generate(np.ones((4, 10, 10, 6, 2)))

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        out_files = os.path.join(td, 'out_{file_id}.h5')
        st_out_dir = os.path.join(td, 'st_gan')
        st_model.save(st_out_dir)

        handler = DataHandlerNC(input_files, features,
                                target=target, shape=shape,
                                sample_shape=(1, 1, 1),
                                val_split=0.0,
                                worker_kwargs=dict(max_workers=1))

        input_handler_kwargs = dict(
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            worker_kwargs=dict(max_workers=1),
            overwrite_cache=True)
        strategy = ForwardPassStrategy(
            input_files, model_kwargs={'model_dir': st_out_dir},
            model_class='Sup3rGan',
            fwp_chunk_shape=(2, 1, 4),
            input_handler_kwargs=input_handler_kwargs,
            spatial_pad=2, temporal_pad=2,
            out_pattern=out_files,
            worker_kwargs=dict(max_workers=1),
            max_nodes=1)

        chunk_lookup = strategy.fwp_slicer.chunk_lookup
        n_s1 = len(strategy.fwp_slicer.s1_lr_slices)
        n_s2 = len(strategy.fwp_slicer.s2_lr_slices)
        n_t = len(strategy.fwp_slicer.t_lr_slices)

        assert chunk_lookup[0, 0, 0] == 0
        assert chunk_lookup[0, 1, 0] == 1
        assert chunk_lookup[0, 2, 0] == 2
        assert chunk_lookup[1, 0, 0] == n_s2
        assert chunk_lookup[1, 1, 0] == n_s2 + 1
        assert chunk_lookup[2, 0, 0] == n_s2 * 2
        assert chunk_lookup[0, 0, 1] == n_s1 * n_s2
        assert chunk_lookup[0, 1, 1] == n_s1 * n_s2 + 1

        for ichunk in range(strategy.chunks):
            forward_pass = ForwardPass(strategy, chunk_index=ichunk)

            s_slices = strategy.lr_pad_slices[forward_pass.spatial_chunk_index]
            lr_data_slice = (s_slices[0], s_slices[1],
                             forward_pass.ti_pad_slice,
                             slice(None))

            # do a manual calculation of what the padding should be.
            # s1 and t axes should have padding of 2 and the borders and
            # padding of 1 when 1 index away from the borders (chunk shape is 1
            # in those axes). s2 should have padding of 2 at the
            # borders and 0 everywhere else.
            ids1, ids2, idt = np.where(chunk_lookup == ichunk)
            ids1, ids2, idt = ids1[0], ids2[0], idt[0]

            start_s1_pad_lookup = {0: 2}
            start_s2_pad_lookup = {0: 2, 1: 1}
            start_t_pad_lookup = {0: 2}
            end_s1_pad_lookup = {n_s1 - 1: 2}
            end_s2_pad_lookup = {n_s2 - 1: 2, n_s2 - 2: 1}
            end_t_pad_lookup = {n_t - 1: 2}

            pad_s1_start = start_s1_pad_lookup.get(ids1, 0)
            pad_s2_start = start_s2_pad_lookup.get(ids2, 0)
            pad_t_start = start_t_pad_lookup.get(idt, 0)
            pad_s1_end = end_s1_pad_lookup.get(ids1, 0)
            pad_s2_end = end_s2_pad_lookup.get(ids2, 0)
            pad_t_end = end_t_pad_lookup.get(idt, 0)

            pad_width = ((pad_s1_start, pad_s1_end),
                         (pad_s2_start, pad_s2_end),
                         (pad_t_start, pad_t_end), (0, 0))

            truth = handler.data[lr_data_slice]
            padded_truth = np.pad(truth, pad_width, mode='reflect')

            assert forward_pass.input_data.shape == padded_truth.shape
            assert np.allclose(forward_pass.input_data, padded_truth)


def test_fwp_single_step_wind_hi_res_topo(plot=False):
    """Test the forward pass with a single spatiotemporal WindGan model
    requiring high-resolution topography input from the exogenous_data
    feature."""
    Sup3rGan.seed()
    gen_model = [{"class": "FlexiblePadding",
                  "paddings": [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]],
                  "mode": "REFLECT"},
                 {"class": "Conv3D", "filters": 64, "kernel_size": 3,
                  "strides": 1},
                 {"class": "Cropping3D", "cropping": 2},
                 {"class": "SpatioTemporalExpansion", "temporal_mult": 2,
                  "temporal_method": "nearest"},

                 {"class": "FlexiblePadding",
                  "paddings": [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]],
                  "mode": "REFLECT"},
                 {"class": "Conv3D", "filters": 64,
                  "kernel_size": 3, "strides": 1},
                 {"class": "Cropping3D", "cropping": 2},
                 {"class": "SpatioTemporalExpansion", "spatial_mult": 2},

                 {"class": "FlexiblePadding",
                  "paddings": [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]],
                  "mode": "REFLECT"},
                 {"class": "Conv3D", "filters": 64,
                  "kernel_size": 3, "strides": 1},
                 {"class": "Cropping3D", "cropping": 2},
                 {"alpha": 0.2, "class": "LeakyReLU"},

                 {"class": "Sup3rConcat"},

                 {"class": "FlexiblePadding",
                  "paddings": [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]],
                  "mode": "REFLECT"},
                 {"class": "Conv3D", "filters": 2,
                  "kernel_size": 3, "strides": 1},
                 {"class": "Cropping3D", "cropping": 2}]

    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    model = WindGan(gen_model, fp_disc, learning_rate=1e-4)
    model.meta['training_features'] = ['U_100m', 'V_100m', 'topography']
    model.meta['output_features'] = ['U_100m', 'V_100m']
    model.meta['s_enhance'] = 2
    model.meta['t_enhance'] = 2
    _ = model.generate(np.random.rand(4, 10, 10, 6, 3),
                       exogenous_data=(None, np.random.rand(4, 20, 20, 6, 1)))

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)

        st_out_dir = os.path.join(td, 'st_gan')
        model.save(st_out_dir)

        exo_kwargs = {'file_paths': input_files,
                      'features': ['topography'],
                      'source_file': FP_WTK,
                      'target': target,
                      'shape': shape,
                      's_enhancements': [1, 2],
                      'agg_factors': [2, 4],
                      }

        model_kwargs = {'model_dir': st_out_dir}
        out_files = os.path.join(td, 'out_{file_id}.h5')
        input_handler_kwargs = dict(
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            worker_kwargs=dict(max_workers=1),
            overwrite_cache=True)

        # should get an error on a bad tensorflow concatenation
        with pytest.raises(RuntimeError):
            exo_kwargs['s_enhancements'] = [1, 1]
            handler = ForwardPassStrategy(
                input_files, model_kwargs=model_kwargs,
                model_class='WindGan',
                fwp_chunk_shape=(4, 4, 8),
                spatial_pad=1, temporal_pad=1,
                input_handler_kwargs=input_handler_kwargs,
                out_pattern=out_files,
                worker_kwargs=dict(max_workers=1),
                exo_kwargs=exo_kwargs,
                max_nodes=1)
            forward_pass = ForwardPass(handler)
            forward_pass.run(handler, node_index=0)

        exo_kwargs['s_enhancements'] = [1, 2]
        handler = ForwardPassStrategy(
            input_files, model_kwargs=model_kwargs,
            model_class='WindGan',
            fwp_chunk_shape=(8, 8, 8),
            spatial_pad=4, temporal_pad=4,
            input_handler_kwargs=input_handler_kwargs,
            out_pattern=out_files,
            worker_kwargs=dict(max_workers=1),
            exo_kwargs=exo_kwargs,
            max_nodes=1)
        forward_pass = ForwardPass(handler)

        if plot:
            for ifeature, feature in enumerate(forward_pass.output_features):
                fig = plt.figure(figsize=(15, 5))
                ax1 = fig.add_subplot(111)
                vmin = np.min(forward_pass.input_data[..., ifeature])
                vmax = np.max(forward_pass.input_data[..., ifeature])
                nc = ax1.imshow(forward_pass.input_data[..., 0, ifeature],
                                vmin=vmin, vmax=vmax)
                fig.colorbar(nc, ax=ax1, shrink=0.6, label=f'{feature}')
                plt.savefig(f'./input_{feature}.png')
                plt.close()

        forward_pass.run(handler, node_index=0)

        for fp in handler.out_files:
            assert os.path.exists(fp)


def test_fwp_multi_step_wind_hi_res_topo():
    """Test the forward pass with multiple WindGan models requiring
    high-resolution topograph input from the exogenous_data feature."""
    Sup3rGan.seed()
    gen_model = [{"class": "FlexiblePadding",
                  "paddings": [[0, 0], [3, 3], [3, 3], [0, 0]],
                  "mode": "REFLECT"},
                 {"class": "Conv2DTranspose", "filters": 64, "kernel_size": 3,
                  "strides": 1, "activation": "relu"},
                 {"class": "Cropping2D", "cropping": 4},

                 {"class": "FlexiblePadding",
                  "paddings": [[0, 0], [3, 3], [3, 3], [0, 0]],
                  "mode": "REFLECT"},
                 {"class": "Conv2DTranspose", "filters": 64,
                  "kernel_size": 3, "strides": 1, "activation": "relu"},
                 {"class": "Cropping2D", "cropping": 4},

                 {"class": "FlexiblePadding",
                  "paddings": [[0, 0], [3, 3], [3, 3], [0, 0]],
                  "mode": "REFLECT"},
                 {"class": "Conv2DTranspose", "filters": 64,
                  "kernel_size": 3, "strides": 1, "activation": "relu"},
                 {"class": "Cropping2D", "cropping": 4},
                 {"class": "SpatialExpansion", "spatial_mult": 2},
                 {"class": "Activation", "activation": "relu"},

                 {"class": "Sup3rConcat"},

                 {"class": "FlexiblePadding",
                  "paddings": [[0, 0], [3, 3], [3, 3], [0, 0]],
                  "mode": "REFLECT"},
                 {"class": "Conv2DTranspose", "filters": 2,
                  "kernel_size": 3, "strides": 1, "activation": "relu"},
                 {"class": "Cropping2D", "cropping": 4}]

    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    s1_model = WindGan(gen_model, fp_disc, learning_rate=1e-4)
    s1_model.meta['training_features'] = ['U_100m', 'V_100m', 'topography']
    s1_model.meta['output_features'] = ['U_100m', 'V_100m']
    s1_model.meta['s_enhance'] = 2
    s1_model.meta['t_enhance'] = 1
    _ = s1_model.generate(np.ones((4, 10, 10, 3)),
                          exogenous_data=(None, np.ones((4, 20, 20, 1))))

    s2_model = WindGan(gen_model, fp_disc, learning_rate=1e-4)
    s2_model.meta['training_features'] = ['U_100m', 'V_100m', 'topography']
    s2_model.meta['output_features'] = ['U_100m', 'V_100m']
    s2_model.meta['s_enhance'] = 2
    s2_model.meta['t_enhance'] = 1
    _ = s2_model.generate(np.ones((4, 10, 10, 3)),
                          exogenous_data=(None, np.ones((4, 20, 20, 1))))

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    st_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    st_model.meta['training_features'] = ['U_100m', 'V_100m', 'topography']
    st_model.meta['output_features'] = ['U_100m', 'V_100m']
    st_model.meta['s_enhance'] = 3
    st_model.meta['t_enhance'] = 4
    _ = st_model.generate(np.ones((4, 10, 10, 6, 3)))

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)

        st_out_dir = os.path.join(td, 'st_gan')
        s1_out_dir = os.path.join(td, 's1_gan')
        s2_out_dir = os.path.join(td, 's2_gan')
        st_model.save(st_out_dir)
        s1_model.save(s1_out_dir)
        s2_model.save(s2_out_dir)

        exo_kwargs = {'file_paths': input_files,
                      'features': ['topography'],
                      'source_file': FP_WTK,
                      'target': target,
                      'shape': shape,
                      's_enhancements': [1, 2, 2],
                      'agg_factors': [2, 4, 12],
                      }

        model_kwargs = {'spatial_model_dirs': [s1_out_dir, s2_out_dir],
                        'temporal_model_dirs': st_out_dir}
        out_files = os.path.join(td, 'out_{file_id}.h5')
        input_handler_kwargs = dict(
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            worker_kwargs=dict(max_workers=1),
            overwrite_cache=True)

        # should get an error on a bad tensorflow concatenation
        with pytest.raises(RuntimeError):
            exo_kwargs['s_enhancements'] = [1, 1, 1]
            handler = ForwardPassStrategy(
                input_files, model_kwargs=model_kwargs,
                model_class='SpatialThenTemporalGan',
                fwp_chunk_shape=(4, 4, 8),
                spatial_pad=1, temporal_pad=1,
                input_handler_kwargs=input_handler_kwargs,
                out_pattern=out_files,
                worker_kwargs=dict(max_workers=1),
                exo_kwargs=exo_kwargs,
                max_nodes=1)
            forward_pass = ForwardPass(handler)
            forward_pass.run(handler, node_index=0)

        exo_kwargs['s_enhancements'] = [1, 2, 2]
        handler = ForwardPassStrategy(
            input_files, model_kwargs=model_kwargs,
            model_class='SpatialThenTemporalGan',
            fwp_chunk_shape=(4, 4, 8),
            spatial_pad=1, temporal_pad=1,
            input_handler_kwargs=input_handler_kwargs,
            out_pattern=out_files,
            worker_kwargs=dict(max_workers=1),
            exo_kwargs=exo_kwargs,
            max_nodes=1)
        forward_pass = ForwardPass(handler)
        forward_pass.run(handler, node_index=0)

        for fp in handler.out_files:
            assert os.path.exists(fp)


def test_fwp_wind_hi_res_topo_plus_linear():
    """Test the forward pass with a WindGan model requiring high-res topo input
    from exo data for spatial enhancement and a linear interpolation model for
    temporal enhancement."""

    Sup3rGan.seed()
    gen_model = [{"class": "FlexiblePadding",
                  "paddings": [[0, 0], [3, 3], [3, 3], [0, 0]],
                  "mode": "REFLECT"},
                 {"class": "Conv2DTranspose", "filters": 64, "kernel_size": 3,
                  "strides": 1},
                 {"class": "Cropping2D", "cropping": 4},

                 {"class": "FlexiblePadding",
                  "paddings": [[0, 0], [3, 3], [3, 3], [0, 0]],
                  "mode": "REFLECT"},
                 {"class": "Conv2DTranspose", "filters": 64,
                  "kernel_size": 3, "strides": 1},
                 {"class": "Cropping2D", "cropping": 4},

                 {"class": "FlexiblePadding",
                  "paddings": [[0, 0], [3, 3], [3, 3], [0, 0]],
                  "mode": "REFLECT"},
                 {"class": "Conv2DTranspose", "filters": 64,
                  "kernel_size": 3, "strides": 1},
                 {"class": "Cropping2D", "cropping": 4},
                 {"class": "SpatialExpansion", "spatial_mult": 2},
                 {"alpha": 0.2, "class": "LeakyReLU"},

                 {"class": "Sup3rConcat"},

                 {"class": "FlexiblePadding",
                  "paddings": [[0, 0], [3, 3], [3, 3], [0, 0]],
                  "mode": "REFLECT"},
                 {"class": "Conv2DTranspose", "filters": 2,
                  "kernel_size": 3, "strides": 1},
                 {"class": "Cropping2D", "cropping": 4}]

    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    s_model = WindGan(gen_model, fp_disc, learning_rate=1e-4)
    s_model.meta['training_features'] = ['U_100m', 'V_100m', 'topography']
    s_model.meta['output_features'] = ['U_100m', 'V_100m']
    s_model.meta['s_enhance'] = 2
    s_model.meta['t_enhance'] = 1
    _ = s_model.generate(np.ones((4, 10, 10, 3)),
                         exogenous_data=(None, np.ones((4, 20, 20, 1))))

    t_model = LinearInterp(features=['U_100m', 'V_100m'], s_enhance=1,
                           t_enhance=4)

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)

        s_out_dir = os.path.join(td, 's_gan')
        t_out_dir = os.path.join(td, 't_interp')
        s_model.save(s_out_dir)
        t_model.save(t_out_dir)

        exo_kwargs = {'file_paths': input_files,
                      'features': ['topography'],
                      'source_file': FP_WTK,
                      'target': target,
                      'shape': shape,
                      's_enhancements': [1, 2],
                      'agg_factors': [2, 4],
                      }

        model_kwargs = {'spatial_model_dirs': s_out_dir,
                        'temporal_model_dirs': t_out_dir}
        out_files = os.path.join(td, 'out_{file_id}.h5')
        input_handler_kwargs = dict(
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            worker_kwargs=dict(max_workers=1),
            overwrite_cache=True)

        exo_kwargs['s_enhancements'] = [1, 2]
        handler = ForwardPassStrategy(
            input_files, model_kwargs=model_kwargs,
            model_class='SpatialThenTemporalGan',
            fwp_chunk_shape=(4, 4, 8),
            spatial_pad=1, temporal_pad=1,
            input_handler_kwargs=input_handler_kwargs,
            out_pattern=out_files,
            worker_kwargs=dict(max_workers=1),
            exo_kwargs=exo_kwargs,
            max_nodes=1)
        forward_pass = ForwardPass(handler)
        forward_pass.run(handler, node_index=0)

        for fp in handler.out_files:
            assert os.path.exists(fp)