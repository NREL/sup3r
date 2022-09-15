# -*- coding: utf-8 -*-
"""pytests for data handling"""
import json
import os
import tempfile
import tensorflow as tf
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from sup3r import TEST_DATA_DIR, CONFIG_DIR, __version__
from sup3r.preprocessing.data_handling import DataHandlerH5, DataHandlerNC
from sup3r.preprocessing.batch_handling import BatchHandler
from sup3r.pipeline.forward_pass import (ForwardPass, ForwardPassStrategy)
from sup3r.models import Sup3rGan
from sup3r.utilities.test_utils import make_fake_nc_files

from rex import ResourceX
from rex import init_logger


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


def test_fwp_nc_cc():
    """Test forward pass handler output for netcdf write with cc data."""

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
        handler = ForwardPassStrategy(
            input_files, model_kwargs={'model_dir': out_dir},
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=1, temporal_pad=1,
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            cache_pattern=cache_pattern,
            overwrite_cache=True, out_pattern=out_files,
            max_workers=max_workers, input_handler='DataHandlerNCforCC')
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
        handler = ForwardPassStrategy(
            input_files, model_kwargs={'model_dir': out_dir},
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=1, temporal_pad=1,
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            cache_pattern=cache_pattern,
            overwrite_cache=True, out_pattern=out_files,
            max_workers=max_workers)
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
        handler = ForwardPassStrategy(
            input_files, model_kwargs={'model_dir': out_dir},
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=1, temporal_pad=1,
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            cache_pattern=cache_pattern,
            overwrite_cache=True, out_pattern=out_files,
            max_workers=max_workers)
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

    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            shape=(20, 20),
                            sample_shape=(18, 18, 24),
                            temporal_slice=slice(None, None, 1),
                            val_split=0.005,
                            max_workers=1)

    batch_handler = BatchHandler([handler], batch_size=4,
                                 s_enhance=s_enhance,
                                 t_enhance=t_enhance,
                                 n_batches=4)

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        model.train(batch_handler, n_epoch=1,
                    weight_gen_advers=0.0,
                    train_gen=True, train_disc=False,
                    checkpoint_int=2,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        max_workers = 1
        cache_pattern = os.path.join(td, 'cache')
        handler = ForwardPassStrategy(
            input_files, model_kwargs={'model_dir': out_dir},
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=1, temporal_pad=1,
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            cache_pattern=cache_pattern,
            overwrite_cache=True,
            max_workers=max_workers)
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

    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            shape=(20, 20),
                            sample_shape=(18, 18, 24),
                            temporal_slice=slice(None, None, 1),
                            val_split=0.005,
                            max_workers=1)

    batch_handler = BatchHandler([handler], batch_size=4,
                                 s_enhance=s_enhance,
                                 t_enhance=t_enhance,
                                 n_batches=4)

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        model.train(batch_handler, n_epoch=2,
                    weight_gen_advers=0.0,
                    train_gen=True, train_disc=False,
                    checkpoint_int=2,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        out_dir = os.path.join(td, 'test_1')

        spatial_pad = 20
        temporal_pad = 20
        cache_pattern = os.path.join(td, 'cache')
        fwp_shape = (4, 4, len(input_files) // 2)
        handler = ForwardPassStrategy(
            input_files, model_kwargs={'model_dir': out_dir},
            fwp_chunk_shape=fwp_shape,
            spatial_pad=spatial_pad, temporal_pad=temporal_pad,
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            cache_pattern=cache_pattern,
            overwrite_cache=True,
            ti_workers=1,
            max_workers=1)

        data_chunked = np.zeros((shape[0] * s_enhance, shape[1] * s_enhance,
                                 len(input_files) * t_enhance,
                                 len(model.output_features)))
        handlerNC = DataHandlerNC(input_files, FEATURES, target=target,
                                  val_split=0.0, shape=shape, ti_workers=1)
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

    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            shape=(20, 20),
                            sample_shape=(18, 18, 24),
                            temporal_slice=slice(None, None, 1),
                            val_split=0.005,
                            max_workers=1)

    batch_handler = BatchHandler([handler], batch_size=4,
                                 s_enhance=s_enhance,
                                 t_enhance=t_enhance,
                                 n_batches=4)

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        model.train(batch_handler, n_epoch=1,
                    weight_gen_advers=0.0,
                    train_gen=True, train_disc=False,
                    checkpoint_int=2,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        cache_pattern = os.path.join(td, 'cache')
        handler = ForwardPassStrategy(
            input_files, model_kwargs={'model_dir': out_dir},
            fwp_chunk_shape=(shape[0], shape[1], list_chunk_size),
            spatial_pad=0, temporal_pad=0,
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            cache_pattern=cache_pattern,
            overwrite_cache=True,
            ti_workers=1,
            max_workers=1)
        forward_pass = ForwardPass(handler)
        data_chunked = forward_pass.run_chunk()

        handlerNC = DataHandlerNC(input_files, FEATURES,
                                  target=target, shape=shape,
                                  temporal_slice=temporal_slice,
                                  max_workers=None,
                                  cache_pattern=None,
                                  time_chunk_size=100,
                                  overwrite_cache=True,
                                  val_split=0.0,
                                  ti_workers=1)

        data_nochunk = model.generate(
            np.expand_dims(handlerNC.data, axis=0))[0]

        assert np.array_equal(data_chunked, data_nochunk)


def test_fwp_multi_step_model_topo_exoskip():
    """Test the forward pass with a multi step model class using exogenous data
    for the first two steps and not the last"""
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
        handler = ForwardPassStrategy(
            input_files, model_kwargs=model_kwargs,
            model_class='SpatialThenTemporalGan',
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=0, temporal_pad=0,
            target=target, shape=shape,
            out_pattern=out_files,
            temporal_slice=temporal_slice,
            max_workers=max_workers,
            exo_kwargs=exo_kwargs,
            max_nodes=1,
            ti_workers=1)

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
        handler = ForwardPassStrategy(
            input_files, model_kwargs=model_kwargs,
            model_class='SpatialThenTemporalGan',
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=1, temporal_pad=1,
            target=target, shape=shape,
            out_pattern=out_files,
            temporal_slice=temporal_slice,
            max_workers=max_workers,
            exo_kwargs=exo_kwargs,
            max_nodes=1,
            ti_workers=1)

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

        handler = ForwardPassStrategy(
            input_files, model_kwargs=model_kwargs,
            model_class='SpatialThenTemporalGan',
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=0, temporal_pad=0,
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            out_pattern=out_files,
            max_workers=max_workers,
            max_nodes=1,
            ti_workers=1)

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


def test_slicing_no_pad():
    """Test the slicing of input data via the ForwardPassStrategy +
    ForwardPassSlicer vs. the actual source data. Does not include any
    reflected padding at the edges."""
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
                                val_split=0.0, max_workers=1)

        strategy = ForwardPassStrategy(
            input_files, model_kwargs={'model_dir': st_out_dir},
            model_class='Sup3rGan',
            fwp_chunk_shape=(3, 2, 4),
            spatial_pad=0, temporal_pad=0,
            target=target, shape=shape,
            temporal_slice=slice(None),
            out_pattern=out_files,
            max_workers=1,
            max_nodes=1)

        for ichunk in range(strategy.chunks):
            forward_pass = ForwardPass(strategy, chunk_index=ichunk)

            s_slices = strategy.lr_pad_slices[forward_pass.spatial_chunk_index]
            lr_data_slice = (s_slices[0], s_slices[1],
                             forward_pass._ti_pad_slice,
                             slice(None))

            truth = handler.data[lr_data_slice]
            assert np.allclose(forward_pass.input_data, truth)


def test_slicing_pad():
    """Test the slicing of input data via the ForwardPassStrategy +
    ForwardPassSlicer vs. the actual source data. Includes reflected padding
    at the edges."""
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
                                val_split=0.0, max_workers=1)

        strategy = ForwardPassStrategy(
            input_files, model_kwargs={'model_dir': st_out_dir},
            model_class='Sup3rGan',
            fwp_chunk_shape=(2, 1, 4),
            spatial_pad=2, temporal_pad=2,
            target=target, shape=shape,
            temporal_slice=slice(None),
            out_pattern=out_files,
            max_workers=1,
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
                             forward_pass._ti_pad_slice,
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
