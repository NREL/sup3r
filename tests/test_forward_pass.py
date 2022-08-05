# -*- coding: utf-8 -*-
"""pytests for data handling"""
import json
import os
import tempfile
import tensorflow as tf
import numpy as np
import xarray as xr

from sup3r import TEST_DATA_DIR, CONFIG_DIR, __version__
from sup3r.preprocessing.data_handling import DataHandlerH5, DataHandlerNC
from sup3r.preprocessing.batch_handling import BatchHandler
from sup3r.pipeline.forward_pass import (ForwardPass, ForwardPassStrategy)
from sup3r.models import Sup3rGan
from sup3r.utilities.test_utils import make_fake_nc_files

from rex import ResourceX


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


def test_forward_pass_nc_cc():
    """Test forward pass handler output for netcdf write with cc data."""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)

    input_files = [os.path.join(TEST_DATA_DIR, 'ua_test.nc'),
                   os.path.join(TEST_DATA_DIR, 'va_test.nc'),
                   os.path.join(TEST_DATA_DIR, 'zg_test.nc')]
    features = ['U_100m', 'V_100m']
    target = (13.67, 125.0)
    _ = model.generate(np.ones((4, 10, 10, 6, len(features))))
    model.meta['training_features'] = features
    model.meta['output_features'] = features
    with tempfile.TemporaryDirectory() as td:
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        cache_pattern = os.path.join(td, 'cache')
        out_files = os.path.join(td, 'out_{file_id}.nc')
        # 1st forward pass
        max_workers = 1
        handler = ForwardPassStrategy(
            input_files, model_args=out_dir,
            s_enhance=3, t_enhance=4,
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=1, temporal_pad=1,
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            cache_pattern=cache_pattern,
            overwrite_cache=True, out_pattern=out_files,
            max_workers=max_workers, input_handler='DataHandlerNCforCC')
        forward_pass = ForwardPass(handler)
        assert forward_pass.pass_workers == max_workers
        assert forward_pass.output_workers == max_workers
        assert forward_pass.data_handler.compute_workers == max_workers
        assert forward_pass.data_handler.load_workers == max_workers
        assert forward_pass.data_handler.norm_workers == max_workers
        assert forward_pass.data_handler.extract_workers == max_workers
        forward_pass.run()

        with xr.open_dataset(handler.out_files[0]) as fh:
            assert fh[FEATURES[0]].shape == (
                t_enhance * len(handler.time_index), s_enhance * shape[0],
                s_enhance * shape[1])
            assert fh[FEATURES[1]].shape == (
                t_enhance * len(handler.time_index), s_enhance * shape[0],
                s_enhance * shape[1])


def test_forward_pass_nc():
    """Test forward pass handler output for netcdf write."""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 10, 10, 6, len(FEATURES))))
    model.meta['training_features'] = FEATURES
    model.meta['output_features'] = ['U_100m', 'V_100m']
    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        cache_pattern = os.path.join(td, 'cache')
        out_files = os.path.join(td, 'out_{file_id}.nc')
        # 1st forward pass
        max_workers = 1
        handler = ForwardPassStrategy(
            input_files, model_args=out_dir,
            s_enhance=3, t_enhance=4,
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=1, temporal_pad=1,
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            cache_pattern=cache_pattern,
            overwrite_cache=True, out_pattern=out_files,
            max_workers=max_workers)
        forward_pass = ForwardPass(handler)
        assert forward_pass.pass_workers == max_workers
        assert forward_pass.output_workers == max_workers
        assert forward_pass.data_handler.compute_workers == max_workers
        assert forward_pass.data_handler.load_workers == max_workers
        assert forward_pass.data_handler.norm_workers == max_workers
        assert forward_pass.data_handler.extract_workers == max_workers
        forward_pass.run()

        with xr.open_dataset(handler.out_files[0]) as fh:
            assert fh[FEATURES[0]].shape == (
                t_enhance * len(handler.time_index), s_enhance * shape[0],
                s_enhance * shape[1])
            assert fh[FEATURES[1]].shape == (
                t_enhance * len(handler.time_index), s_enhance * shape[0],
                s_enhance * shape[1])


def test_forward_pass_temporal_slice():
    """Test forward pass handler output to h5 file. Includes temporal
    slicing."""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 10, 10, 6, 2)))
    model.meta['training_features'] = ['U_100m', 'V_100m']
    model.meta['output_features'] = ['U_100m', 'V_100m']
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
            input_files, model_args=out_dir,
            s_enhance=3, t_enhance=4,
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=1, temporal_pad=1,
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            cache_pattern=cache_pattern,
            overwrite_cache=True, out_pattern=out_files,
            max_workers=max_workers)
        forward_pass = ForwardPass(handler)
        assert forward_pass.pass_workers == max_workers
        assert forward_pass.output_workers == max_workers
        assert forward_pass.data_handler.compute_workers == max_workers
        assert forward_pass.data_handler.load_workers == max_workers
        assert forward_pass.data_handler.norm_workers == max_workers
        assert forward_pass.data_handler.extract_workers == max_workers
        forward_pass.run()

        with ResourceX(handler.out_files[0]) as fh:
            assert fh.shape == (t_enhance * n_tsteps,
                                s_enhance**2 * shape[0] * shape[1])
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


def test_fwd_pass_handler():
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
            input_files, model_args=out_dir,
            s_enhance=3, t_enhance=4,
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=1, temporal_pad=1,
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            cache_pattern=cache_pattern,
            overwrite_cache=True,
            max_workers=max_workers)
        forward_pass = ForwardPass(handler)
        assert forward_pass.pass_workers == max_workers
        assert forward_pass.data_handler.compute_workers == max_workers
        assert forward_pass.data_handler.load_workers == max_workers
        assert forward_pass.data_handler.norm_workers == max_workers
        assert forward_pass.data_handler.extract_workers == max_workers
        data = forward_pass.run()

        assert data.shape == (s_enhance * shape[0],
                              s_enhance * shape[1],
                              t_enhance * len(input_files), 2)


def test_fwd_pass_chunking():
    """Test forward pass chunking. Make sure chunking agrees closely
    with non chunking forward pass.
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
            input_files, model_args=out_dir,
            s_enhance=3, t_enhance=4,
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=1, temporal_pad=1,
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            cache_pattern=cache_pattern,
            overwrite_cache=True)
        forward_pass = ForwardPass(handler)
        data_chunked = forward_pass.run()

        handlerNC = DataHandlerNC(input_files, FEATURES, target=target,
                                  val_split=0.0, shape=shape)

        data_nochunk = model.generate(
            np.expand_dims(handlerNC.data, axis=0))[0]

        assert np.mean(
            (np.abs(data_chunked - data_nochunk)
             / np.product(data_chunked.shape))) < 0.01


def test_fwd_pass_nochunking():
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
            input_files, model_args=out_dir,
            s_enhance=3, t_enhance=4,
            fwp_chunk_shape=(shape[0], shape[1], list_chunk_size),
            spatial_pad=1, temporal_pad=1,
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            cache_pattern=cache_pattern,
            overwrite_cache=True)
        forward_pass = ForwardPass(handler)
        data_chunked = forward_pass.run()

        handlerNC = DataHandlerNC(input_files, FEATURES,
                                  target=target, shape=shape,
                                  temporal_slice=temporal_slice,
                                  max_workers=None,
                                  cache_pattern=None,
                                  time_chunk_size=100,
                                  overwrite_cache=True,
                                  val_split=0.0)

        data_nochunk = model.generate(
            np.expand_dims(handlerNC.data, axis=0))[0]

        assert np.array_equal(data_chunked, data_nochunk)


def test_fwp_multi_step_model_topo():
    """Test the forward pass with a multi step model class"""
    Sup3rGan.seed()
    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    s1_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    s1_model.meta['training_features'] = ['U_100m', 'V_100m', 'topography']
    s1_model.meta['output_features'] = ['U_100m', 'V_100m']
    _ = s1_model.generate(np.ones((4, 10, 10, 3)))

    s2_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    s2_model.meta['training_features'] = ['U_100m', 'V_100m', 'topography']
    s2_model.meta['output_features'] = ['U_100m', 'V_100m']
    _ = s2_model.generate(np.ones((4, 10, 10, 3)))

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    st_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    st_model.meta['training_features'] = ['U_100m', 'V_100m', 'topography']
    st_model.meta['output_features'] = ['U_100m', 'V_100m']
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
        s_enhance = np.product([s for s in s_enhancements if s is not None])
        t_enhance = 4

        exo_kwargs = {'file_paths': input_files,
                      'features': ['topography'],
                      'source_h5': FP_WTK,
                      'target': target,
                      'shape': shape,
                      's_enhancements': s_enhancements,
                      'agg_factors': [2, 4, 12]
                      }

        out_files = os.path.join(td, 'out_{file_id}.h5')
        handler = ForwardPassStrategy(
            input_files, model_args=[[s1_out_dir, s2_out_dir], st_out_dir],
            model_class='SpatialThenTemporalGan',
            s_enhance=s_enhance, t_enhance=t_enhance,
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=0, temporal_pad=0,
            target=target, shape=shape,
            out_pattern=out_files,
            temporal_slice=temporal_slice,
            max_workers=max_workers,
            exo_kwargs=exo_kwargs)

        forward_pass = ForwardPass(handler)

        assert forward_pass.pass_workers == max_workers
        assert forward_pass.output_workers == max_workers
        assert forward_pass.data_handler.compute_workers == max_workers
        assert forward_pass.data_handler.load_workers == max_workers
        assert forward_pass.data_handler.norm_workers == max_workers
        assert forward_pass.data_handler.extract_workers == max_workers

        forward_pass.run()

        with ResourceX(handler.out_files[0]) as fh:
            assert fh.shape == (t_enhance * len(input_files),
                                s_enhance**2 * shape[0] * shape[1])
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
    _ = s_model.generate(np.ones((4, 10, 10, 2)))

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    st_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    st_model.meta['training_features'] = ['U_100m', 'V_100m']
    st_model.meta['output_features'] = ['U_100m', 'V_100m']
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

        handler = ForwardPassStrategy(
            input_files, model_args=[s_out_dir, st_out_dir],
            model_class='SpatialThenTemporalGan',
            s_enhance=s_enhance, t_enhance=t_enhance,
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=0, temporal_pad=0,
            target=target, shape=shape,
            temporal_slice=temporal_slice,
            out_pattern=out_files,
            max_workers=max_workers)

        forward_pass = ForwardPass(handler)
        ones = np.ones((fwp_chunk_shape[2], fwp_chunk_shape[0],
                        fwp_chunk_shape[1], 2))
        out = forward_pass.model.generate(ones)
        assert out.shape == (1, 24, 24, 32, 2)

        assert forward_pass.pass_workers == max_workers
        assert forward_pass.output_workers == max_workers
        assert forward_pass.data_handler.compute_workers == max_workers
        assert forward_pass.data_handler.load_workers == max_workers
        assert forward_pass.data_handler.norm_workers == max_workers
        assert forward_pass.data_handler.extract_workers == max_workers

        forward_pass.run()

        with ResourceX(handler.out_files[0]) as fh:
            assert fh.shape == (t_enhance * len(input_files),
                                s_enhance**2 * shape[0] * shape[1])
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
