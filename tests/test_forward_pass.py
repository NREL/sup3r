# -*- coding: utf-8 -*-
"""pytests for data handling"""

import os
import tempfile
import numpy as np
import xarray as xr

from sup3r import TEST_DATA_DIR, CONFIG_DIR
from sup3r.preprocessing.data_handling import DataHandlerH5, DataHandlerNC
from sup3r.preprocessing.batch_handling import BatchHandler
from sup3r.pipeline.forward_pass import ForwardPass, ForwardPassStrategy
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
fp_chunk_shape = (4, 4, 150)
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
            input_files, target=target, shape=shape,
            temporal_slice=temporal_slice,
            cache_pattern=cache_pattern,
            fp_chunk_shape=fp_chunk_shape,
            overwrite_cache=True, out_pattern=out_files,
            max_workers=max_workers)
        forward_pass = ForwardPass(handler, model_path=out_dir)
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
            input_files, target=target, shape=shape,
            temporal_slice=temporal_slice,
            cache_pattern=cache_pattern,
            fp_chunk_shape=fp_chunk_shape,
            overwrite_cache=True, out_pattern=out_files,
            max_workers=max_workers)
        forward_pass = ForwardPass(handler, model_path=out_dir)
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


def test_forward_pass_h5():
    """Test forward pass handler output with second pass on output files.
    Writing to h5"""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 10, 10, 6, 2)))
    model.meta['training_features'] = ['U_100m', 'V_100m']
    model.meta['output_features'] = ['U_100m', 'V_100m']
    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        cache_pattern = os.path.join(td, 'cache')
        out_files = os.path.join(td, 'out_{file_id}.h5')

        max_workers = 1
        handler = ForwardPassStrategy(
            input_files, target=target, shape=shape,
            temporal_slice=temporal_slice,
            cache_pattern=cache_pattern,
            fp_chunk_shape=fp_chunk_shape,
            overwrite_cache=True, out_pattern=out_files,
            max_workers=max_workers)
        forward_pass = ForwardPass(handler, model_path=out_dir)
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
            input_files, target=target, shape=shape,
            temporal_slice=temporal_slice,
            cache_pattern=cache_pattern,
            fp_chunk_shape=fp_chunk_shape,
            overwrite_cache=True,
            max_workers=max_workers)
        forward_pass = ForwardPass(handler, model_path=out_dir)
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
            input_files, target=target, shape=shape,
            temporal_slice=temporal_slice,
            cache_pattern=cache_pattern,
            fp_chunk_shape=fp_chunk_shape,
            overwrite_cache=True)
        forward_pass = ForwardPass(handler, model_path=out_dir)
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
            input_files, target=target, shape=shape,
            temporal_slice=temporal_slice,
            cache_pattern=cache_pattern,
            fp_chunk_shape=(shape[0], shape[1], list_chunk_size),
            overwrite_cache=True)
        forward_pass = ForwardPass(handler, model_path=out_dir)
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
