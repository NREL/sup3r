# -*- coding: utf-8 -*-
"""pytests for data handling"""

import os
import tempfile
import numpy as np
import glob

from sup3r import TEST_DATA_DIR, CONFIG_DIR
from sup3r.preprocessing.data_handling import DataHandlerH5, DataHandlerNC
from sup3r.preprocessing.batch_handling import BatchHandler
from sup3r.pipeline.forward_pass import ForwardPass, ForwardPassStrategy
from sup3r.models import Sup3rGan


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
sample_shape = (8, 8, 6)
raster_file = os.path.join(tempfile.gettempdir(), 'tmp_raster_nc.txt')
temporal_slice = slice(None, None, 1)
list_chunk_size = 10
forward_pass_chunk_shape = (4, 4, 10)
s_enhance = 3
t_enhance = 4


def test_repeated_forward_pass():
    """Test forward pass handler output with second pass on output files."""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)

    # only use wind features since model output only gives 2 features
    handler = DataHandlerH5(FP_WTK, FEATURES[:2], target=TARGET_COORD,
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
        model.train(batch_handler, n_epoch=1,
                    weight_gen_advers=0.0,
                    train_gen=True, train_disc=False,
                    checkpoint_int=2,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        cache_file_prefix = os.path.join(td, 'cache')
        out_file_prefix = os.path.join(td, 'out')
        # 1st forward pass
        handler = ForwardPassStrategy(
            input_files, target=target, shape=shape,
            temporal_slice=temporal_slice, raster_file=raster_file,
            cache_file_prefix=cache_file_prefix,
            forward_pass_chunk_shape=forward_pass_chunk_shape,
            overwrite_cache=True, out_file_prefix=out_file_prefix)
        forward_pass = ForwardPass(handler, model_path=out_dir)
        forward_pass.run()

        # 2nd forward pass
        new_input_files = glob.glob(f'{out_file_prefix}*')
        new_shape = (s_enhance * shape[0], s_enhance * shape[1])
        handler = ForwardPassStrategy(
            new_input_files, target=target, shape=new_shape,
            temporal_slice=temporal_slice, raster_file=raster_file,
            cache_file_prefix=cache_file_prefix,
            forward_pass_chunk_shape=forward_pass_chunk_shape,
            overwrite_cache=True)
        forward_pass = ForwardPass(handler, model_path=out_dir)
        data = forward_pass.run()
        assert data.shape == (
            s_enhance**2 * shape[0], s_enhance**2 * shape[1],
            t_enhance**2 * len(input_files), 2)


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
                            extract_workers=1,
                            compute_workers=1)

    batch_handler = BatchHandler([handler], batch_size=4,
                                 s_enhance=s_enhance,
                                 t_enhance=t_enhance,
                                 n_batches=4)

    with tempfile.TemporaryDirectory() as td:
        model.train(batch_handler, n_epoch=1,
                    weight_gen_advers=0.0,
                    train_gen=True, train_disc=False,
                    checkpoint_int=2,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        cache_file_prefix = os.path.join(td, 'cache')
        handler = ForwardPassStrategy(
            input_files, target=target, shape=shape,
            temporal_slice=temporal_slice, raster_file=raster_file,
            cache_file_prefix=cache_file_prefix,
            forward_pass_chunk_shape=forward_pass_chunk_shape,
            overwrite_cache=True)
        forward_pass = ForwardPass(handler, model_path=out_dir)
        data = forward_pass.run()

        assert data.shape == (s_enhance * shape[0],
                              s_enhance * shape[1],
                              t_enhance * len(input_files),
                              2)


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
                            extract_workers=1,
                            compute_workers=1)

    batch_handler = BatchHandler([handler], batch_size=4,
                                 s_enhance=s_enhance,
                                 t_enhance=t_enhance,
                                 n_batches=4)

    with tempfile.TemporaryDirectory() as td:
        model.train(batch_handler, n_epoch=1,
                    weight_gen_advers=0.0,
                    train_gen=True, train_disc=False,
                    checkpoint_int=2,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        cache_file_prefix = os.path.join(td, 'cache')
        handler = ForwardPassStrategy(
            input_files, target=target, shape=shape,
            temporal_slice=temporal_slice, raster_file=raster_file,
            cache_file_prefix=cache_file_prefix,
            forward_pass_chunk_shape=forward_pass_chunk_shape,
            overwrite_cache=True)
        forward_pass = ForwardPass(handler, model_path=out_dir)
        data_chunked = forward_pass.run()

        handlerNC = DataHandlerNC(input_files, FEATURES, target=target,
                                  val_split=0.0, shape=shape,
                                  raster_file=raster_file)

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
                            extract_workers=1,
                            compute_workers=1)

    batch_handler = BatchHandler([handler], batch_size=4,
                                 s_enhance=s_enhance,
                                 t_enhance=t_enhance,
                                 n_batches=4)

    with tempfile.TemporaryDirectory() as td:
        model.train(batch_handler, n_epoch=1,
                    weight_gen_advers=0.0,
                    train_gen=True, train_disc=False,
                    checkpoint_int=2,
                    out_dir=os.path.join(td, 'test_{epoch}'))

        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        cache_file_prefix = os.path.join(td, 'cache')
        handler = ForwardPassStrategy(
            input_files, target=target, shape=shape,
            temporal_slice=temporal_slice, raster_file=raster_file,
            cache_file_prefix=cache_file_prefix,
            forward_pass_chunk_shape=(shape[0], shape[1], list_chunk_size),
            overwrite_cache=True)
        forward_pass = ForwardPass(handler, model_path=out_dir)
        data_chunked = forward_pass.run()

        handlerNC = DataHandlerNC(input_files, FEATURES,
                                  target=target, shape=shape,
                                  temporal_slice=temporal_slice,
                                  raster_file=raster_file,
                                  extract_workers=None,
                                  compute_workers=None,
                                  cache_file_prefix=None,
                                  time_chunk_size=100,
                                  overwrite_cache=True,
                                  val_split=0.0)

        data_nochunk = model.generate(
            np.expand_dims(handlerNC.data, axis=0))[0]

        assert np.array_equal(data_chunked, data_nochunk)
