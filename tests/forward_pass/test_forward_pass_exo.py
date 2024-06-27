# -*- coding: utf-8 -*-
"""pytests for data handling"""
import json
import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytest
import tensorflow as tf
from rex import ResourceX, init_logger

from sup3r import CONFIG_DIR, TEST_DATA_DIR, __version__
from sup3r.models import LinearInterp, Sup3rGan, SurfaceSpatialMetModel
from sup3r.pipeline.forward_pass import ForwardPass, ForwardPassStrategy
from sup3r.utilities.pytest import make_fake_nc_files

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


def test_fwp_multi_step_model_topo_exoskip(log=False):
    """Test the forward pass with a multi step model class using exogenous data
    for the first two steps and not the last"""

    if log:
        init_logger('sup3r', log_level='DEBUG')

    Sup3rGan.seed()
    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    s1_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    s1_model.meta['lr_features'] = ['U_100m', 'V_100m', 'topography']
    s1_model.meta['hr_out_features'] = ['U_100m', 'V_100m']
    s1_model.meta['s_enhance'] = 2
    s1_model.meta['t_enhance'] = 1
    s1_model.meta['input_resolution'] = {'spatial': '48km',
                                         'temporal': '60min'}
    _ = s1_model.generate(np.ones((4, 10, 10, 3)))

    s2_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    s2_model.meta['lr_features'] = ['U_100m', 'V_100m', 'topography']
    s2_model.meta['hr_out_features'] = ['U_100m', 'V_100m']
    s2_model.meta['s_enhance'] = 2
    s2_model.meta['t_enhance'] = 1
    s2_model.meta['input_resolution'] = {'spatial': '24km',
                                         'temporal': '60min'}
    _ = s2_model.generate(np.ones((4, 10, 10, 3)))

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    st_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    st_model.meta['lr_features'] = ['U_100m', 'V_100m']
    st_model.meta['hr_out_features'] = ['U_100m', 'V_100m']
    st_model.meta['s_enhance'] = 3
    st_model.meta['t_enhance'] = 4
    st_model.meta['input_resolution'] = {'spatial': '12km',
                                         'temporal': '60min'}
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

        exo_kwargs = {
            'topography': {
                'file_paths': input_files,
                'source_file': FP_WTK,
                'target': target,
                'shape': shape,
                'cache_dir': td,
                'exo_resolution': {'spatial': '4km', 'temporal': '60min'},
                'steps': [
                    {'model': 0, 'combine_type': 'input'},
                    {'model': 1, 'combine_type': 'input'}
                ]
            }
        }

        model_kwargs = {
            'model_dirs': [s1_out_dir, s2_out_dir, st_out_dir]
        }

        out_files = os.path.join(td, 'out_{file_id}.h5')
        input_handler_kwargs = dict(
            target=target,
            shape=shape,
            temporal_slice=temporal_slice,
            worker_kwargs=dict(max_workers=max_workers),
            overwrite_cache=True)
        handler = ForwardPassStrategy(
            input_files,
            model_kwargs=model_kwargs,
            model_class='MultiStepGan',
            fwp_chunk_shape=fwp_chunk_shape,
            input_handler_kwargs=input_handler_kwargs,
            spatial_pad=0,
            temporal_pad=0,
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
            assert fh.shape == (t_enhance * len(input_files), s_enhance**2
                                * fwp_chunk_shape[0] * fwp_chunk_shape[1])
            assert all(f in fh.attrs
                       for f in ('windspeed_100m', 'winddirection_100m'))

            assert fh.global_attrs['package'] == 'sup3r'
            assert fh.global_attrs['version'] == __version__
            assert 'full_version_record' in fh.global_attrs
            version_record = json.loads(fh.global_attrs['full_version_record'])
            assert version_record['tensorflow'] == tf.__version__
            assert 'gan_meta' in fh.global_attrs
            gan_meta = json.loads(fh.global_attrs['gan_meta'])
            assert len(gan_meta) == 3  # three step model
            assert gan_meta[0]['lr_features'] == [
                'U_100m', 'V_100m', 'topography'
            ]


def test_fwp_multi_step_spatial_model_topo_noskip():
    """Test the forward pass with a multi step spatial only model class using
    exogenous data for all model steps"""
    Sup3rGan.seed()
    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    s1_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    s1_model.meta['lr_features'] = ['U_100m', 'V_100m', 'topography']
    s1_model.meta['hr_out_features'] = ['U_100m', 'V_100m']
    s1_model.meta['s_enhance'] = 2
    s1_model.meta['t_enhance'] = 1
    s1_model.meta['input_resolution'] = {'spatial': '16km',
                                         'temporal': '60min'}
    _ = s1_model.generate(np.ones((4, 10, 10, 3)))

    s2_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    s2_model.meta['lr_features'] = ['U_100m', 'V_100m', 'topography']
    s2_model.meta['hr_out_features'] = ['U_100m', 'V_100m']
    s2_model.meta['s_enhance'] = 2
    s2_model.meta['t_enhance'] = 1
    s2_model.meta['input_resolution'] = {'spatial': '8km',
                                         'temporal': '60min'}
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
        s_enhance = np.prod(s_enhancements)

        exo_kwargs = {
            'topography': {
                'file_paths': input_files,
                'source_file': FP_WTK,
                'target': target,
                'shape': shape,
                'cache_dir': td,
                'exo_resolution': {'spatial': '4km', 'temporal': '60min'},
                'steps': [
                    {'model': 0, 'combine_type': 'input'},
                    {'model': 1, 'combine_type': 'input'},
                ]
            }
        }

        model_kwargs = {'model_dirs': [s1_out_dir, s2_out_dir]}

        out_files = os.path.join(td, 'out_{file_id}.h5')
        input_handler_kwargs = dict(
            target=target,
            shape=shape,
            temporal_slice=temporal_slice,
            worker_kwargs=dict(max_workers=max_workers),
            overwrite_cache=True)
        handler = ForwardPassStrategy(
            input_files,
            model_kwargs=model_kwargs,
            model_class='MultiStepGan',
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=1,
            temporal_pad=1,
            input_handler_kwargs=input_handler_kwargs,
            out_pattern=out_files,
            worker_kwargs=dict(max_workers=max_workers),
            exo_kwargs=exo_kwargs,
            max_nodes=1)

        forward_pass = ForwardPass(handler)
        forward_pass.run(handler, node_index=0)

        with ResourceX(handler.out_files[0]) as fh:
            assert fh.shape == (len(input_files), s_enhance**2
                                * fwp_chunk_shape[0] * fwp_chunk_shape[1])
            assert all(f in fh.attrs
                       for f in ('windspeed_100m', 'winddirection_100m'))

            assert fh.global_attrs['package'] == 'sup3r'
            assert fh.global_attrs['version'] == __version__
            assert 'full_version_record' in fh.global_attrs
            version_record = json.loads(fh.global_attrs['full_version_record'])
            assert version_record['tensorflow'] == tf.__version__
            assert 'gan_meta' in fh.global_attrs
            gan_meta = json.loads(fh.global_attrs['gan_meta'])
            assert len(gan_meta) == 2  # two step model
            assert gan_meta[0]['lr_features'] == [
                'U_100m', 'V_100m', 'topography'
            ]


def test_fwp_multi_step_model_topo_noskip():
    """Test the forward pass with a multi step model class using exogenous data
    for all model steps"""
    Sup3rGan.seed()
    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    s1_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    s1_model.meta['lr_features'] = ['U_100m', 'V_100m', 'topography']
    s1_model.meta['hr_out_features'] = ['U_100m', 'V_100m']
    s1_model.meta['s_enhance'] = 2
    s1_model.meta['t_enhance'] = 1
    s1_model.meta['input_resolution'] = {'spatial': '48km',
                                         'temporal': '60min'}
    _ = s1_model.generate(np.ones((4, 10, 10, 3)))

    s2_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    s2_model.meta['lr_features'] = ['U_100m', 'V_100m', 'topography']
    s2_model.meta['hr_out_features'] = ['U_100m', 'V_100m']
    s2_model.meta['s_enhance'] = 2
    s2_model.meta['t_enhance'] = 1
    s2_model.meta['input_resolution'] = {'spatial': '24km',
                                         'temporal': '60min'}
    _ = s2_model.generate(np.ones((4, 10, 10, 3)))

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    st_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    st_model.meta['lr_features'] = ['U_100m', 'V_100m', 'topography']
    st_model.meta['hr_out_features'] = ['U_100m', 'V_100m']
    st_model.meta['s_enhance'] = 3
    st_model.meta['t_enhance'] = 4
    st_model.meta['input_resolution'] = {'spatial': '12km',
                                         'temporal': '60min'}
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
        s_enhance = np.prod(s_enhancements)
        t_enhance = 4

        exo_kwargs = {
            'topography': {
                'file_paths': input_files,
                'source_file': FP_WTK,
                'target': target,
                'shape': shape,
                'cache_dir': td,
                'exo_resolution': {'spatial': '4km', 'temporal': '60min'},
                'steps': [{'model': 0, 'combine_type': 'input'},
                          {'model': 1, 'combine_type': 'input'},
                          {'model': 2, 'combine_type': 'input'}]
            }
        }

        model_kwargs = {
            'model_dirs': [s1_out_dir, s2_out_dir, st_out_dir]
        }

        out_files = os.path.join(td, 'out_{file_id}.h5')
        input_handler_kwargs = dict(
            target=target,
            shape=shape,
            temporal_slice=temporal_slice,
            worker_kwargs=dict(max_workers=max_workers),
            overwrite_cache=True)
        handler = ForwardPassStrategy(
            input_files,
            model_kwargs=model_kwargs,
            model_class='MultiStepGan',
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=1,
            temporal_pad=1,
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
            assert fh.shape == (t_enhance * len(input_files), s_enhance**2
                                * fwp_chunk_shape[0] * fwp_chunk_shape[1])
            assert all(f in fh.attrs
                       for f in ('windspeed_100m', 'winddirection_100m'))

            assert fh.global_attrs['package'] == 'sup3r'
            assert fh.global_attrs['version'] == __version__
            assert 'full_version_record' in fh.global_attrs
            version_record = json.loads(fh.global_attrs['full_version_record'])
            assert version_record['tensorflow'] == tf.__version__
            assert 'gan_meta' in fh.global_attrs
            gan_meta = json.loads(fh.global_attrs['gan_meta'])
            assert len(gan_meta) == 3  # three step model
            assert gan_meta[0]['lr_features'] == [
                'U_100m', 'V_100m', 'topography'
            ]


def test_fwp_single_step_sfc_model(plot=False):
    """Test the forward pass with a single SurfaceSpatialMetModel model
    which requires low and high-resolution topography input from the
    exogenous_data feature."""

    model = SurfaceSpatialMetModel(
        lr_features=['pressure_0m'], s_enhance=2,
        input_resolution={'spatial': '8km', 'temporal': '60min'})

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)

        sfc_out_dir = os.path.join(td, 'sfc')
        model.save(sfc_out_dir)

        exo_kwargs = {
            'topography': {
                'file_paths': input_files,
                'source_file': FP_WTK,
                'target': target,
                'shape': shape,
                'cache_dir': td,
                'exo_resolution': {'spatial': '4km', 'temporal': '60min'},
                'steps': [
                    {'model': 0, 'combine_type': 'input'},
                    {'model': 0, 'combine_type': 'output'}
                ]}}

        out_files = os.path.join(td, 'out_{file_id}.h5')
        input_handler_kwargs = dict(target=target,
                                    shape=shape,
                                    temporal_slice=temporal_slice,
                                    worker_kwargs=dict(max_workers=1),
                                    overwrite_cache=True)

        handler = ForwardPassStrategy(
            input_files,
            model_kwargs=sfc_out_dir,
            model_class='SurfaceSpatialMetModel',
            fwp_chunk_shape=(8, 8, 8),
            spatial_pad=4,
            temporal_pad=4,
            input_handler_kwargs=input_handler_kwargs,
            out_pattern=out_files,
            worker_kwargs=dict(max_workers=1),
            exo_kwargs=exo_kwargs,
            max_nodes=1)
        forward_pass = ForwardPass(handler)

        if plot:
            for ifeature, feature in enumerate(model.hr_out_features):
                fig = plt.figure(figsize=(15, 5))
                ax1 = fig.add_subplot(111)
                vmin = np.min(forward_pass.input_data[..., ifeature])
                vmax = np.max(forward_pass.input_data[..., ifeature])
                nc = ax1.imshow(forward_pass.input_data[..., 0, ifeature],
                                vmin=vmin,
                                vmax=vmax)
                fig.colorbar(nc, ax=ax1, shrink=0.6, label=f'{feature}')
                plt.savefig(f'./input_{feature}.png')
                plt.close()

        forward_pass.run(handler, node_index=0)

        for fp in handler.out_files:
            assert os.path.exists(fp)


def test_fwp_single_step_wind_hi_res_topo(plot=False):
    """Test the forward pass with a single spatiotemporal Sup3rGan model
    requiring high-resolution topography input from the exogenous_data
    feature."""
    Sup3rGan.seed()
    gen_model = [{
        "class": "FlexiblePadding",
        "paddings": [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]],
        "mode": "REFLECT"
    }, {
        "class": "Conv3D",
        "filters": 64,
        "kernel_size": 3,
        "strides": 1
    }, {
        "class": "Cropping3D",
        "cropping": 2
    }, {
        "class": "SpatioTemporalExpansion",
        "temporal_mult": 2,
        "temporal_method": "nearest"
    }, {
        "class": "FlexiblePadding",
        "paddings": [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]],
        "mode": "REFLECT"
    }, {
        "class": "Conv3D",
        "filters": 64,
        "kernel_size": 3,
        "strides": 1
    }, {
        "class": "Cropping3D",
        "cropping": 2
    }, {
        "class": "SpatioTemporalExpansion",
        "spatial_mult": 2
    }, {
        "class": "FlexiblePadding",
        "paddings": [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]],
        "mode": "REFLECT"
    }, {
        "class": "Conv3D",
        "filters": 64,
        "kernel_size": 3,
        "strides": 1
    }, {
        "class": "Cropping3D",
        "cropping": 2
    }, {
        "alpha": 0.2,
        "class": "LeakyReLU"
    }, {
        "class": "Sup3rConcat",
        "name": "topography"
    }, {
        "class": "FlexiblePadding",
        "paddings": [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]],
        "mode": "REFLECT"
    }, {
        "class": "Conv3D",
        "filters": 2,
        "kernel_size": 3,
        "strides": 1
    }, {
        "class": "Cropping3D",
        "cropping": 2
    }]

    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    model = Sup3rGan(gen_model, fp_disc, learning_rate=1e-4)
    model.meta['lr_features'] = ['U_100m', 'V_100m', 'topography']
    model.meta['hr_out_features'] = ['U_100m', 'V_100m']
    model.meta['s_enhance'] = 2
    model.meta['t_enhance'] = 2
    model.meta['input_resolution'] = {'spatial': '8km',
                                      'temporal': '60min'}
    exo_tmp = {
        'topography': {
            'steps': [
                {'model': 0, 'combine_type': 'layer',
                 'data': np.random.rand(4, 20, 20, 12, 1)}]}}
    _ = model.generate(np.random.rand(4, 10, 10, 6, 3),
                       exogenous_data=exo_tmp)

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)

        st_out_dir = os.path.join(td, 'st_gan')
        model.save(st_out_dir)

        exo_kwargs = {
            'topography': {
                'file_paths': input_files,
                'source_file': FP_WTK,
                'target': target,
                'shape': shape,
                'cache_dir': td,
                'exo_resolution': {'spatial': '4km', 'temporal': '60min'},
                'steps': [
                    {'model': 0, 'combine_type': 'input'},
                    {'model': 0, 'combine_type': 'layer'}
                ]}}

        model_kwargs = {'model_dir': st_out_dir}
        out_files = os.path.join(td, 'out_{file_id}.h5')
        input_handler_kwargs = dict(target=target,
                                    shape=shape,
                                    temporal_slice=temporal_slice,
                                    worker_kwargs=dict(max_workers=1),
                                    overwrite_cache=True)

        handler = ForwardPassStrategy(
            input_files,
            model_kwargs=model_kwargs,
            model_class='Sup3rGan',
            fwp_chunk_shape=(8, 8, 8),
            spatial_pad=4,
            temporal_pad=4,
            input_handler_kwargs=input_handler_kwargs,
            out_pattern=out_files,
            worker_kwargs=dict(max_workers=1),
            exo_kwargs=exo_kwargs,
            max_nodes=1)
        forward_pass = ForwardPass(handler)

        if plot:
            for ifeature, feature in enumerate(model.hr_out_features):
                fig = plt.figure(figsize=(15, 5))
                ax1 = fig.add_subplot(111)
                vmin = np.min(forward_pass.input_data[..., ifeature])
                vmax = np.max(forward_pass.input_data[..., ifeature])
                nc = ax1.imshow(forward_pass.input_data[..., 0, ifeature],
                                vmin=vmin,
                                vmax=vmax)
                fig.colorbar(nc, ax=ax1, shrink=0.6, label=f'{feature}')
                plt.savefig(f'./input_{feature}.png')
                plt.close()

        forward_pass.run(handler, node_index=0)

        for fp in handler.out_files:
            assert os.path.exists(fp)


def test_fwp_multi_step_wind_hi_res_topo():
    """Test the forward pass with multiple Sup3rGan models requiring
    high-resolution topograph input from the exogenous_data feature."""
    Sup3rGan.seed()
    gen_model = [{
        "class": "FlexiblePadding",
        "paddings": [[0, 0], [3, 3], [3, 3], [0, 0]],
        "mode": "REFLECT"
    }, {
        "class": "Conv2DTranspose",
        "filters": 64,
        "kernel_size": 3,
        "strides": 1,
        "activation": "relu"
    }, {
        "class": "Cropping2D",
        "cropping": 4
    }, {
        "class": "FlexiblePadding",
        "paddings": [[0, 0], [3, 3], [3, 3], [0, 0]],
        "mode": "REFLECT"
    }, {
        "class": "Conv2DTranspose",
        "filters": 64,
        "kernel_size": 3,
        "strides": 1,
        "activation": "relu"
    }, {
        "class": "Cropping2D",
        "cropping": 4
    }, {
        "class": "FlexiblePadding",
        "paddings": [[0, 0], [3, 3], [3, 3], [0, 0]],
        "mode": "REFLECT"
    }, {
        "class": "Conv2DTranspose",
        "filters": 64,
        "kernel_size": 3,
        "strides": 1,
        "activation": "relu"
    }, {
        "class": "Cropping2D",
        "cropping": 4
    }, {
        "class": "SpatialExpansion",
        "spatial_mult": 2
    }, {
        "class": "Activation",
        "activation": "relu"
    }, {
        "class": "Sup3rConcat",
        "name": "topography"
    }, {
        "class": "FlexiblePadding",
        "paddings": [[0, 0], [3, 3], [3, 3], [0, 0]],
        "mode": "REFLECT"
    }, {
        "class": "Conv2DTranspose",
        "filters": 2,
        "kernel_size": 3,
        "strides": 1,
        "activation": "relu"
    }, {
        "class": "Cropping2D",
        "cropping": 4
    }]

    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    s1_model = Sup3rGan(gen_model, fp_disc, learning_rate=1e-4)
    s1_model.meta['lr_features'] = ['U_100m', 'V_100m', 'topography']
    s1_model.meta['hr_out_features'] = ['U_100m', 'V_100m']
    s1_model.meta['s_enhance'] = 2
    s1_model.meta['t_enhance'] = 1
    s1_model.meta['input_resolution'] = {'spatial': '48km',
                                         'temporal': '60min'}

    exo_tmp = {
        'topography': {
            'steps': [
                {'model': 0, 'combine_type': 'layer',
                 'data': np.random.rand(4, 20, 20, 1)}]}}
    _ = s1_model.generate(np.ones((4, 10, 10, 3)),
                          exogenous_data=exo_tmp)

    s2_model = Sup3rGan(gen_model, fp_disc, learning_rate=1e-4)
    s2_model.meta['lr_features'] = ['U_100m', 'V_100m', 'topography']
    s2_model.meta['hr_out_features'] = ['U_100m', 'V_100m']
    s2_model.meta['s_enhance'] = 2
    s2_model.meta['t_enhance'] = 1
    s2_model.meta['input_resolution'] = {'spatial': '24km',
                                         'temporal': '60min'}
    _ = s2_model.generate(np.ones((4, 10, 10, 3)),
                          exogenous_data=exo_tmp)

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    st_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    st_model.meta['lr_features'] = ['U_100m', 'V_100m', 'topography']
    st_model.meta['hr_out_features'] = ['U_100m', 'V_100m']
    st_model.meta['s_enhance'] = 3
    st_model.meta['t_enhance'] = 4
    st_model.meta['input_resolution'] = {'spatial': '12km',
                                         'temporal': '60min'}
    _ = st_model.generate(np.ones((4, 10, 10, 6, 3)))

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)

        st_out_dir = os.path.join(td, 'st_gan')
        s1_out_dir = os.path.join(td, 's1_gan')
        s2_out_dir = os.path.join(td, 's2_gan')
        st_model.save(st_out_dir)
        s1_model.save(s1_out_dir)
        s2_model.save(s2_out_dir)

        exo_kwargs = {
            'topography': {
                'file_paths': input_files,
                'source_file': FP_WTK,
                'target': target,
                'shape': shape,
                'cache_dir': td,
                'exo_resolution': {'spatial': '4km', 'temporal': '60min'},
            }
        }

        model_kwargs = {
            'model_dirs': [s1_out_dir, s2_out_dir, st_out_dir]
        }
        out_files = os.path.join(td, 'out_{file_id}.h5')
        input_handler_kwargs = dict(target=target,
                                    shape=shape,
                                    temporal_slice=temporal_slice,
                                    worker_kwargs=dict(max_workers=1),
                                    overwrite_cache=True)

        with pytest.raises(RuntimeError):
            # should raise error since steps doesn't include
            # {'model': 2, 'combine_type': 'input'}
            steps = [{'model': 0, 'combine_type': 'input'},
                     {'model': 0, 'combine_type': 'layer'},
                     {'model': 1, 'combine_type': 'input'},
                     {'model': 1, 'combine_type': 'layer'}]
            exo_kwargs['topography']['steps'] = steps
            handler = ForwardPassStrategy(
                input_files,
                model_kwargs=model_kwargs,
                model_class='MultiStepGan',
                fwp_chunk_shape=(4, 4, 8),
                spatial_pad=1,
                temporal_pad=1,
                input_handler_kwargs=input_handler_kwargs,
                out_pattern=out_files,
                worker_kwargs=dict(max_workers=1),
                exo_kwargs=exo_kwargs,
                max_nodes=1)
            forward_pass = ForwardPass(handler)
            forward_pass.run(handler, node_index=0)

        steps = [{'model': 0, 'combine_type': 'input'},
                 {'model': 0, 'combine_type': 'layer'},
                 {'model': 1, 'combine_type': 'input'},
                 {'model': 1, 'combine_type': 'layer'},
                 {'model': 2, 'combine_type': 'input'}]
        exo_kwargs['topography']['steps'] = steps
        handler = ForwardPassStrategy(
            input_files,
            model_kwargs=model_kwargs,
            model_class='MultiStepGan',
            fwp_chunk_shape=(4, 4, 8),
            spatial_pad=1,
            temporal_pad=1,
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
    """Test the forward pass with a Sup3rGan model requiring high-res topo
    input from exo data for spatial enhancement and a linear interpolation
    model for temporal enhancement."""

    Sup3rGan.seed()
    gen_model = [{
        "class": "FlexiblePadding",
        "paddings": [[0, 0], [3, 3], [3, 3], [0, 0]],
        "mode": "REFLECT"
    }, {
        "class": "Conv2DTranspose",
        "filters": 64,
        "kernel_size": 3,
        "strides": 1
    }, {
        "class": "Cropping2D",
        "cropping": 4
    }, {
        "class": "FlexiblePadding",
        "paddings": [[0, 0], [3, 3], [3, 3], [0, 0]],
        "mode": "REFLECT"
    }, {
        "class": "Conv2DTranspose",
        "filters": 64,
        "kernel_size": 3,
        "strides": 1
    }, {
        "class": "Cropping2D",
        "cropping": 4
    }, {
        "class": "FlexiblePadding",
        "paddings": [[0, 0], [3, 3], [3, 3], [0, 0]],
        "mode": "REFLECT"
    }, {
        "class": "Conv2DTranspose",
        "filters": 64,
        "kernel_size": 3,
        "strides": 1
    }, {
        "class": "Cropping2D",
        "cropping": 4
    }, {
        "class": "SpatialExpansion",
        "spatial_mult": 2
    }, {
        "alpha": 0.2,
        "class": "LeakyReLU"
    }, {
        "class": "Sup3rConcat",
        "name": "topography"
    }, {
        "class": "FlexiblePadding",
        "paddings": [[0, 0], [3, 3], [3, 3], [0, 0]],
        "mode": "REFLECT"
    }, {
        "class": "Conv2DTranspose",
        "filters": 2,
        "kernel_size": 3,
        "strides": 1
    }, {
        "class": "Cropping2D",
        "cropping": 4
    }]

    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    s_model = Sup3rGan(gen_model, fp_disc, learning_rate=1e-4)
    s_model.meta['lr_features'] = ['U_100m', 'V_100m', 'topography']
    s_model.meta['hr_out_features'] = ['U_100m', 'V_100m']
    s_model.meta['s_enhance'] = 2
    s_model.meta['t_enhance'] = 1
    s_model.meta['input_resolution'] = {'spatial': '12km',
                                        'temporal': '60min'}
    exo_tmp = {
        'topography': {
            'steps': [
                {'combine_type': 'layer', 'data': np.ones((4, 20, 20, 1))}]}}
    _ = s_model.generate(np.ones((4, 10, 10, 3)),
                         exogenous_data=exo_tmp)

    t_model = LinearInterp(lr_features=['U_100m', 'V_100m'],
                           s_enhance=1,
                           t_enhance=4)
    t_model.meta['input_resolution'] = {'spatial': '4km',
                                        'temporal': '60min'}

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)

        s_out_dir = os.path.join(td, 's_gan')
        t_out_dir = os.path.join(td, 't_interp')
        s_model.save(s_out_dir)
        t_model.save(t_out_dir)

        exo_kwargs = {
            'topography': {
                'file_paths': input_files,
                'source_file': FP_WTK,
                'target': target,
                'shape': shape,
                'cache_dir': td,
                'exo_resolution': {'spatial': '4km', 'temporal': '60min'},
                'steps': [{'model': 0, 'combine_type': 'input'},
                          {'model': 0, 'combine_type': 'layer'}]
            }
        }

        model_kwargs = {
            'model_dirs': [s_out_dir, t_out_dir]
        }
        out_files = os.path.join(td, 'out_{file_id}.h5')
        input_handler_kwargs = dict(target=target,
                                    shape=shape,
                                    temporal_slice=temporal_slice,
                                    worker_kwargs=dict(max_workers=1),
                                    overwrite_cache=True)

        handler = ForwardPassStrategy(
            input_files,
            model_kwargs=model_kwargs,
            model_class='MultiStepGan',
            fwp_chunk_shape=(4, 4, 8),
            spatial_pad=1,
            temporal_pad=1,
            input_handler_kwargs=input_handler_kwargs,
            out_pattern=out_files,
            worker_kwargs=dict(max_workers=1),
            exo_kwargs=exo_kwargs,
            max_nodes=1)
        forward_pass = ForwardPass(handler)
        forward_pass.run(handler, node_index=0)

        for fp in handler.out_files:
            assert os.path.exists(fp)


def test_fwp_multi_step_model_multi_exo():
    """Test the forward pass with a multi step model class using 2 exogenous
    data features"""
    Sup3rGan.seed()
    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    s1_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    s1_model.meta['lr_features'] = [
        'U_100m', 'V_100m', 'topography'
    ]
    s1_model.meta['hr_out_features'] = ['U_100m', 'V_100m']
    s1_model.meta['s_enhance'] = 2
    s1_model.meta['t_enhance'] = 1
    s1_model.meta['input_resolution'] = {'spatial': '48km',
                                         'temporal': '60min'}
    _ = s1_model.generate(np.ones((4, 10, 10, 3)))

    s2_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    s2_model.meta['lr_features'] = [
        'U_100m', 'V_100m', 'topography'
    ]
    s2_model.meta['hr_out_features'] = ['U_100m', 'V_100m']
    s2_model.meta['s_enhance'] = 2
    s2_model.meta['t_enhance'] = 1
    s2_model.meta['input_resolution'] = {'spatial': '24km',
                                         'temporal': '60min'}
    _ = s2_model.generate(np.ones((4, 10, 10, 3)))

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    st_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    st_model.meta['input_resolution'] = {'spatial': '12km',
                                         'temporal': '60min'}
    st_model.meta['lr_features'] = [
        'U_100m', 'V_100m', 'sza'
    ]
    st_model.meta['hr_out_features'] = ['U_100m', 'V_100m']
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
        s_enhance = np.prod(s_enhancements)
        t_enhance = 4

        exo_kwargs = {
            'topography': {
                'file_paths': input_files,
                'source_file': FP_WTK,
                'target': target,
                'shape': shape,
                'cache_dir': td,
                'exo_resolution': {'spatial': '4km', 'temporal': '60min'},
                'steps': [{'model': 0, 'combine_type': 'input'},
                          {'model': 1, 'combine_type': 'input'}]
            },
            'sza': {
                'file_paths': input_files,
                'target': target,
                'shape': shape,
                'cache_dir': td,
                'exo_handler': 'SzaExtract',
                'exo_resolution': {'spatial': '4km', 'temporal': '60min'},
                'steps': [{'model': 2, 'combine_type': 'input'}]
            }
        }

        model_kwargs = {
            'model_dirs': [s1_out_dir, s2_out_dir, st_out_dir]
        }

        out_files = os.path.join(td, 'out_{file_id}.h5')
        input_handler_kwargs = dict(
            target=target,
            shape=shape,
            temporal_slice=temporal_slice,
            worker_kwargs=dict(max_workers=max_workers),
            overwrite_cache=True)
        handler = ForwardPassStrategy(
            input_files,
            model_kwargs=model_kwargs,
            model_class='MultiStepGan',
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=1,
            temporal_pad=1,
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
            assert fh.shape == (t_enhance * len(input_files), s_enhance**2
                                * fwp_chunk_shape[0] * fwp_chunk_shape[1])
            assert all(f in fh.attrs
                       for f in ('windspeed_100m', 'winddirection_100m'))

            assert fh.global_attrs['package'] == 'sup3r'
            assert fh.global_attrs['version'] == __version__
            assert 'full_version_record' in fh.global_attrs
            version_record = json.loads(fh.global_attrs['full_version_record'])
            assert version_record['tensorflow'] == tf.__version__
            assert 'gan_meta' in fh.global_attrs
            gan_meta = json.loads(fh.global_attrs['gan_meta'])
            assert len(gan_meta) == 3  # three step model
            assert gan_meta[0]['lr_features'] == [
                'U_100m', 'V_100m', 'topography'
            ]

    shutil.rmtree('./exo_cache', ignore_errors=True)


def test_fwp_multi_step_exo_hi_res_topo_and_sza():
    """Test the forward pass with multiple ExoGan models requiring
    high-resolution topography and sza input from the exogenous_data
    feature."""
    Sup3rGan.seed()
    gen_s_model = [{
        "class": "FlexiblePadding",
        "paddings": [[0, 0], [3, 3], [3, 3], [0, 0]],
        "mode": "REFLECT"
    }, {
        "class": "Conv2DTranspose",
        "filters": 64,
        "kernel_size": 3,
        "strides": 1,
        "activation": "relu"
    }, {
        "class": "Cropping2D",
        "cropping": 4
    }, {
        "class": "FlexiblePadding",
        "paddings": [[0, 0], [3, 3], [3, 3], [0, 0]],
        "mode": "REFLECT"
    }, {
        "class": "Conv2DTranspose",
        "filters": 64,
        "kernel_size": 3,
        "strides": 1,
        "activation": "relu"
    }, {
        "class": "Cropping2D",
        "cropping": 4
    }, {
        "class": "FlexiblePadding",
        "paddings": [[0, 0], [3, 3], [3, 3], [0, 0]],
        "mode": "REFLECT"
    }, {
        "class": "Conv2DTranspose",
        "filters": 64,
        "kernel_size": 3,
        "strides": 1,
        "activation": "relu"
    }, {
        "class": "Cropping2D",
        "cropping": 4
    }, {
        "class": "SpatialExpansion",
        "spatial_mult": 2
    }, {
        "class": "Activation",
        "activation": "relu"
    }, {
        "class": "Sup3rConcat",
        "name": "topography"
    }, {
        "class": "Sup3rConcat",
        "name": "sza"
    }, {
        "class": "FlexiblePadding",
        "paddings": [[0, 0], [3, 3], [3, 3], [0, 0]],
        "mode": "REFLECT"
    }, {
        "class": "Conv2DTranspose",
        "filters": 2,
        "kernel_size": 3,
        "strides": 1,
        "activation": "relu"
    }, {
        "class": "Cropping2D",
        "cropping": 4
    }]

    gen_t_model = [{
        "class": "FlexiblePadding",
        "paddings": [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]],
        "mode": "REFLECT"
    }, {
        "class": "Conv3D", "filters": 1, "kernel_size": 3, "strides": 1
    }, {
        "class": "Cropping3D", "cropping": 2
    }, {
        "alpha": 0.2, "class": "LeakyReLU"
    }, {
        "class": "SpatioTemporalExpansion", "temporal_mult": 2,
        "temporal_method": "nearest"
    }, {
        "class": "FlexiblePadding",
        "paddings": [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]],
        "mode": "REFLECT"
    }, {
        "class": "Conv3D", "filters": 1, "kernel_size": 3, "strides": 1
    }, {
        "class": "Cropping3D", "cropping": 2
    }, {
        "class": "FlexiblePadding",
        "paddings": [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]],
        "mode": "REFLECT"
    }, {
        "class": "Conv3D", "filters": 36, "kernel_size": 3, "strides": 1
    }, {
        "class": "Cropping3D", "cropping": 2
    }, {
        "class": "SpatioTemporalExpansion", "spatial_mult": 3
    }, {
        "alpha": 0.2, "class": "LeakyReLU"
    }, {
        "class": "Sup3rConcat", "name": "sza"
    }, {
        "class": "FlexiblePadding",
        "paddings": [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]],
        "mode": "REFLECT"
    }, {
        "class": "Conv3D", "filters": 2, "kernel_size": 3, "strides": 1
    }, {
        "class": "Cropping3D", "cropping": 2
    }]

    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    s1_model = Sup3rGan(gen_s_model, fp_disc, learning_rate=1e-4)
    s1_model.meta['lr_features'] = [
        'U_100m', 'V_100m', 'topography', 'sza'
    ]
    s1_model.meta['hr_out_features'] = ['U_100m', 'V_100m']
    s1_model.meta['s_enhance'] = 2
    s1_model.meta['t_enhance'] = 1
    s1_model.meta['input_resolution'] = {'spatial': '48km',
                                         'temporal': '60min'}
    exo_tmp = {
        'topography': {
            'steps': [{'model': 0, 'combine_type': 'layer',
                       'data': np.ones((4, 20, 20, 1))}]},
        'sza': {
            'steps': [{'model': 0, 'combine_type': 'layer',
                       'data': np.ones((4, 20, 20, 1))}]}
    }
    _ = s1_model.generate(np.ones((4, 10, 10, 4)),
                          exogenous_data=exo_tmp)

    s2_model = Sup3rGan(gen_s_model, fp_disc, learning_rate=1e-4)
    s2_model.meta['lr_features'] = [
        'U_100m', 'V_100m', 'topography', 'sza'
    ]
    s2_model.meta['hr_out_features'] = ['U_100m', 'V_100m']
    s2_model.meta['s_enhance'] = 2
    s2_model.meta['t_enhance'] = 1
    s2_model.meta['input_resolution'] = {'spatial': '24km',
                                         'temporal': '60min'}
    _ = s2_model.generate(np.ones((4, 10, 10, 4)),
                          exogenous_data=exo_tmp)

    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    st_model = Sup3rGan(gen_t_model, fp_disc, learning_rate=1e-4)
    st_model.meta['lr_features'] = [
        'U_100m', 'V_100m', 'sza'
    ]
    st_model.meta['hr_out_features'] = ['U_100m', 'V_100m']
    st_model.meta['s_enhance'] = 3
    st_model.meta['t_enhance'] = 2
    st_model.meta['input_resolution'] = {'spatial': '12km',
                                         'temporal': '60min'}
    exo_tmp = {
        'sza': {
            'steps': [{'model': 0, 'combine_type': 'layer',
                       'data': np.ones((4, 30, 30, 12, 1))}]}
    }
    _ = st_model.generate(np.ones((4, 10, 10, 6, 3)),
                          exogenous_data=exo_tmp)

    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)

        st_out_dir = os.path.join(td, 'st_gan')
        s1_out_dir = os.path.join(td, 's1_gan')
        s2_out_dir = os.path.join(td, 's2_gan')
        st_model.save(st_out_dir)
        s1_model.save(s1_out_dir)
        s2_model.save(s2_out_dir)

        exo_kwargs = {
            'topography': {
                'file_paths': input_files,
                'source_file': FP_WTK,
                'target': target,
                'shape': shape,
                'cache_dir': td,
                'exo_resolution': {'spatial': '4km', 'temporal': '60min'},
                'steps': [{'model': 0, 'combine_type': 'input'},
                          {'model': 0, 'combine_type': 'layer'},
                          {'model': 1, 'combine_type': 'input'},
                          {'model': 1, 'combine_type': 'layer'}]
            },
            'sza': {
                'file_paths': input_files,
                'exo_handler': 'SzaExtract',
                'target': target,
                'shape': shape,
                'cache_dir': td,
                'exo_resolution': {'spatial': '4km', 'temporal': '60min'},
                'steps': [{'model': 0, 'combine_type': 'input'},
                          {'model': 0, 'combine_type': 'layer'},
                          {'model': 1, 'combine_type': 'input'},
                          {'model': 1, 'combine_type': 'layer'},
                          {'model': 2, 'combine_type': 'input'},
                          {'model': 2, 'combine_type': 'layer'}]
            }
        }

        model_kwargs = {
            'model_dirs': [s1_out_dir, s2_out_dir, st_out_dir]
        }
        out_files = os.path.join(td, 'out_{file_id}.h5')
        input_handler_kwargs = dict(target=target,
                                    shape=shape,
                                    temporal_slice=temporal_slice,
                                    worker_kwargs=dict(max_workers=1),
                                    overwrite_cache=True)

        handler = ForwardPassStrategy(
            input_files,
            model_kwargs=model_kwargs,
            model_class='MultiStepGan',
            fwp_chunk_shape=(4, 4, 8),
            spatial_pad=1,
            temporal_pad=1,
            input_handler_kwargs=input_handler_kwargs,
            out_pattern=out_files,
            worker_kwargs=dict(max_workers=1),
            exo_kwargs=exo_kwargs,
            max_nodes=1)
        forward_pass = ForwardPass(handler)
        forward_pass.run(handler, node_index=0)

        for fp in handler.out_files:
            assert os.path.exists(fp)

    shutil.rmtree('./exo_cache', ignore_errors=True)
