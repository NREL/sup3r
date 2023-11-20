# -*- coding: utf-8 -*-
"""pytests for data handling"""
import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import xarray as xr
from rex import Resource, init_logger

from sup3r import CONFIG_DIR, TEST_DATA_DIR
from sup3r.models import Sup3rGan
from sup3r.pipeline.forward_pass import ForwardPass, ForwardPassStrategy
from sup3r.qa.qa import Sup3rQa
from sup3r.qa.stats import Sup3rStatsMulti
from sup3r.qa.utilities import continuous_dist
from sup3r.utilities.pytest import make_fake_nc_files

FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
TARGET_COORD = (39.01, -105.15)
TRAIN_FEATURES = ['U_100m', 'V_100m', 'BVF2_200m']
MODEL_OUT_FEATURES = ['U_100m', 'V_100m']
FOUT_FEATURES = ['windspeed_100m', 'winddirection_100m']
INPUT_FILE = os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_00_00_00')
TARGET = (19.3, -123.5)
SHAPE = (8, 8)
TEMPORAL_SLICE = slice(None, None, 1)
FWP_CHUNK_SHAPE = (8, 8, int(1e6))
S_ENHANCE = 3
T_ENHANCE = 4


def test_qa_nc():
    """Test forward pass strategy output for netcdf write."""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 10, 10, 6, len(TRAIN_FEATURES))))
    model.meta['lr_features'] = TRAIN_FEATURES
    model.meta['hr_out_features'] = MODEL_OUT_FEATURES
    model.meta['s_enhance'] = 3
    model.meta['t_enhance'] = 4
    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        out_files = os.path.join(td, 'out_{file_id}.nc')
        strategy = ForwardPassStrategy(
            input_files, model_kwargs={'model_dir': out_dir},
            fwp_chunk_shape=FWP_CHUNK_SHAPE,
            spatial_pad=1, temporal_pad=1,
            input_handler_kwargs=dict(target=TARGET, shape=SHAPE,
                                      temporal_slice=TEMPORAL_SLICE,
                                      worker_kwargs=dict(max_workers=1)),
            out_pattern=out_files,
            worker_kwargs=dict(max_workers=1),
            max_nodes=1)

        forward_pass = ForwardPass(strategy)
        forward_pass.run(strategy, node_index=0)

        assert len(strategy.out_files) == 1

        args = [input_files, strategy.out_files[0]]
        qa_fp = os.path.join(td, 'qa.h5')
        kwargs = dict(s_enhance=S_ENHANCE, t_enhance=T_ENHANCE,
                      temporal_coarsening_method='subsample',
                      temporal_slice=TEMPORAL_SLICE,
                      target=TARGET, shape=SHAPE,
                      qa_fp=qa_fp, save_sources=True,
                      worker_kwargs=dict(max_workers=1))
        with Sup3rQa(*args, **kwargs) as qa:
            data = qa.output_handler[qa.features[0]]
            data = qa.get_dset_out(qa.features[0])
            data = qa.coarsen_data(0, qa.features[0], data)

            assert isinstance(qa.meta, pd.DataFrame)
            assert isinstance(qa.time_index, pd.DatetimeIndex)
            for i in range(3):
                assert data.shape[i] == qa.source_handler.data.shape[i]

            qa.run()

            assert os.path.exists(qa_fp)

            with xr.open_dataset(strategy.out_files[0]) as fwp_out:
                with Resource(qa_fp) as qa_out:

                    for dset in MODEL_OUT_FEATURES:
                        idf = qa.source_handler.features.index(dset)
                        qa_true = qa_out[dset + '_true'].flatten()
                        qa_syn = qa_out[dset + '_synthetic'].flatten()
                        qa_diff = qa_out[dset + '_error'].flatten()

                        wtk_source = qa.source_handler.data[..., idf]
                        wtk_source = np.transpose(wtk_source, axes=(2, 0, 1))
                        wtk_source = wtk_source.flatten()

                        fwp_data = fwp_out[dset].values
                        fwp_data = np.transpose(fwp_data, axes=(1, 2, 0))
                        fwp_data = qa.coarsen_data(idf, dset, fwp_data)
                        fwp_data = np.transpose(fwp_data, axes=(2, 0, 1))
                        fwp_data = fwp_data.flatten()

                        test_diff = fwp_data - wtk_source

                        assert np.allclose(qa_true, wtk_source, atol=0.01)
                        assert np.allclose(qa_syn, fwp_data, atol=0.01)
                        assert np.allclose(test_diff, qa_diff, atol=0.01)


def test_qa_h5():
    """Test the QA module with forward pass output to h5 file."""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 10, 10, 6, len(TRAIN_FEATURES))))
    model.meta['lr_features'] = TRAIN_FEATURES
    model.meta['hr_out_features'] = MODEL_OUT_FEATURES
    model.meta['s_enhance'] = 3
    model.meta['t_enhance'] = 4
    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        out_files = os.path.join(td, 'out_{file_id}.h5')
        input_handler_kwargs = dict(target=TARGET, shape=SHAPE,
                                    temporal_slice=TEMPORAL_SLICE,
                                    worker_kwargs=dict(max_workers=1))
        strategy = ForwardPassStrategy(
            input_files, model_kwargs={'model_dir': out_dir},
            fwp_chunk_shape=FWP_CHUNK_SHAPE,
            spatial_pad=1, temporal_pad=1,
            input_handler_kwargs=input_handler_kwargs,
            out_pattern=out_files,
            worker_kwargs=dict(max_workers=1),
            max_nodes=1)

        forward_pass = ForwardPass(strategy)
        forward_pass.run(strategy, node_index=0)

        assert len(strategy.out_files) == 1

        qa_fp = os.path.join(td, 'qa.h5')
        args = [input_files, strategy.out_files[0]]
        kwargs = dict(s_enhance=S_ENHANCE, t_enhance=T_ENHANCE,
                      temporal_coarsening_method='subsample',
                      temporal_slice=TEMPORAL_SLICE,
                      target=TARGET, shape=SHAPE,
                      qa_fp=qa_fp, save_sources=True,
                      worker_kwargs=dict(max_workers=1))
        with Sup3rQa(*args, **kwargs) as qa:
            data = qa.output_handler[qa.features[0]]
            data = qa.get_dset_out(qa.features[0])
            data = qa.coarsen_data(0, qa.features[0], data)

            assert isinstance(qa.meta, pd.DataFrame)
            assert isinstance(qa.time_index, pd.DatetimeIndex)
            for i in range(3):
                assert data.shape[i] == qa.source_handler.data.shape[i]

            qa.run()

            assert os.path.exists(qa_fp)

            with Resource(strategy.out_files[0]) as fwp_out:
                with Resource(qa_fp) as qa_out:

                    for dset in FOUT_FEATURES:
                        idf = qa.source_handler.features.index(dset)
                        qa_true = qa_out[dset + '_true'].flatten()
                        qa_syn = qa_out[dset + '_synthetic'].flatten()
                        qa_diff = qa_out[dset + '_error'].flatten()

                        wtk_source = qa.source_handler.data[..., idf]
                        wtk_source = np.transpose(wtk_source, axes=(2, 0, 1))
                        wtk_source = wtk_source.flatten()

                        shape = (qa.source_handler.shape[0] * S_ENHANCE,
                                 qa.source_handler.shape[1] * S_ENHANCE,
                                 qa.source_handler.shape[2] * T_ENHANCE)
                        fwp_data = np.transpose(fwp_out[dset])
                        fwp_data = fwp_data.reshape(shape)
                        fwp_data = qa.coarsen_data(idf, dset, fwp_data)
                        fwp_data = np.transpose(fwp_data, axes=(2, 0, 1))
                        fwp_data = fwp_data.flatten()

                        test_diff = fwp_data - wtk_source

                        assert np.allclose(qa_true, wtk_source, atol=0.01)
                        assert np.allclose(qa_syn, fwp_data, atol=0.01)
                        assert np.allclose(test_diff, qa_diff, atol=0.01)


def test_stats(log=False):
    """Test the WindStats module with forward pass output to h5 file."""

    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 10, 10, 6, len(TRAIN_FEATURES))))
    model.meta['lr_features'] = TRAIN_FEATURES
    model.meta['hr_out_features'] = MODEL_OUT_FEATURES
    model.meta['s_enhance'] = 3
    model.meta['t_enhance'] = 4
    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        out_files = os.path.join(td, 'out_{file_id}.h5')
        strategy = ForwardPassStrategy(
            input_files, model_kwargs={'model_dir': out_dir},
            fwp_chunk_shape=(100, 100, 100),
            spatial_pad=1, temporal_pad=1,
            input_handler_kwargs=dict(temporal_slice=TEMPORAL_SLICE,
                                      worker_kwargs=dict(max_workers=1)),
            out_pattern=out_files,
            worker_kwargs=dict(max_workers=1),
            max_nodes=1)

        forward_pass = ForwardPass(strategy)
        forward_pass.run_chunk()

        qa_fp = os.path.join(td, 'stats.pkl')
        features = ['U_100m', 'V_100m', 'vorticity_100m']
        include_stats = ['direct', 'time_derivative', 'gradient',
                         'avg_spectrum_k']
        kwargs = dict(features=features, shape=(4, 4),
                      target=(19.4, -123.4),
                      s_enhance=S_ENHANCE, t_enhance=T_ENHANCE,
                      synth_t_slice=TEMPORAL_SLICE,
                      qa_fp=qa_fp, include_stats=include_stats,
                      worker_kwargs=dict(max_workers=1), n_bins=10,
                      get_interp=True, max_values={'time_derivative': 10},
                      max_delta=2)
        with Sup3rStatsMulti(lr_file_paths=input_files,
                             synth_file_paths=strategy.out_files[0],
                             **kwargs) as qa:
            qa.run()
            assert os.path.exists(qa_fp)
            with open(qa_fp, 'rb') as fh:
                qa_out = pickle.load(fh)
                names = ['low_res', 'interp', 'synth']
                assert all(name in qa_out for name in names)
                for key in qa_out:
                    assert all(feature in qa_out[key] for feature in features)
                    for feature in features:
                        assert all(metric in qa_out[key][feature]
                                   for metric in include_stats)


def test_continuous_dist():
    """Test distribution interpolation function"""

    a = np.linspace(-6, 6, 10)
    counts, centers = continuous_dist(a, bins=20, range=[-10, 10])
    assert not all(np.isnan(counts))
    assert centers[0] < -9.0
    assert centers[-1] > 9.0
