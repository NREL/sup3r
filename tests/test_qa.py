# -*- coding: utf-8 -*-
"""pytests for data handling"""
import os
import tempfile
import pandas as pd
import numpy as np
from rex import Resource
import xarray as xr

from sup3r import TEST_DATA_DIR, CONFIG_DIR
from sup3r.pipeline.forward_pass import ForwardPass, ForwardPassStrategy
from sup3r.models import Sup3rGan
from sup3r.utilities.test_utils import make_fake_nc_files
from sup3r.qa.qa import Sup3rQa


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
    model.meta['training_features'] = TRAIN_FEATURES
    model.meta['output_features'] = MODEL_OUT_FEATURES
    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        out_files = os.path.join(td, 'out_{file_id}.nc')
        strategy = ForwardPassStrategy(
            input_files, model_args=out_dir,
            s_enhance=S_ENHANCE, t_enhance=T_ENHANCE,
            fwp_chunk_shape=FWP_CHUNK_SHAPE,
            spatial_pad=1, temporal_pad=1,
            target=TARGET, shape=SHAPE,
            temporal_slice=TEMPORAL_SLICE,
            out_pattern=out_files,
            max_workers=1)

        forward_pass = ForwardPass(strategy)
        forward_pass.run()

        assert len(strategy.out_files) == 1

        args = [input_files, strategy.out_files[0]]
        qa_fp = os.path.join(td, 'qa.h5')
        kwargs = dict(s_enhance=S_ENHANCE, t_enhance=T_ENHANCE,
                      temporal_coarsening_method='subsample',
                      temporal_slice=TEMPORAL_SLICE,
                      target=TARGET, shape=SHAPE,
                      qa_fp=qa_fp, save_sources=True,
                      max_workers=1)
        with Sup3rQa(*args, **kwargs) as qa:
            data = qa.output_handler[qa.features[0]]
            data = qa.get_dset_out(qa.features[0])
            data = qa.coarsen_data(data)

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
                        fwp_data = qa.coarsen_data(fwp_data)
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
    model.meta['training_features'] = TRAIN_FEATURES
    model.meta['output_features'] = MODEL_OUT_FEATURES
    with tempfile.TemporaryDirectory() as td:
        input_files = make_fake_nc_files(td, INPUT_FILE, 8)
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        out_files = os.path.join(td, 'out_{file_id}.h5')
        strategy = ForwardPassStrategy(
            input_files, model_args=out_dir,
            s_enhance=S_ENHANCE, t_enhance=T_ENHANCE,
            fwp_chunk_shape=FWP_CHUNK_SHAPE,
            spatial_pad=1, temporal_pad=1,
            target=TARGET, shape=SHAPE,
            temporal_slice=TEMPORAL_SLICE,
            out_pattern=out_files,
            max_workers=1)

        forward_pass = ForwardPass(strategy)
        forward_pass.run()

        assert len(strategy.out_files) == 1

        qa_fp = os.path.join(td, 'qa.h5')
        args = [input_files, strategy.out_files[0]]
        kwargs = dict(s_enhance=S_ENHANCE, t_enhance=T_ENHANCE,
                      temporal_coarsening_method='subsample',
                      temporal_slice=TEMPORAL_SLICE,
                      target=TARGET, shape=SHAPE,
                      qa_fp=qa_fp, save_sources=True,
                      max_workers=1)
        with Sup3rQa(*args, **kwargs) as qa:
            data = qa.output_handler[qa.features[0]]
            data = qa.get_dset_out(qa.features[0])
            data = qa.coarsen_data(data)

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
                        fwp_data = qa.coarsen_data(fwp_data)
                        fwp_data = np.transpose(fwp_data, axes=(2, 0, 1))
                        fwp_data = fwp_data.flatten()

                        test_diff = fwp_data - wtk_source

                        assert np.allclose(qa_true, wtk_source, atol=0.01)
                        assert np.allclose(qa_syn, fwp_data, atol=0.01)
                        assert np.allclose(test_diff, qa_diff, atol=0.01)