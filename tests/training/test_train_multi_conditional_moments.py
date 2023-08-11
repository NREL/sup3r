# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN"""
import os
# import json
import numpy as np
import pytest
import tempfile
import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from rex import init_logger

from sup3r import TEST_DATA_DIR
from sup3r import CONFIG_DIR
from sup3r.models import MultiSup3rCondMom
from sup3r.preprocessing.data_handling import DataHandlerH5
from sup3r.preprocessing.conditional_moment_batch_handling import (
    SpatialBatchHandlerMom1,
    SpatialBatchHandlerMom1SF,
    SpatialBatchHandlerMom2,
    SpatialBatchHandlerMom2Sep,
    SpatialBatchHandlerMom2SF,
    SpatialBatchHandlerMom2SepSF,
    BatchHandlerMom1,
    BatchHandlerMom1SF,
    BatchHandlerMom2,
    BatchHandlerMom2Sep,
    BatchHandlerMom2SF,
    BatchHandlerMom2SepSF)


FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
TARGET_COORD = (39.01, -105.15)
FEATURES = ['U_100m', 'V_100m']
TRAIN_FEATURES = None



@pytest.mark.parametrize('FEATURES, end_t_padding',
                         [(['U_100m', 'V_100m'], False),
                          (['U_100m', 'V_100m'], True)])
def test_train_st_mom1(FEATURES,
                       end_t_padding,
                       log=False, full_shape=(20, 20),
                       sample_shape=(12, 12, 24), n_epoch=2,
                       batch_size=2, n_batches=2,
                       out_dir_root=None):
    """Test basic spatiotemporal model training
    for first conditional moment."""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen1 = os.path.join(CONFIG_DIR,
                          'spatiotemporal',
                          'gen_3x_1x_2f.json')
    fp_gen2 = os.path.join(CONFIG_DIR,
                          'spatiotemporal',
                          'gen_1x_4x_2f.json')

    MultiSup3rCondMom.seed()
    model = MultiSup3rCondMom(gen_layers_list=[fp_gen1, fp_gen2], gen_s_en_list=[3, 1], gen_t_en_list=[1, 4], learning_rate=1e-4)

    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 1),
                            val_split=0.005,
                            worker_kwargs=dict(max_workers=1))

    batch_handler = BatchHandlerMom1([handler], batch_size=batch_size,
                                     s_enhance=3, t_enhance=4,
                                     n_batches=n_batches,
                                     end_t_padding=end_t_padding)

    with tempfile.TemporaryDirectory() as td:
        if out_dir_root is None:
            out_dir_root = td
        model.train(batch_handler, n_epoch=n_epoch,
                    checkpoint_int=2,
                    out_dir=os.path.join(out_dir_root, 'test_{epoch}'))

        # test save/load functionality
        out_dir = os.path.join(out_dir_root, 'st_cond_mom')
        model.save(out_dir)

if __name__ == "__main__":
    test_train_st_mom1(['U_100m', 'V_100m'], end_t_padding=True, log=True)
