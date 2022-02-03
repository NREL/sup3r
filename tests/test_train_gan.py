# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN"""
import os
import numpy as np

from rex import init_logger

from sup3r import TEST_DATA_DIR
from sup3r import CONFIG_DIR
from sup3r.models.models import SpatialGan
from sup3r.data_handling.preprocessing import (DataHandler,
                                               SpatialBatchHandler)


input_file = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
target = (39.01, -105.15)
shape = (20, 20)
features = ['windspeed_100m', 'winddirection_100m']


def test_train_spatial(log=False):
    """Test basic model training with only gen content loss."""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')

    model = SpatialGan(fp_gen, fp_disc,
                       weight_gen_content=1.0,
                       weight_gen_advers=0.0,
                       weight_disc=0.0,
                       learning_rate=1e-4)

    data_handler = DataHandler([input_file], target, shape, features,
                               max_delta=20)
    reduced_data = data_handler.data[:, :, ::100, :]
    batch_handler = SpatialBatchHandler(reduced_data, batch_size=32,
                                        spatial_res=2)

    model.train(batch_handler, n_epoch=4)
    assert len(model.history) == 4
    assert (np.diff(model.history['training_loss'].values) < 0).all()
