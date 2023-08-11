# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN"""
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from rex import init_logger

from sup3r import TEST_DATA_DIR
from sup3r.preprocessing.data_handling import (
    DataHandlerH5,
    DataHandlerNC,
    DualDataHandler,
)
from sup3r.preprocessing.dual_batch_handling import SpatialDualBatchHandler

FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
FP_ERA = os.path.join(TEST_DATA_DIR, 'test_era5_co_2012.nc')
TARGET_COORD = (39.01, -105.15)
FEATURES = ['U_100m', 'V_100m']


def test_dual_data_handler(
    log=False, full_shape=(20, 20), sample_shape=(10, 10, 1), plot=True
):
    """Test basic spatial model training with only gen content loss."""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    # need to reduce the number of temporal examples to test faster
    hr_handler = DataHandlerH5(
        FP_WTK,
        FEATURES,
        target=TARGET_COORD,
        shape=full_shape,
        sample_shape=sample_shape,
        temporal_slice=slice(None, None, 10),
        worker_kwargs=dict(max_workers=1),
    )
    lr_handler = DataHandlerNC(
        FP_ERA,
        FEATURES,
        sample_shape=(sample_shape[0] // 2, sample_shape[1] // 2, 1),
        temporal_slice=slice(None, None, 10),
        worker_kwargs=dict(max_workers=1),
    )

    dual_handler = DualDataHandler(
        hr_handler, lr_handler, s_enhance=2, t_enhance=1, val_split=0.1
    )

    batch_handler = SpatialDualBatchHandler(
        [dual_handler], batch_size=2, s_enhance=2, n_batches=10
    )

    with tempfile.TemporaryDirectory() as td:
        td = '/home/bbenton/dual_data_handler_test_figs/'
        if plot:
            for i, batch in enumerate(batch_handler):
                fig, ax = plt.subplots(1, 2, figsize=(5, 10))
                fig.suptitle(f'High vs Low Res ({dual_handler.features[-1]})')
                ax[0].set_title('High Res')
                ax[0].imshow(np.mean(batch.high_res[..., -1], axis=0))
                ax[1].set_title('Low Res')
                ax[1].imshow(np.mean(batch.low_res[..., -1], axis=0))
                fig.savefig(f'{td}/high_vs_low_{str(i).zfill(3)}.png')
