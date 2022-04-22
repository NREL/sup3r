# -*- coding: utf-8 -*-
"""pytests for data handling with NSRDB files"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sup3r import TEST_DATA_DIR
from sup3r.preprocessing.data_handling import DataHandlerNsrdb
from sup3r.preprocessing.batch_handling import NsrdbBatchHandler


INPUT_FILE = os.path.join(TEST_DATA_DIR, 'test_nsrdb_co_2018.h5')
TARGET = (39.01, -105.13)
SHAPE = (20, 20)
FEATURES = ['clearsky_ratio', 'ghi', 'clearsky_ghi']


def test_handler(plot=False):
    """Test loading irrad data from NSRDB file and calculating clearsky ratio
    with NaN values for nighttime."""
    handler = DataHandlerNsrdb(INPUT_FILE, FEATURES,
                               target=TARGET, shape=SHAPE,
                               time_pruning=2,
                               time_roll=-7,
                               val_split=0.1,
                               temporal_sample_shape=24,
                               spatial_sample_shape=(20, 20),
                               max_extract_workers=1,
                               max_compute_workers=1)

    assert handler.val_data.shape[2] % 24 == 0

    # some of the raw clearsky ghi and clearsky ratio data should be loaded in
    # the handler as NaN
    assert np.isnan(handler.data).any()

    for _ in range(10):
        obs = handler.get_next()
        cs_ratio_profile = obs[0, 0, :, 0]
        cs_ghi_profile = obs[0, 0, :, 2]
        assert np.isnan(cs_ratio_profile[0]) & np.isnan(cs_ratio_profile[-1])
        assert np.isnan(cs_ghi_profile[0]) & np.isnan(cs_ghi_profile[-1])

        nan_mask = np.isnan(cs_ratio_profile)
        assert all((cs_ratio_profile <= 1)[~nan_mask])
        assert all((cs_ratio_profile >= 0)[~nan_mask])

        nan_mask = np.isnan(cs_ghi_profile)
        assert all((cs_ghi_profile <= 1200)[~nan_mask])
        assert all((cs_ghi_profile >= 0)[~nan_mask])

    if plot:
        obs = handler.get_next()
        for i in range(obs.shape[2]):
            _, axes = plt.subplots(1, 3, figsize=(15, 4))

            a = axes[0].imshow(obs[:, :, i, 0], vmin=0, vmax=1)
            plt.colorbar(a, ax=axes[0])
            axes[0].set_title('Clearsky Ratio')

            a = axes[1].imshow(obs[:, :, i, 1], vmin=0, vmax=1100)
            plt.colorbar(a, ax=axes[1])
            axes[1].set_title('GHI')

            a = axes[2].imshow(obs[:, :, i, 2], vmin=0, vmax=1100)
            plt.colorbar(a, ax=axes[2])
            axes[2].set_title('Clearsky GHI')

            plt.title(i)
            plt.savefig('./test_fig_{}.png'.format(i), dpi=300,
                        bbox_inches='tight')
            plt.close()
