# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN"""
import os

from sup3r import TEST_DATA_DIR

from test_train_conditional_moments import (test_train_st_mom1,
                                            test_train_st_mom1_sf,
                                            test_train_st_mom2,
                                            test_train_st_mom2_sf)

FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
TARGET_COORD = (39.01, -105.15)
FEATURES = ['U_100m', 'V_100m']

if __name__ == "__main__":
    test_train_st_mom1(n_epoch=2, log=True,
                       full_shape=(20, 20),
                       sample_shape=(12, 12, 24),
                       batch_size=2, n_batches=2,
                       out_dir_root='st_mom1',
                       FEATURES=FEATURES)
    test_train_st_mom1_sf(n_epoch=2, log=True,
                          full_shape=(20, 20),
                          sample_shape=(12, 12, 24),
                          batch_size=2, n_batches=2,
                          out_dir_root='st_mom1_sf',
                          FEATURES=FEATURES)
    test_train_st_mom2(n_epoch=2, log=True,
                       full_shape=(20, 20),
                       sample_shape=(12, 12, 24),
                       batch_size=2, n_batches=2,
                       out_dir_root='st_mom2',
                       model_mom1_dir='st_mom1/st_cond_mom',
                       FEATURES=FEATURES)
    test_train_st_mom2_sf(n_epoch=2, log=True,
                          full_shape=(20, 20),
                          sample_shape=(12, 12, 24),
                          batch_size=2, n_batches=2,
                          out_dir_root='st_mom2_sf',
                          model_mom1_dir='st_mom1_sf/st_cond_mom',
                          FEATURES=FEATURES)
    # pass
