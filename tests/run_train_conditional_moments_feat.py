# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN"""
import os

from sup3r import TEST_DATA_DIR
from test_train_conditional_moments import (test_train_spatial_mom1,
                                            test_train_spatial_mom2,
                                            test_train_spatial_mom2_sep,
                                            test_train_spatial_mom1_sf,
                                            test_train_spatial_mom2_sf,
                                            test_train_spatial_mom2_sep_sf)


FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
TARGET_COORD = (39.01, -105.15)
FEATURES = ['U_100m', 'V_100m', 'BVF2_200m']
TRAIN_FEATURES = ['BVF2_200m']


if __name__ == "__main__":
    test_train_spatial_mom1(n_epoch=2, log=True, full_shape=(20, 20),
                            sample_shape=(10, 10, 1),
                            batch_size=8, n_batches=5,
                            out_dir_root='s_mom1_feat',
                            FEATURES=FEATURES,
                            TRAIN_FEATURES=TRAIN_FEATURES,
                            s_padding=None,
                            t_padding=None)

    test_train_spatial_mom2(n_epoch=2, log=True, full_shape=(20, 20),
                            sample_shape=(10, 10, 1),
                            batch_size=8, n_batches=5,
                            out_dir_root='s_mom2_feat',
                            model_mom1_dir='s_mom1_feat/spatial_cond_mom',
                            FEATURES=FEATURES,
                            TRAIN_FEATURES=TRAIN_FEATURES,
                            s_padding=None,
                            t_padding=None)

    test_train_spatial_mom2_sep(n_epoch=2, log=True, full_shape=(20, 20),
                                sample_shape=(10, 10, 1),
                                batch_size=8, n_batches=5,
                                out_dir_root='s_mom2_sep_feat',
                                FEATURES=FEATURES,
                                TRAIN_FEATURES=TRAIN_FEATURES,
                                s_padding=None,
                                t_padding=None)

    test_train_spatial_mom1_sf(n_epoch=2, log=True,
                               full_shape=(20, 20),
                               sample_shape=(10, 10, 1),
                               batch_size=8, n_batches=5,
                               out_dir_root='s_mom1_sf_feat',
                               FEATURES=FEATURES,
                               TRAIN_FEATURES=TRAIN_FEATURES,
                               s_padding=None,
                               t_padding=None)

    test_train_spatial_mom2_sf(n_epoch=2, log=True,
                               full_shape=(20, 20),
                               sample_shape=(10, 10, 1),
                               batch_size=8, n_batches=5,
                               out_dir_root='s_mom2_sf_feat',
                               model_mom1_dir=os.path.join('s_mom1_sf_feat',
                                                           'spatial_cond_mom'),
                               FEATURES=FEATURES,
                               TRAIN_FEATURES=TRAIN_FEATURES,
                               s_padding=None,
                               t_padding=None)

    test_train_spatial_mom2_sep_sf(n_epoch=2, log=True,
                                   full_shape=(20, 20),
                                   sample_shape=(10, 10, 1),
                                   batch_size=8, n_batches=5,
                                   out_dir_root='s_mom2_sep_sf_feat',
                                   FEATURES=FEATURES,
                                   TRAIN_FEATURES=TRAIN_FEATURES,
                                   s_padding=None,
                                   t_padding=None)
    # pass
