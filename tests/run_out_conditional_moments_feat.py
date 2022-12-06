# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN"""
import os

from sup3r import TEST_DATA_DIR

from test_out_conditional_moments import (test_out_spatial_mom1,
                                          test_out_spatial_mom1_sf,
                                          test_out_spatial_mom2,
                                          test_out_spatial_mom2_sep,
                                          test_out_spatial_mom2_sf,
                                          test_out_spatial_mom2_sep_sf,
                                          test_out_loss)

FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
TARGET_COORD = (39.01, -105.15)
FEATURES = ['U_100m', 'V_100m', 'BVF2_200m']
TRAIN_FEATURES = ['BVF2_200m']
n_feat_in = len(FEATURES)
n_feat_out = len(FEATURES) - len(TRAIN_FEATURES)


if __name__ == "__main__":

    test_out_loss(plot=True, model_dirs=['s_mom1_feat/s_cond_mom'],
                  figureDir='feat_mom1_loss')

    test_out_loss(plot=True, model_dirs=['s_mom2_feat/s_cond_mom'],
                  figureDir='feat_mom2_loss')

    test_out_loss(plot=True, model_dirs=['s_mom1_sf_feat/s_cond_mom'],
                  figureDir='feat_mom1_sf_loss')

    test_out_loss(plot=True, model_dirs=['s_mom2_sf_feat/s_cond_mom'],
                  figureDir='feat_mom2_sf_loss')

    test_out_spatial_mom1(plot=True, full_shape=(20, 20),
                          sample_shape=(10, 10, 1),
                          batch_size=4, n_batches=2,
                          s_enhance=2,
                          FEATURES=FEATURES,
                          model_dir='s_mom1_feat/s_cond_mom',
                          TRAIN_FEATURES=TRAIN_FEATURES)

    test_out_spatial_mom2(plot=True, full_shape=(20, 20),
                          sample_shape=(10, 10, 1),
                          batch_size=4, n_batches=2,
                          s_enhance=2,
                          FEATURES=FEATURES,
                          model_dir='s_mom2_feat/s_cond_mom',
                          model_mom1_dir='s_mom1_feat/s_cond_mom',
                          TRAIN_FEATURES=TRAIN_FEATURES)

    test_out_spatial_mom2_sep(plot=True, full_shape=(20, 20),
                              sample_shape=(10, 10, 1),
                              batch_size=4, n_batches=2,
                              s_enhance=2,
                              FEATURES=FEATURES,
                              model_dir='s_mom2_sep_feat/s_cond_mom',
                              model_mom1_dir='s_mom1_feat/s_cond_mom',
                              TRAIN_FEATURES=TRAIN_FEATURES)

    test_out_spatial_mom1_sf(plot=True, full_shape=(20, 20),
                             sample_shape=(10, 10, 1),
                             batch_size=4, n_batches=2,
                             s_enhance=2,
                             FEATURES=FEATURES,
                             model_dir='s_mom1_sf_feat/s_cond_mom',
                             TRAIN_FEATURES=TRAIN_FEATURES)

    test_out_spatial_mom2_sf(plot=True, full_shape=(20, 20),
                             sample_shape=(10, 10, 1),
                             batch_size=4, n_batches=2,
                             s_enhance=2,
                             FEATURES=FEATURES,
                             model_mom1_dir='s_mom1_sf_feat/s_cond_mom',
                             model_dir='s_mom2_sf_feat/s_cond_mom',
                             TRAIN_FEATURES=TRAIN_FEATURES)

    test_out_spatial_mom2_sep_sf(
        plot=True,
        full_shape=(20, 20),
        sample_shape=(10, 10, 1),
        batch_size=4, n_batches=2,
        s_enhance=2,
        FEATURES=FEATURES,
        model_mom1_dir='s_mom1_sf_feat/s_cond_mom',
        model_dir='s_mom2_sep_sf_feat/s_cond_mom',
        TRAIN_FEATURES=TRAIN_FEATURES)
