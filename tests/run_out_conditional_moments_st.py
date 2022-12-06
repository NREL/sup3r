# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN"""
import os

from sup3r import TEST_DATA_DIR
from test_out_conditional_moments import (test_out_loss,
                                          test_out_st_mom1,
                                          test_out_st_mom2,
                                          test_out_st_mom2_sep,
                                          test_out_st_mom1_sf,
                                          test_out_st_mom2_sf,
                                          test_out_st_mom2_sep_sf)

FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
TARGET_COORD = (39.01, -105.15)
FEATURES = ['U_100m', 'V_100m']


if __name__ == "__main__":

    test_out_loss(plot=True, model_dirs=['st_mom1/st_cond_mom'],
                  figureDir='st_mom1_loss')

    test_out_loss(plot=True, model_dirs=['st_mom2/st_cond_mom'],
                  figureDir='st_mom2_loss')

    test_out_loss(plot=True, model_dirs=['st_mom2_sep/st_cond_mom'],
                  figureDir='st_mom2_sep_loss')

    test_out_loss(plot=True, model_dirs=['st_mom1_sf/st_cond_mom'],
                  figureDir='st_mom1_sf_loss')

    test_out_loss(plot=True, model_dirs=['st_mom2_sf/st_cond_mom'],
                  figureDir='st_mom2_sf_loss')

    test_out_loss(plot=True, model_dirs=['st_mom2_sep_sf/st_cond_mom'],
                  figureDir='st_mom2_sep_sf_loss')

    test_out_st_mom1(plot=True, full_shape=(20, 20),
                     sample_shape=(12, 12, 24),
                     batch_size=1, n_batches=1,
                     s_enhance=3, t_enhance=4,
                     model_dir='st_mom1/st_cond_mom')

    test_out_st_mom2(plot=True, full_shape=(20, 20),
                     sample_shape=(12, 12, 24),
                     batch_size=1, n_batches=1,
                     s_enhance=3, t_enhance=4,
                     model_dir='st_mom2/st_cond_mom',
                     model_mom1_dir='st_mom1/st_cond_mom')

    test_out_st_mom2_sep(plot=True, full_shape=(20, 20),
                         sample_shape=(12, 12, 24),
                         batch_size=1, n_batches=1,
                         s_enhance=3, t_enhance=4,
                         model_dir='st_mom2_sep/st_cond_mom',
                         model_mom1_dir='st_mom1/st_cond_mom')

    test_out_st_mom1_sf(plot=True, full_shape=(20, 20),
                        sample_shape=(12, 12, 24),
                        batch_size=1, n_batches=1,
                        s_enhance=3, t_enhance=4,
                        model_dir='st_mom1_sf/st_cond_mom')

    test_out_st_mom2_sf(plot=True, full_shape=(20, 20),
                        sample_shape=(12, 12, 24),
                        batch_size=1, n_batches=1,
                        s_enhance=3, t_enhance=4,
                        model_mom1_dir='st_mom1_sf/st_cond_mom',
                        model_dir='st_mom2_sf/st_cond_mom')

    test_out_st_mom2_sep_sf(plot=True, full_shape=(20, 20),
                            sample_shape=(12, 12, 24),
                            batch_size=1, n_batches=1,
                            s_enhance=3, t_enhance=4,
                            model_mom1_dir='st_mom1_sf/st_cond_mom',
                            model_dir='st_mom2_sep_sf/st_cond_mom')
