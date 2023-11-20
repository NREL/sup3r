# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN"""
import os
import pytest
import json
import numpy as np
from pandas import read_csv

from sup3r import TEST_DATA_DIR
from sup3r import CONFIG_DIR
from sup3r.models import Sup3rCondMom
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
from sup3r.utilities.utilities import (spatial_simple_enhancing,
                                       temporal_simple_enhancing)

FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
TARGET_COORD = (39.01, -105.15)
FEATURES = ['U_100m', 'V_100m']
TRAIN_FEATURES = None


@pytest.mark.parametrize('FEATURES, TRAIN_FEATURES',
                         [(['U_100m', 'V_100m'],
                           None),
                          (['U_100m', 'V_100m', 'BVF2_200m'],
                           ['BVF2_200m'])])
def test_out_s_mom1(FEATURES, TRAIN_FEATURES,
                    plot=False, full_shape=(20, 20),
                    sample_shape=(10, 10, 1),
                    batch_size=4, n_batches=4,
                    s_enhance=2, model_dir=None):
    """Test basic spatial model outputing."""
    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            lr_only_features=TRAIN_FEATURES,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 10),
                            val_split=0,
                            worker_kwargs=dict(max_workers=1))

    batch_handler = SpatialBatchHandlerMom1([handler],
                                            batch_size=batch_size,
                                            s_enhance=s_enhance,
                                            n_batches=n_batches)

    # Load Model
    if model_dir is None:
        fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
        model = Sup3rCondMom(fp_gen)
    else:
        fp_gen = os.path.join(model_dir, 'model_params.json')
        model = Sup3rCondMom(fp_gen).load(model_dir)

    # Feature counting
    n_feat_in = len(FEATURES)
    n_train_features = (len(TRAIN_FEATURES)
                        if isinstance(TRAIN_FEATURES, list)
                        else 0)
    n_feat_out = len(FEATURES) - n_train_features

    # Check sizes
    for batch in batch_handler:
        assert batch.high_res.shape == (batch_size, sample_shape[0],
                                        sample_shape[1], n_feat_out)
        assert batch.output.shape == (batch_size, sample_shape[0],
                                      sample_shape[1], n_feat_out)
        assert batch.low_res.shape == (batch_size,
                                       sample_shape[0] // s_enhance,
                                       sample_shape[1] // s_enhance, n_feat_in)
        out = model._tf_generate(batch.low_res)
        assert out.shape == (batch_size, sample_shape[0], sample_shape[1],
                             n_feat_out)
        break
    if plot:
        import matplotlib.pyplot as plt
        from sup3r.utilities.plotting import (plot_multi_contour,
                                              make_movie)
        figureFolder = 'Figures'
        os.makedirs(figureFolder, exist_ok=True)
        movieFolder = os.path.join(figureFolder, 'Movie')
        os.makedirs(movieFolder, exist_ok=True)
        mom_name = r'$\mathbb{E}$(HR|LR)'
        n_snap = 0
        for p, batch in enumerate(batch_handler):
            out = model.generate(batch.low_res,
                                 norm_in=False,
                                 un_norm_out=False)
            for i in range(batch.output.shape[0]):
                lr = (batch.low_res[i, :, :, 0] * batch_handler.stds[0]
                      + batch_handler.means[0])
                hr = (batch.output[i, :, :, 0] * batch_handler.stds[0]
                      + batch_handler.means[0])
                gen = (out[i, :, :, 0] * batch_handler.stds[0]
                       + batch_handler.means[0])
                fig = plot_multi_contour(
                    [lr, hr, gen],
                    [0, batch.output.shape[1]],
                    [0, batch.output.shape[2]],
                    ['U [m/s]', 'U [m/s]', 'U [m/s]'],
                    ['LR', 'HR', mom_name],
                    ['x [m]', 'x [m]', 'x [m]'],
                    ['y [m]', 'y [m]', 'y [m]'],
                    [np.amin(lr), np.amin(hr), np.amin(hr)],
                    [np.amax(lr), np.amax(hr), np.amax(hr)],
                )
                fig.savefig(os.path.join(movieFolder,
                                         "im_{}.png".format(n_snap)),
                            dpi=100, bbox_inches='tight')
                plt.close(fig)
                n_snap += 1
            if p > 4:
                break
        make_movie(n_snap, movieFolder, os.path.join(figureFolder, 'mom1.gif'),
                   fps=6)


@pytest.mark.parametrize('FEATURES, TRAIN_FEATURES',
                         [(['U_100m', 'V_100m'],
                           None),
                          (['U_100m', 'V_100m', 'BVF2_200m'],
                           ['BVF2_200m'])])
def test_out_s_mom1_sf(FEATURES, TRAIN_FEATURES,
                       plot=False, full_shape=(20, 20),
                       sample_shape=(10, 10, 1),
                       batch_size=4, n_batches=4,
                       s_enhance=2, model_dir=None):
    """Test basic spatial model outputing."""
    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            lr_only_features=TRAIN_FEATURES,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 10),
                            val_split=0,
                            worker_kwargs=dict(max_workers=1))

    batch_handler = SpatialBatchHandlerMom1SF([handler],
                                              batch_size=batch_size,
                                              s_enhance=s_enhance,
                                              n_batches=n_batches)

    # Load Model
    if model_dir is None:
        fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
        model = Sup3rCondMom(fp_gen)
    else:
        fp_gen = os.path.join(model_dir, 'model_params.json')
        model = Sup3rCondMom(fp_gen).load(model_dir)

    if plot:
        import matplotlib.pyplot as plt
        from sup3r.utilities.plotting import plot_multi_contour, make_movie
        figureFolder = 'Figures'
        os.makedirs(figureFolder, exist_ok=True)
        movieFolder = os.path.join(figureFolder, 'Movie')
        os.makedirs(movieFolder, exist_ok=True)
        mom_name = r'$\mathbb{E}$(HR|LR)'
        mom_name2 = r'$\mathbb{E}$(SF|LR)'
        n_snap = 0
        for p, batch in enumerate(batch_handler):
            out = model.generate(batch.low_res,
                                 norm_in=False,
                                 un_norm_out=False)
            for i in range(batch.output.shape[0]):
                lr = (batch.low_res[i, :, :, 0] * batch_handler.stds[0]
                      + batch_handler.means[0])
                blr_aug_shape = (1,) + lr.shape + (1,)
                blr_aug = np.reshape(batch.low_res[i, :, :, 0],
                                     blr_aug_shape)
                up_lr = spatial_simple_enhancing(blr_aug,
                                                 s_enhance=s_enhance)
                up_lr = up_lr[0, :, :, 0]
                hr = (batch.high_res[i, :, :, 0]
                      * batch_handler.stds[0]
                      + batch_handler.means[0])
                sf = (batch.output[i, :, :, 0]
                      * batch_handler.stds[0])
                sf_pred = (out[i, :, :, 0]
                           * batch_handler.stds[0])
                hr_pred = (up_lr
                           * batch_handler.stds[0]
                           + batch_handler.means[0]
                           + sf_pred)
                fig = plot_multi_contour(
                    [lr, hr, hr_pred, sf, sf_pred],
                    [0, batch.output.shape[1]],
                    [0, batch.output.shape[2]],
                    ['U [m/s]', 'U [m/s]', 'U [m/s]',
                     'U [m/s]', 'U [m/s]'],
                    ['LR', 'HR', mom_name, 'SF', mom_name2],
                    ['x [m]', 'x [m]', 'x [m]', 'x [m]', 'x [m]'],
                    ['y [m]', 'y [m]', 'y [m]', 'y [m]', 'y [m]'],
                    [np.amin(lr), np.amin(hr),
                     np.amin(hr), np.amin(sf),
                     np.amin(sf)],
                    [np.amax(lr), np.amax(hr),
                     np.amax(hr), np.amax(sf),
                     np.amax(sf)],
                )
                fig.savefig(os.path.join(movieFolder,
                                         "im_{}.png".format(n_snap)),
                            dpi=100, bbox_inches='tight')
                plt.close(fig)
                n_snap += 1
            if p > 4:
                break
        make_movie(n_snap, movieFolder,
                   os.path.join(figureFolder, 'mom1_sf.gif'),
                   fps=6)


@pytest.mark.parametrize('FEATURES, TRAIN_FEATURES',
                         [(['U_100m', 'V_100m'],
                           None),
                          (['U_100m', 'V_100m', 'BVF2_200m'],
                           ['BVF2_200m'])])
def test_out_s_mom2(FEATURES, TRAIN_FEATURES,
                    plot=False, full_shape=(20, 20),
                    sample_shape=(10, 10, 1),
                    batch_size=4, n_batches=4,
                    s_enhance=2, model_dir=None,
                    model_mom1_dir=None):
    """Test basic spatial model outputing."""
    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            lr_only_features=TRAIN_FEATURES,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 10),
                            val_split=0,
                            worker_kwargs=dict(max_workers=1))

    # Load Mom 1 Model
    if model_mom1_dir is None:
        fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
        model_mom1 = Sup3rCondMom(fp_gen)
    else:
        fp_gen = os.path.join(model_mom1_dir, 'model_params.json')
        model_mom1 = Sup3rCondMom(fp_gen).load(model_mom1_dir)

    batch_handler = SpatialBatchHandlerMom2([handler],
                                            batch_size=batch_size,
                                            s_enhance=s_enhance,
                                            n_batches=n_batches,
                                            model_mom1=model_mom1)

    # Load Mom2 Model
    if model_dir is None:
        fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
        model = Sup3rCondMom(fp_gen)
    else:
        fp_gen = os.path.join(model_dir, 'model_params.json')
        model = Sup3rCondMom(fp_gen).load(model_dir)

    if plot:
        import matplotlib.pyplot as plt
        from sup3r.utilities.plotting import plot_multi_contour, make_movie
        figureFolder = 'Figures'
        os.makedirs(figureFolder, exist_ok=True)
        movieFolder = os.path.join(figureFolder, 'Movie')
        os.makedirs(movieFolder, exist_ok=True)
        mom_name = r'$\sigma$(HR|LR)'
        hr_name = r'|HR - $\mathbb{E}$(HR|LR)|'
        n_snap = 0
        for p, batch in enumerate(batch_handler):
            out = np.clip(model.generate(batch.low_res,
                                         norm_in=False,
                                         un_norm_out=False),
                          a_min=0, a_max=None)
            for i in range(batch.output.shape[0]):
                lr = (batch.low_res[i, :, :, 0] * batch_handler.stds[0]
                      + batch_handler.means[0])
                hr = (batch.high_res[i, :, :, 0] * batch_handler.stds[0]
                      + batch_handler.means[0])
                hr_to_mean = np.sqrt(batch.output[i, :, :, 0]
                                     * batch_handler.stds[0]**2)
                sigma = np.sqrt(out[i, :, :, 0]
                                * batch_handler.stds[0]**2)
                fig = plot_multi_contour(
                    [lr, hr, hr_to_mean, sigma],
                    [0, batch.output.shape[1]],
                    [0, batch.output.shape[2]],
                    ['U [m/s]', 'U [m/s]', 'U [m/s]', 'U [m/s]'],
                    ['LR', 'HR', hr_name, mom_name],
                    ['x [m]', 'x [m]', 'x [m]', 'x [m]'],
                    ['y [m]', 'y [m]', 'y [m]', 'y [m]'],
                    [np.amin(lr), np.amin(hr),
                     np.amin(hr_to_mean), np.amin(sigma)],
                    [np.amax(lr), np.amax(hr),
                     np.amax(hr_to_mean), np.amax(sigma)],
                )
                fig.savefig(os.path.join(movieFolder,
                                         "im_{}.png".format(n_snap)),
                            dpi=100, bbox_inches='tight')
                plt.close(fig)
                n_snap += 1
            if p > 4:
                break
        make_movie(n_snap, movieFolder, os.path.join(figureFolder, 'mom2.gif'),
                   fps=6)


@pytest.mark.parametrize('FEATURES, TRAIN_FEATURES',
                         [(['U_100m', 'V_100m'],
                           None),
                          (['U_100m', 'V_100m', 'BVF2_200m'],
                           ['BVF2_200m'])])
def test_out_s_mom2_sf(FEATURES, TRAIN_FEATURES,
                       plot=False, full_shape=(20, 20),
                       sample_shape=(10, 10, 1),
                       batch_size=4, n_batches=4,
                       s_enhance=2, model_dir=None,
                       model_mom1_dir=None):
    """Test basic spatial model outputing."""
    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            lr_only_features=TRAIN_FEATURES,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 10),
                            val_split=0,
                            worker_kwargs=dict(max_workers=1))

    # Load Mom 1 Model
    if model_mom1_dir is None:
        fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
        model_mom1 = Sup3rCondMom(fp_gen)
    else:
        fp_gen = os.path.join(model_mom1_dir, 'model_params.json')
        model_mom1 = Sup3rCondMom(fp_gen).load(model_mom1_dir)

    batch_handler = SpatialBatchHandlerMom2SF([handler],
                                              batch_size=batch_size,
                                              s_enhance=s_enhance,
                                              n_batches=n_batches,
                                              model_mom1=model_mom1)

    # Load Mom2 Model
    if model_dir is None:
        fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
        model = Sup3rCondMom(fp_gen)
    else:
        fp_gen = os.path.join(model_dir, 'model_params.json')
        model = Sup3rCondMom(fp_gen).load(model_dir)

    if plot:
        import matplotlib.pyplot as plt
        from sup3r.utilities.plotting import plot_multi_contour, make_movie
        figureFolder = 'Figures'
        os.makedirs(figureFolder, exist_ok=True)
        movieFolder = os.path.join(figureFolder, 'Movie')
        os.makedirs(movieFolder, exist_ok=True)
        mom_name1 = r'|SF - $\mathbb{E}$(SF|LR)|'
        mom_name2 = r'$\sigma$(SF|LR)'
        n_snap = 0
        for p, batch in enumerate(batch_handler):
            out = np.clip(model.generate(batch.low_res,
                                         norm_in=False,
                                         un_norm_out=False),
                          a_min=0, a_max=None)
            for i in range(batch.output.shape[0]):
                lr = (batch.low_res[i, :, :, 0] * batch_handler.stds[0]
                      + batch_handler.means[0])
                blr_aug_shape = (1,) + lr.shape + (1,)
                blr_aug = np.reshape(batch.low_res[i, :, :, 0],
                                     blr_aug_shape)
                up_lr = spatial_simple_enhancing(blr_aug,
                                                 s_enhance=s_enhance)
                up_lr = up_lr[0, :, :, 0]
                hr = (batch.high_res[i, :, :, 0]
                      * batch_handler.stds[0]
                      + batch_handler.means[0])
                sf = (hr
                      - up_lr
                      * batch_handler.stds[0]
                      - batch_handler.means[0])
                sf_to_mean = np.sqrt(batch.output[i, :, :, 0]
                                     * batch_handler.stds[0]**2)
                sigma = np.sqrt(out[i, :, :, 0]
                                * batch_handler.stds[0]**2)
                fig = plot_multi_contour(
                    [lr, hr, sf, sf_to_mean, sigma],
                    [0, batch.output.shape[1]],
                    [0, batch.output.shape[2]],
                    ['U [m/s]', 'U [m/s]', 'U [m/s]',
                     'U [m/s]', 'U [m/s]'],
                    ['LR', 'HR', 'SF', mom_name1, mom_name2],
                    ['x [m]', 'x [m]', 'x [m]', 'x [m]', 'x [m]'],
                    ['y [m]', 'y [m]', 'y [m]', 'y [m]', 'y [m]'],
                    [np.amin(lr), np.amin(hr),
                     np.amin(sf), np.amin(sf_to_mean),
                     np.amin(sigma)],
                    [np.amax(lr), np.amax(hr),
                     np.amax(sf), np.amax(sf_to_mean),
                     np.amax(sigma)],
                )
                fig.savefig(os.path.join(movieFolder,
                                         "im_{}.png".format(n_snap)),
                            dpi=100, bbox_inches='tight')
                plt.close(fig)
                n_snap += 1
            if p > 4:
                break
        make_movie(n_snap, movieFolder,
                   os.path.join(figureFolder, 'mom2_sf.gif'),
                   fps=6)


@pytest.mark.parametrize('FEATURES, TRAIN_FEATURES',
                         [(['U_100m', 'V_100m'],
                           None),
                          (['U_100m', 'V_100m', 'BVF2_200m'],
                           ['BVF2_200m'])])
def test_out_s_mom2_sep(FEATURES, TRAIN_FEATURES,
                        plot=False, full_shape=(20, 20),
                        sample_shape=(10, 10, 1),
                        batch_size=4, n_batches=4,
                        s_enhance=2, model_dir=None,
                        model_mom1_dir=None):
    """Test basic spatial model outputing for second conditional,
    moment separate from the first moment"""
    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            lr_only_features=TRAIN_FEATURES,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 10),
                            val_split=0,
                            worker_kwargs=dict(max_workers=1))

    # Load Mom 1 Model
    if model_mom1_dir is None:
        fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
        model_mom1 = Sup3rCondMom(fp_gen)
    else:
        fp_gen = os.path.join(model_mom1_dir, 'model_params.json')
        model_mom1 = Sup3rCondMom(fp_gen).load(model_mom1_dir)

    batch_handler = SpatialBatchHandlerMom2Sep([handler],
                                               batch_size=batch_size,
                                               s_enhance=s_enhance,
                                               n_batches=n_batches,
                                               model_mom1=model_mom1)

    # Load Mom2 Model
    if model_dir is None:
        fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
        model = Sup3rCondMom(fp_gen)
    else:
        fp_gen = os.path.join(model_dir, 'model_params.json')
        model = Sup3rCondMom(fp_gen).load(model_dir)

    if plot:
        import matplotlib.pyplot as plt
        from sup3r.utilities.plotting import plot_multi_contour, make_movie
        figureFolder = 'Figures'
        os.makedirs(figureFolder, exist_ok=True)
        movieFolder = os.path.join(figureFolder, 'Movie')
        os.makedirs(movieFolder, exist_ok=True)
        mom_name = r'$\sigma$(HR|LR)'
        hr_name = r'|HR - $\mathbb{E}$(HR|LR)|'
        n_snap = 0
        for p, batch in enumerate(batch_handler):
            out = np.clip(model.generate(batch.low_res,
                                         norm_in=False,
                                         un_norm_out=False),
                          a_min=0, a_max=None)
            out_mom1 = model_mom1.generate(batch.low_res,
                                           norm_in=False,
                                           un_norm_out=False)
            for i in range(batch.output.shape[0]):
                lr = (batch.low_res[i, :, :, 0] * batch_handler.stds[0]
                      + batch_handler.means[0])
                hr = (batch.high_res[i, :, :, 0] * batch_handler.stds[0]
                      + batch_handler.means[0])
                hr_pred = (out_mom1[i, :, :, 0] * batch_handler.stds[0]
                           + batch_handler.means[0])
                hr_to_mean = np.abs(hr - hr_pred)
                hr2_pred = (out[i, :, :, 0] * batch_handler.stds[0]**2
                            + (2 * batch_handler.means[0]
                               * hr_pred)
                            - batch_handler.means[0]**2)
                hr2_pred = np.clip(hr2_pred,
                                   a_min=0,
                                   a_max=None)
                sigma_pred = np.sqrt(np.clip(hr2_pred - hr_pred**2,
                                             a_min=0,
                                             a_max=None))
                fig = plot_multi_contour(
                    [lr, hr, hr_to_mean, sigma_pred],
                    [0, batch.output.shape[1]],
                    [0, batch.output.shape[2]],
                    ['U [m/s]', 'U [m/s]', 'U [m/s]', 'U [m/s]'],
                    ['LR', 'HR', hr_name, mom_name],
                    ['x [m]', 'x [m]', 'x [m]', 'x [m]'],
                    ['y [m]', 'y [m]', 'y [m]', 'y [m]'],
                    [np.amin(lr), np.amin(hr),
                     np.amin(hr_to_mean), np.amin(sigma_pred)],
                    [np.amax(lr), np.amax(hr),
                     np.amax(hr_to_mean), np.amax(sigma_pred)],
                )
                fig.savefig(os.path.join(movieFolder,
                                         "im_{}.png".format(n_snap)),
                            dpi=100, bbox_inches='tight')
                plt.close(fig)
                n_snap += 1
            if p > 4:
                break
        make_movie(n_snap, movieFolder, os.path.join(figureFolder,
                                                     'mom2_sep.gif'),
                   fps=6)


@pytest.mark.parametrize('FEATURES, TRAIN_FEATURES',
                         [(['U_100m', 'V_100m'],
                           None),
                          (['U_100m', 'V_100m', 'BVF2_200m'],
                           ['BVF2_200m'])])
def test_out_s_mom2_sep_sf(FEATURES, TRAIN_FEATURES,
                           plot=False, full_shape=(20, 20),
                           sample_shape=(10, 10, 1),
                           batch_size=4, n_batches=4,
                           s_enhance=2, model_dir=None,
                           model_mom1_dir=None):
    """Test basic spatial model outputing."""
    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            lr_only_features=TRAIN_FEATURES,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 10),
                            val_split=0,
                            worker_kwargs=dict(max_workers=1))

    # Load Mom 1 Model
    if model_mom1_dir is None:
        fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
        model_mom1 = Sup3rCondMom(fp_gen)
    else:
        fp_gen = os.path.join(model_mom1_dir, 'model_params.json')
        model_mom1 = Sup3rCondMom(fp_gen).load(model_mom1_dir)

    batch_handler = SpatialBatchHandlerMom2SepSF([handler],
                                                 batch_size=batch_size,
                                                 s_enhance=s_enhance,
                                                 n_batches=n_batches,
                                                 model_mom1=model_mom1)

    # Load Mom2 Model
    if model_dir is None:
        fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
        model = Sup3rCondMom(fp_gen)
    else:
        fp_gen = os.path.join(model_dir, 'model_params.json')
        model = Sup3rCondMom(fp_gen).load(model_dir)

    if plot:
        import matplotlib.pyplot as plt
        from sup3r.utilities.plotting import plot_multi_contour, make_movie
        figureFolder = 'Figures'
        os.makedirs(figureFolder, exist_ok=True)
        movieFolder = os.path.join(figureFolder, 'Movie')
        os.makedirs(movieFolder, exist_ok=True)
        mom_name1 = r'|SF - $\mathbb{E}$(SF|LR)|'
        mom_name2 = r'$\sigma$(SF|LR)'
        n_snap = 0
        for p, batch in enumerate(batch_handler):
            out = np.clip(model.generate(batch.low_res,
                                         norm_in=False,
                                         un_norm_out=False),
                          a_min=0, a_max=None)
            out_mom1 = model_mom1.generate(batch.low_res,
                                           norm_in=False,
                                           un_norm_out=False)
            for i in range(batch.output.shape[0]):
                lr = (batch.low_res[i, :, :, 0] * batch_handler.stds[0]
                      + batch_handler.means[0])
                blr_aug_shape = (1,) + lr.shape + (1,)
                blr_aug = np.reshape(batch.low_res[i, :, :, 0],
                                     blr_aug_shape)
                up_lr = spatial_simple_enhancing(blr_aug,
                                                 s_enhance=s_enhance)
                up_lr = up_lr[0, :, :, 0]
                hr = (batch.high_res[i, :, :, 0]
                      * batch_handler.stds[0]
                      + batch_handler.means[0])
                sf = (hr
                      - up_lr
                      * batch_handler.stds[0]
                      - batch_handler.means[0])
                sf2_pred = (out[i, :, :, 0]
                            * batch_handler.stds[0]**2)
                sf_pred = (out_mom1[i, :, :, 0]
                           * batch_handler.stds[0])
                sf_to_mean = np.abs(sf - sf_pred)
                sigma_pred = np.sqrt(np.clip(sf2_pred - sf_pred**2,
                                             a_min=0,
                                             a_max=None))
                fig = plot_multi_contour(
                    [lr, hr, sf, sf_to_mean, sigma_pred],
                    [0, batch.output.shape[1]],
                    [0, batch.output.shape[2]],
                    ['U [m/s]', 'U [m/s]', 'U [m/s]',
                     'U [m/s]', 'U [m/s]'],
                    ['LR', 'HR', 'SF', mom_name1, mom_name2],
                    ['x [m]', 'x [m]', 'x [m]', 'x [m]', 'x [m]'],
                    ['y [m]', 'y [m]', 'y [m]', 'y [m]', 'y [m]'],
                    [np.amin(lr), np.amin(hr),
                     np.amin(sf), np.amin(sf_to_mean),
                     np.amin(sigma_pred)],
                    [np.amax(lr), np.amax(hr),
                     np.amax(sf), np.amax(sf_to_mean),
                     np.amax(sigma_pred)],
                )
                fig.savefig(os.path.join(movieFolder,
                                         "im_{}.png".format(n_snap)),
                            dpi=100, bbox_inches='tight')
                plt.close(fig)
                n_snap += 1
            if p > 4:
                break
        make_movie(n_snap, movieFolder,
                   os.path.join(figureFolder, 'mom2_sep_sf.gif'),
                   fps=6)


def test_out_loss(plot=False, model_dirs=None,
                  model_names=None,
                  figureDir=None):
    """Loss convergence plotting of multiple models"""
    # Load history
    if model_dirs is not None:
        history_files = [os.path.join(path, 'history.csv')
                         for path in model_dirs]
        param_files = [os.path.join(path, 'model_params.json')
                       for path in model_dirs]
    else:
        print("No history file provided")
        return

    # Get model names
    if model_names is None:
        model_names_tmp = ["model_" + str(i)
                           for i in range(len(history_files))]
    else:
        model_names_tmp = model_names

    def get_num_params(param_file):
        with open(param_file, 'r') as f:
            model_params = json.load(f)
        return model_params['num_par']

    num_params = [get_num_params(param) for param in param_files]

    model_names = [name + " (%.3f M par)" % (num_par / 1e6)
                   for name, num_par
                   in zip(model_names_tmp, num_params)]

    # Read csv
    histories = [read_csv(file) for file in history_files]
    if plot:
        import matplotlib.pyplot as plt
        import matplotlib.pylab as pl
        from sup3r.utilities.plotting import pretty_labels, plot_legend
        figureFolder = 'Figures'
        os.makedirs(figureFolder, exist_ok=True)
        if figureDir is None:
            figureDir = 'loss'
        figureLossFolder = os.path.join(figureFolder, figureDir)
        os.makedirs(figureLossFolder, exist_ok=True)

        epoch_id = histories[0].columns.get_loc('epoch')
        time_id = histories[0].columns.get_loc('elapsed_time')
        train_loss_id = histories[0].columns.get_loc('train_loss_gen')
        test_loss_id = histories[0].columns.get_loc('val_loss_gen')
        datas = [history.values for history in histories]

        colors = pl.cm.jet(np.linspace(0, 1, len(histories)))

        _ = plt.figure()
        for idata, data in enumerate(datas):
            plt.plot(data[:, epoch_id], np.diff(data[:, time_id],
                     prepend=0),
                     color=colors[idata], linewidth=3,
                     label=model_names[idata])
        pretty_labels('Epoch', 'Wall clock [s]', 14)
        plt.savefig(os.path.join(figureLossFolder, 'timing.png'))
        plt.close()

        _ = plt.figure()
        # test train labels
        plt.plot(datas[0][:, epoch_id], datas[0][:, train_loss_id],
                 color='k', linewidth=3, label='train')
        plt.plot(datas[0][:, epoch_id], datas[0][:, test_loss_id],
                 '--', color='k', linewidth=1, label='test')
        # model labels
        for idata, data in enumerate(datas):
            plt.plot(data[:, epoch_id], data[:, train_loss_id],
                     color=colors[idata], linewidth=3,
                     label=model_names[idata])
            plt.plot(data[:, epoch_id], data[:, test_loss_id],
                     '--', color=colors[idata], linewidth=3)
        pretty_labels('Epoch', 'Loss', 14)
        plot_legend()
        plt.savefig(os.path.join(figureLossFolder, 'loss_lin.png'))
        ax = plt.gca()
        ax.set_yscale('log')
        plt.savefig(os.path.join(figureLossFolder, 'loss_log.png'))
        plt.close()


def test_out_st_mom1(plot=False, full_shape=(20, 20),
                     sample_shape=(12, 12, 24),
                     batch_size=4, n_batches=4,
                     s_enhance=3, t_enhance=4,
                     end_t_padding=False,
                     model_dir=None):
    """Test basic spatiotemporal model outputing for
    first conditional moment."""
    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 1),
                            val_split=0,
                            worker_kwargs=dict(max_workers=1))

    batch_handler = BatchHandlerMom1([handler],
                                     batch_size=batch_size,
                                     s_enhance=s_enhance,
                                     t_enhance=t_enhance,
                                     n_batches=n_batches,
                                     end_t_padding=end_t_padding)

    # Load Model
    if model_dir is None:
        fp_gen = os.path.join(CONFIG_DIR,
                              'spatiotemporal',
                              'gen_3x_4x_2f.json')
        model = Sup3rCondMom(fp_gen)
    else:
        fp_gen = os.path.join(model_dir, 'model_params.json')
        model = Sup3rCondMom(fp_gen).load(model_dir)

    # Check sizes
    for batch in batch_handler:
        assert batch.high_res.shape == (batch_size,
                                        sample_shape[0],
                                        sample_shape[1],
                                        sample_shape[2], 2)
        assert batch.output.shape == (batch_size,
                                      sample_shape[0],
                                      sample_shape[1],
                                      sample_shape[2], 2)
        assert batch.low_res.shape == (batch_size,
                                       sample_shape[0] // s_enhance,
                                       sample_shape[1] // s_enhance,
                                       sample_shape[2] // t_enhance,
                                       2)
        out = model._tf_generate(batch.low_res)
        assert out.shape == (batch_size,
                             sample_shape[0],
                             sample_shape[1],
                             sample_shape[2], 2)
        break
    if plot:
        import matplotlib.pyplot as plt
        from sup3r.utilities.plotting import plot_multi_contour, make_movie
        figureFolder = 'Figures'
        os.makedirs(figureFolder, exist_ok=True)
        movieFolder = os.path.join(figureFolder, 'Movie')
        os.makedirs(movieFolder, exist_ok=True)
        mom_name = r'$\mathbb{E}$(HR|LR)'
        n_snap = 0
        for p, batch in enumerate(batch_handler):
            out = model.generate(batch.low_res,
                                 norm_in=False,
                                 un_norm_out=False)
            for i in range(batch.output.shape[0]):
                lr = (batch.low_res[i, :, :, :, 0] * batch_handler.stds[0]
                      + batch_handler.means[0])
                aug_lr = np.reshape(lr, (1,) + lr.shape + (1,))
                tup_lr = temporal_simple_enhancing(aug_lr,
                                                   t_enhance=t_enhance,
                                                   mode='constant')
                tup_lr = tup_lr[0, :, :, :, 0]
                hr = (batch.output[i, :, :, :, 0] * batch_handler.stds[0]
                      + batch_handler.means[0])
                gen = (out[i, :, :, :, 0] * batch_handler.stds[0]
                       + batch_handler.means[0])
                max_t_ind = batch.output.shape[3]
                if end_t_padding:
                    max_t_ind -= t_enhance
                for j in range(max_t_ind):
                    fig = plot_multi_contour(
                        [tup_lr[:, :, j], hr[:, :, j], gen[:, :, j]],
                        [0, batch.output.shape[1]],
                        [0, batch.output.shape[2]],
                        ['U [m/s]', 'U [m/s]', 'U [m/s]'],
                        ['LR', 'HR', mom_name],
                        ['x [m]', 'x [m]', 'x [m]'],
                        ['y [m]', 'y [m]', 'y [m]'],
                        [np.amin(tup_lr), np.amin(hr), np.amin(hr)],
                        [np.amax(tup_lr), np.amax(hr), np.amax(hr)],
                    )
                    fig.savefig(os.path.join(movieFolder,
                                             "im_{}.png".format(n_snap)),
                                dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    n_snap += 1
            if p > 4:
                break
        make_movie(n_snap, movieFolder,
                   os.path.join(figureFolder, 'st_mom1.gif'),
                   fps=6)


def test_out_st_mom1_sf(plot=False, full_shape=(20, 20),
                        sample_shape=(12, 12, 24),
                        batch_size=4, n_batches=4,
                        s_enhance=3, t_enhance=4,
                        end_t_padding=False,
                        t_enhance_mode='constant',
                        model_dir=None):
    """Test basic spatiotemporal model outputing for first conditional moment
    of subfilter velocity."""
    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 1),
                            val_split=0,
                            worker_kwargs=dict(max_workers=1))

    batch_handler = BatchHandlerMom1SF(
        [handler],
        batch_size=batch_size,
        s_enhance=s_enhance,
        t_enhance=t_enhance,
        n_batches=n_batches,
        end_t_padding=end_t_padding,
        temporal_enhancing_method=t_enhance_mode)

    # Load Model
    if model_dir is None:
        fp_gen = os.path.join(CONFIG_DIR,
                              'spatiotemporal',
                              'gen_3x_4x_2f.json')
        model = Sup3rCondMom(fp_gen)
    else:
        fp_gen = os.path.join(model_dir, 'model_params.json')
        model = Sup3rCondMom(fp_gen).load(model_dir)

    if plot:
        import matplotlib.pyplot as plt
        from sup3r.utilities.plotting import plot_multi_contour, make_movie
        figureFolder = 'Figures'
        os.makedirs(figureFolder, exist_ok=True)
        movieFolder = os.path.join(figureFolder, 'Movie')
        os.makedirs(movieFolder, exist_ok=True)
        mom_name = r'$\mathbb{E}$(HR|LR)'
        mom_name2 = r'$\mathbb{E}$(SF|LR)'
        n_snap = 0
        for p, batch in enumerate(batch_handler):
            out = model.generate(batch.low_res,
                                 norm_in=False,
                                 un_norm_out=False)
            for i in range(batch.output.shape[0]):

                b_lr = batch.low_res[i, :, :, :, 0]
                b_lr_aug = np.reshape(b_lr, (1,) + b_lr.shape + (1,))

                tup_lr = temporal_simple_enhancing(b_lr_aug,
                                                   t_enhance=t_enhance,
                                                   mode='constant')
                tup_lr = (tup_lr[0, :, :, :, 0]
                          * batch_handler.stds[0]
                          + batch_handler.means[0])
                up_lr_tmp = spatial_simple_enhancing(b_lr_aug,
                                                     s_enhance=s_enhance)
                up_lr = temporal_simple_enhancing(up_lr_tmp,
                                                  t_enhance=t_enhance,
                                                  mode=t_enhance_mode)
                up_lr = up_lr[0, :, :, :, 0]

                hr = (batch.high_res[i, :, :, :, 0]
                      * batch_handler.stds[0]
                      + batch_handler.means[0])

                sf = (batch.output[i, :, :, :, 0]
                      * batch_handler.stds[0])

                sf_pred = (out[i, :, :, :, 0]
                           * batch_handler.stds[0])

                hr_pred = (up_lr
                           * batch_handler.stds[0]
                           + batch_handler.means[0]
                           + sf_pred)
                max_t_ind = batch.output.shape[3]
                if end_t_padding:
                    max_t_ind -= t_enhance
                for j in range(max_t_ind):
                    fig = plot_multi_contour(
                        [tup_lr[:, :, j], hr[:, :, j],
                         hr_pred[:, :, j], sf[:, :, j],
                         sf_pred[:, :, j]],
                        [0, batch.output.shape[1]],
                        [0, batch.output.shape[2]],
                        ['U [m/s]', 'U [m/s]', 'U [m/s]',
                         'U [m/s]', 'U [m/s]'],
                        ['LR', 'HR', mom_name, 'SF', mom_name2],
                        ['x [m]', 'x [m]', 'x [m]', 'x [m]', 'x [m]'],
                        ['y [m]', 'y [m]', 'y [m]', 'y [m]', 'y [m]'],
                        [np.amin(tup_lr), np.amin(hr),
                         np.amin(hr), np.amin(sf),
                         np.amin(sf)],
                        [np.amax(tup_lr), np.amax(hr),
                         np.amax(hr), np.amax(sf),
                         np.amax(sf)],
                    )
                    fig.savefig(os.path.join(movieFolder,
                                             "im_{}.png".format(n_snap)),
                                dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    n_snap += 1
            if p > 4:
                break
        make_movie(n_snap, movieFolder,
                   os.path.join(figureFolder, 'st_mom1_sf.gif'),
                   fps=6)


def test_out_st_mom2(plot=False, full_shape=(20, 20),
                     sample_shape=(12, 12, 24),
                     batch_size=4, n_batches=4,
                     s_enhance=3, t_enhance=4,
                     end_t_padding=False,
                     model_dir=None,
                     model_mom1_dir=None):
    """Test basic spatiotemporal model outputing
    for second conditional moment."""
    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 1),
                            val_split=0,
                            worker_kwargs=dict(max_workers=1))

    # Load Mom 1 Model
    if model_mom1_dir is None:
        fp_gen = os.path.join(CONFIG_DIR,
                              'spatiotemporal',
                              'gen_3x_4x_2f.json')
        model_mom1 = Sup3rCondMom(fp_gen)
    else:
        fp_gen = os.path.join(model_mom1_dir, 'model_params.json')
        model_mom1 = Sup3rCondMom(fp_gen).load(model_mom1_dir)

    batch_handler = BatchHandlerMom2([handler],
                                     batch_size=batch_size,
                                     s_enhance=s_enhance,
                                     t_enhance=t_enhance,
                                     n_batches=n_batches,
                                     model_mom1=model_mom1,
                                     end_t_padding=end_t_padding)

    # Load Mom2 Model
    if model_dir is None:
        fp_gen = os.path.join(CONFIG_DIR,
                              'spatiotemporal',
                              'gen_3x_4x_2f.json')
        model = Sup3rCondMom(fp_gen)
    else:
        fp_gen = os.path.join(model_dir, 'model_params.json')
        model = Sup3rCondMom(fp_gen).load(model_dir)

    if plot:
        import matplotlib.pyplot as plt
        from sup3r.utilities.plotting import (plot_multi_contour,
                                              pretty_labels, make_movie)
        figureFolder = 'Figures'
        os.makedirs(figureFolder, exist_ok=True)
        movieFolder = os.path.join(figureFolder, 'Movie')
        os.makedirs(movieFolder, exist_ok=True)
        mom_name = r'$\sigma$(HR|LR)'
        hr_name = r'|HR - $\mathbb{E}$(HR|LR)|'
        n_snap = 0
        integratedSigma = []
        for p, batch in enumerate(batch_handler):
            out = np.clip(model.generate(batch.low_res,
                                         norm_in=False,
                                         un_norm_out=False),
                          a_min=0, a_max=None)
            for i in range(batch.output.shape[0]):

                b_lr = batch.low_res[i, :, :, :, 0]
                b_lr_aug = np.reshape(b_lr, (1,) + b_lr.shape + (1,))

                tup_lr = temporal_simple_enhancing(b_lr_aug,
                                                   t_enhance=t_enhance,
                                                   mode='constant')
                tup_lr = (tup_lr[0, :, :, :, 0]
                          * batch_handler.stds[0]
                          + batch_handler.means[0])
                hr = (batch.high_res[i, :, :, :, 0]
                      * batch_handler.stds[0]
                      + batch_handler.means[0])
                hr_to_mean = np.sqrt(batch.output[i, :, :, :, 0]
                                     * batch_handler.stds[0]**2)
                sigma = np.sqrt(out[i, :, :, :, 0]
                                * batch_handler.stds[0]**2)
                integratedSigma.append(np.mean(sigma, axis=(0, 1)))

                max_t_ind = batch.output.shape[3]
                if end_t_padding:
                    max_t_ind -= t_enhance
                for j in range(max_t_ind):
                    fig = plot_multi_contour(
                        [tup_lr[:, :, j], hr[:, :, j],
                         hr_to_mean[:, :, j], sigma[:, :, j]],
                        [0, batch.output.shape[1]],
                        [0, batch.output.shape[2]],
                        ['U [m/s]', 'U [m/s]', 'U [m/s]', 'U [m/s]'],
                        ['LR', 'HR', hr_name, mom_name],
                        ['x [m]', 'x [m]', 'x [m]', 'x [m]'],
                        ['y [m]', 'y [m]', 'y [m]', 'y [m]'],
                        [np.amin(tup_lr), np.amin(hr),
                         np.amin(hr_to_mean), np.amin(sigma)],
                        [np.amax(tup_lr), np.amax(hr),
                         np.amax(hr_to_mean), np.amax(sigma)],
                    )
                    fig.savefig(os.path.join(movieFolder,
                                             "im_{}.png".format(n_snap)),
                                dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    n_snap += 1
            if p > 4:
                break
        make_movie(n_snap, movieFolder,
                   os.path.join(figureFolder, 'st_mom2.gif'),
                   fps=6)

        fig = plt.figure()
        for sigma_xy in integratedSigma:
            plt.plot(sigma_xy, color='k', linewidth=3)
        pretty_labels('t', r'$\langle \sigma \rangle_{x,y}$ [m/s]', 14)
        plt.savefig(os.path.join(figureFolder, 'st_mom2_int_sig.png'))
        plt.close(fig)

        fig = plt.figure()
        for sigma_xy in integratedSigma:
            plt.plot(sigma_xy / np.mean(sigma_xy),
                     color='k', linewidth=3)
        ylabel = r'$\langle \sigma \rangle_{x,y}$'
        ylabel += r'$\langle \sigma \rangle_{x,y,t}$'
        pretty_labels('t', ylabel, 14)
        plt.savefig(os.path.join(figureFolder, 'st_mom2_int_sig_resc.png'))
        plt.close(fig)


def test_out_st_mom2_sf(plot=False, full_shape=(20, 20),
                        sample_shape=(12, 12, 24),
                        batch_size=4, n_batches=4,
                        s_enhance=3, t_enhance=4,
                        end_t_padding=False,
                        t_enhance_mode='constant',
                        model_dir=None,
                        model_mom1_dir=None):
    """Test basic spatiotemporal model outputing for second conditional moment
    of subfilter velocity."""
    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 1),
                            val_split=0,
                            worker_kwargs=dict(max_workers=1))

    # Load Mom 1 Model
    if model_mom1_dir is None:
        fp_gen = os.path.join(CONFIG_DIR,
                              'spatiotemporal',
                              'gen_3x_4x_2f.json')
        model_mom1 = Sup3rCondMom(fp_gen)
    else:
        fp_gen = os.path.join(model_mom1_dir, 'model_params.json')
        model_mom1 = Sup3rCondMom(fp_gen).load(model_mom1_dir)

    batch_handler = BatchHandlerMom2SF(
        [handler],
        batch_size=batch_size,
        s_enhance=s_enhance,
        t_enhance=t_enhance,
        n_batches=n_batches,
        model_mom1=model_mom1,
        end_t_padding=end_t_padding,
        temporal_enhancing_method=t_enhance_mode)

    # Load Mom2 Model
    if model_dir is None:
        fp_gen = os.path.join(CONFIG_DIR,
                              'spatiotemporal',
                              'gen_3x_4x_2f.json')
        model = Sup3rCondMom(fp_gen)
    else:
        fp_gen = os.path.join(model_dir, 'model_params.json')
        model = Sup3rCondMom(fp_gen).load(model_dir)

    if plot:
        import matplotlib.pyplot as plt
        from sup3r.utilities.plotting import (plot_multi_contour,
                                              pretty_labels, make_movie)
        figureFolder = 'Figures'
        os.makedirs(figureFolder, exist_ok=True)
        movieFolder = os.path.join(figureFolder, 'Movie')
        os.makedirs(movieFolder, exist_ok=True)
        mom_name1 = r'|SF - $\mathbb{E}$(SF|LR)|'
        mom_name2 = r'$\sigma$(SF|LR)'
        n_snap = 0
        integratedSigma = []
        for p, batch in enumerate(batch_handler):
            out = np.clip(model.generate(batch.low_res,
                                         norm_in=False,
                                         un_norm_out=False),
                          a_min=0, a_max=None)
            for i in range(batch.output.shape[0]):
                b_lr = batch.low_res[i, :, :, :, 0]
                b_lr_aug = np.reshape(b_lr, (1,) + b_lr.shape + (1,))

                tup_lr = temporal_simple_enhancing(b_lr_aug,
                                                   t_enhance=t_enhance,
                                                   mode='constant')
                tup_lr = (tup_lr[0, :, :, :, 0]
                          * batch_handler.stds[0]
                          + batch_handler.means[0])

                up_lr_tmp = spatial_simple_enhancing(b_lr_aug,
                                                     s_enhance=s_enhance)
                up_lr = temporal_simple_enhancing(up_lr_tmp,
                                                  t_enhance=t_enhance,
                                                  mode=t_enhance_mode)
                up_lr = up_lr[0, :, :, :, 0]

                hr = (batch.high_res[i, :, :, :, 0]
                      * batch_handler.stds[0]
                      + batch_handler.means[0])
                sf = (hr
                      - up_lr
                      * batch_handler.stds[0]
                      - batch_handler.means[0])
                sf_to_mean = np.sqrt(batch.output[i, :, :, :, 0]
                                     * batch_handler.stds[0]**2)
                sigma = np.sqrt(out[i, :, :, :, 0]
                                * batch_handler.stds[0]**2)
                integratedSigma.append(np.mean(sigma, axis=(0, 1)))

                max_t_ind = batch.output.shape[3]
                if end_t_padding:
                    max_t_ind -= t_enhance
                for j in range(max_t_ind):
                    fig = plot_multi_contour(
                        [tup_lr[:, :, j], hr[:, :, j],
                         sf[:, :, j], sf_to_mean[:, :, j],
                         sigma[:, :, j]],
                        [0, batch.output.shape[1]],
                        [0, batch.output.shape[2]],
                        ['U [m/s]', 'U [m/s]', 'U [m/s]',
                         'U [m/s]', 'U [m/s]'],
                        ['LR', 'HR', 'SF', mom_name1, mom_name2],
                        ['x [m]', 'x [m]', 'x [m]', 'x [m]', 'x [m]'],
                        ['y [m]', 'y [m]', 'y [m]', 'y [m]', 'y [m]'],
                        [np.amin(tup_lr), np.amin(hr),
                         np.amin(sf), np.amin(sf_to_mean),
                         np.amin(sigma)],
                        [np.amax(tup_lr), np.amax(hr),
                         np.amax(sf), np.amax(sf_to_mean),
                         np.amax(sigma)],
                    )
                    fig.savefig(os.path.join(movieFolder,
                                             "im_{}.png".format(n_snap)),
                                dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    n_snap += 1
            if p > 4:
                break
        make_movie(n_snap, movieFolder,
                   os.path.join(figureFolder, 'st_mom2_sf.gif'),
                   fps=6)

        fig = plt.figure()
        for sigma_xy in integratedSigma:
            plt.plot(sigma_xy, color='k', linewidth=3)
        pretty_labels('t', r'$\langle \sigma \rangle_{x,y}$ [m/s]', 14)
        plt.savefig(os.path.join(figureFolder, 'st_mom2_sf_int_sig.png'))
        plt.close(fig)

        fig = plt.figure()
        for sigma_xy in integratedSigma:
            plt.plot(sigma_xy / np.mean(sigma_xy),
                     color='k', linewidth=3)
        ylabel = r'$\langle \sigma \rangle_{x,y}$'
        ylabel += r'$\langle \sigma \rangle_{x,y,t}$'
        pretty_labels('t', ylabel, 14)
        plt.savefig(os.path.join(figureFolder, 'st_mom2_sf_int_sig_resc.png'))
        plt.close(fig)


def test_out_st_mom2_sep(plot=False, full_shape=(20, 20),
                         sample_shape=(12, 12, 24),
                         batch_size=4, n_batches=4,
                         s_enhance=3, t_enhance=4,
                         end_t_padding=False,
                         model_dir=None,
                         model_mom1_dir=None):
    """Test basic spatiotemporal model outputing
    for second conditional moment."""
    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 1),
                            val_split=0,
                            worker_kwargs=dict(max_workers=1))

    # Load Mom 1 Model
    if model_mom1_dir is None:
        fp_gen = os.path.join(CONFIG_DIR,
                              'spatiotemporal',
                              'gen_3x_4x_2f.json')
        model_mom1 = Sup3rCondMom(fp_gen)
    else:
        fp_gen = os.path.join(model_mom1_dir, 'model_params.json')
        model_mom1 = Sup3rCondMom(fp_gen).load(model_mom1_dir)

    batch_handler = BatchHandlerMom2Sep([handler],
                                        batch_size=batch_size,
                                        s_enhance=s_enhance,
                                        t_enhance=t_enhance,
                                        n_batches=n_batches,
                                        end_t_padding=end_t_padding)

    # Load Mom2 Model
    if model_dir is None:
        fp_gen = os.path.join(CONFIG_DIR,
                              'spatiotemporal',
                              'gen_3x_4x_2f.json')
        model = Sup3rCondMom(fp_gen)
    else:
        fp_gen = os.path.join(model_dir, 'model_params.json')
        model = Sup3rCondMom(fp_gen).load(model_dir)

    if plot:
        import matplotlib.pyplot as plt
        from sup3r.utilities.plotting import (plot_multi_contour,
                                              pretty_labels, make_movie)
        figureFolder = 'Figures'
        os.makedirs(figureFolder, exist_ok=True)
        movieFolder = os.path.join(figureFolder, 'Movie')
        os.makedirs(movieFolder, exist_ok=True)
        mom_name = r'$\sigma$(HR|LR)'
        hr_name = r'|HR - $\mathbb{E}$(HR|LR)|'
        n_snap = 0
        integratedSigma = []
        for p, batch in enumerate(batch_handler):
            out = np.clip(model.generate(batch.low_res,
                                         norm_in=False,
                                         un_norm_out=False),
                          a_min=0, a_max=None)
            out_mom1 = model_mom1.generate(batch.low_res,
                                           norm_in=False,
                                           un_norm_out=False)
            for i in range(batch.output.shape[0]):

                b_lr = batch.low_res[i, :, :, :, 0]
                b_lr_aug = np.reshape(b_lr, (1,) + b_lr.shape + (1,))

                tup_lr = temporal_simple_enhancing(b_lr_aug,
                                                   t_enhance=t_enhance,
                                                   mode='constant')
                tup_lr = (tup_lr[0, :, :, :, 0]
                          * batch_handler.stds[0]
                          + batch_handler.means[0])
                hr = (batch.high_res[i, :, :, :, 0]
                      * batch_handler.stds[0]
                      + batch_handler.means[0])
                hr_pred = (out_mom1[i, :, :, :, 0] * batch_handler.stds[0]
                           + batch_handler.means[0])
                hr_to_mean = np.abs(hr - hr_pred)
                hr2_pred = (out[i, :, :, :, 0] * batch_handler.stds[0]**2
                            + (2 * batch_handler.means[0]
                               * hr_pred)
                            - batch_handler.means[0]**2)
                hr2_pred = np.clip(hr2_pred,
                                   a_min=0,
                                   a_max=None)
                sigma_pred = np.sqrt(np.clip(hr2_pred - hr_pred**2,
                                             a_min=0,
                                             a_max=None))
                integratedSigma.append(np.mean(sigma_pred, axis=(0, 1)))
                max_t_ind = batch.output.shape[3]
                if end_t_padding:
                    max_t_ind -= t_enhance
                for j in range(max_t_ind):
                    fig = plot_multi_contour(
                        [tup_lr[:, :, j], hr[:, :, j],
                         hr_to_mean[:, :, j], sigma_pred[:, :, j]],
                        [0, batch.output.shape[1]],
                        [0, batch.output.shape[2]],
                        ['U [m/s]', 'U [m/s]', 'U [m/s]', 'U [m/s]'],
                        ['LR', 'HR', hr_name, mom_name],
                        ['x [m]', 'x [m]', 'x [m]', 'x [m]'],
                        ['y [m]', 'y [m]', 'y [m]', 'y [m]'],
                        [np.amin(tup_lr), np.amin(hr),
                         np.amin(hr_to_mean), np.amin(sigma_pred)],
                        [np.amax(tup_lr), np.amax(hr),
                         np.amax(hr_to_mean), np.amax(sigma_pred)],
                    )
                    fig.savefig(os.path.join(movieFolder,
                                             "im_{}.png".format(n_snap)),
                                dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    n_snap += 1
            if p > 4:
                break
        make_movie(n_snap, movieFolder,
                   os.path.join(figureFolder, 'st_mom2_sep.gif'),
                   fps=6)

        fig = plt.figure()
        for sigma_xy in integratedSigma:
            plt.plot(sigma_xy, color='k', linewidth=3)
        pretty_labels('t', r'$\langle \sigma \rangle_{x,y}$ [m/s]', 14)
        plt.savefig(os.path.join(figureFolder, 'st_mom2_sep_int_sig.png'))
        plt.close(fig)

        fig = plt.figure()
        for sigma_xy in integratedSigma:
            plt.plot(sigma_xy / np.mean(sigma_xy),
                     color='k', linewidth=3)
        ylabel = r'$\langle \sigma \rangle_{x,y}$'
        ylabel += r'$\langle \sigma \rangle_{x,y,t}$'
        pretty_labels('t', ylabel, 14)
        plt.savefig(os.path.join(figureFolder, 'st_mom2_sep_int_sig_resc.png'))
        plt.close(fig)


def test_out_st_mom2_sep_sf(plot=False, full_shape=(20, 20),
                            sample_shape=(12, 12, 24),
                            batch_size=4, n_batches=4,
                            s_enhance=3, t_enhance=4,
                            end_t_padding=False,
                            t_enhance_mode='constant',
                            model_dir=None,
                            model_mom1_dir=None):
    """Test basic spatiotemporal model outputing for second conditional moment
    of subfilter velocity."""
    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 1),
                            val_split=0,
                            worker_kwargs=dict(max_workers=1))

    # Load Mom 1 Model
    if model_mom1_dir is None:
        fp_gen = os.path.join(CONFIG_DIR,
                              'spatiotemporal',
                              'gen_3x_4x_2f.json')
        model_mom1 = Sup3rCondMom(fp_gen)
    else:
        fp_gen = os.path.join(model_mom1_dir, 'model_params.json')
        model_mom1 = Sup3rCondMom(fp_gen).load(model_mom1_dir)

    batch_handler = BatchHandlerMom2SepSF(
        [handler],
        batch_size=batch_size,
        s_enhance=s_enhance,
        t_enhance=t_enhance,
        n_batches=n_batches,
        end_t_padding=end_t_padding,
        temporal_enhancing_method=t_enhance_mode)

    # Load Mom2 Model
    if model_dir is None:
        fp_gen = os.path.join(CONFIG_DIR,
                              'spatiotemporal',
                              'gen_3x_4x_2f.json')
        model = Sup3rCondMom(fp_gen)
    else:
        fp_gen = os.path.join(model_dir, 'model_params.json')
        model = Sup3rCondMom(fp_gen).load(model_dir)

    if plot:
        import matplotlib.pyplot as plt
        from sup3r.utilities.plotting import (plot_multi_contour,
                                              pretty_labels, make_movie)
        figureFolder = 'Figures'
        os.makedirs(figureFolder, exist_ok=True)
        movieFolder = os.path.join(figureFolder, 'Movie')
        os.makedirs(movieFolder, exist_ok=True)
        mom_name1 = r'|SF - $\mathbb{E}$(SF|LR)|'
        mom_name2 = r'$\sigma$(SF|LR)'
        n_snap = 0
        integratedSigma = []
        for p, batch in enumerate(batch_handler):
            out = np.clip(model.generate(batch.low_res,
                                         norm_in=False,
                                         un_norm_out=False),
                          a_min=0, a_max=None)
            out_mom1 = model_mom1.generate(batch.low_res,
                                           norm_in=False,
                                           un_norm_out=False)
            for i in range(batch.output.shape[0]):

                b_lr = batch.low_res[i, :, :, :, 0]
                b_lr_aug = np.reshape(b_lr, (1,) + b_lr.shape + (1,))

                tup_lr = temporal_simple_enhancing(b_lr_aug,
                                                   t_enhance=t_enhance,
                                                   mode='constant')
                tup_lr = (tup_lr[0, :, :, :, 0]
                          * batch_handler.stds[0]
                          + batch_handler.means[0])

                up_lr_tmp = spatial_simple_enhancing(b_lr_aug,
                                                     s_enhance=s_enhance)
                up_lr = temporal_simple_enhancing(up_lr_tmp,
                                                  t_enhance=t_enhance,
                                                  mode=t_enhance_mode)
                up_lr = up_lr[0, :, :, :, 0]

                hr = (batch.high_res[i, :, :, :, 0]
                      * batch_handler.stds[0]
                      + batch_handler.means[0])
                sf = (hr
                      - up_lr
                      * batch_handler.stds[0]
                      - batch_handler.means[0])

                sf2_pred = (out[i, :, :, :, 0]
                            * batch_handler.stds[0]**2)
                sf_pred = (out_mom1[i, :, :, :, 0]
                           * batch_handler.stds[0])
                sf_to_mean = np.abs(sf - sf_pred)

                sigma_pred = np.sqrt(np.clip(sf2_pred - sf_pred**2,
                                             a_min=0,
                                             a_max=None))
                integratedSigma.append(np.mean(sigma_pred, axis=(0, 1)))
                max_t_ind = batch.output.shape[3]
                if end_t_padding:
                    max_t_ind -= t_enhance
                for j in range(max_t_ind):
                    fig = plot_multi_contour(
                        [tup_lr[:, :, j], hr[:, :, j],
                         sf[:, :, j], sf_to_mean[:, :, j],
                         sigma_pred[:, :, j]],
                        [0, batch.output.shape[1]],
                        [0, batch.output.shape[2]],
                        ['U [m/s]', 'U [m/s]', 'U [m/s]',
                         'U [m/s]', 'U [m/s]'],
                        ['LR', 'HR', 'SF', mom_name1, mom_name2],
                        ['x [m]', 'x [m]', 'x [m]', 'x [m]', 'x [m]'],
                        ['y [m]', 'y [m]', 'y [m]', 'y [m]', 'y [m]'],
                        [np.amin(tup_lr), np.amin(hr),
                         np.amin(sf), np.amin(sf_to_mean),
                         np.amin(sigma_pred)],
                        [np.amax(tup_lr), np.amax(hr),
                         np.amax(sf), np.amax(sf_to_mean),
                         np.amax(sigma_pred)],
                    )
                    fig.savefig(os.path.join(movieFolder,
                                             "im_{}.png".format(n_snap)),
                                dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    n_snap += 1
            if p > 4:
                break
        make_movie(n_snap, movieFolder,
                   os.path.join(figureFolder, 'st_mom2_sep_sf.gif'),
                   fps=6)

        fig = plt.figure()
        for sigma_xy in integratedSigma:
            plt.plot(sigma_xy, color='k', linewidth=3)
        pretty_labels('t', r'$\langle \sigma \rangle_{x,y}$ [m/s]', 14)
        plt.savefig(os.path.join(figureFolder, 'st_mom2_sep_sf_int_sig.png'))
        plt.close(fig)

        fig = plt.figure()
        for sigma_xy in integratedSigma:
            plt.plot(sigma_xy / np.mean(sigma_xy),
                     color='k', linewidth=3)
        ylabel = r'$\langle \sigma \rangle_{x,y}$'
        ylabel += r'$\langle \sigma \rangle_{x,y,t}$'
        pretty_labels('t', ylabel, 14)
        plt.savefig(os.path.join(figureFolder,
                                 'st_mom2_sep_sf_int_sig_resc.png'))
        plt.close(fig)
