# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN"""
import os
import json
import numpy as np
from pandas import read_csv

from sup3r import TEST_DATA_DIR
from sup3r import CONFIG_DIR
from sup3r.models import Sup3rCondMom
from sup3r.preprocessing.data_handling import DataHandlerH5
from sup3r.preprocessing.batch_handling import (SpatialBatchHandler,
                                                SpatialBatchHandlerMom1SF,
                                                SpatialBatchHandlerMom2,
                                                SpatialBatchHandlerMom2SF)
from sup3r.utilities.utilities import (spatial_simple_enhancing)

FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
TARGET_COORD = (39.01, -105.15)
FEATURES = ['U_100m', 'V_100m']
TRAIN_FEATURES = None
n_feat_in = len(FEATURES)
n_feat_out = len(FEATURES)


def test_out_spatial_mom1(plot=False, full_shape=(20, 20),
                          sample_shape=(10, 10, 1),
                          batch_size=4, n_batches=4,
                          s_enhance=2, model_dir=None,
                          FEATURES=None,
                          TRAIN_FEATURES=None,
                          n_feat_in=n_feat_in, n_feat_out=n_feat_out):
    """Test basic spatial model outputing."""
    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            train_only_features=TRAIN_FEATURES,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 10),
                            val_split=0,
                            max_workers=1)

    batch_handler = SpatialBatchHandler([handler],
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
        from sup3r.utilities.plot_utilities import (plot_multi_contour,
                                                    makeMovie)
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
        makeMovie(n_snap, movieFolder, os.path.join(figureFolder, 'mom1.gif'),
                  fps=6)


def test_out_spatial_mom1_sf(plot=False, full_shape=(20, 20),
                             sample_shape=(10, 10, 1),
                             batch_size=4, n_batches=4,
                             s_enhance=2, model_dir=None,
                             FEATURES=None,
                             TRAIN_FEATURES=None):
    """Test basic spatial model outputing."""
    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            train_only_features=TRAIN_FEATURES,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 10),
                            val_split=0,
                            max_workers=1)

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
        from sup3r.utilities.plot_utilities import (plot_multi_contour,
                                                    makeMovie)
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
        makeMovie(n_snap, movieFolder,
                  os.path.join(figureFolder, 'mom1_sf.gif'),
                  fps=6)


def test_out_spatial_mom2(plot=False, full_shape=(20, 20),
                          sample_shape=(10, 10, 1),
                          batch_size=4, n_batches=4,
                          s_enhance=2, model_dir=None,
                          model_mom1_dir=None,
                          FEATURES=None,
                          TRAIN_FEATURES=None):
    """Test basic spatial model outputing."""
    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            train_only_features=TRAIN_FEATURES,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 10),
                            val_split=0,
                            max_workers=1)

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
        fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f_mom2.json')
        model = Sup3rCondMom(fp_gen)
    else:
        fp_gen = os.path.join(model_dir, 'model_params.json')
        model = Sup3rCondMom(fp_gen).load(model_dir)

    if plot:
        import matplotlib.pyplot as plt
        from sup3r.utilities.plot_utilities import (plot_multi_contour,
                                                    makeMovie)
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
        makeMovie(n_snap, movieFolder, os.path.join(figureFolder, 'mom2.gif'),
                  fps=6)


def test_out_spatial_mom2_sf(plot=False, full_shape=(20, 20),
                             sample_shape=(10, 10, 1),
                             batch_size=4, n_batches=4,
                             s_enhance=2, model_dir=None,
                             model_mom1_dir=None,
                             FEATURES=None,
                             TRAIN_FEATURES=None):
    """Test basic spatial model outputing."""
    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            train_only_features=TRAIN_FEATURES,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 10),
                            val_split=0,
                            max_workers=1)

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
        fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f_mom2.json')
        model = Sup3rCondMom(fp_gen)
    else:
        fp_gen = os.path.join(model_dir, 'model_params.json')
        model = Sup3rCondMom(fp_gen).load(model_dir)

    if plot:
        import matplotlib.pyplot as plt
        from sup3r.utilities.plot_utilities import (plot_multi_contour,
                                                    makeMovie)
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
        makeMovie(n_snap, movieFolder,
                  os.path.join(figureFolder, 'mom2_sf.gif'),
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
        from sup3r.utilities.plot_utilities import (prettyLabels,
                                                    plotLegend)
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
        prettyLabels('Epoch', 'Wall clock [s]', 14)
        plt.savefig(os.path.join(figureLossFolder, 'timing.png'))
        plt.close()

        _ = plt.figure()
        # test train labels
        plt.plot(datas[0][:, epoch_id], datas[0][:, train_loss_id],
                 color='k', linewidth=3, label='train')
        plt.plot(datas[0][:, epoch_id], datas[0][:, test_loss_id],
                 '--', color='k', linewidth=3, label='test')
        # model labels
        for idata, data in enumerate(datas):
            plt.plot(data[:, epoch_id], data[:, train_loss_id],
                     color=colors[idata], linewidth=3,
                     label=model_names[idata])
            plt.plot(data[:, epoch_id], data[:, test_loss_id],
                     '--', color=colors[idata], linewidth=3)
        prettyLabels('Epoch', 'Loss', 14)
        plotLegend()
        plt.savefig(os.path.join(figureLossFolder, 'loss_lin.png'))
        ax = plt.gca()
        ax.set_yscale('log')
        plt.savefig(os.path.join(figureLossFolder, 'loss_log.png'))
        plt.close()


if __name__ == "__main__":

    test_out_loss(plot=True, model_dirs=['s_mom1/spatial_cond_mom'],
                  figureDir='mom1_loss')

    test_out_loss(plot=True, model_dirs=['s_mom2/spatial_cond_mom'],
                  figureDir='mom2_loss')

    test_out_loss(plot=True, model_dirs=['s_mom1_sf/spatial_cond_mom'],
                  figureDir='mom1_sf_loss')

    test_out_loss(plot=True, model_dirs=['s_mom2_sf/spatial_cond_mom'],
                  figureDir='mom2_sf_loss')

    test_out_spatial_mom1(plot=True, full_shape=(20, 20),
                          sample_shape=(10, 10, 1),
                          batch_size=4, n_batches=2,
                          s_enhance=2, model_dir='s_mom1/spatial_cond_mom',
                          FEATURES=FEATURES,
                          n_feat_in=n_feat_in, n_feat_out=n_feat_out,
                          TRAIN_FEATURES=TRAIN_FEATURES)

    test_out_spatial_mom2(plot=True, full_shape=(20, 20),
                          sample_shape=(10, 10, 1),
                          batch_size=4, n_batches=2,
                          s_enhance=2, model_dir='s_mom2/spatial_cond_mom',
                          FEATURES=FEATURES,
                          model_mom1_dir='s_mom1/spatial_cond_mom',
                          TRAIN_FEATURES=TRAIN_FEATURES)

    test_out_spatial_mom1_sf(plot=True, full_shape=(20, 20),
                             sample_shape=(10, 10, 1),
                             batch_size=4, n_batches=2,
                             s_enhance=2,
                             FEATURES=FEATURES,
                             model_dir='s_mom1_sf/spatial_cond_mom',
                             TRAIN_FEATURES=TRAIN_FEATURES)

    test_out_spatial_mom2_sf(plot=True, full_shape=(20, 20),
                             sample_shape=(10, 10, 1),
                             batch_size=4, n_batches=2,
                             s_enhance=2,
                             FEATURES=FEATURES,
                             model_mom1_dir='s_mom1_sf/spatial_cond_mom',
                             model_dir='s_mom2_sf/spatial_cond_mom',
                             TRAIN_FEATURES=TRAIN_FEATURES)
