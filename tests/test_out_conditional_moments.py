# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN"""
import os
# import json
import numpy as np
from pandas import read_csv

from sup3r import TEST_DATA_DIR
from sup3r import CONFIG_DIR
from sup3r.models import Sup3rCondMom
from sup3r.preprocessing.data_handling import DataHandlerH5
from sup3r.preprocessing.batch_handling import (SpatialBatchHandler,
                                                SpatialBatchHandler_sf,
                                                SpatialBatchHandler_mom2)
from sup3r.utilities.utilities import (spatial_upsampling)

FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
TARGET_COORD = (39.01, -105.15)
FEATURES = ['U_100m', 'V_100m']


def test_out_spatial_mom1(plot=False, full_shape=(20, 20),
                          sample_shape=(10, 10, 1),
                          batch_size=4, n_batches=4,
                          s_enhance=2, model_dir=None):
    """Test basic spatial model outputing."""
    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
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
                                        sample_shape[1], 2)
        assert batch.low_res.shape == (batch_size,
                                       sample_shape[0] // s_enhance,
                                       sample_shape[1] // s_enhance, 2)
        out = model._tf_generate(batch.low_res)
        assert out.shape == (batch_size, sample_shape[0], sample_shape[1], 2)
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
            out = model._tf_generate(batch.low_res).numpy()
            for i in range(batch.high_res.shape[0]):
                lr = (batch.low_res[i, :, :, 0] * batch_handler.stds[0]
                      + batch_handler.means[0])
                hr = (batch.high_res[i, :, :, 0] * batch_handler.stds[0]
                      + batch_handler.means[0])
                gen = (out[i, :, :, 0] * batch_handler.stds[0]
                       + batch_handler.means[0])
                fig = plot_multi_contour(
                    [lr, hr, gen],
                    [0, batch.high_res.shape[1]],
                    [0, batch.high_res.shape[2]],
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
                             s_enhance=2, model_dir=None):
    """Test basic spatial model outputing."""
    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 10),
                            val_split=0,
                            max_workers=1)

    batch_handler = SpatialBatchHandler_sf([handler],
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
            out = model._tf_generate(batch.low_res).numpy()
            for i in range(batch.high_res.shape[0]):
                lr = (batch.low_res[i, :, :, 0] * batch_handler.stds[0]
                      + batch_handler.means[0])
                up_lr = spatial_upsampling(batch.low_res,
                                           s_enhance=s_enhance)
                hr_truth = ((batch.high_res[i, :, :, 0]
                            + up_lr[i, :, :, 0])
                            * batch_handler.stds[0]
                            + batch_handler.means[0])
                sf_truth = (batch.high_res[i, :, :, 0]
                            * batch_handler.stds[0])
                sf_pred = (out[i, :, :, 0]
                           * batch_handler.stds[0])
                hr_pred = ((out[i, :, :, 0]
                           + up_lr[i, :, :, 0])
                           * batch_handler.stds[0]
                           + batch_handler.means[0])
                fig = plot_multi_contour(
                    [lr, hr_truth, hr_pred, sf_truth, sf_pred],
                    [0, batch.high_res.shape[1]],
                    [0, batch.high_res.shape[2]],
                    ['U [m/s]', 'U [m/s]', 'U [m/s]',
                     'U [m/s]', 'U [m/s]'],
                    ['LR', 'HR', mom_name, 'SF', mom_name2],
                    ['x [m]', 'x [m]', 'x [m]', 'x [m]', 'x [m]'],
                    ['y [m]', 'y [m]', 'y [m]', 'y [m]', 'y [m]'],
                    [np.amin(lr), np.amin(hr_truth),
                     np.amin(hr_truth), np.amin(sf_truth),
                     np.amin(sf_truth)],
                    [np.amax(lr), np.amax(hr_truth),
                     np.amax(hr_truth), np.amax(sf_truth),
                     np.amax(sf_truth)],
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
                          model_mom1_dir=None):
    """Test basic spatial model outputing."""
    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
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

    batch_handler = SpatialBatchHandler_mom2([handler],
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

    # Check sizes
    for batch in batch_handler:
        assert batch.high_res.shape == (batch_size, sample_shape[0],
                                        sample_shape[1], 2)
        assert batch.low_res.shape == (batch_size,
                                       sample_shape[0] // s_enhance,
                                       sample_shape[1] // s_enhance, 2)
        out = model._tf_generate(batch.low_res)
        assert out.shape == (batch_size, sample_shape[0], sample_shape[1], 2)
        break
    if plot:
        import matplotlib.pyplot as plt
        from sup3r.utilities.plot_utilities import (plot_multi_contour,
                                                    makeMovie)
        figureFolder = 'Figures'
        os.makedirs(figureFolder, exist_ok=True)
        movieFolder = os.path.join(figureFolder, 'Movie')
        os.makedirs(movieFolder, exist_ok=True)
        mom_name = r'$\sigma$(HR|LR)'
        hr_name = r'HR - $\mathbb{E}$(HR|LR)'
        n_snap = 0
        for p, batch in enumerate(batch_handler):
            out = model._tf_generate(batch.low_res).numpy()
            for i in range(batch.high_res.shape[0]):
                lr = (batch.low_res[i, :, :, 0] * batch_handler.stds[0]
                      + batch_handler.means[0])
                hr = np.sqrt(batch.high_res[i, :, :, 0])
                gen = np.sqrt(np.clip(out[i, :, :, 0], a_min=0, a_max=None))
                fig = plot_multi_contour(
                    [lr, hr, gen],
                    [0, batch.high_res.shape[1]],
                    [0, batch.high_res.shape[2]],
                    ['U [m/s]', 'U [m/s]', 'U [m/s]'],
                    ['LR', hr_name, mom_name],
                    ['x [m]', 'x [m]', 'x [m]'],
                    ['y [m]', 'y [m]', 'y [m]'],
                    [np.amin(lr), np.amin(hr), np.amin(gen)],
                    [np.amax(lr), np.amax(hr), np.amax(gen)],
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


def test_out_loss(plot=False, model_dirs=None,
                  history_files=None, model_names=None):
    """Loss convergence plotting of multiple models"""
    # Load history
    if model_dirs is not None:
        history_files = [os.path.join(path, 'history.csv')
                         for path in model_dirs]
    elif history_files is not None:
        pass
    else:
        print("No history file provided")
        return

    # Get model names
    if model_names is None:
        model_names = ["model_" + str(i)
                       for i in range(len(history_files))]

    # Read csv
    histories = [read_csv(file) for file in history_files]
    if plot:
        import matplotlib.pyplot as plt
        import matplotlib.pylab as pl
        from sup3r.utilities.plot_utilities import (prettyLabels,
                                                    plotLegend)
        figureFolder = 'Figures'
        os.makedirs(figureFolder, exist_ok=True)
        figureLossFolder = os.path.join(figureFolder, 'loss')
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

    # test_out_loss(plot=True, model_dirs=['s_mom2.save/spatial_cond_mom'])

    # test_out_spatial_mom1(plot=True, full_shape=(20, 20),
    #                       sample_shape=(10, 10, 1),
    #                       batch_size=4, n_batches=4,
    #                       s_enhance=2, model_dir='s_mom1/spatial_cond_mom')
    test_out_spatial_mom1_sf(plot=True, full_shape=(20, 20),
                             sample_shape=(10, 10, 1),
                             batch_size=4, n_batches=4,
                             s_enhance=2,
                             model_dir='s_mom1_sf/spatial_cond_mom')

    # test_out_spatial_mom2(plot=True, full_shape=(20, 20),
    #                       sample_shape=(10, 10, 1),
    #                       batch_size=4, n_batches=4,
    #                       s_enhance=2, model_dir='s_mom2/spatial_cond_mom',
    #                       model_mom1_dir='s_mom1/spatial_cond_mom')
