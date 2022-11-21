# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN"""
import os
import numpy as np

from sup3r import TEST_DATA_DIR
from sup3r import CONFIG_DIR
from sup3r.models import Sup3rCondMom
from sup3r.preprocessing.data_handling import DataHandlerH5
from sup3r.preprocessing.batch_handling import (BatchHandler,
                                                BatchHandlerMom1SF,
                                                BatchHandlerMom2,
                                                BatchHandlerMom2SF)
from sup3r.utilities.utilities import (spatial_simple_enhancing,
                                       temporal_simple_enhancing)
from test_out_conditional_moments import test_out_loss

FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
TARGET_COORD = (39.01, -105.15)
FEATURES = ['U_100m', 'V_100m']


def test_out_st_mom1(plot=False, full_shape=(20, 20),
                     sample_shape=(12, 12, 4),
                     batch_size=4, n_batches=4,
                     s_enhance=2, t_enhance=4,
                     model_dir=None):
    """Test basic spatiotemporal model outputing for
    first conditional moment."""
    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 1),
                            val_split=0,
                            max_workers=1)

    batch_handler = BatchHandler([handler],
                                 batch_size=batch_size,
                                 s_enhance=s_enhance,
                                 t_enhance=t_enhance,
                                 n_batches=n_batches)

    # Load Model
    if model_dir is None:
        fp_gen = os.path.join(CONFIG_DIR,
                              'spatiotemporal',
                              'gen_3x_4x_2f_simple.json')
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
                lr = (batch.low_res[i, :, :, :, 0] * batch_handler.stds[0]
                      + batch_handler.means[0])
                aug_lr = np.reshape(lr, (1,) + lr.shape + (1,))
                tup_lr = temporal_simple_enhancing(aug_lr,
                                                   t_enhance=t_enhance)
                tup_lr = tup_lr[0, :, :, :, 0]
                hr = (batch.output[i, :, :, :, 0] * batch_handler.stds[0]
                      + batch_handler.means[0])
                gen = (out[i, :, :, :, 0] * batch_handler.stds[0]
                       + batch_handler.means[0])
                for j in range(batch.output.shape[3]):
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
        makeMovie(n_snap, movieFolder,
                  os.path.join(figureFolder, 'st_mom1.gif'),
                  fps=6)


def test_out_st_mom1_sf(plot=False, full_shape=(20, 20),
                        sample_shape=(12, 12, 24),
                        batch_size=4, n_batches=4,
                        s_enhance=2, t_enhance=4,
                        model_dir=None):
    """Test basic spatiotemporal model outputing for first conditional moment
    of subfilter velocity."""
    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 1),
                            val_split=0,
                            max_workers=1)

    batch_handler = BatchHandlerMom1SF([handler],
                                       batch_size=batch_size,
                                       s_enhance=s_enhance,
                                       t_enhance=t_enhance,
                                       n_batches=n_batches)

    # Load Model
    if model_dir is None:
        fp_gen = os.path.join(CONFIG_DIR,
                              'spatiotemporal',
                              'gen_3x_4x_2f_simple.json')
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

                b_lr = batch.low_res[i, :, :, :, 0]
                b_lr_aug = np.reshape(b_lr, (1,) + b_lr.shape + (1,))

                tup_lr = temporal_simple_enhancing(b_lr_aug,
                                                   t_enhance=t_enhance)
                tup_lr = (tup_lr[0, :, :, :, 0]
                          * batch_handler.stds[0]
                          + batch_handler.means[0])
                up_lr_tmp = spatial_simple_enhancing(b_lr_aug,
                                                     s_enhance=s_enhance)
                up_lr = temporal_simple_enhancing(up_lr_tmp,
                                                  t_enhance=t_enhance)
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

                for j in range(batch.output.shape[3]):
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
        makeMovie(n_snap, movieFolder,
                  os.path.join(figureFolder, 'st_mom1_sf.gif'),
                  fps=6)


def test_out_st_mom2(plot=False, full_shape=(20, 20),
                     sample_shape=(12, 12, 24),
                     batch_size=4, n_batches=4,
                     s_enhance=2, t_enhance=4,
                     model_dir=None,
                     model_mom1_dir=None):
    """Test basic spatiotemporal model outputing
    for second conditional moment."""
    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 1),
                            val_split=0,
                            max_workers=1)

    # Load Mom 1 Model
    if model_mom1_dir is None:
        fp_gen = os.path.join(CONFIG_DIR,
                              'spatiotemporal',
                              'gen_3x_4x_2f_simple.json')
        model_mom1 = Sup3rCondMom(fp_gen)
    else:
        fp_gen = os.path.join(model_mom1_dir, 'model_params.json')
        model_mom1 = Sup3rCondMom(fp_gen).load(model_mom1_dir)

    batch_handler = BatchHandlerMom2([handler],
                                     batch_size=batch_size,
                                     s_enhance=s_enhance,
                                     t_enhance=t_enhance,
                                     n_batches=n_batches,
                                     model_mom1=model_mom1)

    # Load Mom2 Model
    if model_dir is None:
        fp_gen = os.path.join(CONFIG_DIR,
                              'spatiotemporal',
                              'gen_3x_4x_2f_simple.json')
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

                b_lr = batch.low_res[i, :, :, :, 0]
                b_lr_aug = np.reshape(b_lr, (1,) + b_lr.shape + (1,))

                tup_lr = temporal_simple_enhancing(b_lr_aug,
                                                   t_enhance=t_enhance)
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
                for j in range(batch.output.shape[3]):
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
        makeMovie(n_snap, movieFolder,
                  os.path.join(figureFolder, 'st_mom2.gif'),
                  fps=6)


def test_out_st_mom2_sf(plot=False, full_shape=(20, 20),
                        sample_shape=(12, 12, 24),
                        batch_size=4, n_batches=4,
                        s_enhance=2, t_enhance=4,
                        model_dir=None,
                        model_mom1_dir=None):
    """Test basic spatiotemporal model outputing for second conditional moment
    of subfilter velocity."""
    handler = DataHandlerH5(FP_WTK, FEATURES, target=TARGET_COORD,
                            shape=full_shape,
                            sample_shape=sample_shape,
                            temporal_slice=slice(None, None, 1),
                            val_split=0,
                            max_workers=1)

    # Load Mom 1 Model
    if model_mom1_dir is None:
        fp_gen = os.path.join(CONFIG_DIR,
                              'spatiotemporal',
                              'gen_3x_4x_2f_simple.json')
        model_mom1 = Sup3rCondMom(fp_gen)
    else:
        fp_gen = os.path.join(model_mom1_dir, 'model_params.json')
        model_mom1 = Sup3rCondMom(fp_gen).load(model_mom1_dir)

    batch_handler = BatchHandlerMom2SF([handler],
                                       batch_size=batch_size,
                                       s_enhance=s_enhance,
                                       t_enhance=t_enhance,
                                       n_batches=n_batches,
                                       model_mom1=model_mom1)

    # Load Mom2 Model
    if model_dir is None:
        fp_gen = os.path.join(CONFIG_DIR,
                              'spatiotemporal',
                              'gen_3x_4x_2f_simple_mom2.json')
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
                b_lr = batch.low_res[i, :, :, :, 0]
                b_lr_aug = np.reshape(b_lr, (1,) + b_lr.shape + (1,))

                tup_lr = temporal_simple_enhancing(b_lr_aug,
                                                   t_enhance=t_enhance)
                tup_lr = (tup_lr[0, :, :, :, 0]
                          * batch_handler.stds[0]
                          + batch_handler.means[0])

                up_lr_tmp = spatial_simple_enhancing(b_lr_aug,
                                                     s_enhance=s_enhance)
                up_lr = temporal_simple_enhancing(up_lr_tmp,
                                                  t_enhance=t_enhance)
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
                for j in range(batch.output.shape[3]):
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
        makeMovie(n_snap, movieFolder,
                  os.path.join(figureFolder, 'st_mom2_sf.gif'),
                  fps=6)


if __name__ == "__main__":

    test_out_loss(plot=True, model_dirs=['st_mom1/st_cond_mom'],
                  figureDir='st_mom1_loss')

    test_out_loss(plot=True, model_dirs=['st_mom2/st_cond_mom'],
                  figureDir='st_mom2_loss')

    test_out_loss(plot=True, model_dirs=['st_mom1_sf/st_cond_mom'],
                  figureDir='st_mom1_sf_loss')

    test_out_loss(plot=True, model_dirs=['st_mom2_sf/st_cond_mom'],
                  figureDir='st_mom2_sf_loss')

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
