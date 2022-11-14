# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN"""
import os
# import json
import numpy as np

from sup3r import TEST_DATA_DIR
from sup3r import CONFIG_DIR
from sup3r.models import Sup3rCondMom
from sup3r.preprocessing.data_handling import DataHandlerH5
from sup3r.preprocessing.batch_handling import (SpatialBatchHandler,
                                                SpatialBatchHandler_mom2)


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


if __name__ == "__main__":

    test_out_spatial_mom1(plot=True, full_shape=(20, 20),
                          sample_shape=(10, 10, 1),
                          batch_size=4, n_batches=4,
                          s_enhance=2, model_dir='s_mom1/spatial_cond_mom')

    test_out_spatial_mom2(plot=True, full_shape=(20, 20),
                          sample_shape=(10, 10, 1),
                          batch_size=4, n_batches=4,
                          s_enhance=2, model_dir='s_mom2/spatial_cond_mom',
                          model_mom1_dir='s_mom1/spatial_cond_mom')
