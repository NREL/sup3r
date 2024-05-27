"""Test the basic training of super resolution GAN for solar climate change
applications"""

import os
import tempfile

import numpy as np
import pytest
from rex import init_logger

from sup3r import CONFIG_DIR, TEST_DATA_DIR
from sup3r.containers import (
    BatchHandlerCC,
    DataHandlerH5WindCC,
)
from sup3r.models import Sup3rGan

SHAPE = (20, 20)

INPUT_FILE_S = os.path.join(TEST_DATA_DIR, 'test_nsrdb_co_2018.h5')
FEATURES_S = ['clearsky_ratio', 'ghi', 'clearsky_ghi']
TARGET_S = (39.01, -105.13)

INPUT_FILE_W = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
FEATURES_W = ['temperature_100m', 'U_100m', 'V_100m', 'topography']
TARGET_W = (39.01, -105.15)

FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
TARGET_COORD = (39.01, -105.15)


init_logger('sup3r', log_level='DEBUG')

np.random.seed(42)


@pytest.mark.parametrize([('CustomLayer', 'features', 'lr_only_features')],
                         [('Sup3rAdder', FEATURES_W, ['temperature_100m']),
                         ('Sup3rConcat', FEATURES_W, ['temperature_100m']),
                         ('Sup3rAdder', FEATURES_W[1:], []),
                         ('Sup3rConcat', FEATURES_W[1:], [])])
def test_wind_hi_res_topo(CustomLayer, features, lr_only_features):
    """Test a special wind cc model with the custom Sup3rAdder or Sup3rConcat
    layer that adds/concatenates hi-res topography in the middle of the
    network. The first two parameter sets include an lr only feature."""

    handler = DataHandlerH5WindCC(
        INPUT_FILE_W,
        features,
        target=TARGET_W,
        shape=SHAPE,
        time_slice=slice(None, None, 2),
        time_roll=-7,
    )
    batcher = BatchHandlerCC(
        [handler],
        batch_size=2,
        n_batches=2,
        s_enhance=2,
        sample_shape=(20, 20),
        feature_sets={
            'lr_only_features': lr_only_features,
            'hr_exo_features': ['topography'],
        },
    )

    gen_model = [
        {
            'class': 'FlexiblePadding',
            'paddings': [[0, 0], [3, 3], [3, 3], [0, 0]],
            'mode': 'REFLECT',
        },
        {
            'class': 'Conv2DTranspose',
            'filters': 64,
            'kernel_size': 3,
            'strides': 1,
            'activation': 'relu',
        },
        {'class': 'Cropping2D', 'cropping': 4},
        {
            'class': 'FlexiblePadding',
            'paddings': [[0, 0], [3, 3], [3, 3], [0, 0]],
            'mode': 'REFLECT',
        },
        {
            'class': 'Conv2DTranspose',
            'filters': 64,
            'kernel_size': 3,
            'strides': 1,
            'activation': 'relu',
        },
        {'class': 'Cropping2D', 'cropping': 4},
        {
            'class': 'FlexiblePadding',
            'paddings': [[0, 0], [3, 3], [3, 3], [0, 0]],
            'mode': 'REFLECT',
        },
        {
            'class': 'Conv2DTranspose',
            'filters': 64,
            'kernel_size': 3,
            'strides': 1,
            'activation': 'relu',
        },
        {'class': 'Cropping2D', 'cropping': 4},
        {'class': 'SpatialExpansion', 'spatial_mult': 2},
        {'class': 'Activation', 'activation': 'relu'},
        {'class': CustomLayer, 'name': 'topography'},
        {
            'class': 'FlexiblePadding',
            'paddings': [[0, 0], [3, 3], [3, 3], [0, 0]],
            'mode': 'REFLECT',
        },
        {
            'class': 'Conv2DTranspose',
            'filters': 2,
            'kernel_size': 3,
            'strides': 1,
            'activation': 'relu',
        },
        {'class': 'Cropping2D', 'cropping': 4},
    ]

    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(gen_model, fp_disc, learning_rate=1e-4)

    with tempfile.TemporaryDirectory() as td:
        model.train(
            batcher,
            input_resolution={'spatial': '16km', 'temporal': '3600min'},
            n_epoch=1,
            weight_gen_advers=0.0,
            train_gen=True,
            train_disc=False,
            checkpoint_int=None,
            out_dir=os.path.join(td, 'test_{epoch}'),
        )

        assert model.lr_features == features
        assert model.hr_out_features == ['U_100m', 'V_100m']
        assert model.hr_exo_features == ['topography']
        assert 'test_0' in os.listdir(td)
        assert model.meta['hr_out_features'] == ['U_100m', 'V_100m']
        assert model.meta['class'] == 'Sup3rGan'
        assert 'topography' in batcher.hr_exo_features
        assert 'topography' not in model.hr_out_features

    x = np.random.uniform(0, 1, (4, 30, 30, len(features)))
    hi_res_topo = np.random.uniform(0, 1, (4, 60, 60, 1))

    with pytest.raises(RuntimeError):
        y = model.generate(x, exogenous_data=None)

    exo_tmp = {
        'topography': {
            'steps': [
                {'model': 0, 'combine_type': 'layer', 'data': hi_res_topo}
            ]
        }
    }
    y = model.generate(x, exogenous_data=exo_tmp)

    assert y.dtype == np.float32
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == x.shape[1] * 2
    assert y.shape[2] == x.shape[2] * 2
    assert y.shape[3] == x.shape[3] - 2
