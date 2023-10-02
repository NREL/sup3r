# -*- coding: utf-8 -*-
"""pytests for exogenous data handling"""
import os
import shutil

import numpy as np
import pytest

from sup3r import TEST_DATA_DIR
from sup3r.preprocessing.data_handling import ExogenousDataHandler

FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')

FILE_PATHS = [os.path.join(TEST_DATA_DIR, 'ua_test.nc'),
              os.path.join(TEST_DATA_DIR, 'va_test.nc'),
              os.path.join(TEST_DATA_DIR, 'orog_test.nc'),
              os.path.join(TEST_DATA_DIR, 'zg_test.nc')]
TARGET = (13.67, 125.0)
SHAPE = (8, 8)
S_ENHANCE = [1, 4]
T_ENHANCE = [1, 1]
S_AGG_FACTORS = [4, 1]
T_AGG_FACTORS = [1, 1]


@pytest.mark.parametrize('feature', ['topography', 'sza'])
def test_exo_cache(feature):
    """Test exogenous data caching and re-load"""
    # no cached data
    steps = []
    for s_en, t_en, s_agg, t_agg in zip(S_ENHANCE, T_ENHANCE, S_AGG_FACTORS,
                                        T_AGG_FACTORS):
        steps.append({'s_enhance': s_en,
                      't_enhance': t_en,
                      's_agg_factor': s_agg,
                      't_agg_factor': t_agg})
    try:
        base = ExogenousDataHandler(FILE_PATHS, feature,
                                    source_file=FP_WTK,
                                    steps=steps,
                                    target=TARGET, shape=SHAPE,
                                    input_handler='DataHandlerNCforCC')
        for i, arr in enumerate(base.data):
            assert arr.shape[0] == SHAPE[0] * S_ENHANCE[i]
            assert arr.shape[1] == SHAPE[1] * S_ENHANCE[i]
    except Exception as e:
        if os.path.exists('./exo_cache/'):
            shutil.rmtree('./exo_cache/')
        raise e
    else:
        assert os.path.exists('./exo_cache/')
        assert len(os.listdir('./exo_cache')) == 2

    # load cached data
    try:
        cache = ExogenousDataHandler(FILE_PATHS, feature,
                                     source_file=FP_WTK,
                                     steps=steps,
                                     target=TARGET, shape=SHAPE,
                                     input_handler='DataHandlerNCforCC')
    except Exception as e:
        if os.path.exists('./exo_cache/'):
            shutil.rmtree('./exo_cache/')
        raise e
    else:
        assert os.path.exists('./exo_cache/')
        assert len(os.listdir('./exo_cache')) == 2
        for arr1, arr2 in zip(base.data, cache.data):
            assert np.allclose(arr1, arr2)
        shutil.rmtree('./exo_cache/')
