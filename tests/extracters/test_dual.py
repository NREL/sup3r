# -*- coding: utf-8 -*-
"""Test the basic training of super resolution GAN"""

import os
import tempfile

import dask.array as da
from rex import init_logger

from sup3r import TEST_DATA_DIR
from sup3r.containers import (
    DataHandlerH5,
    DataHandlerNC,
    DualExtracter,
    LoaderH5,
)
from sup3r.utilities.pytest.helpers import execute_pytest

FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
FP_ERA = os.path.join(TEST_DATA_DIR, 'test_era5_co_2012.nc')
TARGET_COORD = (39.01, -105.15)
FEATURES = ['U_100m', 'V_100m']


init_logger('sup3r')


def test_pair_extracter_shapes(log=False, full_shape=(20, 20)):
    """Test basic spatial model training with only gen content loss."""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    # need to reduce the number of temporal examples to test faster
    hr_container = DataHandlerH5(
        file_paths=FP_WTK,
        features=FEATURES,
        target=TARGET_COORD,
        shape=full_shape,
        time_slice=slice(None, None, 10),
    )
    lr_container = DataHandlerNC(
        file_paths=FP_ERA,
        load_features=FEATURES,
        features=FEATURES,
        time_slice=slice(None, None, 10),
    )

    pair_extracter = DualExtracter(
        lr_container, hr_container, s_enhance=2, t_enhance=1
    )
    assert pair_extracter.lr_container.shape == (
        pair_extracter.hr_container.shape[0] // 2,
        pair_extracter.hr_container.shape[1] // 2,
        *pair_extracter.hr_container.shape[2:],
    )


def test_regrid_caching(log=False, full_shape=(20, 20)):
    """Test caching and loading of regridded data"""
    if log:
        init_logger('sup3r', log_level='DEBUG')

    # need to reduce the number of temporal examples to test faster
    with tempfile.TemporaryDirectory() as td:
        hr_container = DataHandlerH5(
            file_paths=FP_WTK,
            features=FEATURES,
            target=TARGET_COORD,
            shape=full_shape,
            time_slice=slice(None, None, 10),
        )
        lr_container = DataHandlerNC(
            file_paths=FP_ERA,
            load_features=FEATURES,
            features=FEATURES,
            time_slice=slice(None, None, 10),
        )
        lr_cache_pattern = os.path.join(td, 'lr_{feature}.h5')
        hr_cache_pattern = os.path.join(td, 'hr_{feature}.h5')
        pair_extracter = DualExtracter(
            lr_container,
            hr_container,
            s_enhance=2,
            t_enhance=1,
            lr_cache_kwargs={'cache_pattern': lr_cache_pattern},
            hr_cache_kwargs={'cache_pattern': hr_cache_pattern},
        )

        # Load handlers again
        lr_container_new = LoaderH5(
            [
                lr_cache_pattern.format(feature=f)
                for f in lr_container.features
            ],
            lr_container.features,
        )
        hr_container_new = LoaderH5(
            [
                hr_cache_pattern.format(feature=f)
                for f in hr_container.features
            ],
            features=hr_container.features,
        )

        assert da.map_blocks(
            lambda x, y: x == y,
            lr_container_new.data,
            pair_extracter.lr_container.data,
        ).all()
        assert da.map_blocks(
            lambda x, y: x == y,
            hr_container_new.data,
            pair_extracter.hr_container.data,
        ).all()


if __name__ == '__main__':
    execute_pytest(__file__)
