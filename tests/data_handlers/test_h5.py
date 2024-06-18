"""Tets H5 data handling by composite handler objects"""

import os

import numpy as np

from sup3r import TEST_DATA_DIR
from sup3r.preprocessing import BatchHandler, DataHandlerH5, Sampler
from sup3r.utilities.pytest.helpers import execute_pytest

sample_shape = (10, 10, 12)
t_enhance = 2
s_enhance = 5


def test_solar_spatial_h5():
    """Test solar spatial batch handling with NaN drop."""
    input_file_s = os.path.join(TEST_DATA_DIR, 'test_nsrdb_co_2018.h5')
    features_s = ['clearsky_ratio']
    target_s = (39.01, -105.13)
    dh_nan = DataHandlerH5(
        input_file_s, features=features_s, target=target_s, shape=(20, 20)
    )
    dh = DataHandlerH5(
        input_file_s, features=features_s, target=target_s, shape=(20, 20),
        nan_mask=True)

    assert np.nanmax(dh.as_array()) == 1
    assert np.nanmin(dh.as_array()) == 0
    assert not np.isnan(dh.as_array()).any()
    assert np.isnan(dh_nan.as_array()).any()
    sampler = Sampler(dh.data, sample_shape=(10, 10, 12))
    for _ in range(10):
        x = next(sampler)
        assert x.shape == (10, 10, 12, 1)
        assert not np.isnan(x).any()

    batch_handler = BatchHandler(
        [dh],
        val_containers=[],
        batch_size=8,
        n_batches=20,
        sample_shape=(10, 10, 1),
        s_enhance=s_enhance,
        t_enhance=1,
    )
    for batch in batch_handler:
        assert not np.isnan(batch.low_res).any()
        assert not np.isnan(batch.high_res).any()
        assert batch.low_res.shape == (8, 2, 2, 1)
        assert batch.high_res.shape == (8, 10, 10, 1)

    batch_handler.stop()


if __name__ == '__main__':
    execute_pytest(__file__)
