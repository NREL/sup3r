"""pytests for data handling"""

import os
import numpy as np
import pytest
import matplotlib.pyplot as plt

from sup3r import TEST_DATA_DIR
from sup3r.data_handling.preprocessing import (DataHandler,
                                               SpatialBatchHandler)
from sup3r.utilities import utilities

input_file = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
target = (39.01, -105.15)
shape = (20, 20)
features = ['windspeed_100m', 'winddirection_100m']
batch_size = 14
spatial_res = 5
max_delta = 20
val_split = 0.2

batch_handler = SpatialBatchHandler.make([input_file], target,
                                         shape, features,
                                         batch_size=batch_size,
                                         spatial_res=spatial_res,
                                         max_delta=max_delta,
                                         val_split=val_split)


def test_data_extraction():
    """Test data extraction class"""

    handler = DataHandler([input_file], target, shape, features, max_delta)
    print(features)

    assert handler.data.shape == (shape[0], shape[1],
                                  handler.data.shape[2], len(features))


def test_batch_handling():
    """Test spatial batch handling class"""

    for batch in batch_handler:
        assert batch.low_res.shape == (batch_size, shape[0] // spatial_res,
                                       shape[1] // spatial_res, len(features))
        assert batch.high_res.shape == (batch_size, shape[0],
                                        shape[1], len(features))


@pytest.mark.parametrize(
    ('spatial_res', 'plot'),
    ((10, False),
     (5, False),
     (4, False),
     (2, False),
     (2, True))
)
def test_spatial_coarsening(spatial_res, plot):
    """Test spatial coarsening"""

    handler = DataHandler([input_file], target, shape, features, max_delta)

    coarse_data = utilities.spatial_coarsening(handler.data, spatial_res)
    direct_avg = np.zeros((handler.data.shape[0] // spatial_res,
                           handler.data.shape[1] // spatial_res,
                           handler.data.shape[2],
                           handler.data.shape[3]), dtype=np.float32) \

    for i in range(direct_avg.shape[0]):
        for j in range(direct_avg.shape[1]):
            direct_avg[i, j, :, :] = \
                np.mean(handler.data[spatial_res * i:spatial_res * (i + 1),
                                     spatial_res * j:spatial_res * (j + 1),
                                     :, :], axis=(0, 1))

    np.testing.assert_equal(coarse_data, direct_avg)

    if plot:
        _, ax = plt.subplots(1, 2)
        ax[0].imshow(handler.data[:, :, 0, 0])
        ax[1].imshow(coarse_data[:, :, 0, 0])
        plt.show()
