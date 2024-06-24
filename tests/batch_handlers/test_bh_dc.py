"""Smoke tests for batcher objects. Just make sure things run without errors"""

import numpy as np
import pytest
from rex import init_logger

from sup3r.utilities.pytest.helpers import (
    DummyData,
    TestBatchHandlerDC,
    execute_pytest,
)

init_logger('sup3r', log_level='DEBUG')

FEATURES = ['windspeed', 'winddirection']
means = dict.fromkeys(FEATURES, 0)
stds = dict.fromkeys(FEATURES, 1)


np.random.seed(42)


@pytest.mark.parametrize(
    ('s_weights', 't_weights'),
    [
        ([0.25, 0.25, 0.25, 0.25], [1.0]),
        ([0.5, 0.0, 0.25, 0.25], [1.0]),
        ([0, 1, 0, 0], [0.25, 0.25, 0.25, 0.25]),
        ([0, 0.5, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]),
        ([0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]),
        ([0.25, 0.25, 0.25, 0.25], [0.0, 0.0, 0.5, 0.5]),
        ([0.75, 0.25, 0.0, 0.0], [0.0, 0.0, 0.75, 0.25]),
    ],
)
def test_counts(s_weights, t_weights):
    """Make sure dc batch handler returns the correct number of samples for
    each bin."""

    dat = DummyData((10, 10, 100), FEATURES)
    n_batches = 4
    batch_size = 50
    batcher = TestBatchHandlerDC(
        train_containers=[dat],
        val_containers=[dat],
        sample_shape=(4, 4, 4),
        batch_size=batch_size,
        n_batches=n_batches,
        queue_cap=1,
        s_enhance=2,
        t_enhance=1,
        means=means,
        stds=stds,
        max_workers=1,
        n_time_bins=len(t_weights),
        n_space_bins=len(s_weights),
    )
    assert batcher.val_data.n_batches == len(s_weights) * len(t_weights)
    batcher.update_spatial_weights(s_weights)
    batcher.update_temporal_weights(t_weights)

    for _ in batcher:
        assert batcher.spatial_weights == s_weights
        assert batcher.temporal_weights == t_weights

    assert np.allclose(
        batcher._space_norm_count(),
        batcher.spatial_weights,
        atol=2 * batcher._space_norm_count().std(),
    )
    assert np.allclose(
        batcher._time_norm_count(),
        batcher.temporal_weights,
        atol=2 * batcher._time_norm_count().std(),
    )
    batcher.stop()


if __name__ == '__main__':
    execute_pytest(__file__)
