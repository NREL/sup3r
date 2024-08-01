"""Smoke tests for batcher objects. Just make sure things run without errors"""


import numpy as np
import pytest

from sup3r.utilities.pytest.helpers import (
    BatchHandlerTesterDC,
    DummyData,
)

FEATURES = ['windspeed', 'winddirection']
means = dict.fromkeys(FEATURES, 0)
stds = dict.fromkeys(FEATURES, 1)


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
    batcher = BatchHandlerTesterDC(
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
    batcher.update_weights(
        spatial_weights=s_weights, temporal_weights=t_weights
    )

    for _ in batcher:
        assert batcher.spatial_weights == s_weights
        assert batcher.temporal_weights == t_weights
    batcher.stop()

    s_normed = batcher._mean_record_normed(batcher.space_bin_record)
    assert np.allclose(
        s_normed,
        batcher._mean_record_normed(batcher.spatial_weights_record),
        atol=2 * s_normed.std(),
    )

    t_normed = batcher._mean_record_normed(batcher.time_bin_record)
    assert np.allclose(
        t_normed,
        batcher._mean_record_normed(batcher.temporal_weights_record),
        atol=2 * t_normed.std(),
    )
