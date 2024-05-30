"""Smoke tests for batcher objects. Just make sure things run without errors"""

import numpy as np
import pytest
from rex import init_logger
from scipy.ndimage import gaussian_filter

from sup3r.preprocessing import (
    BatchHandler,
)
from sup3r.utilities.pytest.helpers import (
    DummyData,
    execute_pytest,
)
from sup3r.utilities.utilities import spatial_coarsening, temporal_coarsening

init_logger('sup3r', log_level='DEBUG')

FEATURES = ['windspeed', 'winddirection']
means = dict.fromkeys(FEATURES, 0)
stds = dict.fromkeys(FEATURES, 1)


def test_batch_handler_with_validation():
    """Smoke test for batch queue."""

    coarsen_kwargs = {'smoothing_ignore': [], 'smoothing': None}
    batcher = BatchHandler(
        train_containers=[DummyData((10, 10, 100), FEATURES)],
        val_containers=[DummyData((10, 10, 100), FEATURES)],
        sample_shape=(8, 8, 4),
        batch_size=4,
        n_batches=3,
        s_enhance=2,
        t_enhance=1,
        queue_cap=10,
        means=means,
        stds=stds,
        max_workers=1,
        coarsen_kwargs=coarsen_kwargs,
    )

    assert len(batcher) == 3
    for b in batcher:
        assert b.low_res.shape == (4, 4, 4, 4, len(FEATURES))
        assert b.high_res.shape == (4, 8, 8, 4, len(FEATURES))
        assert b.low_res.dtype == np.float32
        assert b.high_res.dtype == np.float32

    assert len(batcher.val_data) == 3
    for b in batcher.val_data:
        assert b.low_res.shape == (4, 4, 4, 4, len(FEATURES))
        assert b.high_res.shape == (4, 8, 8, 4, len(FEATURES))
        assert b.low_res.dtype == np.float32
        assert b.high_res.dtype == np.float32
    batcher.stop()


@pytest.mark.parametrize(
    'method, t_enhance',
    [
        ('subsample', 2),
        ('average', 2),
        ('total', 2),
        ('subsample', 3),
        ('average', 3),
        ('total', 3),
        ('subsample', 4),
        ('average', 4),
        ('total', 4),
    ],
)
def test_temporal_coarsening(method, t_enhance):
    """Test temporal coarsening of batches"""

    sample_shape = (8, 8, 12)
    s_enhance = 2
    batch_size = 4
    coarsen_kwargs = {
        'smoothing_ignore': [],
        'smoothing': None,
        'temporal_coarsening_method': method,
    }
    batcher = BatchHandler(
        train_containers=[DummyData((10, 10, 100), FEATURES)],
        val_containers=[DummyData((10, 10, 100), FEATURES)],
        sample_shape=sample_shape,
        batch_size=batch_size,
        n_batches=3,
        s_enhance=s_enhance,
        t_enhance=t_enhance,
        queue_cap=10,
        means=means,
        stds=stds,
        max_workers=1,
        coarsen_kwargs=coarsen_kwargs,
    )

    for batch in batcher:
        assert batch.low_res.shape[0] == batch.high_res.shape[0]
        assert batch.low_res.shape == (
            batch_size,
            sample_shape[0] // s_enhance,
            sample_shape[1] // s_enhance,
            sample_shape[2] // t_enhance,
            len(FEATURES),
        )
        assert batch.high_res.shape == (
            batch_size,
            sample_shape[0],
            sample_shape[1],
            sample_shape[2],
            len(FEATURES),
        )
    batcher.stop()


def test_smoothing():
    """Check gaussian filtering on low res"""

    coarsen_kwargs = {
        'smoothing_ignore': [],
        'smoothing': 0.6,
    }
    s_enhance = 2
    t_enhance = 2
    sample_shape = (10, 10, 12)
    batch_size = 4
    batcher = BatchHandler(
        train_containers=[DummyData((10, 10, 100), FEATURES)],
        val_containers=[DummyData((10, 10, 100), FEATURES)],
        sample_shape=sample_shape,
        batch_size=batch_size,
        n_batches=3,
        s_enhance=s_enhance,
        t_enhance=t_enhance,
        queue_cap=10,
        means=means,
        stds=stds,
        max_workers=1,
        coarsen_kwargs=coarsen_kwargs,
    )

    for batch in batcher:
        high_res = batch.high_res
        low_res = spatial_coarsening(high_res, s_enhance)
        low_res = temporal_coarsening(low_res, t_enhance)
        low_res_no_smooth = low_res.copy()
        for i in range(low_res_no_smooth.shape[0]):
            for j in range(low_res_no_smooth.shape[-1]):
                for t in range(low_res_no_smooth.shape[-2]):
                    low_res[i, ..., t, j] = gaussian_filter(
                        low_res_no_smooth[i, ..., t, j], 0.6, mode='nearest')
        assert np.array_equal(batch.low_res, low_res)
        assert not np.array_equal(low_res, low_res_no_smooth)
    batcher.stop()


if __name__ == '__main__':
    execute_pytest(__file__)
