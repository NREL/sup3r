"""Smoke tests for batcher objects. Just make sure things run without errors"""

import copy

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from sup3r.preprocessing import (
    BatchHandler,
)
from sup3r.preprocessing.base import Container
from sup3r.utilities.pytest.helpers import (
    BatchHandlerTesterFactory,
    DummyData,
    SamplerTester,
)
from sup3r.utilities.utilities import spatial_coarsening, temporal_coarsening

FEATURES = ['windspeed', 'winddirection']
means = dict.fromkeys(FEATURES, 0)
stds = dict.fromkeys(FEATURES, 1)

np.random.seed(42)


BatchHandlerTester = BatchHandlerTesterFactory(BatchHandler, SamplerTester)


def test_eager_vs_lazy():
    """Make sure eager and lazy loading agree."""

    eager_data = DummyData((10, 10, 100), FEATURES)
    lazy_data = Container(copy.deepcopy(eager_data.data))
    kwargs = {
        'val_containers': [],
        'sample_shape': (8, 8, 4),
        'batch_size': 4,
        'n_batches': 4,
        's_enhance': 2,
        't_enhance': 1,
        'queue_cap': 3,
        'means': means,
        'stds': stds,
        'max_workers': 1,
    }

    lazy_batcher = BatchHandlerTester(
        [lazy_data],
        **kwargs,
        mode='lazy',
    )
    eager_batcher = BatchHandlerTester(
        train_containers=[eager_data],
        **kwargs,
        mode='eager',
    )

    assert eager_batcher.loaded
    assert not lazy_batcher.loaded

    assert np.array_equal(
        eager_batcher.data[0].as_array(),
        lazy_batcher.data[0].as_array().compute(),
    )

    np.random.seed(42)
    eager_batches = list(eager_batcher)
    eager_batcher.stop()
    np.random.seed(42)
    lazy_batches = list(lazy_batcher)
    lazy_batcher.stop()

    for eb, lb in zip(eager_batches, lazy_batches):
        assert np.array_equal(eb.high_res, lb.high_res)
        assert np.array_equal(eb.low_res, lb.low_res)


@pytest.mark.parametrize('n_epochs', [1, 2, 3, 4])
def test_sample_counter(n_epochs):
    """Make sure samples are counted correctly, over multiple epochs."""

    dat = DummyData((10, 10, 100), FEATURES)
    batcher = BatchHandlerTester(
        train_containers=[dat],
        val_containers=[],
        sample_shape=(8, 8, 4),
        batch_size=4,
        n_batches=4,
        s_enhance=2,
        t_enhance=1,
        queue_cap=1,
        means=means,
        stds=stds,
        max_workers=1,
        mode='eager',
    )

    for _ in range(n_epochs):
        for _ in batcher:
            pass
    batcher.stop()

    assert (
        batcher.sample_count // batcher.batch_size
        == n_epochs * batcher.n_batches + batcher.queue.size().numpy()
    )


def test_normalization():
    """Smoke test for batch queue."""

    means = {'windspeed': 2, 'winddirection': 5}
    stds = {'windspeed': 6.5, 'winddirection': 8.2}

    dat = DummyData((10, 10, 100), FEATURES)
    dat.data['windspeed', ...] = 1
    dat.data['windspeed', 0:4] = np.nan
    dat.data['winddirection', ...] = 1
    dat.data['winddirection', 0:4] = np.nan

    transform_kwargs = {'smoothing_ignore': [], 'smoothing': None}
    batcher = BatchHandler(
        train_containers=[dat],
        val_containers=[dat],
        sample_shape=(8, 8, 4),
        batch_size=4,
        n_batches=3,
        s_enhance=2,
        t_enhance=1,
        queue_cap=10,
        means=means,
        stds=stds,
        max_workers=1,
        transform_kwargs=transform_kwargs,
    )

    means = list(means.values())
    stds = list(stds.values())

    assert len(batcher) == 3
    for b in batcher:
        assert round(np.nanmean(b.low_res[..., 0]) * stds[0] + means[0]) == 1
        assert round(np.nanmean(b.low_res[..., 1]) * stds[1] + means[1]) == 1
        assert round(np.nanmean(b.high_res[..., 0]) * stds[0] + means[0]) == 1
        assert round(np.nanmean(b.high_res[..., 1]) * stds[1] + means[1]) == 1

    assert len(batcher.val_data) == 3
    for b in batcher.val_data:
        assert round(np.nanmean(b.low_res[..., 0]) * stds[0] + means[0]) == 1
        assert round(np.nanmean(b.low_res[..., 1]) * stds[1] + means[1]) == 1
        assert round(np.nanmean(b.high_res[..., 0]) * stds[0] + means[0]) == 1
        assert round(np.nanmean(b.high_res[..., 1]) * stds[1] + means[1]) == 1
    batcher.stop()


def test_batch_handler_with_validation():
    """Smoke test for batch queue."""

    transform_kwargs = {'smoothing_ignore': [], 'smoothing': None}
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
        transform_kwargs=transform_kwargs,
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
    transform_kwargs = {
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
        transform_kwargs=transform_kwargs,
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

    transform_kwargs = {
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
        transform_kwargs=transform_kwargs,
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
                        low_res_no_smooth[i, ..., t, j], 0.6, mode='nearest'
                    )
        assert np.array_equal(batch.low_res, low_res)
        assert not np.array_equal(low_res, low_res_no_smooth)
    batcher.stop()
