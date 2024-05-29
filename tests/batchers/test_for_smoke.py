"""Smoke tests for batcher objects. Just make sure things run without errors"""

import numpy as np
import pytest
from rex import init_logger
from scipy.ndimage import gaussian_filter

from sup3r.preprocessing import (
    BatchHandler,
    DualBatchQueue,
    DualContainer,
    DualSampler,
    SingleBatchQueue,
)
from sup3r.utilities.pytest.helpers import (
    DummyData,
    DummySampler,
    execute_pytest,
)
from sup3r.utilities.utilities import spatial_coarsening, temporal_coarsening

init_logger('sup3r', log_level='DEBUG')

FEATURES = ['windspeed', 'winddirection']
means = dict.fromkeys(FEATURES, 0)
stds = dict.fromkeys(FEATURES, 1)


def test_not_enough_stats_for_batch_queue():
    """Negative test for not enough means / stds for given features."""

    samplers = [
        DummySampler(
            sample_shape=(8, 8, 10), data_shape=(10, 10, 20), features=FEATURES
        ),
        DummySampler(
            sample_shape=(8, 8, 10), data_shape=(12, 12, 15), features=FEATURES
        ),
    ]
    coarsen_kwargs = {'smoothing_ignore': [], 'smoothing': None}

    with pytest.raises(AssertionError):
        _ = SingleBatchQueue(
            samplers=samplers,
            n_batches=3,
            batch_size=4,
            s_enhance=2,
            t_enhance=2,
            means={'windspeed': 4},
            stds={'windspeed': 2},
            queue_cap=10,
            max_workers=1,
            coarsen_kwargs=coarsen_kwargs,
        )


def test_batch_queue():
    """Smoke test for batch queue."""

    sample_shape = (8, 8, 10)
    samplers = [
        DummySampler(sample_shape, data_shape=(10, 10, 20), features=FEATURES),
        DummySampler(sample_shape, data_shape=(12, 12, 15), features=FEATURES),
    ]
    coarsen_kwargs = {'smoothing_ignore': [], 'smoothing': None}
    batcher = SingleBatchQueue(
        samplers=samplers,
        n_batches=3,
        batch_size=4,
        s_enhance=2,
        t_enhance=2,
        means=means,
        stds=stds,
        queue_cap=10,
        max_workers=1,
        coarsen_kwargs=coarsen_kwargs,
    )
    batcher.start()
    assert len(batcher) == 3
    for b in batcher:
        assert b.low_res.shape == (4, 4, 4, 5, len(FEATURES))
        assert b.high_res.shape == (4, 8, 8, 10, len(FEATURES))
    batcher.stop()


def test_spatial_batch_queue():
    """Smoke test for spatial batch queue. A batch queue returns batches for
    spatial models if the sample shapes have 1 for the time axis"""
    sample_shape = (8, 8)
    s_enhance = 2
    t_enhance = 1
    batch_size = 4
    queue_cap = 10
    n_batches = 3
    coarsen_kwargs = {'smoothing_ignore': [], 'smoothing': None}
    samplers = [
        DummySampler(sample_shape, data_shape=(10, 10, 20), features=FEATURES),
        DummySampler(sample_shape, data_shape=(12, 12, 15), features=FEATURES),
    ]
    batcher = SingleBatchQueue(
        samplers=samplers,
        s_enhance=s_enhance,
        t_enhance=t_enhance,
        n_batches=n_batches,
        batch_size=batch_size,
        queue_cap=queue_cap,
        means=means,
        stds=stds,
        max_workers=1,
        coarsen_kwargs=coarsen_kwargs,
    )
    batcher.start()
    assert len(batcher) == 3
    for b in batcher:
        assert b.low_res.shape == (
            batch_size,
            sample_shape[0] // s_enhance,
            sample_shape[1] // s_enhance,
            len(FEATURES),
        )
        assert b.high_res.shape == (batch_size, *sample_shape, len(FEATURES))
    batcher.stop()


def test_dual_batch_queue():
    """Smoke test for paired batch queue."""
    lr_sample_shape = (4, 4, 5)
    hr_sample_shape = (8, 8, 10)
    lr_containers = [
        DummyData(
            data_shape=(10, 10, 20),
            features=FEATURES,
        ),
        DummyData(
            data_shape=(12, 12, 15),
            features=FEATURES,
        ),
    ]
    hr_containers = [
        DummyData(
            data_shape=(20, 20, 40),
            features=FEATURES,
        ),
        DummyData(
            data_shape=(24, 24, 30),
            features=FEATURES,
        ),
    ]
    sampler_pairs = [
        DualSampler(
            DualContainer(lr, hr), hr_sample_shape, s_enhance=2, t_enhance=2
        )
        for lr, hr in zip(lr_containers, hr_containers)
    ]
    batcher = DualBatchQueue(
        samplers=sampler_pairs,
        s_enhance=2,
        t_enhance=2,
        n_batches=3,
        batch_size=4,
        queue_cap=10,
        means=means,
        stds=stds,
        max_workers=1,
    )
    batcher.start()
    assert len(batcher) == 3
    for b in batcher:
        assert b.low_res.shape == (4, *lr_sample_shape, len(FEATURES))
        assert b.high_res.shape == (4, *hr_sample_shape, len(FEATURES))
    batcher.stop()


def test_pair_batch_queue_with_lr_only_features():
    """Smoke test for paired batch queue with an extra lr_only_feature."""
    lr_sample_shape = (4, 4, 5)
    hr_sample_shape = (8, 8, 10)
    lr_only_features = ['dummy_lr_feat']
    lr_features = [*lr_only_features, *FEATURES]
    lr_containers = [
        DummyData(
            data_shape=(10, 10, 20),
            features=lr_features,
        ),
        DummyData(
            data_shape=(12, 12, 15),
            features=lr_features,
        ),
    ]
    hr_containers = [
        DummyData(
            data_shape=(20, 20, 40),
            features=FEATURES,
        ),
        DummyData(
            data_shape=(24, 24, 30),
            features=FEATURES,
        ),
    ]
    sampler_pairs = [
        DualSampler(
            DualContainer(lr, hr),
            hr_sample_shape,
            s_enhance=2,
            t_enhance=2,
            feature_sets={'lr_only_features': lr_only_features},
        )
        for lr, hr in zip(lr_containers, hr_containers)
    ]
    means = dict.fromkeys(lr_features, 0)
    stds = dict.fromkeys(lr_features, 1)
    batcher = DualBatchQueue(
        samplers=sampler_pairs,
        s_enhance=2,
        t_enhance=2,
        n_batches=3,
        batch_size=4,
        queue_cap=10,
        means=means,
        stds=stds,
        max_workers=1,
    )
    batcher.start()
    assert len(batcher) == 3
    for b in batcher:
        assert b.low_res.shape == (4, *lr_sample_shape, len(lr_features))
        assert b.high_res.shape == (4, *hr_sample_shape, len(FEATURES))
    batcher.stop()


def test_bad_enhancement_factors():
    """Failure when enhancement factors given to BatchQueue do not match those
    given to the contained DualSamplers, and when those given to DualSampler
    are not consistent with the low / high res shapes."""
    hr_sample_shape = (8, 8, 10)
    lr_containers = [
        DummyData(
            data_shape=(10, 10, 20),
            features=FEATURES,
        ),
        DummyData(
            data_shape=(12, 12, 15),
            features=FEATURES,
        ),
    ]
    hr_containers = [
        DummyData(
            data_shape=(20, 20, 40),
            features=FEATURES,
        ),
        DummyData(
            data_shape=(24, 24, 30),
            features=FEATURES,
        ),
    ]
    for s_enhance, t_enhance in zip([2, 4], [2, 6]):
        with pytest.raises(AssertionError):
            sampler_pairs = [
                DualSampler(
                    DualContainer(lr, hr),
                    hr_sample_shape,
                    s_enhance=s_enhance,
                    t_enhance=t_enhance,
                )
                for lr, hr in zip(lr_containers, hr_containers)
            ]
            _ = DualBatchQueue(
                samplers=sampler_pairs,
                s_enhance=4,
                t_enhance=6,
                n_batches=3,
                batch_size=4,
                queue_cap=10,
                means=means,
                stds=stds,
                max_workers=1,
            )


def test_bad_sample_shapes():
    """Failure when sample shapes are not consistent across a collection of
    samplers."""

    samplers = [
        DummySampler(
            sample_shape=(4, 4, 5), data_shape=(10, 10, 20), features=FEATURES
        ),
        DummySampler(
            sample_shape=(3, 3, 5), data_shape=(12, 12, 15), features=FEATURES
        ),
    ]

    with pytest.raises(AssertionError):
        _ = SingleBatchQueue(
            samplers=samplers,
            s_enhance=4,
            t_enhance=6,
            n_batches=3,
            batch_size=4,
            queue_cap=10,
            means=means,
            stds=stds,
            max_workers=1,
        )


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
