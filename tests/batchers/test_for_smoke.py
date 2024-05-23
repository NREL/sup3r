"""Smoke tests for batcher objects. Just make sure things run without errors"""

import pytest
from rex import init_logger

from sup3r.containers import (
    BatchQueue,
    DualBatchQueue,
    DualContainer,
    DualSampler,
)
from sup3r.utilities.pytest.helpers import (
    DummyCroppedSampler,
    DummyData,
    DummySampler,
    execute_pytest,
)

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
        _ = BatchQueue(
            train_containers=samplers,
            val_containers=[],
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
    batcher = BatchQueue(
        train_containers=samplers,
        val_containers=[],
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
    batcher = BatchQueue(
        train_containers=samplers,
        val_containers=[],
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


def test_pair_batch_queue():
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
        train_containers=sampler_pairs,
        val_containers=[],
        s_enhance=2,
        t_enhance=2,
        n_batches=3,
        batch_size=4,
        queue_cap=10,
        means=means,
        stds=stds,
        max_workers=1,
    )
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
        train_containers=sampler_pairs,
        val_containers=[],
        s_enhance=2,
        t_enhance=2,
        n_batches=3,
        batch_size=4,
        queue_cap=10,
        means=means,
        stds=stds,
        max_workers=1,
    )
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
                train_containers=sampler_pairs,
                val_containers=[],
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
        _ = BatchQueue(
            train_containers=samplers,
            val_containers=[],
            s_enhance=4,
            t_enhance=6,
            n_batches=3,
            batch_size=4,
            queue_cap=10,
            means=means,
            stds=stds,
            max_workers=1,
        )


def test_split_batch_queue():
    """Smoke test for batch queue."""

    train_sampler = DummyCroppedSampler(
        sample_shape=(8, 8, 4),
        data_shape=(10, 10, 100),
        features=FEATURES,
        crop_slice=slice(0, 90),
    )
    val_sampler = DummyCroppedSampler(
        sample_shape=(8, 8, 4),
        data_shape=(10, 10, 100),
        features=FEATURES,
        crop_slice=slice(90, 100),
    )
    coarsen_kwargs = {'smoothing_ignore': [], 'smoothing': None}
    batcher = BatchQueue(
        train_containers=[train_sampler],
        val_containers=[val_sampler],
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

    assert len(batcher.val_data) == 3
    for b in batcher.val_data:
        assert b.low_res.shape == (4, 4, 4, 4, len(FEATURES))
        assert b.high_res.shape == (4, 8, 8, 4, len(FEATURES))
    batcher.stop()


if __name__ == '__main__':
    # test_batch_queue()
    if True:
        execute_pytest(__file__)
