"""Smoke tests for batcher objects. Just make sure things run without errors"""

import os

import pytest
from rex import init_logger

from sup3r.containers.batchers import (
    BatchQueue,
    PairBatchQueue,
    SplitBatchQueue,
)
from sup3r.containers.samplers import SamplerPair
from sup3r.utilities.pytest.helpers import DummyCroppedSampler, DummySampler

init_logger('sup3r', log_level='DEBUG')


def test_batch_queue():
    """Smoke test for batch queue."""

    samplers = [
        DummySampler(sample_shape=(8, 8, 10), data_shape=(10, 10, 20)),
        DummySampler(sample_shape=(8, 8, 10), data_shape=(12, 12, 15)),
    ]
    coarsen_kwargs = {'smoothing_ignore': [], 'smoothing': None}
    batcher = BatchQueue(
        containers=samplers,
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
    batcher.start()
    assert len(batcher) == 3
    for b in batcher:
        assert b.low_res.shape == (4, 4, 4, 5, 1)
        assert b.high_res.shape == (4, 8, 8, 10, 1)
    batcher.stop()


def test_spatial_batch_queue():
    """Smoke test for spatial batch queue. A batch queue returns batches for
    spatial models if the sample shapes have 1 for the time axis"""
    samplers = [
        DummySampler(sample_shape=(8, 8, 1), data_shape=(10, 10, 20)),
        DummySampler(sample_shape=(8, 8, 1), data_shape=(12, 12, 15)),
    ]
    coarsen_kwargs = {'smoothing_ignore': [], 'smoothing': None}
    batcher = BatchQueue(
        containers=samplers,
        s_enhance=2,
        t_enhance=1,
        n_batches=3,
        batch_size=4,
        queue_cap=10,
        means={'windspeed': 4},
        stds={'windspeed': 2},
        max_workers=1,
        coarsen_kwargs=coarsen_kwargs,
    )
    batcher.start()
    assert len(batcher) == 3
    for b in batcher:
        assert b.low_res.shape == (4, 4, 4, 1)
        assert b.high_res.shape == (4, 8, 8, 1)
    batcher.stop()


def test_pair_batch_queue():
    """Smoke test for paired batch queue."""
    lr_samplers = [
        DummySampler(sample_shape=(4, 4, 5), data_shape=(10, 10, 20)),
        DummySampler(sample_shape=(4, 4, 5), data_shape=(12, 12, 15)),
    ]
    hr_samplers = [
        DummySampler(sample_shape=(8, 8, 10), data_shape=(20, 20, 40)),
        DummySampler(sample_shape=(8, 8, 10), data_shape=(24, 24, 30)),
    ]
    sampler_pairs = [
        SamplerPair(lr, hr, s_enhance=2, t_enhance=2)
        for lr, hr in zip(lr_samplers, hr_samplers)
    ]
    batcher = PairBatchQueue(
        containers=sampler_pairs,
        s_enhance=2,
        t_enhance=2,
        n_batches=3,
        batch_size=4,
        queue_cap=10,
        means={'windspeed': 4},
        stds={'windspeed': 2},
        max_workers=1,
    )
    batcher.start()
    assert len(batcher) == 3
    for b in batcher:
        assert b.low_res.shape == (4, 4, 4, 5, 1)
        assert b.high_res.shape == (4, 8, 8, 10, 1)
    batcher.stop()


def test_bad_enhancement_factors():
    """Failure when enhancement factors given to BatchQueue do not match those
    given to the contained SamplerPairs, and when those given to SamplerPair
    are not consistent with the low / high res shapes."""

    lr_samplers = [
        DummySampler(sample_shape=(4, 4, 5), data_shape=(10, 10, 20)),
        DummySampler(sample_shape=(4, 4, 5), data_shape=(12, 12, 15)),
    ]
    hr_samplers = [
        DummySampler(sample_shape=(8, 8, 10), data_shape=(20, 20, 40)),
        DummySampler(sample_shape=(8, 8, 10), data_shape=(24, 24, 30)),
    ]

    for s_enhance, t_enhance in zip([2, 4], [2, 6]):
        with pytest.raises(AssertionError):
            sampler_pairs = [
                SamplerPair(lr, hr, s_enhance=s_enhance, t_enhance=t_enhance)
                for lr, hr in zip(lr_samplers, hr_samplers)
            ]
            _ = PairBatchQueue(
                containers=sampler_pairs,
                s_enhance=4,
                t_enhance=6,
                n_batches=3,
                batch_size=4,
                queue_cap=10,
                means={'windspeed': 4},
                stds={'windspeed': 2},
                max_workers=1,
            )


def test_bad_sample_shapes():
    """Failure when sample shapes are not consistent across a collection of
    samplers."""

    samplers = [
        DummySampler(sample_shape=(4, 4, 5), data_shape=(10, 10, 20)),
        DummySampler(sample_shape=(3, 3, 5), data_shape=(12, 12, 15)),
    ]

    with pytest.raises(AssertionError):
        _ = BatchQueue(
            containers=samplers,
            s_enhance=4,
            t_enhance=6,
            n_batches=3,
            batch_size=4,
            queue_cap=10,
            means={'windspeed': 4},
            stds={'windspeed': 2},
            max_workers=1,
        )


def test_split_batch_queue():
    """Smoke test for batch queue."""

    samplers = [
        DummyCroppedSampler(
            sample_shape=(8, 8, 4), data_shape=(10, 10, 100)
        ),
        DummyCroppedSampler(
            sample_shape=(8, 8, 4), data_shape=(12, 12, 100)
        ),
    ]
    coarsen_kwargs = {'smoothing_ignore': [], 'smoothing': None}
    batcher = SplitBatchQueue(
        containers=samplers,
        val_split=0.2,
        batch_size=4,
        n_batches=3,
        s_enhance=2,
        t_enhance=1,
        queue_cap=10,
        means={'windspeed': 4},
        stds={'windspeed': 2},
        max_workers=1,
        coarsen_kwargs=coarsen_kwargs,
    )
    test_train_slices = batcher.get_test_train_slices()

    for i, (test_s, train_s) in enumerate(test_train_slices):
        assert batcher.containers[i].crop_slice == train_s
        assert batcher.val_data.containers[i].crop_slice == test_s

    batcher.start()
    assert len(batcher) == 3
    for b in batcher:
        assert b.low_res.shape == (4, 4, 4, 4, 1)
        assert b.high_res.shape == (4, 8, 8, 4, 1)

    assert len(batcher.val_data) == 3
    for b in batcher.val_data:
        assert b.low_res.shape == (4, 4, 4, 4, 1)
        assert b.high_res.shape == (4, 8, 8, 4, 1)
    batcher.stop()


def execute_pytest(capture='all', flags='-rapP'):
    """Execute module as pytest with detailed summary report.

    Parameters
    ----------
    capture : str
        Log or stdout/stderr capture option. ex: log (only logger),
        all (includes stdout/stderr)
    flags : str
        Which tests to show logs and results for.
    """

    fname = os.path.basename(__file__)
    pytest.main(['-q', '--show-capture={}'.format(capture), fname, flags])


if __name__ == '__main__':
    execute_pytest()
