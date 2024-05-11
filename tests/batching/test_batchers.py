"""Smoke tests for batcher objects. Just make sure things run without errors"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from rex import init_logger

from sup3r.containers.batchers import (
    BatchQueue,
    PairBatchQueue,
    SpatialBatchQueue,
)
from sup3r.containers.samplers import Sampler, SamplerPair

init_logger('sup3r', log_level='DEBUG')


class DummyData:
    """Dummy container with random data."""

    def __init__(self, features, data_shape):
        self.features = features
        self.shape = data_shape
        self._data = None

    @property
    def data(self):
        """Dummy data property."""
        if self._data is None:
            lons, lats = np.meshgrid(
                np.linspace(0, 1, self.shape[1]),
                np.linspace(0, 1, self.shape[0]),
            )
            times = pd.date_range('2024-01-01', periods=self.shape[2])
            dim_names = ['time', 'south_north', 'west_east']
            coords = {'time': times,
                      'latitude': (dim_names[1:], lats),
                      'longitude': (dim_names[1:], lons)}
            ws = np.zeros((len(times), *lats.shape))
            self._data = xr.Dataset(
                data_vars={'windspeed': (dim_names, ws)}, coords=coords
            )
        return self._data

    def __getitem__(self, key):
        out = self.data.isel(
            south_north=key[0],
            west_east=key[1],
            time=key[2],
        )
        out = out.to_dataarray().values
        out = np.transpose(out, axes=(2, 3, 1, 0))
        return out


class DummySampler(Sampler):
    """Dummy container with random data."""

    def __init__(self, sample_shape, data_shape):
        data = DummyData(features=['windspeed'], data_shape=data_shape)
        super().__init__(data, sample_shape)


def test_batch_queue():
    """Smoke test for batch queue."""

    samplers = [
        DummySampler(sample_shape=(8, 8, 10), data_shape=(10, 10, 20)),
        DummySampler(sample_shape=(8, 8, 10), data_shape=(12, 12, 15)),
    ]
    coarsen_kwargs = {'smoothing_ignore': [], 'smoothing': None}
    batcher = BatchQueue(
        containers=samplers,
        s_enhance=2,
        t_enhance=2,
        n_batches=3,
        batch_size=4,
        queue_cap=10,
        means={'windspeed': 4},
        stds={'windspeed': 2},
        max_workers=1,
        coarsen_kwargs=coarsen_kwargs
    )
    batcher.start()
    assert len(batcher) == 3
    for b in batcher:
        assert b.low_res.shape == (4, 4, 4, 5, 1)
        assert b.high_res.shape == (4, 8, 8, 10, 1)
    batcher.stop()


def test_spatial_batch_queue():
    """Smoke test for spatial batch queue."""
    samplers = [
        DummySampler(sample_shape=(8, 8, 1), data_shape=(10, 10, 20)),
        DummySampler(sample_shape=(8, 8, 1), data_shape=(12, 12, 15)),
    ]
    coarsen_kwargs = {'smoothing_ignore': [], 'smoothing': None}
    batcher = SpatialBatchQueue(
        containers=samplers,
        s_enhance=2,
        t_enhance=1,
        n_batches=3,
        batch_size=4,
        queue_cap=10,
        means={'windspeed': 4},
        stds={'windspeed': 2},
        max_workers=1,
        coarsen_kwargs=coarsen_kwargs
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


if __name__ == '__main__':
    test_batch_queue()
    test_bad_enhancement_factors()
