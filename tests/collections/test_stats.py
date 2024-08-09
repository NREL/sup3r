"""Test :class:`Collection` objects, specifically stats calculations."""

import os
from tempfile import TemporaryDirectory

import numpy as np
from rex import safe_json_load

from sup3r import TEST_DATA_DIR
from sup3r.preprocessing import Rasterizer, StatsCollection
from sup3r.preprocessing.accessor import Sup3rX
from sup3r.preprocessing.base import Sup3rDataset
from sup3r.utilities.pytest.helpers import DummyData

input_files = [
    os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5'),
    os.path.join(TEST_DATA_DIR, 'test_wtk_co_2013.h5'),
]
target = (39.01, -105.15)
shape = (20, 20)
features = ['u_100m', 'v_100m']
kwargs = {
    'target': target,
    'shape': shape,
    'max_delta': 20,
    'time_slice': slice(None, None, 1),
}


def test_stats_dual_data():
    """Check accuracy of stats calcs across multiple containers with
    `type(self.data) == type(Sup3rDataset)` (e.g. a dual dataset)."""

    feats = ['windspeed', 'winddirection']
    dat = DummyData((10, 10, 100), feats)
    dat.data = Sup3rDataset(
        low_res=Sup3rX(dat.data[0]._ds), high_res=Sup3rX(dat.data[0]._ds)
    )

    og_means = {f: np.nanmean(dat[f]) for f in feats}
    og_stds = {f: np.nanstd(dat[f]) for f in feats}

    direct_means = {
        'windspeed': dat.data.mean(
            features='windspeed', skipna=True
        ).compute(),
        'winddirection': dat.data.mean(
            features='winddirection', skipna=True
        ).compute(),
    }
    direct_stds = {
        'windspeed': dat.data.std(features='windspeed', skipna=True).compute(),
        'winddirection': dat.data.std(
            features='winddirection', skipna=True
        ).compute(),
    }

    with TemporaryDirectory() as td:
        means = os.path.join(td, 'means.json')
        stds = os.path.join(td, 'stds.json')
        stats = StatsCollection([dat, dat], means=means, stds=stds)

        means = safe_json_load(means)
        stds = safe_json_load(stds)
        assert means == stats.means
        assert stds == stats.stds

        for k in set(means):
            assert np.allclose(means[k], og_means[k])
            assert np.allclose(stds[k], og_stds[k])
            assert np.allclose(means[k], direct_means[k])
            assert np.allclose(stds[k], direct_stds[k])


def test_stats_known():
    """Check accuracy of stats calcs across multiple containers with known
    means / stds."""

    feats = ['windspeed', 'winddirection']
    dat = DummyData((10, 10, 100), feats)

    og_means = {f: np.nanmean(dat[f]) for f in feats}
    og_stds = {f: np.nanstd(dat[f]) for f in feats}

    with TemporaryDirectory() as td:
        means = os.path.join(td, 'means.json')
        stds = os.path.join(td, 'stds.json')
        stats = StatsCollection([dat, dat], means=means, stds=stds)

        means = safe_json_load(means)
        stds = safe_json_load(stds)
        assert means == stats.means
        assert stds == stats.stds

        assert np.allclose(means['windspeed'], og_means['windspeed'])
        assert np.allclose(means['winddirection'], og_means['winddirection'])
        assert np.allclose(stds['windspeed'], og_stds['windspeed'])
        assert np.allclose(stds['winddirection'], og_stds['winddirection'])


def test_stats_calc():
    """Check accuracy of stats calcs across multiple rasterizers and caching
    stats files."""
    features = ['windspeed_100m', 'winddirection_100m']
    rasterizers = [
        Rasterizer(file, features=features, **kwargs) for file in input_files
    ]
    with TemporaryDirectory() as td:
        means = os.path.join(td, 'means.json')
        stds = os.path.join(td, 'stds.json')
        stats = StatsCollection(rasterizers, means=means, stds=stds)

        means = safe_json_load(means)
        stds = safe_json_load(stds)
        assert means == stats.means
        assert stds == stats.stds

        # reload unnormalized rasterizers
        rasterizers = [
            Rasterizer(file, features=features, **kwargs)
            for file in input_files
        ]

        means = {
            f: np.sum(
                [
                    wgt * c.data[f].mean()
                    for wgt, c in zip(stats.container_weights, rasterizers)
                ]
            )
            for f in features
        }
        stds = {
            f: np.sqrt(
                np.sum(
                    [
                        wgt * c.data[f].std() ** 2
                        for wgt, c in zip(stats.container_weights, rasterizers)
                    ]
                )
            )
            for f in features
        }

        assert means == stats.means
        assert stds == stats.stds
