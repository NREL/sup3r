"""Test :class:`Collection` objects, specifically stats calculations."""

import os
from tempfile import TemporaryDirectory

import numpy as np
from rex import safe_json_load

from sup3r import TEST_DATA_DIR
from sup3r.preprocessing import ExtracterH5, StatsCollection
from sup3r.preprocessing.accessor import Sup3rX
from sup3r.preprocessing.base import Sup3rDataset
from sup3r.utilities.pytest.helpers import DummyData

input_files = [
    os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5'),
    os.path.join(TEST_DATA_DIR, 'test_wtk_co_2013.h5'),
]
target = (39.01, -105.15)
shape = (20, 20)
features = ['U_100m', 'V_100m']
kwargs = {
    'target': target,
    'shape': shape,
    'max_delta': 20,
    'time_slice': slice(None, None, 1),
}


def test_stats_dual_data():
    """Check accuracy of stats calcs across multiple containers with
    `type(self.data) == type(Sup3rDataset)` (e.g. a dual dataset)."""

    dat = DummyData((10, 10, 100), ['windspeed', 'winddirection'])
    dat.data = Sup3rDataset(
        low_res=Sup3rX(dat.data[0]._ds), high_res=Sup3rX(dat.data[0]._ds)
    )

    og_means = {
        'windspeed': np.nanmean(dat[..., 0]),
        'winddirection': np.nanmean(dat[..., 1]),
    }
    og_stds = {
        'windspeed': np.nanstd(dat[..., 0]),
        'winddirection': np.nanstd(dat[..., 1]),
    }

    direct_means = {
        'windspeed': dat.data.mean(features='windspeed', skipna=True),
        'winddirection': dat.data.mean(features='winddirection', skipna=True)
    }
    direct_stds = {
        'windspeed': dat.data.std(features='windspeed', skipna=True),
        'winddirection': dat.data.std(features='winddirection', skipna=True)
    }

    with TemporaryDirectory() as td:
        means = os.path.join(td, 'means.json')
        stds = os.path.join(td, 'stds.json')
        stats = StatsCollection([dat, dat], means=means, stds=stds)

        means = safe_json_load(means)
        stds = safe_json_load(stds)
        assert means == stats.means
        assert stds == stats.stds

        assert np.allclose(list(means.values()), list(og_means.values()))
        assert np.allclose(list(stds.values()), list(og_stds.values()))

        assert np.allclose(list(means.values()), list(direct_means.values()))
        assert np.allclose(list(stds.values()), list(direct_stds.values()))


def test_stats_known():
    """Check accuracy of stats calcs across multiple containers with known
    means / stds."""

    dat = DummyData((10, 10, 100), ['windspeed', 'winddirection'])

    og_means = {
        'windspeed': np.nanmean(dat[..., 0]),
        'winddirection': np.nanmean(dat[..., 1]),
    }
    og_stds = {
        'windspeed': np.nanstd(dat[..., 0]),
        'winddirection': np.nanstd(dat[..., 1]),
    }

    with TemporaryDirectory() as td:
        means = os.path.join(td, 'means.json')
        stds = os.path.join(td, 'stds.json')
        stats = StatsCollection([dat, dat], means=means, stds=stds)

        means = safe_json_load(means)
        stds = safe_json_load(stds)
        assert means == stats.means
        assert stds == stats.stds

        assert means['windspeed'] == og_means['windspeed']
        assert means['winddirection'] == og_means['winddirection']
        assert stds['windspeed'] == og_stds['windspeed']
        assert stds['winddirection'] == og_stds['winddirection']


def test_stats_calc():
    """Check accuracy of stats calcs across multiple extracters and caching
    stats files."""
    features = ['windspeed_100m', 'winddirection_100m']
    extracters = [
        ExtracterH5(file, features=features, **kwargs) for file in input_files
    ]
    with TemporaryDirectory() as td:
        means = os.path.join(td, 'means.json')
        stds = os.path.join(td, 'stds.json')
        stats = StatsCollection(extracters, means=means, stds=stds)

        means = safe_json_load(means)
        stds = safe_json_load(stds)
        assert means == stats.means
        assert stds == stats.stds

        means = {
            f: np.sum(
                [
                    wgt * c.data[f].mean()
                    for wgt, c in zip(stats.container_weights, extracters)
                ]
            )
            for f in features
        }
        stds = {
            f: np.sqrt(
                np.sum(
                    [
                        wgt * c.data[f].std() ** 2
                        for wgt, c in zip(stats.container_weights, extracters)
                    ]
                )
            )
            for f in features
        }

        assert means == stats.means
        assert stds == stats.stds
