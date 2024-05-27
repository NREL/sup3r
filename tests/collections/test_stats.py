# -*- coding: utf-8 -*-
"""pytests for data handling"""

import os
from tempfile import TemporaryDirectory

import numpy as np
from rex import safe_json_load

from sup3r import TEST_DATA_DIR
from sup3r.containers import ExtracterH5, StatsCollection
from sup3r.utilities.pytest.helpers import execute_pytest

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


def test_stats_calc():
    """Check accuracy of stats calcs across multiple extracters and caching
    stats files."""
    features = ['windspeed_100m', 'winddirection_100m']
    extracters = [
        ExtracterH5(file, features=features, **kwargs)
        for file in input_files
    ]
    with TemporaryDirectory() as td:
        means = os.path.join(td, 'means.json')
        stds = os.path.join(td, 'stds.json')
        stats = StatsCollection(
            extracters, means=means, stds=stds
        )

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


if __name__ == '__main__':
    execute_pytest()
