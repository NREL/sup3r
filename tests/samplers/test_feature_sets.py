# -*- coding: utf-8 -*-
"""pytests for data handling"""

import pytest

from sup3r.containers import Sampler
from sup3r.utilities.pytest.helpers import DummyData, execute_pytest


@pytest.mark.parametrize(
    ['features', 'lr_only_features', 'hr_exo_features'],
    [
        (['V_100m'], ['V_100m'], []),
        (['U_100m'], ['V_100m'], ['V_100m']),
        (['U_100m'], [], ['U_100m']),
        (['U_100m', 'V_100m'], [], ['U_100m']),
        (['U_100m', 'V_100m'], [], ['V_100m', 'U_100m']),
    ],
)
def test_feature_errors(features, lr_only_features, hr_exo_features):
    """Each of these feature combinations should raise an error due to no
    features left in hr output or bad ordering"""
    sampler = Sampler(
        DummyData(data_shape=(20, 20, 10), features=features),
        sample_shape=(5, 5, 4),
        feature_sets={
            'lr_only_features': lr_only_features,
            'hr_exo_features': hr_exo_features,
        },
    )

    with pytest.raises(Exception):
        _ = sampler.lr_features
        _ = sampler.hr_out_features
        _ = sampler.hr_exo_features


if __name__ == '__main__':
    execute_pytest(__file__)
