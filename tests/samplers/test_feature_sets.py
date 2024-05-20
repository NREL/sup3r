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


@pytest.mark.parametrize(
    ['lr_features', 'hr_features', 'hr_exo_features'],
    [
        (['U_100m'], ['U_100m', 'V_100m'], ['V_100m']),
        (['U_100m'], ['U_100m', 'V_100m'], ('V_100m',)),
        (['U_100m'], ['V_100m', 'BVF2_200m'], ['BVF2_200m']),
        (['U_100m'], ('V_100m', 'BVF2_200m'), ['BVF2_200m']),
        (['U_100m'], ['V_100m', 'BVF2_200m'], []),
    ],
)
def test_mixed_lr_hr_features(lr_features, hr_features, hr_exo_features):
    """Test weird mixes of low-res and high-res features that should work with
    the dual dh"""
    lr_handler = DataHandlerNC(
        FP_ERA,
        lr_features,
        sample_shape=(5, 5, 4),
        time_slice=slice(None, None, 1),
    )
    hr_handler = DataHandlerH5(
        FP_WTK,
        hr_features,
        hr_exo_features=hr_exo_features,
        target=TARGET_COORD,
        shape=(20, 20),
        time_slice=slice(None, None, 1),
    )

    dual_handler = DualDataHandler(
        hr_handler, lr_handler, s_enhance=1, t_enhance=1, val_split=0.0
    )

    batch_handler = DualBatchHandler(
        dual_handler,
        batch_size=2,
        s_enhance=1,
        t_enhance=1,
        n_batches=10,
        worker_kwargs={'max_workers': 2},
    )

    n_hr_features = len(batch_handler.hr_out_features) + len(
        batch_handler.hr_exo_features
    )
    hr_only_features = [fn for fn in hr_features if fn not in lr_features]
    hr_out_true = [fn for fn in hr_features if fn not in hr_exo_features]
    assert batch_handler.features == lr_features + hr_only_features
    assert batch_handler.lr_features == list(lr_features)
    assert batch_handler.hr_exo_features == list(hr_exo_features)
    assert batch_handler.hr_out_features == list(hr_out_true)

    for batch in batch_handler:
        assert batch.high_res.shape[-1] == n_hr_features
        assert batch.low_res.shape[-1] == len(batch_handler.lr_features)

        if batch_handler.lr_features == lr_features + hr_only_features:
            assert np.allclose(batch.low_res, batch.high_res)
        elif batch_handler.lr_features != lr_features + hr_only_features:
            assert not np.allclose(batch.low_res, batch.high_res)


if __name__ == '__main__':
    execute_pytest(__file__)
