# -*- coding: utf-8 -*-
"""Test forward passes through multi-step GAN models"""
import os
import numpy as np
import pytest
import tempfile

from sup3r import CONFIG_DIR
from sup3r.models import Sup3rGan, MultiStepGan, SpatialThenTemporalGan

FEATURES = ['U_100m', 'V_100m']


def test_multi_step_model():
    """Test a basic forward pass through a multi step model with 2 steps"""
    Sup3rGan.seed(0)
    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    model1 = Sup3rGan(fp_gen, fp_disc)
    model2 = Sup3rGan(fp_gen, fp_disc)

    _ = model1.generate(np.ones((4, 10, 10, 6, len(FEATURES))))
    _ = model2.generate(np.ones((4, 10, 10, 6, len(FEATURES))))

    with tempfile.TemporaryDirectory() as td:
        fp1 = os.path.join(td, 'model1')
        fp2 = os.path.join(td, 'model2')
        model1.save(fp1)
        model2.save(fp2)

        ms_model = MultiStepGan.load([fp1, fp2])

        x = np.ones((4, 5, 5, 6, len(FEATURES)))
        out = ms_model.generate(x)
        assert out.shape == (4, 45, 45, 96, len(FEATURES))

        out1 = model1.generate(x)
        out2 = model2.generate(out1)
        assert np.allclose(out, out2)


@pytest.mark.parametrize('norm_option', ['same_stats', 'diff_stats'])
def test_multi_step_norm(norm_option):
    """Test the multi step model with 3 GAN's with the same and different norm
    stats for each model"""
    Sup3rGan.seed(0)
    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    model1 = Sup3rGan(fp_gen, fp_disc)
    model2 = Sup3rGan(fp_gen, fp_disc)
    model3 = Sup3rGan(fp_gen, fp_disc)

    if norm_option == 'diff_stats':
        # models have different norm stats
        model1.set_norm_stats([0.1, 0.2], [0.04, 0.02])
        model2.set_norm_stats([0.1, 0.2], [0.04, 0.02])
        model3.set_norm_stats([0.3, 0.9], [0.02, 0.07])
    else:
        # all models have the same norm stats
        model1.set_norm_stats([0.1, 0.8], [0.04, 0.02])
        model2.set_norm_stats([0.1, 0.8], [0.04, 0.02])
        model3.set_norm_stats([0.1, 0.8], [0.04, 0.02])

    model1.set_feature_names(FEATURES, FEATURES)
    model2.set_feature_names(FEATURES, FEATURES)
    model3.set_feature_names(FEATURES, FEATURES)

    _ = model1.generate(np.ones((4, 10, 10, 6, len(FEATURES))))
    _ = model2.generate(np.ones((4, 10, 10, 6, len(FEATURES))))
    _ = model3.generate(np.ones((4, 10, 10, 6, len(FEATURES))))

    with tempfile.TemporaryDirectory() as td:
        fp1 = os.path.join(td, 'model1')
        fp2 = os.path.join(td, 'model2')
        fp3 = os.path.join(td, 'model3')
        model1.save(fp1)
        model2.save(fp2)
        model3.save(fp3)

        ms_model = MultiStepGan.load([fp1, fp2, fp3])

        x = np.ones((1, 4, 4, 4, len(FEATURES)))
        out = ms_model.generate(x)
        assert out.shape == (1, 108, 108, 256, len(FEATURES))

        # make sure the multistep generate is the same as several manual
        # norm_in/unnorm_out forward passes
        out1 = model1.generate(x)
        out2 = model2.generate(out1)
        out3 = model3.generate(out2)
        assert np.allclose(out, out3, atol=5e-4)


def test_spatial_then_temporal_gan():
    """Test the 2-step spatial-then-spatiotemporal GAN"""
    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    model1 = Sup3rGan(fp_gen, fp_disc)
    _ = model1.generate(np.ones((4, 10, 10, len(FEATURES))))

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    model2 = Sup3rGan(fp_gen, fp_disc)
    _ = model2.generate(np.ones((4, 10, 10, 6, len(FEATURES))))

    model1.set_norm_stats([0.1, 0.2], [0.04, 0.02])
    model2.set_norm_stats([0.3, 0.9], [0.02, 0.07])
    model1.set_feature_names(FEATURES, FEATURES)
    model2.set_feature_names(FEATURES, FEATURES)

    with tempfile.TemporaryDirectory() as td:
        fp1 = os.path.join(td, 'model1')
        fp2 = os.path.join(td, 'model2')
        model1.save(fp1)
        model2.save(fp2)

        ms_model = SpatialThenTemporalGan.load(fp1, fp2)

        x = np.ones((4, 10, 10, len(FEATURES)))
        out = ms_model.generate(x)
        assert out.shape == (1, 60, 60, 16, 2)
