# -*- coding: utf-8 -*-
"""Test the sample super resolution GAN configs"""
import os
import numpy as np
import tensorflow as tf

from sup3r import CONFIG_DIR
from sup3r.models.models import SpatialGan, SpatioTemporalGan


def test_load_spatial():
    """Test the loading of a sample the spatial gan model."""
    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_10x.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')

    model = SpatialGan(fp_gen, fp_disc)

    coarse_shape = (32, 5, 5, 2)
    coarse_input = np.ones(coarse_shape)
    x = np.ones(coarse_shape)

    for layer in model.generator:
        x = layer(x)

    gen_out = tf.identity(x)

    assert len(gen_out.shape) == 4
    assert gen_out.shape[0] == coarse_input.shape[0]
    assert gen_out.shape[1] == 10 * coarse_input.shape[1]
    assert gen_out.shape[2] == 10 * coarse_input.shape[2]
    assert gen_out.shape[3] == coarse_input.shape[3]

    for layer in model.disc:
        x = layer(x)

    disc_out = tf.identity(x)
    assert len(disc_out.shape) == 2
    assert disc_out.shape[0] == coarse_input.shape[0]
    assert disc_out.shape[1] == 1


def test_load_spatiotemporal():
    """Test loading of a sample spatiotemporal gan model"""
    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_2x_24x.json')
    fp_disc_s = os.path.join(CONFIG_DIR, 'spatiotemporal/disc_space.json')
    fp_disc_t = os.path.join(CONFIG_DIR, 'spatiotemporal/disc_time.json')

    model = SpatioTemporalGan(fp_gen, fp_disc_s, fp_disc_t)

    coarse_shape = (32, 5, 5, 4, 2)
    coarse_input = np.ones(coarse_shape)
    x = np.ones(coarse_shape)

    for layer in model.generator:
        x = layer(x)

    gen_out = tf.identity(x)

    assert len(gen_out.shape) == 5
    assert gen_out.shape[0] == coarse_input.shape[0]
    assert gen_out.shape[1] == 2 * coarse_input.shape[1]
    assert gen_out.shape[2] == 2 * coarse_input.shape[2]
    assert gen_out.shape[3] == 24 * coarse_input.shape[3]
    assert gen_out.shape[4] == coarse_input.shape[4]

    for disc in (model.disc_spatial, model.disc_temporal):
        x = tf.identity(gen_out)
        for layer in disc:
            x = layer(x)

        disc_out = tf.identity(x)
        assert len(disc_out.shape) == 2
        assert disc_out.shape[0] == coarse_input.shape[0]
        assert disc_out.shape[1] == 1
