# -*- coding: utf-8 -*-
"""Test the sample super resolution GAN configs"""
import os
import numpy as np
import pytest
import tensorflow as tf

from sup3r import CONFIG_DIR
from sup3r.models.models import SpatialGan, SpatioTemporalGan


@pytest.mark.parametrize('spatial_len', (5, 6, 7))
def test_load_spatial(spatial_len):
    """Test the loading of a sample the spatial gan model."""
    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_10x.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')

    model = SpatialGan(fp_gen, fp_disc)

    coarse_shapes = [(32, spatial_len, spatial_len, 2),
                     (16, 2 * spatial_len, 2 * spatial_len, 2)]

    for coarse_shape in coarse_shapes:
        x = np.ones(coarse_shape)
        gen_out = model.generate(x)

        assert len(gen_out.shape) == 4
        assert gen_out.shape[0] == coarse_shape[0]
        assert gen_out.shape[1] == 10 * coarse_shape[1]
        assert gen_out.shape[2] == 10 * coarse_shape[2]
        assert gen_out.shape[3] == coarse_shape[3]

        disc_out = model.discriminate(x)
        assert len(disc_out.shape) == 2
        assert disc_out.shape[0] == coarse_shape[0]
        assert disc_out.shape[1] == 1


def test_load_spatiotemporal():
    """Test loading of a sample spatiotemporal gan model"""
    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_2x_24x.json')
    fp_disc_s = os.path.join(CONFIG_DIR, 'spatiotemporal/disc_space.json')
    fp_disc_t = os.path.join(CONFIG_DIR, 'spatiotemporal/disc_time.json')

    model = SpatioTemporalGan(fp_gen, fp_disc_t, fp_disc_s)

    coarse_shape = (32, 5, 5, 4, 2)
    x = np.ones(coarse_shape)

    for layer in model.generator:
        x = layer(x)

    gen_out = tf.identity(x)

    assert len(gen_out.shape) == 5
    assert gen_out.shape[0] == coarse_shape[0]
    assert gen_out.shape[1] == 2 * coarse_shape[1]
    assert gen_out.shape[2] == 2 * coarse_shape[2]
    assert gen_out.shape[3] == 24 * coarse_shape[3]
    assert gen_out.shape[4] == coarse_shape[4]

    x = tf.identity(gen_out)
    for layer in model.disc_temporal:
        x = layer(x)
    disc_out = tf.identity(x)
    assert len(disc_out.shape) == 2
    assert disc_out.shape[0] == coarse_shape[0]
    assert disc_out.shape[1] == 1

    # only take one temporal slice for the spatial disc
    x = tf.identity(gen_out[:, :, :, 0, :])
    for layer in model.disc_spatial:
        x = layer(x)
    disc_out = tf.identity(x)
    assert len(disc_out.shape) == 2
    assert disc_out.shape[0] == coarse_shape[0]
    assert disc_out.shape[1] == 1
