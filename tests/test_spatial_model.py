# -*- coding: utf-8 -*-
"""Test the spatial super resolution GAN configs"""
import os
import json
import numpy as np
from sup3r import CONFIG_DIR
from phygnn.layers.handlers import HiddenLayers

fp_gen = os.path.join(CONFIG_DIR, 'spatial/spatial_generator.json')
fp_disc = os.path.join(CONFIG_DIR, 'spatial/spatial_discriminator.json')

with open(fp_gen, 'r') as f:
    GEN_CONFIG = json.load(f)

with open(fp_disc, 'r') as f:
    DISC_CONFIG = json.load(f)


if __name__ == '__main__':

    x = np.ones((32, 100, 100, 2))

    layers = HiddenLayers(DISC_CONFIG['hidden_layers'])
    for i, layer in enumerate(layers):
        x = layer(x)
        print(i, layer, x.shape)
    print(layers)
