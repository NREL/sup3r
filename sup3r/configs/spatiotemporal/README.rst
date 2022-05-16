******************************************
Spatiotemporal Super Resolving GAN configs
******************************************

This directory saves standard configs for spatiotemporal super resolving GAN
models. All generator model configs should start with "gen_*" and should have
two "4x" tags (4 for example) that represents the spatial and temporal
enhancements that the generator is designed for respectively and one "2f" tag
that represents the number of output features.

For example "gen_2x_24x_2f.json" is a model that would enhance a 4km daily
spatiotemporal field to 2km hourly with 2 output features.
