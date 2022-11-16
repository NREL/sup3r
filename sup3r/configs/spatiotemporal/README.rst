******************************************
Spatiotemporal Super Resolving GAN configs
******************************************

This directory saves example configs for spatiotemporal super resolving GAN
models. All generator model configs should start with "gen_*" and should have
two "4x" tags (4 for example) that represents the spatial and temporal
enhancements that the generator is designed for respectively and one "2f" tag
that represents the number of output features.

For example "gen_2x_24x_2f.json" is a model that would enhance a 4km daily
spatiotemporal field to 2km hourly with 2 output features.

These configs are only examples and are not guaranteed to be the models used in
producing actual production datasets. For the final model architectures, see
the global file attributes associated with sup3r output h5 files which should
contain all model meta data.
