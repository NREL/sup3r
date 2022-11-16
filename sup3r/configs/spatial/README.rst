***********************************
Spatial Super Resolving GAN configs
***********************************

This directory saves example configs for spatial-only super resolving GAN
models. All generator model configs should start with "gen_*" and should have a
"4x" tag (for example) that represents the spatial enhancement that the
generator is designed for and one "2f" tag that represents the number of output

For example "gen_2x_2f.json" is a model that would enhance a 4km hourly
spatial field to 2km hourly with 2 output features.

These configs are only examples and are not guaranteed to be the models used in
producing actual production datasets. For the final model architectures, see
the global file attributes associated with sup3r output h5 files which should
contain all model meta data.
