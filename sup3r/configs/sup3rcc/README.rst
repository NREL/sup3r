**************************************************************************
Spatiotemporal Super Resolving GAN configs for Climate Change Applications
**************************************************************************

This directory saves example configs for spatiotemporal super resolving GAN
models for wind, solar, and temperature climate change data. All generator
model configs should start with "gen_*" and should have two "_4x" tags (4 for
example) that represents the spatial and temporal enhancements that the
generator is designed for respectively and one "_3f" tag that represents the
number of output features.

For example "gen_2x_24x_2f.json" is a model that would enhance a 4km daily
spatiotemporal field to 2km hourly with 2 output features.

Unique model designs are utilized for each unique variable set. For example,
when doing spatial super resolution of wind fields, a custom model with
mid-network topography injection via a "Sup3rConcat" layer is used. For wind
temporal super resolution, a 24x enhancement is used to go from daily to
hourly, but for solar an 8x enhancement is used to go from 3 days to 24 hours
of the middle day. Also, note that the "_trh_" model tag stands for temperature
and relative humidity.

These configs are only examples and are not guaranteed to be the models used in
producing actual production datasets. For the final model architectures, see
the global file attributes associated with sup3r output h5 files which should
contain all model meta data.
