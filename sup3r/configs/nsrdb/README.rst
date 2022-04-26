********************************************************************************
Spatiotemporal Super Resolving GAN configs for Climate Change NSRDB Applications
********************************************************************************

This directory saves standard configs for spatiotemporal super resolving GAN
models based on the NSRDB for climate change applications. All generator model
configs should start with "gen_*" and should have two "4x" tags (for example)
that represents the spatial and temporal enhancements that the generator is
designed for respectively (e.g. "gen_2x_24x.json" is a model that would enhance
a 4km daily spatiotemporal field to 2km hourly).

These models are different from the wind-based spatiotemporal models because
they anticipate only a single low-res observation in the time dimension
(Axis=3) to correspond with climate change data.
