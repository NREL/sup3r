# -*- coding: utf-8 -*-
"""Sup3r Model Software"""
from .base import Sup3rGan
from .conditional_moments import Sup3rCondMom
from .data_centric import Sup3rGanDC
from .linear import LinearInterp
from .multi_exo import MultiExoGan
from .multi_step import (
    MultiStepGan,
    MultiStepSurfaceMetGan,
    SolarMultiStepGan,
    SpatialThenTemporalGan,
    TemporalThenSpatialGan,
)
from .solar_cc import SolarCC
from .surface import SurfaceSpatialMetModel
from .wind_conditional_moments import WindCondMom

SPATIAL_FIRST_MODELS = (SpatialThenTemporalGan,
                        MultiStepSurfaceMetGan,
                        SolarMultiStepGan)
