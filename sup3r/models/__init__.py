# -*- coding: utf-8 -*-
"""Sup3r Model Software"""
from .base import Sup3rGan
from .solar_cc import SolarCC
from .wind_cc import WindCC
from .data_centric import Sup3rGanDC
from .multi_step import (MultiStepGan, SpatialThenTemporalGan,
                         MultiStepSurfaceMetGan, SolarMultiStepGan)
from .surface import SurfaceSpatialMetModel

SPATIAL_FIRST_MODELS = (SpatialThenTemporalGan, MultiStepSurfaceMetGan,
                        SolarMultiStepGan)
