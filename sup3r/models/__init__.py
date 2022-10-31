# -*- coding: utf-8 -*-
"""Sup3r Model Software"""
from .base import Sup3rGan
from .wind import WindGan
from .solar_cc import SolarCC
from .data_centric import Sup3rGanDC
from .multi_step import (MultiStepGan, SpatialThenTemporalGan,
                         MultiStepSurfaceMetGan, SolarMultiStepGan)
from .surface import SurfaceSpatialMetModel
from .linear import LinearInterp

SPATIAL_FIRST_MODELS = (SpatialThenTemporalGan,
                        MultiStepSurfaceMetGan,
                        SolarMultiStepGan)
