"""Sup3r Model Software"""
from .base import Sup3rGan
from .conditional import Sup3rCondMom
from .dc import Sup3rGanDC
from .linear import LinearInterp
from .multi_step import MultiStepGan, MultiStepSurfaceMetGan, SolarMultiStepGan
from .solar_cc import SolarCC
from .surface import SurfaceSpatialMetModel

SPATIAL_FIRST_MODELS = (MultiStepSurfaceMetGan,
                        SolarMultiStepGan)
