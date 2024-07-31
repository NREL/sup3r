"""Container subclass with methods for extracting a specific spatiotemporal
extents from data. :class:`Rasterizer` objects mostly operate on
:class:`Loader` objects, which just load data from files but do not do anything
else to the data. :class:`Rasterizer` objects are mostly operated on by
:class:`Deriver` objects, which derive new features from the data contained in
:class:`Rasterizer` objects."""

from .base import BaseRasterizer
from .dual import DualRasterizer
from .exo import (
    SzaRasterizer,
    TopoRasterizer,
    TopoRasterizerH5,
    TopoRasterizerNC,
)
from .extended import Rasterizer
