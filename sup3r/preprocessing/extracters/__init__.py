"""Container subclass with methods for extracting a specific spatiotemporal
extents from data. :class:`Extracter` objects mostly operate on :class:`Loader`
objects, which just load data from files but do not do anything else to the
data. :class:`Extracter` objects are mostly operated on by :class:`Deriver`
objects, which derive new features from the data contained in
:class:`Extracter` objects."""

from .base import Extracter
from .dual import DualExtracter
from .exo import SzaExtract, TopoExtractH5, TopoExtractNC
from .factory import ExtracterH5, ExtracterNC
from .h5 import BaseExtracterH5
from .nc import BaseExtracterNC