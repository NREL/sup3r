"""General `Loader` class which parses file type and returns a type specific
loader."""

import logging
from typing import ClassVar

from sup3r.preprocessing.base import TypeGeneralClass
from sup3r.preprocessing.utilities import (
    get_composite_signature,
)

from .h5 import LoaderH5
from .nc import LoaderNC

logger = logging.getLogger(__name__)


class Loader(TypeGeneralClass):
    """`Loader` class which parses input file type and returns
    appropriate `TypeSpecificLoader`."""

    _legos = (LoaderNC, LoaderH5)
    __signature__ = get_composite_signature(_legos)
    TypeSpecificClass: ClassVar = dict(zip(['nc', 'h5'], _legos))
