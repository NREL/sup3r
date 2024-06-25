"""Composite objects built from loaders and extracters."""

import logging
from typing import ClassVar

from sup3r.preprocessing.base import FactoryMeta, TypeAgnosticClass
from sup3r.preprocessing.loaders import LoaderH5, LoaderNC
from sup3r.preprocessing.utilities import (
    get_class_kwargs,
    get_composite_signature,
)

from .extended import ExtendedExtracter

logger = logging.getLogger(__name__)


def ExtracterFactory(LoaderClass, BaseLoader=None, name='DirectExtracter'):
    """Build composite :class:`Extracter` objects that also load from
    file_paths. Inputs are required to be provided as keyword args so that they
    can be split appropriately across different classes.

    Parameters
    ----------
    LoaderClass : class
        :class:`Loader` class to use in this object composition.
    BaseLoader : function
        Optional base loader method update. This is a function which takes
        `file_paths` and `**kwargs` and returns an initialized base loader with
        those arguments. The default for h5 is a method which returns
        MultiFileWindX(file_paths, **kwargs) and for nc the default is
        xarray.open_mfdataset(file_paths, **kwargs)
    name : str
        Optional name for class built from factory. This will display in
        logging.
    """

    class TypeSpecificExtracter(ExtendedExtracter, metaclass=FactoryMeta):
        """Extracter object built from factory arguments."""

        __name__ = name
        _legos = (LoaderClass, ExtendedExtracter)
        __signature__ = get_composite_signature(_legos, exclude=['loader'])

        if BaseLoader is not None:
            BASE_LOADER = BaseLoader

        def __init__(self, file_paths, **kwargs):
            """
            Parameters
            ----------
            file_paths : str | list | pathlib.Path
                file_paths input to LoaderClass
            **kwargs : dict
                Dictionary of keyword args for Extracter and Loader
            """
            self.loader = LoaderClass(
                file_paths, **get_class_kwargs(LoaderClass, kwargs)
            )
            super().__init__(
                loader=self.loader,
                **get_class_kwargs(ExtendedExtracter, kwargs),
            )

    return TypeSpecificExtracter


ExtracterH5 = ExtracterFactory(LoaderClass=LoaderH5, name='ExtracterH5')
ExtracterNC = ExtracterFactory(LoaderClass=LoaderNC, name='ExtracterNC')


class Extracter(TypeAgnosticClass):
    """`DirectExtracter` class which parses input file type and returns
    appropriate `TypeSpecificExtracter`."""

    TypeSpecificClasses: ClassVar = {'nc': ExtracterNC, 'h5': ExtracterH5}
