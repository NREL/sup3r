"""Composite objects built from loaders and extracters."""

import logging

from sup3r.preprocessing.extracters.h5 import (
    BaseExtracterH5,
)
from sup3r.preprocessing.extracters.nc import (
    BaseExtracterNC,
)
from sup3r.preprocessing.loaders import LoaderH5, LoaderNC
from sup3r.preprocessing.utilities import (
    FactoryMeta,
    get_class_kwargs,
)

logger = logging.getLogger(__name__)


def ExtracterFactory(
    ExtracterClass, LoaderClass, BaseLoader=None, name='DirectExtracter'
):
    """Build composite :class:`Extracter` objects that also load from
    file_paths. Inputs are required to be provided as keyword args so that they
    can be split appropriately across different classes.

    Parameters
    ----------
    ExtracterClass : class
        :class:`Extracter` class to use in this object composition.
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

    class DirectExtracter(ExtracterClass, metaclass=FactoryMeta):
        __name__ = name

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
            [loader_kwargs, extracter_kwargs] = get_class_kwargs(
                [LoaderClass, ExtracterClass], kwargs
            )
            self.loader = LoaderClass(file_paths, **loader_kwargs)
            super().__init__(loader=self.loader, **extracter_kwargs)

    return DirectExtracter


ExtracterH5 = ExtracterFactory(BaseExtracterH5, LoaderH5, name='ExtracterH5')
ExtracterNC = ExtracterFactory(BaseExtracterNC, LoaderNC, name='ExtracterNC')
