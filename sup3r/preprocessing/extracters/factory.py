"""Composite objects built from loaders and extracters."""

import logging

from sup3r.preprocessing.loaders import LoaderH5, LoaderNC
from sup3r.preprocessing.utilities import (
    FactoryMeta,
    get_class_kwargs,
    get_source_type,
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
                [LoaderClass, ExtendedExtracter], kwargs
            )
            self.loader = LoaderClass(file_paths, **loader_kwargs)
            super().__init__(loader=self.loader, **extracter_kwargs)

    return TypeSpecificExtracter


ExtracterH5 = ExtracterFactory(LoaderH5, name='ExtracterH5')
ExtracterNC = ExtracterFactory(LoaderNC, name='ExtracterNC')


class DirectExtracter:
    """`DirectExtracter` class which parses input file type and returns
    appropriate `TypeSpecificExtracter`."""

    _legos = (ExtracterH5, ExtracterNC)

    def __new__(cls, file_paths, *args, **kwargs):
        """Return a new `DirectExtracter` based on input file type."""
        source_type = get_source_type(file_paths)
        if source_type == 'h5':
            return ExtracterH5(file_paths, *args, **kwargs)
        if source_type == 'nc':
            return ExtracterNC(file_paths, *args, **kwargs)
        msg = (f'Can only handle H5 or NETCDF files. Received '
               f'"{source_type}" for file_paths: {file_paths}')
        logger.error(msg)
        raise ValueError(msg)
