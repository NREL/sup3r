"""Data handling for H5 files.
@author: bbenton
"""

import logging

from rex import MultiFileNSRDBX

from sup3r.preprocessing.data_handlers.factory import (
    DailyDataHandlerFactory,
    DataHandlerFactory,
)
from sup3r.preprocessing.derivers.methods import (
    RegistryH5SolarCC,
    RegistryH5WindCC,
)
from sup3r.preprocessing.extracters import BaseExtracterH5
from sup3r.preprocessing.loaders import LoaderH5

logger = logging.getLogger(__name__)


BaseH5WindCC = DataHandlerFactory(
    BaseExtracterH5, LoaderH5, FeatureRegistry=RegistryH5WindCC
)
DailyH5WindCC = DailyDataHandlerFactory(
    BaseExtracterH5, LoaderH5, FeatureRegistry=RegistryH5WindCC
)


def _base_loader(file_paths, **kwargs):
    return MultiFileNSRDBX(file_paths, **kwargs)


BaseH5SolarCC = DataHandlerFactory(
    BaseExtracterH5,
    LoaderH5,
    BaseLoader=_base_loader,
    FeatureRegistry=RegistryH5SolarCC,
)
DailyH5SolarCC = DailyDataHandlerFactory(
    BaseExtracterH5,
    LoaderH5,
    BaseLoader=_base_loader,
    FeatureRegistry=RegistryH5SolarCC,
)


class DataHandlerH5WindCC(BaseH5WindCC):
    """Composite handler which includes daily data derived with a
    :class:`DailyDataHandler`, stored in the `.daily_data` attribute."""

    def __init__(self, file_paths, features, **kwargs):
        """
        Parameters
        ----------
        file_paths : str | list | pathlib.Path
            file_paths input to Loader
        features : list
            Features to derive from loaded data.
        **kwargs : dict
            Dictionary of keyword args for Loader, Extracter, Deriver, and
            Cacher
        """
        super().__init__(file_paths, features, **kwargs)

        self.daily_data = DailyH5WindCC(file_paths, features, **kwargs).data


class DataHandlerH5SolarCC(BaseH5SolarCC):
    """Composite handler which includes daily data derived with a
    :class:`DailyDataHandler`, stored in the `.daily_data` attribute."""

    def __init__(self, file_paths, features, **kwargs):
        """
        Parameters
        ----------
        file_paths : str | list | pathlib.Path
            file_paths input to Loader
        features : list
            Features to derive from loaded data.
        **kwargs : dict
            Dictionary of keyword args for Loader, Extracter, Deriver, and
            Cacher
        """

        required = ['ghi', 'clearsky_ghi', 'clearsky_ratio']
        missing = [dset for dset in required if dset not in features]
        if any(missing):
            msg = (
                'Cannot initialize DataHandlerH5SolarCC without required '
                'features {}. All three are necessary to get the daily '
                'average clearsky ratio (ghi sum / clearsky ghi sum), '
                'even though only the clearsky ratio will be passed to the '
                'GAN.'.format(required)
            )
            logger.error(msg)
            raise KeyError(msg)

        super().__init__(file_paths, features, **kwargs)

        self.daily_data = DailyH5SolarCC(file_paths, features, **kwargs)
        features = [
            f
            for f in self.daily_data.features
            if f not in ('clearsky_ghi', 'ghi')
        ]
        self.daily_data = self.daily_data.slice_dset(features=features)
