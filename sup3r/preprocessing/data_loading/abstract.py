"""Abstract data loaders"""
import logging
from abc import abstractmethod

import xarray as xr

from sup3r.preprocessing.mixin import (
    HandlerFeatureSets,
    InputMixIn,
    TrainingPrep,
)

logger = logging.getLogger(__name__)


class AbstractLoader(InputMixIn, TrainingPrep, HandlerFeatureSets):
    """Abstract Loader. Takes netcdf files that have been preprocessed to
    select only the region and time period that will be used for training.
    These files usually come from using the data munging classes to
    extract/compute specific features for specified regions and then calling
    the to_netcdf() method for these """

    def __init__(
        self, file_paths, features, sample_shape, lr_only_features=(),
        hr_exo_features=(), res_kwargs=None, mode='lazy'
    ):
        self.features = features
        self.sample_shape = sample_shape
        self.file_paths = file_paths
        self._lr_only_features = lr_only_features
        self._hr_exo_features = hr_exo_features
        self._res_kwargs = res_kwargs or {}
        self._data = None
        self._mode = mode
        self.shape = (*self.data["latitude"].shape, len(self.data["time"]))

        logger.info(f'Initialized {self.__class__.__name__} with '
                    f'files = {self.file_paths}, features = {self.features}, '
                    f'sample_shape = {self.sample_shape}.')

    @property
    def data(self):
        """Xarray dataset either lazily loaded (mode = 'lazy') or loaded into
        memory right away (mode = 'eager').

        Returns
        -------
        xr.Dataset()
            xarray dataset with the requested features
        """
        if self._data is None:
            self._data = xr.open_mfdataset(self.file_paths, **self._res_kwargs)
            msg = (f'Loading {self.file_paths} with kwargs = '
                   f'{self._res_kwargs} and mode = {self._mode}')
            logger.info(msg)

            if self._mode == 'eager':
                self._data = self._data.compute()

            self._data = self._data[self.features]
        return self._data

    @abstractmethod
    def get_observation(self, obs_index):
        """Get observation/sample. Should return a single sample from the
        underlying data with shape (spatial_1, spatial_2, temporal,
        features)."""

    def get_next(self):
        """Get next observation sample."""
        obs_index = self.get_observation_index(self.shape, self.sample_shape)
        return self.get_observation(obs_index)
