"""Base loading classes. These are containers which also load data from
file_paths and include some sampling ability to interface with batcher
classes."""

import logging

import numpy as np
import xarray as xr

from sup3r.containers.loaders.abstract import AbstractLoader
from sup3r.containers.samplers.base import Sampler

logger = logging.getLogger(__name__)


class LoaderNC(AbstractLoader, Sampler):
    """Base loader. Loads precomputed netcdf files (usually from
    a DataHandler.to_netcdf() call after populating DataHandler.data) and can
    retrieve samples from this data for use in batch building."""

    def __init__(
        self, file_paths, features, sample_shape, lr_only_features=(),
        hr_exo_features=(), res_kwargs=None, mode='lazy'
    ):
        """
        Parameters
        ----------
        file_paths : str | pathlib.Path | list
            Location(s) of files to load
        features : list
            list of all features extracted or to extract.
        sample_shape : tuple
            Size of spatiotemporal extent of samples used to build batches.
        lr_only_features : list | tuple
            List of feature names or patt*erns that should only be included in
            the low-res training set and not the high-res observations.
        hr_exo_features : list | tuple
            List of feature names or patt*erns that should be included in the
            high-resolution observation but not expected to be output from the
            generative model. An example is high-res topography that is to be
            injected mid-network.
        res_kwargs : dict
            kwargs for xr.open_mfdataset()
        mode : str
            Options are ('lazy', 'eager') for how to load data.
        """
        super().__init__(file_paths, features, lr_only_features,
                         hr_exo_features)
        self.features = features
        self.sample_shape = sample_shape
        self._lr_only_features = lr_only_features
        self._hr_exo_features = hr_exo_features
        self._res_kwargs = res_kwargs or {}
        self._mode = mode
        self._shape = None

        logger.info(f'Initialized {self.__class__.__name__} with '
                    f'files = {self.file_paths}, features = {self.features}, '
                    f'sample_shape = {self.sample_shape}.')

    @property
    def features(self):
        """Return set of features loaded from file_paths."""
        return self._features

    @features.setter
    def features(self, features):
        self._features = features

    @property
    def sample_shape(self):
        """Return shape of samples which can be used to build batches."""
        return self._sample_shape

    @sample_shape.setter
    def sample_shape(self, sample_shape):
        self._sample_shape = sample_shape

    @property
    def shape(self):
        """Return shape of extent available for sampling."""
        if self._shape is None:
            self._shape = (*self.data["latitude"].shape,
                           len(self.data["time"]))
        return self._shape

    def load(self):
        """Xarray dataset either lazily loaded (mode = 'lazy') or loaded into
        memory right away (mode = 'eager').

        Returns
        -------
        xr.Dataset()
            xarray dataset with the requested features
        """
        data = xr.open_mfdataset(self.file_paths, **self._res_kwargs)
        msg = (f'Loading {self.file_paths} with kwargs = '
               f'{self._res_kwargs} and mode = {self._mode}')
        logger.info(msg)

        if self._mode == 'eager':
            data = data.compute()

        return data[self.features]

    def __getitem__(self, key):
        """Get observation/sample. Should return a single sample from the
        underlying data with shape (spatial_1, spatial_2, temporal,
        features)."""

        out = self.data.isel(
            south_north=key[0],
            west_east=key[1],
            time=key[2],
        )

        if self._mode == 'lazy':
            out = out.compute()

        out = out.to_dataarray().values
        return np.transpose(out, axes=(2, 3, 1, 0))
