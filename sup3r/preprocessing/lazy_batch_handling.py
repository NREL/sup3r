"""Batch handling classes for queued batch loads"""
import logging
import threading

import numpy as np
import tensorflow as tf
import xarray as xr

from sup3r.preprocessing.data_handling import DualDataHandler
from sup3r.preprocessing.data_handling.base import DataHandler

logger = logging.getLogger(__name__)


class LazyDataHandler(DataHandler):
    """Lazy loading data handler. Uses precomputed netcdf files (usually from
    a DataHandler.to_netcdf() call after populating DataHandler.data) to create
    batches on the fly during training without previously loading to memory."""

    def __init__(
        self, files, features, sample_shape, lr_only_features=(),
        hr_exo_features=(), chunk_kwargs=None, mode='lazy'
    ):
        self.features = features
        self.sample_shape = sample_shape
        self._lr_only_features = lr_only_features
        self._hr_exo_features = hr_exo_features
        self.chunk_kwargs = (
            chunk_kwargs if chunk_kwargs is not None
            else {'south_north': 10, 'west_east': 10, 'time': 3})
        self.data = xr.open_mfdataset(files, chunks=chunk_kwargs)
        self._shape = (*self.data["latitude"].shape, len(self.data["time"]))
        self._i = 0
        self.mode = mode
        if mode == 'eager':
            logger.info(f'Loading {files} in eager mode.')
            self.data = self.data.compute()

        logger.info(f'Initialized {self.__class__.__name__} with '
                    f'files = {files}, features = {features}, '
                    f'sample_shape = {sample_shape}.')

    def get_observation(self, obs_index):
        out = self.data[self.features].isel(
            south_north=obs_index[0],
            west_east=obs_index[1],
            time=obs_index[2],
        )
        if self.mode == 'lazy':
            out = out.compute()
        out = out.to_dataarray().values
        out = np.transpose(out, axes=(2, 3, 1, 0))
        #out = tf.transpose(out, perm=[2, 3, 1, 0]).numpy()
        #out = np.zeros((*self.sample_shape, len(self.features)))
        return out

    def get_next(self):
        """Get next observation sample."""
        obs_index = self.get_observation_index()
        return self.get_observation(obs_index)

    def __getitem__(self, index):
        return self.get_next()

    def __next__(self):
        if self._i < self.epoch_samples:
            out = self.get_next()
            self._i += 1
            return out
        else:
            raise StopIteration


class LazyDualDataHandler(DualDataHandler):
    """Lazy loading dual data handler. Matches sample regions for low res and
    high res lazy data handlers."""

    def __init__(self, lr_dh, hr_dh, s_enhance=1, t_enhance=1,
                 epoch_samples=1024):
        self.lr_dh = lr_dh
        self.hr_dh = hr_dh
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        self.current_obs_index = None
        self._means = None
        self._stds = None
        self.epoch_samples = epoch_samples
        self.check_shapes()

        logger.info(f'Finished initializing {self.__class__.__name__}.')

    @property
    def means(self):
        """Get dictionary of means for all features available in low-res and
        high-res handlers."""
        if self._means is None:
            lr_features = self.lr_dh.features
            hr_only_features = [f for f in self.hr_dh.features
                                if f not in lr_features]
            self._means = dict(zip(lr_features,
                                   self.lr_dh.data[lr_features].mean(axis=0)))
            hr_means = dict(zip(hr_only_features,
                                self.hr_dh[hr_only_features].mean(axis=0)))
            self._means.update(hr_means)
        return self._means

    @property
    def stds(self):
        """Get dictionary of standard deviations for all features available in
        low-res and high-res handlers."""
        if self._stds is None:
            lr_features = self.lr_dh.features
            hr_only_features = [f for f in self.hr_dh.features
                                if f not in lr_features]
            self._stds = dict(zip(lr_features,
                              self.lr_dh.data[lr_features].std(axis=0)))
            hr_stds = dict(zip(hr_only_features,
                               self.hr_dh[hr_only_features].std(axis=0)))
            self._stds.update(hr_stds)
        return self._stds

    def __iter__(self):
        self._i = 0
        return self

    def __len__(self):
        return self.epoch_samples

    @property
    def size(self):
        """'Size' of data handler. Used to compute handler weights for batch
        sampling."""
        return np.prod(self.lr_dh.shape)

    def check_shapes(self):
        """Make sure data handler shapes are compatible with enhancement
        factors."""
        hr_shape = self.hr_dh.shape
        lr_shape = self.lr_dh.shape
        enhanced_shape = (lr_shape[0] * self.s_enhance,
                          lr_shape[1] * self.s_enhance,
                          lr_shape[2] * self.t_enhance)
        msg = (f'hr_dh.shape {hr_shape} and enhanced lr_dh.shape '
               f'{enhanced_shape} are not compatible')
        assert hr_shape == enhanced_shape, msg

    def get_next(self):
        """Get next pair of low-res / high-res samples ensuring that low-res
        and high-res sampling regions match.

        Returns
        -------
        tuple
            (low_res, high_res) pair
        """
        lr_obs_idx = self.lr_dh._get_observation_index()
        hr_obs_idx = [slice(s.start * self.s_enhance, s.stop * self.s_enhance)
                      for s in lr_obs_idx[:2]]
        hr_obs_idx += [slice(s.start * self.t_enhance, s.stop * self.t_enhance)
                       for s in lr_obs_idx[2:]]
        out = (self.lr_dh._get_observation(lr_obs_idx),
               self.hr_dh._get_observation(hr_obs_idx))
        return out

    def __getitem__(self, index):
        logger.info(f'Getting sample {index + 1}.')
        return self.get_next()

    def __next__(self):
        if self._i < self.epoch_samples:
            out = self.get_next()
            self._i += 1
            return out
        else:
            raise StopIteration

    def __call__(self):
        """Call method to enable Dataset.from_generator() call."""
        for i in range(self.epoch_samples):
            yield self.__getitem__(i)

    @property
    def data(self):
        """Return tensorflow dataset generator."""
        lr_shape = (*self.lr_dh.sample_shape, len(self.lr_dh.features))
        hr_shape = (*self.hr_dh.sample_shape, len(self.hr_dh.features))
        return tf.data.Dataset.from_generator(
            self.__call__,
            output_signature=(tf.TensorSpec(lr_shape, tf.float32),
                              tf.TensorSpec(hr_shape, tf.float32)))


class TrainingSession:

    def __init__(self, batch_handler, model, kwargs):
        self.model = model
        self.batch_handler = batch_handler
        self.kwargs = kwargs
        self.train_thread = threading.Thread(target=self.train)

        self.batch_handler.start()
        self.train_thread.start()

        self.batch_handler.stop()
        self.train_thread.join()

    def train(self):
        self.model.train(self.batch_handler, **self.kwargs)

