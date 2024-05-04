"""Batch handling classes for queued batch loads"""
import logging

import numpy as np
import tensorflow as tf
import xarray as xr

from sup3r.preprocessing.data_handling import DualDataHandler
from sup3r.preprocessing.data_handling.base import DataHandler
from sup3r.preprocessing.dual_batch_handling import DualBatchHandler
from sup3r.utilities.utilities import uniform_box_sampler, uniform_time_sampler

logger = logging.getLogger(__name__)


class LazyDataHandler(tf.keras.utils.Sequence, DataHandler):
    """Lazy loading data handler. Uses precomputed netcdf files (usually from
    a DataHandler.to_netcdf() call after populating DataHandler.data) to create
    batches on the fly during training without previously loading to memory."""

    def __init__(
        self, files, features, sample_shape, epoch_samples=1024,
        lr_only_features=tuple(), hr_exo_features=tuple()
    ):
        self.ds = xr.open_mfdataset(
            files, chunks={'south_north': 200, 'west_east': 200, 'time': 20})
        self.features = features
        self.sample_shape = sample_shape
        self.epoch_samples = epoch_samples
        self._lr_only_features = lr_only_features
        self._hr_exo_features = hr_exo_features
        self._shape = (*self.ds["latitude"].shape, len(self.ds["time"]))

        logger.info(f'Initialized {self.__class__.__name__} with '
                    f'files = {files}, features = {features}, '
                    f'sample_shape = {sample_shape}, '
                    f'epoch_samples = {epoch_samples}.')

    def __iter__(self):
        self._i = 0
        return self

    def __len__(self):
        return self.epoch_samples

    def _get_observation_index(self):
        spatial_slice = uniform_box_sampler(
            self.shape, self.sample_shape[:2]
        )
        temporal_slice = uniform_time_sampler(
            self.shape, self.sample_shape[2]
        )
        return (*spatial_slice, temporal_slice)

    def _get_observation(self, obs_index):
        out = self.ds[self.features].isel(
            south_north=obs_index[0],
            west_east=obs_index[1],
            time=obs_index[2],
        )
        out = tf.convert_to_tensor(out.to_dataarray())
        out = tf.transpose(out, perm=[2, 3, 1, 0])
        return out

    def get_next(self):
        """Get next observation sample."""
        obs_index = self._get_observation_index()
        return self._get_observation(obs_index)

    def __get_item__(self, index):
        return self.get_next()

    def __next__(self):
        return self.get_next()

    def __call__(self):
        """Call method to enable Dataset.from_generator() call."""
        for _ in range(self.epoch_samples):
            yield(self.get_next())

    @classmethod
    def gen(cls, files, features, sample_shape=(10, 10, 5),
            epoch_samples=1024):
        """Return tensorflow dataset generator."""

        return tf.data.Dataset.from_generator(
            cls(files, features, sample_shape, epoch_samples),
            output_types=(tf.float32),
            output_shapes=(*sample_shape, len(features)))


class LazyDualDataHandler(tf.keras.utils.Sequence, DualDataHandler):
    """Lazy loading dual data handler. Matches sample regions for low res and
    high res lazy data handlers."""

    def __init__(self, lr_dh, hr_dh, s_enhance=1, t_enhance=1):
        self.lr_dh = lr_dh
        self.hr_dh = hr_dh
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        self.current_obs_index = None
        self.check_shapes()

        logger.info(f'Finished initializing {self.__class__.__name__}.')

    def __iter__(self):
        self._i = 0
        return self

    def __len__(self):
        return self.lr_dh.epoch_samples

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
        and high-res sampling regions match."""
        lr_obs_idx = self.lr_dh._get_observation_index()
        hr_obs_idx = [slice(s.start * self.s_enhance, s.stop * self.s_enhance)
                      for s in lr_obs_idx[:2]]
        hr_obs_idx += [slice(s.start * self.t_enhance, s.stop * self.t_enhance)
                       for s in lr_obs_idx[2:]]
        logger.debug(f'Getting observation for lr_obs_index = {lr_obs_idx}, '
                     f'hr_obs_index = {hr_obs_idx}.')
        out = (self.hr_dh._get_observation(hr_obs_idx),
               self.lr_dh._get_observation(lr_obs_idx))
        return out

    def __get_item__(self, index):
        return self.get_next()

    def __next__(self):
        return self.get_next()


class LazyDualBatchHandler(DualBatchHandler):
    """Dual batch handler which uses lazy data handlers to load data as
    needed rather than all in memory at once."""

    def __init__(self, data_handlers, batch_size=32, n_batches=100):
        self.data_handlers = data_handlers
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.s_enhance = self.data_handlers[0].s_enhance
        self.t_enhance = self.data_handlers[0].t_enhance
        self._means = None
        self._stds = None

    @property
    def means(self):
        """Means used to normalize the data."""
        if self._means is None:
            self._means = {}
            for k in self.data_handlers[0].lr_dh.features:
                logger.info(f'Getting mean for {k}.')
                self._means[k] = self.data_handlers[0].lr_dh.ds[k].mean()
            for k in self.data_handlers[0].hr_dh.features:
                if k not in self._means:
                    logger.info(f'Getting mean for {k}.')
                    self._means[k] = self.data_handlers[0].hr_dh.ds[k].mean()
        return self._means

    @means.setter
    def means(self, means):
        self._means = means

    @property
    def stds(self):
        """Standard deviations used to normalize the data."""
        if self._stds is None:
            self._stds = {}
            for k in self.data_handlers[0].lr_dh.features:
                logger.info(f'Getting stdev for {k}.')
                self._stds[k] = self.data_handlers[0].lr_dh.ds[k].std()
            for k in self.data_handlers[0].hr_dh.features:
                if k not in self._stds:
                    logger.info(f'Getting stdev for {k}.')
                    self._stds[k] = self.data_handlers[0].hr_dh.ds[k].std()
        return self._stds

    @stds.setter
    def stds(self, stds):
        self._stds = stds
