"""Batch handling classes for dual data handlers"""
import logging
from datetime import datetime as dt

import numpy as np
import tensorflow as tf

from sup3r.preprocessing.batch_handling.base import (
    Batch,
    BatchHandler,
    ValidationData,
)
from sup3r.preprocessing.mixin import (
    MultiDualMixIn,
    MultiHandlerMixIn,
)
from sup3r.utilities.utilities import uniform_box_sampler, uniform_time_sampler

logger = logging.getLogger(__name__)


class DualValidationData(ValidationData, MultiHandlerMixIn):
    """Iterator for validation data for training with dual data handler"""

    # Classes to use for handling an individual batch obj.
    BATCH_CLASS = Batch

    def _get_val_indices(self):
        """List of dicts to index each validation data observation across all
        handlers

        Returns
        -------
        val_indices : list[dict]
            List of dicts with handler_index and tuple_index. The tuple index
            is used to get validation data observation with
        data[tuple_index]
        """

        val_indices = []
        for i, h in enumerate(self.data_handlers):
            if h.hr_val_data is not None:
                for _ in range(h.hr_val_data.shape[2]):
                    spatial_slice = uniform_box_sampler(
                        h.lr_val_data.shape, self.lr_sample_shape[:2])
                    temporal_slice = uniform_time_sampler(
                        h.lr_val_data.shape, self.lr_sample_shape[2])
                    lr_index = (*spatial_slice, temporal_slice,
                                np.arange(h.lr_val_data.shape[-1]))
                    hr_index = [slice(s.start * self.s_enhance,
                                      s.stop * self.s_enhance)
                                for s in lr_index[:2]]
                    hr_index += [slice(s.start * self.t_enhance,
                                       s.stop * self.t_enhance)
                                 for s in lr_index[2:-1]]
                    hr_index.append(lr_index[-1])
                    hr_index = tuple(hr_index)
                    val_indices.append({
                        'handler_index': i,
                        'hr_index': hr_index,
                        'lr_index': lr_index
                    })
        return val_indices

    @property
    def shape(self):
        """Shape of full validation dataset across all handlers

        Returns
        -------
        shape : tuple
            (spatial_1, spatial_2, temporal, features)
            With temporal extent equal to the sum across all data handlers time
            dimension
        """
        time_steps = np.sum([h.hr_val_data.shape[2]
                             for h in self.data_handlers])
        return (self.data_handlers[0].hr_val_data.shape[0],
                self.data_handlers[0].hr_val_data.shape[1], time_steps,
                self.data_handlers[0].hr_val_data.shape[3])

    def __next__(self):
        """Get validation data batch

        Returns
        -------
        batch : Batch
            validation data batch with low and high res data each with
            n_observations = batch_size
        """
        self.current_batch_indices = []
        if self._remaining_observations > 0:
            if self._remaining_observations > self.batch_size:
                n_obs = self.batch_size
            else:
                n_obs = self._remaining_observations

            high_res = np.zeros(
                (n_obs, self.hr_sample_shape[0], self.hr_sample_shape[1],
                 self.hr_sample_shape[2], self.data_handlers[0].shape[-1]),
                dtype=np.float32,
            )
            low_res = np.zeros(
                (n_obs, self.lr_sample_shape[0], self.lr_sample_shape[1],
                 self.lr_sample_shape[2], self.data_handlers[0].shape[-1]),
                dtype=np.float32,
            )
            for i in range(high_res.shape[0]):
                val_index = self.val_indices[self._i + i]
                high_res[i, ...] = self.data_handlers[val_index[
                    'handler_index']].hr_val_data[val_index['hr_index']]
                low_res[i, ...] = self.data_handlers[val_index[
                    'handler_index']].lr_val_data[val_index['lr_index']]
                self._remaining_observations -= 1
                self.current_batch_indices.append(val_index)

            # This checks if there is only a single timestep. If so this means
            # we are using a spatial batch handler which uses 4D batches.
            if self.sample_shape[2] == 1:
                high_res = high_res[..., 0, :]
                low_res = low_res[..., 0, :]

            high_res = high_res[..., self.hr_features_ind]
            batch = self.BATCH_CLASS(low_res=low_res, high_res=high_res)
            self._i += 1
            return batch
        else:
            raise StopIteration


class DualBatchHandler(BatchHandler, MultiDualMixIn):
    """Batch handling class for dual data handlers"""

    BATCH_CLASS = Batch
    VAL_CLASS = DualValidationData

    def __next__(self):
        """Get the next batch of observations.

        Returns
        -------
        batch : Batch
            Batch object with batch.low_res and batch.high_res attributes
            with the appropriate subsampling of interpolated ERA.
        """
        self.current_batch_indices = []
        start = dt.now()
        if self._i < self.n_batches:
            handler = self.get_rand_handler()
            hr_list = []
            lr_list = []
            for _ in range(self.batch_size):
                lr_sample, hr_sample = handler.get_next()
                lr_list.append(tf.expand_dims(lr_sample, axis=0))
                hr_list.append(tf.expand_dims(hr_sample, axis=0))
                self.current_batch_indices.append(handler.current_obs_idx)

            batch = self.BATCH_CLASS(
                low_res=tf.concat(lr_list, axis=0),
                high_res=tf.concat(hr_list, axis=0))

            self._i += 1
            logger.debug(f'Built batch in {dt.now() - start}.')
            return batch
        else:
            raise StopIteration


class SpatialDualBatchHandler(DualBatchHandler):
    """Batch handling class for h5 data as high res (usually WTK) and ERA5 as
    low res"""

    BATCH_CLASS = Batch
    VAL_CLASS = DualValidationData

    def __next__(self):
        """Get the next iterator output.

        Returns
        -------
        batch : Batch
            Batch object with batch.low_res and batch.high_res attributes
            with the appropriate subsampling of interpolated ERA.
        """
        self.current_batch_indices = []
        if self._i < self.n_batches:
            handler = self.get_rand_handler()
            hr_list = []
            lr_list = []
            for i in range(self.batch_size):
                logger.debug(f'Making batch, observation: {i + 1} / '
                             f'{self.batch_size}.')
                hr_sample, lr_sample = handler.get_next()
                hr_list.append(np.expand_dims(hr_sample[..., 0, :], axis=0))
                lr_list.append(np.expand_dims(lr_sample[..., 0, :], axis=0))
                self.current_batch_indices.append(handler.current_obs_index)

            batch = self.BATCH_CLASS(
                low_res=np.concatenate(lr_list, axis=0, dtype=np.float32),
                high_res=np.concatenate(hr_list, axis=0, dtype=np.float32))

            self._i += 1
            return batch
        else:
            raise StopIteration
