"""Batch handling classes for dual data handlers"""
import logging

import numpy as np

from sup3r.preprocessing.batch_handling import (
    Batch,
    BatchHandler,
    ValidationData,
)
from sup3r.utilities.utilities import uniform_box_sampler, uniform_time_sampler

logger = logging.getLogger(__name__)


class DualValidationData(ValidationData):
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
        for i, h in enumerate(self.handlers):
            if h.hr_val_data is not None:
                for _ in range(h.hr_val_data.shape[2]):
                    spatial_slice = uniform_box_sampler(
                        h.hr_val_data, self.sample_shape[:2]
                    )
                    temporal_slice = uniform_time_sampler(
                        h.hr_val_data, self.sample_shape[2]
                    )
                    hr_index = tuple(
                        [
                            *spatial_slice,
                            temporal_slice,
                            np.arange(h.hr_val_data.shape[-1]),
                        ]
                    )
                    lr_index = []
                    for s in hr_index[:2]:
                        lr_index.append(
                            slice(
                                s.start // self.s_enhance,
                                s.stop // self.s_enhance + 1,
                            )
                        )
                    for s in hr_index[2:-1]:
                        lr_index.append(
                            slice(
                                s.start // self.t_enhance,
                                s.stop // self.t_enhance + 1,
                            )
                        )
                    lr_index.append(hr_index[-1])
                    lr_index = tuple(lr_index)
                    val_indices.append(
                        {
                            'handler_index': i,
                            'hr_index': hr_index,
                            'lr_index': lr_index,
                        }
                    )
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
        time_steps = 0
        for h in self.handlers:
            time_steps += h.hr_val_data.shape[2]
        return (
            self.handlers[0].hr_val_data.shape[0],
            self.handlers[0].hr_val_data.shape[1],
            time_steps,
            self.handlers[0].hr_val_data.shape[3],
        )

    def __next__(self):
        """Get validation data batch

        Returns
        -------
        batch : Batch
            validation data batch with low and high res data each with
            n_observations = batch_size
        """
        if self._remaining_observations > 0:
            if self._remaining_observations > self.batch_size:
                n_obs = self.batch_size
            else:
                n_obs = self._remaining_observations

            high_res = np.zeros(
                (
                    n_obs,
                    self.sample_shape[0],
                    self.sample_shape[1],
                    self.sample_shape[2],
                    self.handlers[0].shape[-1],
                ),
                dtype=np.float32,
            )
            low_res = np.zeros(
                (
                    n_obs,
                    self.sample_shape[0] // self.s_enhance,
                    self.sample_shape[1] // self.s_enhance,
                    self.sample_shape[2] // self.t_enhance,
                    self.handlers[0].shape[-1],
                ),
                dtype=np.float32,
            )
            for i in range(high_res.shape[0]):
                val_index = self.val_indices[self._i + i]
                high_res[i, ...] = self.handlers[
                    val_index['handler_index']
                ].hr_val_data[val_index['hr_index']]
                low_res[i, ...] = self.handlers[
                    val_index['handler_index']
                ].lr_val_data[val_index['lr_index']]
                self._remaining_observations -= 1

            if self.sample_shape[2] == 1:
                high_res = high_res[..., 0, :]
                low_res = low_res[..., 0, :]

            high_res = self.BATCH_CLASS.reduce_features(
                high_res, self.output_features_ind
            )
            batch = self.BATCH_CLASS(low_res=low_res, high_res=high_res)
            self._i += 1
            return batch
        else:
            raise StopIteration


class DualBatchHandler(BatchHandler):
    """Batch handling class for dual data handlers"""

    BATCH_CLASS = Batch

    def __iter__(self):
        self._i = 0
        return self

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
            handler_index = np.random.randint(0, len(self.data_handlers))
            self.current_handler_index = handler_index
            handler = self.data_handlers[handler_index]
            high_res = np.zeros(
                (
                    self.batch_size,
                    self.sample_shape[0],
                    self.sample_shape[1],
                    self.sample_shape[2],
                    self.shape[-1],
                ),
                dtype=np.float32,
            )
            low_res = np.zeros(
                (
                    self.batch_size,
                    self.sample_shape[0] // self.s_enhance,
                    self.sample_shape[1] // self.s_enhance,
                    self.sample_shape[2] // self.t_enhance,
                    self.shape[-1],
                ),
                dtype=np.float32,
            )

            for i in range(self.batch_size):
                high_res[i, ...], low_res[i, ...] = handler.get_next()
                self.current_batch_indices.append(handler.current_obs_index)

            high_res = self.BATCH_CLASS.reduce_features(
                high_res, self.output_features_ind
            )
            batch = self.BATCH_CLASS(low_res=low_res, high_res=high_res)

            self._i += 1
            return batch
        else:
            raise StopIteration


class SpatialDualBatchHandler(DualBatchHandler):
    """Batch handling class for h5 data as high res (usually WTK) and ERA5 as
    low res"""

    BATCH_CLASS = Batch
    VAL_CLASS = DualValidationData

    def __iter__(self):
        self._i = 0
        return self

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
            handler_index = np.random.randint(0, len(self.data_handlers))
            self.current_handler_index = handler_index
            handler = self.data_handlers[handler_index]
            high_res = np.zeros(
                (
                    self.batch_size,
                    self.sample_shape[0],
                    self.sample_shape[1],
                    self.shape[-1],
                ),
                dtype=np.float32,
            )
            low_res = np.zeros(
                (
                    self.batch_size,
                    self.sample_shape[0] // self.s_enhance,
                    self.sample_shape[1] // self.s_enhance,
                    self.shape[-1],
                ),
                dtype=np.float32,
            )

            for i in range(self.batch_size):
                hr, lr = handler.get_next()
                high_res[i, ...] = hr[..., 0, :]
                low_res[i, ...] = lr[..., 0, :]
                self.current_batch_indices.append(handler.current_obs_index)

            high_res = self.BATCH_CLASS.reduce_features(
                high_res, self.output_features_ind
            )
            batch = self.BATCH_CLASS(low_res=low_res, high_res=high_res)

            self._i += 1
            return batch
        else:
            raise StopIteration
