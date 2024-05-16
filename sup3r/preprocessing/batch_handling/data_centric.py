"""
Sup3r batch_handling module.
@author: bbenton
"""
import logging

import numpy as np

from sup3r.containers.batchers.abstract import Batch
from sup3r.preprocessing.batch_handling.base import (
    BatchHandler,
    ValidationData,
)
from sup3r.preprocessing.data_handling import (
    DataHandlerDCforH5,
)
from sup3r.utilities.utilities import (
    uniform_box_sampler,
    uniform_time_sampler,
    weighted_box_sampler,
    weighted_time_sampler,
)

np.random.seed(42)

logger = logging.getLogger(__name__)


class ValidationDataDC(ValidationData):
    """Iterator for data-centric validation data"""

    N_TIME_BINS = 12
    N_SPACE_BINS = 4

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

        val_indices = {}
        for t in range(self.N_TIME_BINS):
            val_indices[t] = []
            h_idx = self.get_handler_index()
            h = self.data_handlers[h_idx]
            for _ in range(self.batch_size):
                spatial_slice = uniform_box_sampler(h.data.shape,
                                                    self.sample_shape[:2])
                weights = np.zeros(self.N_TIME_BINS)
                weights[t] = 1
                time_slice = weighted_time_sampler(h.data.shape,
                                                       self.sample_shape[2],
                                                       weights)
                tuple_index = (
                    *spatial_slice, time_slice,
                    np.arange(h.data.shape[-1])
                )
                val_indices[t].append({
                    'handler_index': h_idx,
                    'tuple_index': tuple_index
                })
        for s in range(self.N_SPACE_BINS):
            val_indices[s + self.N_TIME_BINS] = []
            h_idx = self.get_handler_index()
            h = self.data_handlers[h_idx]
            for _ in range(self.batch_size):
                weights = np.zeros(self.N_SPACE_BINS)
                weights[s] = 1
                spatial_slice = weighted_box_sampler(h.data.shape,
                                                     self.sample_shape[:2],
                                                     weights)
                time_slice = uniform_time_sampler(h.data.shape,
                                                      self.sample_shape[2])
                tuple_index = (
                    *spatial_slice, time_slice,
                    np.arange(h.data.shape[-1])
                )
                val_indices[s + self.N_TIME_BINS].append({
                    'handler_index': h_idx,
                    'tuple_index': tuple_index
                })
        return val_indices

    def __next__(self):
        if self._i < len(self.val_indices.keys()):
            high_res = np.zeros(
                (self.batch_size, self.sample_shape[0], self.sample_shape[1],
                 self.sample_shape[2], self.data_handlers[0].shape[-1]),
                dtype=np.float32)
            val_indices = self.val_indices[self._i]
            for i, idx in enumerate(val_indices):
                high_res[i, ...] = self.data_handlers[
                    idx['handler_index']].data[idx['tuple_index']]

            batch = self.BATCH_CLASS.get_coarse_batch(
                high_res,
                self.s_enhance,
                t_enhance=self.t_enhance,
                temporal_coarsening_method=self.temporal_coarsening_method,
                hr_features_ind=self.hr_features_ind,
                smoothing=self.smoothing,
                smoothing_ignore=self.smoothing_ignore)
            self._i += 1
            return batch
        else:
            raise StopIteration


class ValidationDataTemporalDC(ValidationDataDC):
    """Iterator for data-centric temporal validation data"""

    N_SPACE_BINS = 0


class ValidationDataSpatialDC(ValidationDataDC):
    """Iterator for data-centric spatial validation data"""

    N_TIME_BINS = 0

    def __next__(self):
        if self._i < len(self.val_indices.keys()):
            high_res = np.zeros(
                (self.batch_size, self.sample_shape[0], self.sample_shape[1],
                 self.data_handlers[0].shape[-1]),
                dtype=np.float32)
            val_indices = self.val_indices[self._i]
            for i, idx in enumerate(val_indices):
                high_res[i, ...] = self.data_handlers[
                    idx['handler_index']].data[idx['tuple_index']][..., 0, :]

            batch = self.BATCH_CLASS.get_coarse_batch(
                high_res,
                self.s_enhance,
                hr_features_ind=self.hr_features_ind,
                smoothing=self.smoothing,
                smoothing_ignore=self.smoothing_ignore)
            self._i += 1
            return batch
        else:
            raise StopIteration


class BatchHandlerDC(BatchHandler):
    """Data-centric batch handler"""

    VAL_CLASS = ValidationDataTemporalDC
    BATCH_CLASS = Batch
    DATA_HANDLER_CLASS = DataHandlerDCforH5

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args : list
            Same positional args as BatchHandler
        **kwargs : dict
            Same keyword args as BatchHandler
        """
        super().__init__(*args, **kwargs)

        self.temporal_weights = np.ones(self.val_data.N_TIME_BINS)
        self.temporal_weights /= np.sum(self.temporal_weights)
        self.old_temporal_weights = [0] * self.val_data.N_TIME_BINS
        bin_range = self.data_handlers[0].data.shape[2]
        bin_range -= self.sample_shape[2] - 1
        self.temporal_bins = np.array_split(np.arange(0, bin_range),
                                            self.val_data.N_TIME_BINS)
        self.temporal_bins = [b[0] for b in self.temporal_bins]

        logger.info('Using temporal weights: '
                    f'{[round(w, 3) for w in self.temporal_weights]}')
        self.temporal_sample_record = [0] * self.val_data.N_TIME_BINS
        self.norm_temporal_record = [0] * self.val_data.N_TIME_BINS

    def update_training_sample_record(self):
        """Keep track of number of observations from each temporal bin"""
        handler = self.data_handlers[self.handler_index]
        t_start = handler.current_obs_index[2].start
        t_bin_number = np.digitize(t_start, self.temporal_bins)
        self.temporal_sample_record[t_bin_number - 1] += 1

    def __iter__(self):
        self._i = 0
        self.temporal_sample_record = [0] * self.val_data.N_TIME_BINS
        return self

    def __next__(self):
        self.current_batch_indices = []
        if self._i < self.n_batches:
            handler = self.get_rand_handler()
            high_res = np.zeros(
                (self.batch_size, self.sample_shape[0], self.sample_shape[1],
                 self.sample_shape[2], self.shape[-1]),
                dtype=np.float32)

            for i in range(self.batch_size):
                high_res[i, ...] = handler.get_next(
                    temporal_weights=self.temporal_weights)
                self.current_batch_indices.append(handler.current_obs_index)

                self.update_training_sample_record()

            batch = self.BATCH_CLASS.get_coarse_batch(
                high_res,
                self.s_enhance,
                t_enhance=self.t_enhance,
                temporal_coarsening_method=self.temporal_coarsening_method,
                hr_features_ind=self.hr_features_ind,
                features=self.features,
                smoothing=self.smoothing,
                smoothing_ignore=self.smoothing_ignore)

            self._i += 1
            return batch
        else:
            total_count = self.n_batches * self.batch_size
            self.norm_temporal_record = [
                c / total_count for c in self.temporal_sample_record.copy()
            ]
            self.old_temporal_weights = self.temporal_weights.copy()
            raise StopIteration


class BatchHandlerSpatialDC(BatchHandler):
    """Data-centric batch handler"""

    VAL_CLASS = ValidationDataSpatialDC
    BATCH_CLASS = Batch
    DATA_HANDLER_CLASS = DataHandlerDCforH5

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args : list
            Same positional args as BatchHandler
        **kwargs : dict
            Same keyword args as BatchHandler
        """
        super().__init__(*args, **kwargs)

        self.spatial_weights = np.ones(self.val_data.N_SPACE_BINS)
        self.spatial_weights /= np.sum(self.spatial_weights)
        self.old_spatial_weights = [0] * self.val_data.N_SPACE_BINS
        self.max_rows = self.data_handlers[0].data.shape[0] + 1
        self.max_rows -= self.sample_shape[0]
        self.max_cols = self.data_handlers[0].data.shape[1] + 1
        self.max_cols -= self.sample_shape[1]
        bin_range = self.max_rows * self.max_cols
        self.spatial_bins = np.array_split(np.arange(0, bin_range),
                                           self.val_data.N_SPACE_BINS)
        self.spatial_bins = [b[0] for b in self.spatial_bins]

        logger.info('Using spatial weights: '
                    f'{[round(w, 3) for w in self.spatial_weights]}')

        self.spatial_sample_record = [0] * self.val_data.N_SPACE_BINS
        self.norm_spatial_record = [0] * self.val_data.N_SPACE_BINS

    def update_training_sample_record(self):
        """Keep track of number of observations from each temporal bin"""
        handler = self.data_handlers[self.handler_index]
        row = handler.current_obs_index[0].start
        col = handler.current_obs_index[1].start
        s_start = self.max_rows * row + col
        s_bin_number = np.digitize(s_start, self.spatial_bins)
        self.spatial_sample_record[s_bin_number - 1] += 1

    def __iter__(self):
        self._i = 0
        self.spatial_sample_record = [0] * self.val_data.N_SPACE_BINS
        return self

    def __next__(self):
        self.current_batch_indices = []
        if self._i < self.n_batches:
            handler = self.get_rand_handler()
            high_res = np.zeros((self.batch_size, self.sample_shape[0],
                                 self.sample_shape[1], self.shape[-1],
                                 ),
                                dtype=np.float32,
                                )

            for i in range(self.batch_size):
                high_res[i, ...] = handler.get_next(
                    spatial_weights=self.spatial_weights)[..., 0, :]
                self.current_batch_indices.append(handler.current_obs_index)

                self.update_training_sample_record()

            batch = self.BATCH_CLASS.get_coarse_batch(
                high_res,
                self.s_enhance,
                hr_features_ind=self.hr_features_ind,
                features=self.features,
                smoothing=self.smoothing,
                smoothing_ignore=self.smoothing_ignore,
            )

            self._i += 1
            return batch
        else:
            total_count = self.n_batches * self.batch_size
            self.norm_spatial_record = [
                c / total_count for c in self.spatial_sample_record
            ]
            self.old_spatial_weights = self.spatial_weights.copy()
            raise StopIteration
