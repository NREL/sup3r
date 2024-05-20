"""
Sup3r batch_handling module.
@author: bbenton
"""
import logging

import numpy as np

from sup3r.containers import (
    BatchHandler,
    DataCentricSampler,
)

np.random.seed(42)

logger = logging.getLogger(__name__)


class BatchHandlerDC(BatchHandler):
    """Data-centric batch handler"""

    SAMPLER = DataCentricSampler

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
        bin_range = self.containers[0].data.shape[2]
        bin_range -= self.sample_shape[2] - 1
        self.temporal_bins = np.array_split(
            np.arange(0, bin_range), self.val_data.N_TIME_BINS
        )
        self.temporal_bins = [b[0] for b in self.temporal_bins]

        logger.info(
            'Using temporal weights: '
            f'{[round(w, 3) for w in self.temporal_weights]}'
        )
        self.temporal_sample_record = [0] * self.val_data.N_TIME_BINS
        self.norm_temporal_record = [0] * self.val_data.N_TIME_BINS

    def update_training_sample_record(self):
        """Keep track of number of observations from each temporal bin"""
        handler = self.containers[self.current_handler_index]
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
            handler = self.get_random_container()
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

            for i in range(self.batch_size):
                high_res[i, ...] = handler.get_next(
                    temporal_weights=self.temporal_weights
                )

                self.update_training_sample_record()

            batch = self.coarsen(
                high_res,
                temporal_coarsening_method=self.temporal_coarsening_method,
                smoothing=self.smoothing,
                smoothing_ignore=self.smoothing_ignore,
            )

            self._i += 1
            return batch
        total_count = self.n_batches * self.batch_size
        self.norm_temporal_record = [
            c / total_count for c in self.temporal_sample_record.copy()
        ]
        self.old_temporal_weights = self.temporal_weights.copy()
        raise StopIteration
