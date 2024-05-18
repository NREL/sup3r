"""
Sup3r batch_handling module.
@author: bbenton
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
from scipy.ndimage import gaussian_filter

from sup3r.containers import (
    BatchQueueWithValidation,
    Container,
    DataCentricSampler,
    Sampler,
)
from sup3r.utilities.utilities import (
    nn_fill_array,
    nsrdb_reduce_daily_data,
    spatial_coarsening,
    uniform_box_sampler,
    uniform_time_sampler,
    weighted_box_sampler,
    weighted_time_sampler,
)

np.random.seed(42)

logger = logging.getLogger(__name__)


class BatchHandler(BatchQueueWithValidation):
    """BatchHandler object built from two lists of class:`Container` objects,
    one with training data and one with validation data. These lists will be
    used to initialize lists of class:`Sampler` objects that will then be used
    to build batches at run time.

    Notes
    -----
    These lists of containers can contain data from the same underlying data
    source (e.g. CONUS WTK) (by using `CroppedSampler(...,
    crop_slice=crop_slice)` with `crop_slice` selecting different time periods
    to prevent cross-contamination), or they can be used to sample from
    completely different data sources (e.g. train on CONUS WTK while validating
    on Canada WTK)."""

    SAMPLER = Sampler

    def __init__(
        self,
        train_containers: List[Container],
        val_containers: List[Container],
        batch_size,
        n_batches,
        s_enhance,
        t_enhance,
        means: Union[Dict, str],
        stds: Union[Dict, str],
        sample_shape,
        feature_sets,
        queue_cap: Optional[int] = None,
        max_workers: Optional[int] = None,
        coarsen_kwargs: Optional[Dict] = None,
        default_device: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        train_containers : List[Container]
            List of Container instances containing training data
        val_containers : List[Container]
            List of Container instances containing validation data
        batch_size : int
            Number of observations / samples in a batch
        n_batches : int
            Number of batches in an epoch, this sets the iteration limit for
            this object.
        s_enhance : int
            Integer factor by which the spatial axes is to be enhanced.
        t_enhance : int
            Integer factor by which the temporal axes is to be enhanced.
        means : Union[Dict, str]
            Either a .json path containing a dictionary or a dictionary of
            means which will be used to normalize batches as they are built.
        stds : Union[Dict, str]
            Either a .json path containing a dictionary or a dictionary of
            standard deviations which will be used to normalize batches as they
            are built.
        sample_shape : tuple
            Shape of samples to select from containers to build batches.
            Batches will be of shape (batch_size, *sample_shape, len(features))
        feature_sets : dict
            Dictionary of feature sets. This must include a 'features' entry
            and optionally can include 'lr_only_features' and/or
            'hr_only_features'

            The allowed keys are:
                lr_only_features : list | tuple
                    List of feature names or patt*erns that should only be
                    included in the low-res training set and not the high-res
                    observations.
                hr_exo_features : list | tuple
                    List of feature names or patt*erns that should be included
                    in the high-resolution observation but not expected to be
                    output from the generative model. An example is high-res
                    topography that is to be injected mid-network.
        queue_cap : int
            Maximum number of batches the batch queue can store.
        max_workers : int
            Number of workers / threads to use for getting samples used to
            build batches. This goes into a call to data.map(...,
            num_parallel_calls=max_workers) before prefetching samples from the
            tensorflow dataset generator.
        coarsen_kwargs : Union[Dict, None]
            Dictionary of kwargs to be passed to `self.coarsen`.
        default_device : str
            Default device to use for batch queue (e.g. /cpu:0, /gpu:0). If
            None this will use the first GPU if GPUs are available otherwise
            the CPU.
        """
        train_samplers = [
            self.SAMPLER(c, sample_shape, feature_sets)
            for c in train_containers
        ]
        val_samplers = [
            self.SAMPLER(c, sample_shape, feature_sets) for c in val_containers
        ]
        super().__init__(
            train_samplers,
            val_samplers,
            batch_size=batch_size,
            n_batches=n_batches,
            s_enhance=s_enhance,
            t_enhance=t_enhance,
            means=means,
            stds=stds,
            queue_cap=queue_cap,
            max_workers=max_workers,
            coarsen_kwargs=coarsen_kwargs,
            default_device=default_device,
        )


class BatchHandlerCC(BatchHandler):
    """Batch handling class for climate change data with daily averages as the
    coarse dataset."""

    def __init__(self, *args, sub_daily_shape=None, **kwargs):
        """
        Parameters
        ----------
        *args : list
            Same positional args as BatchHandler
        sub_daily_shape : int
            Number of hours to use in the high res sample output. This is the
            shape of the temporal dimension of the high res batch observation.
            This time window will be sampled for the daylight hours on the
            middle day of the data handler observation.
        **kwargs : dict
            Same keyword args as BatchHandler
        """
        super().__init__(*args, **kwargs)
        self.sub_daily_shape = sub_daily_shape

    def __next__(self):
        """Get the next iterator output.

        Returns
        -------
        batch : Batch
            Batch object with batch.low_res and batch.high_res attributes
            with the appropriate coarsening.
        """
        self.current_batch_indices = []

        if self._i >= self.n_batches:
            raise StopIteration

        handler = self.get_random_container()

        low_res = None
        high_res = None

        for i in range(self.batch_size):
            obs_hourly, obs_daily_avg = handler.get_next()
            self.current_batch_indices.append(handler.current_obs_index)

            obs_hourly = obs_hourly[..., self.hr_features_ind]

            if low_res is None:
                lr_shape = (self.batch_size, *obs_daily_avg.shape)
                hr_shape = (self.batch_size, *obs_hourly.shape)
                low_res = np.zeros(lr_shape, dtype=np.float32)
                high_res = np.zeros(hr_shape, dtype=np.float32)

            low_res[i] = obs_daily_avg
            high_res[i] = obs_hourly

        high_res = self.reduce_high_res_sub_daily(high_res)
        low_res = spatial_coarsening(low_res, self.s_enhance)

        if (
            self.hr_out_features is not None
            and 'clearsky_ratio' in self.hr_out_features
        ):
            i_cs = self.hr_out_features.index('clearsky_ratio')
            if np.isnan(high_res[..., i_cs]).any():
                high_res[..., i_cs] = nn_fill_array(high_res[..., i_cs])

        if self.smoothing is not None:
            feat_iter = [
                j
                for j in range(low_res.shape[-1])
                if self.features[j] not in self.smoothing_ignore
            ]
            for i in range(low_res.shape[0]):
                for j in feat_iter:
                    low_res[i, ..., j] = gaussian_filter(
                        low_res[i, ..., j], self.smoothing, mode='nearest'
                    )

        batch = self.BATCH_CLASS(low_res, high_res)

        self._i += 1
        return batch

    def reduce_high_res_sub_daily(self, high_res):
        """Take an hourly high-res observation and reduce the temporal axis
        down to the self.sub_daily_shape using only daylight hours on the
        center day.

        Parameters
        ----------
        high_res : np.ndarray
            5D array with dimensions (n_obs, spatial_1, spatial_2, temporal,
            n_features) where temporal >= 24 (set by the data handler).

        Returns
        -------
        high_res : np.ndarray
            5D array with dimensions (n_obs, spatial_1, spatial_2, temporal,
            n_features) where temporal has been reduced down to the integer
            self.sub_daily_shape. For example if the input temporal shape is 72
            (3 days) and sub_daily_shape=9, the center daylight 9 hours from
            the second day will be returned in the output array.
        """

        if self.sub_daily_shape is not None:
            n_days = int(high_res.shape[3] / 24)
            if n_days > 1:
                ind = np.arange(high_res.shape[3])
                day_slices = np.array_split(ind, n_days)
                day_slices = [slice(x[0], x[-1] + 1) for x in day_slices]
                assert n_days % 2 == 1, 'Need odd days'
                i_mid = int((n_days - 1) / 2)
                high_res = high_res[:, :, :, day_slices[i_mid], :]

            high_res = nsrdb_reduce_daily_data(high_res, self.sub_daily_shape)

        return high_res


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
            h = self.containers[h_idx]
            for _ in range(self.batch_size):
                spatial_slice = uniform_box_sampler(
                    h.data, self.sample_shape[:2]
                )
                weights = np.zeros(self.N_TIME_BINS)
                weights[t] = 1
                time_slice = weighted_time_sampler(
                    h.data, self.sample_shape[2], weights
                )
                tuple_index = (
                    *spatial_slice,
                    time_slice,
                    np.arange(h.data.shape[-1]),
                )
                val_indices[t].append(
                    {'handler_index': h_idx, 'tuple_index': tuple_index}
                )
        for s in range(self.N_SPACE_BINS):
            val_indices[s + self.N_TIME_BINS] = []
            h_idx = self.get_handler_index()
            h = self.containers[h_idx]
            for _ in range(self.batch_size):
                weights = np.zeros(self.N_SPACE_BINS)
                weights[s] = 1
                spatial_slice = weighted_box_sampler(
                    h.data, self.sample_shape[:2], weights
                )
                time_slice = uniform_time_sampler(h.data, self.sample_shape[2])
                tuple_index = (
                    *spatial_slice,
                    time_slice,
                    np.arange(h.data.shape[-1]),
                )
                val_indices[s + self.N_TIME_BINS].append(
                    {'handler_index': h_idx, 'tuple_index': tuple_index}
                )
        return val_indices

    def __next__(self):
        if self._i < len(self.val_indices.keys()):
            high_res = np.zeros(
                (
                    self.batch_size,
                    self.sample_shape[0],
                    self.sample_shape[1],
                    self.sample_shape[2],
                    self.containers[0].shape[-1],
                ),
                dtype=np.float32,
            )
            val_indices = self.val_indices[self._i]
            for i, idx in enumerate(val_indices):
                high_res[i, ...] = self.containers[idx['handler_index']].data[
                    idx['tuple_index']
                ]

            batch = self.coarsen(
                high_res,
                temporal_coarsening_method=self.temporal_coarsening_method,
                smoothing=self.smoothing,
                smoothing_ignore=self.smoothing_ignore,
            )
            self._i += 1
            return batch
        raise StopIteration


class ValidationDataTemporalDC(ValidationDataDC):
    """Iterator for data-centric temporal validation data"""

    N_SPACE_BINS = 0


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
