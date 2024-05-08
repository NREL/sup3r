"""
Sup3r batch_handling module.
@author: bbenton
"""
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime as dt

import numpy as np
from rex.utilities import log_mem
from scipy.ndimage import gaussian_filter

from sup3r.containers.batchers.abstract import AbstractBatchBuilder, Batch
from sup3r.preprocessing.mixin import MultiHandlerMixIn
from sup3r.utilities.utilities import (
    nn_fill_array,
    nsrdb_reduce_daily_data,
    spatial_coarsening,
    uniform_box_sampler,
    uniform_time_sampler,
)

np.random.seed(42)

logger = logging.getLogger(__name__)


class ValidationData(AbstractBatchBuilder):
    """Iterator for validation data"""

    # Classes to use for handling an individual batch obj.
    BATCH_CLASS = Batch

    def __init__(self,
                 data_handlers,
                 batch_size=8,
                 s_enhance=1,
                 t_enhance=1,
                 temporal_coarsening_method='subsample',
                 hr_features_ind=None,
                 smoothing=None,
                 smoothing_ignore=None):
        """
        Parameters
        ----------
        data_handlers : list[DataHandler]
            List of DataHandler instances
        batch_size : int
            Size of validation data batches
        s_enhance : int
            Factor by which to coarsen spatial dimensions of the high
            resolution data
        t_enhance : int
            Factor by which to coarsen temporal dimension of the high
            resolution data
        temporal_coarsening_method : str
            [subsample, average, total, min, max]
            Subsample will take every t_enhance-th time step, average will
            average over t_enhance time steps, total will sum over t_enhance
            time steps
        hr_features_ind : list | np.ndarray | None
            List/array of feature channel indices that are used for generative
            output, without any feature indices used only for training.
        smoothing : float | None
            Standard deviation to use for gaussian filtering of the coarse
            data. This can be tuned by matching the kinetic energy of a low
            resolution simulation with the kinetic energy of a coarsened and
            smoothed high resolution simulation. If None no smoothing is
            performed.
        smoothing_ignore : list | None
            List of features to ignore for the smoothing filter. None will
            smooth all features if smoothing kwarg is not None
        """

        handler_shapes = np.array([d.sample_shape for d in data_handlers])
        assert np.all(handler_shapes[0] == handler_shapes)

        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        self.data_handlers = data_handlers
        self.batch_size = batch_size
        self.sample_shape = handler_shapes[0]
        self.val_indices = self._get_val_indices()
        self.max = np.ceil(len(self.val_indices) / (batch_size))
        self._remaining_observations = len(self.val_indices)
        self.temporal_coarsening_method = temporal_coarsening_method
        self.hr_features_ind = hr_features_ind
        self.smoothing = smoothing
        self.smoothing_ignore = smoothing_ignore
        self.current_batch_indices = []

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
            if h.val_data is not None:
                for _ in range(h.val_data.shape[2]):
                    spatial_slice = uniform_box_sampler(
                        h.val_data.shape, self.sample_shape[:2])
                    temporal_slice = uniform_time_sampler(
                        h.val_data.shape, self.sample_shape[2])
                    tuple_index = (
                        *spatial_slice, temporal_slice,
                        np.arange(h.val_data.shape[-1]),
                    )
                    val_indices.append({
                        'handler_index': i,
                        'tuple_index': tuple_index
                    })
        return val_indices

    def any(self):
        """Return True if any validation data exists"""
        return any(self.val_indices)

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
        for h in self.data_handlers:
            time_steps += h.val_data.shape[2]
        return (self.data_handlers[0].val_data.shape[0],
                self.data_handlers[0].val_data.shape[1], time_steps,
                self.data_handlers[0].val_data.shape[3])

    def __iter__(self):
        self._i = 0
        self._remaining_observations = len(self.val_indices)
        return self

    def __len__(self):
        """
        Returns
        -------
        len : int
            Number of total batches
        """
        return int(self.max)

    def batch_next(self, high_res):
        """Assemble the next batch

        Parameters
        ----------
        high_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        batch : Batch
        """
        return self.BATCH_CLASS.get_coarse_batch(
            high_res,
            self.s_enhance,
            t_enhance=self.t_enhance,
            temporal_coarsening_method=self.temporal_coarsening_method,
            hr_features_ind=self.hr_features_ind,
            smoothing=self.smoothing,
            smoothing_ignore=self.smoothing_ignore)

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
            n_obs = self._remaining_observations
            if self._remaining_observations > self.batch_size:
                n_obs = self.batch_size
            hr_list = []
            for i in range(n_obs):
                val_idx = self.val_indices[self._i + i]
                h_idx = val_idx['handler_index']
                tuple_idx = val_idx['tuple_index']
                hr_sample = self.data_handlers[h_idx].val_data[tuple_idx]
                hr_list.append(np.expand_dims(hr_sample, axis=0))
                self._remaining_observations -= 1
                self.current_batch_indices.append(h_idx)
            high_res = np.concatenate(hr_list, axis=0)
            if self.sample_shape[2] == 1:
                high_res = high_res[..., 0, :]
            batch = self.batch_next(high_res)
            self._i += 1
            return batch
        else:
            raise StopIteration


class BatchHandler(MultiHandlerMixIn, AbstractBatchBuilder):
    """Sup3r base batch handling class"""

    # Classes to use for handling an individual batch obj.
    VAL_CLASS = ValidationData
    BATCH_CLASS = Batch
    DATA_HANDLER_CLASS = None

    def __init__(self,
                 data_handlers,
                 batch_size=8,
                 s_enhance=1,
                 t_enhance=1,
                 means=None,
                 stds=None,
                 norm=True,
                 n_batches=10,
                 temporal_coarsening_method='subsample',
                 stdevs_file=None,
                 means_file=None,
                 overwrite_stats=False,
                 smoothing=None,
                 smoothing_ignore=None,
                 worker_kwargs=None):
        """
        Parameters
        ----------
        data_handlers : list[DataHandler]
            List of DataHandler instances
        batch_size : int
            Number of observations in a batch
        s_enhance : int
            Factor by which to coarsen spatial dimensions of the high
            resolution data to generate low res data
        t_enhance : int
            Factor by which to coarsen temporal dimension of the high
            resolution data to generate low res data
        means : dict | none
            Dictionary of means for all features with keys: feature names and
            values: mean values. if None, this will be calculated. if norm is
            true these will be used for data normalization
        stds : dict | none
            dictionary of standard deviation values for all features with keys:
            feature names and values: standard deviations. if None, this will
            be calculated. if norm is true these will be used for data
            normalization
        norm : bool
            Whether to normalize the data or not
        n_batches : int
            Number of batches in an epoch, this sets the iteration limit for
            this object.
        temporal_coarsening_method : str
            [subsample, average, total, min, max]
            Subsample will take every t_enhance-th time step, average will
            average over t_enhance time steps, total will sum over t_enhance
            time steps
        stdevs_file : str | None
            Optional .json path to stdevs data or where to save data after
            calling get_stats
        means_file : str | None
            Optional .json path to means data or where to save data after
            calling get_stats
        overwrite_stats : bool
            Whether to overwrite stats cache files.
        smoothing : float | None
            Standard deviation to use for gaussian filtering of the coarse
            data. This can be tuned by matching the kinetic energy of a low
            resolution simulation with the kinetic energy of a coarsened and
            smoothed high resolution simulation. If None no smoothing is
            performed.
        smoothing_ignore : list | None
            List of features to ignore for the smoothing filter. None will
            smooth all features if smoothing kwarg is not None
        worker_kwargs : dict | None
            Dictionary of worker values. Can include max_workers,
            norm_workers, stats_workers, and load_workers. Each argument needs
            to be an integer or None.

            Providing a value for max workers will be used to set the value of
            all other worker arguments. If max_workers == 1 then all processes
            will be serialized. If None then other workers arguments will use
            their own provided values.

            `load_workers` is the max number of workers to use for loading
            data handlers. `norm_workers` is the max number of workers to use
            for normalizing data handlers. `stats_workers` is the max number
            of workers to use for computing stats across data handlers.
        """
        worker_kwargs = worker_kwargs or {}
        max_workers = worker_kwargs.get('max_workers', None)
        norm_workers = stats_workers = load_workers = None
        if max_workers is not None:
            norm_workers = stats_workers = load_workers = max_workers
        self.stats_workers = worker_kwargs.get('stats_workers', stats_workers)
        self.norm_workers = worker_kwargs.get('norm_workers', norm_workers)
        self.load_workers = worker_kwargs.get('load_workers', load_workers)

        data_handlers = (data_handlers
                         if isinstance(data_handlers, (list, tuple))
                         else [data_handlers])
        msg = 'All data handlers must have the same sample_shape'
        handler_shapes = np.array([d.sample_shape for d in data_handlers])
        assert np.all(handler_shapes[0] == handler_shapes), msg

        self.data_handlers = data_handlers
        self._i = 0
        self.low_res = None
        self.high_res = None
        self.batch_size = batch_size
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        self.sample_shape = handler_shapes[0]
        self.means = means
        self.stds = stds
        self.n_batches = n_batches
        self.temporal_coarsening_method = temporal_coarsening_method
        self.current_batch_indices = None
        self.handler_index = self.get_handler_index()
        self.stdevs_file = stdevs_file
        self.means_file = means_file
        self.overwrite_stats = overwrite_stats
        self.smoothing = smoothing
        self.smoothing_ignore = smoothing_ignore or []
        self.smoothed_features = [
            f for f in self.features if f not in self.smoothing_ignore
        ]
        FeatureSets.__init__(self, data_handlers)

        logger.info(f'Initializing BatchHandler with '
                    f'{len(self.data_handlers)} data handlers with handler '
                    f'weights={self.handler_weights}, smoothing={smoothing}. '
                    f'Using stats_workers={self.stats_workers}, '
                    f'norm_workers={self.norm_workers}, '
                    f'load_workers={self.load_workers}.')

        now = dt.now()
        self.load_handler_data()
        logger.debug(f'Finished loading data of shape {self.shape} '
                     f'for BatchHandler in {dt.now() - now}.')
        log_mem(logger, log_level='INFO')

        if norm:
            self.means, self.stds = self.check_cached_stats()
            self.normalize(self.means, self.stds)

        logger.debug('Getting validation data for BatchHandler.')
        self.val_data = self.VAL_CLASS(
            data_handlers,
            batch_size=batch_size,
            s_enhance=s_enhance,
            t_enhance=t_enhance,
            temporal_coarsening_method=temporal_coarsening_method,
            hr_features_ind=self.hr_features_ind,
            smoothing=self.smoothing,
            smoothing_ignore=self.smoothing_ignore,
        )

        logger.info('Finished initializing BatchHandler.')
        log_mem(logger, log_level='INFO')

    @property
    def shape(self):
        """Shape of full dataset across all handlers

        Returns
        -------
        shape : tuple
            (spatial_1, spatial_2, temporal, features)
            With spatiotemporal extent equal to the sum across all data handler
            dimensions
        """
        time_steps = np.sum([h.shape[-2] for h in self.data_handlers])
        n_lons = self.data_handlers[0].shape[1]
        n_lats = self.data_handlers[0].shape[0]
        return (n_lats, n_lons, time_steps, self.data_handlers[0].shape[-1])

    def _parallel_normalization(self):
        """Normalize data in all data handlers in parallel or serial depending
        on norm_workers."""
        logger.info(f'Normalizing {len(self.data_handlers)} data handlers.')
        max_workers = self.norm_workers
        if max_workers == 1:
            for dh in self.data_handlers:
                dh.normalize(self.means, self.stds,
                             max_workers=dh.norm_workers)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                futures = {}
                now = dt.now()
                for idh, dh in enumerate(self.data_handlers):
                    future = exe.submit(dh.normalize, self.means, self.stds,
                                        max_workers=1)
                    futures[future] = idh

                logger.info(f'Started normalizing {len(self.data_handlers)} '
                            f'data handlers in {dt.now() - now}.')

                for i, _ in enumerate(as_completed(futures)):
                    try:
                        future.result()
                    except Exception as e:
                        msg = ('Error normalizing data handler number '
                               f'{futures[future]}')
                        logger.exception(msg)
                        raise RuntimeError(msg) from e
                    logger.debug(f'{i + 1} out of {len(futures)} data handlers'
                                 ' normalized.')

    def load_handler_data(self):
        """Load data handler data in parallel or serial"""
        logger.info(f'Loading {len(self.data_handlers)} data handlers')
        max_workers = self.load_workers
        if max_workers == 1:
            for d in self.data_handlers:
                if d.data is None:
                    d.load_cached_data()
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                futures = {}
                now = dt.now()
                for i, d in enumerate(self.data_handlers):
                    if d.data is None:
                        future = exe.submit(d.load_cached_data)
                        futures[future] = i

                logger.info(f'Started loading all {len(self.data_handlers)} '
                            f'data handlers in {dt.now() - now}.')

                for i, future in enumerate(as_completed(futures)):
                    try:
                        future.result()
                    except Exception as e:
                        msg = ('Error loading data handler number '
                               f'{futures[future]}')
                        logger.exception(msg)
                        raise RuntimeError(msg) from e
                    logger.debug(f'{i + 1} out of {len(futures)} handlers '
                                 'loaded.')

    def _get_stats(self):
        """Get standard deviations and means for training features in
        parallel."""
        logger.info(f'Calculating stats for {len(self.features)} '
                    'features.')
        for feature in self.features:
            logger.debug(f'Calculating mean/stdev for "{feature}"')
            self.means[feature] = np.float32(0)
            self.stds[feature] = np.float32(0)
            max_workers = self.stats_workers

            if max_workers is None or max_workers >= 1:
                with ThreadPoolExecutor(max_workers=max_workers) as exe:
                    futures = {}
                    for idh, dh in enumerate(self.data_handlers):
                        future = exe.submit(dh._get_stats)
                        futures[future] = idh

                    for i, future in enumerate(as_completed(futures)):
                        _ = future.result()
                        logger.debug(
                            f'{i + 1} out of {len(self.data_handlers)} '
                            'means calculated.')

            self.means[feature] = self._get_feature_means(feature)
            self.stds[feature] = self._get_feature_stdev(feature)

    def __len__(self):
        """Use user input of n_batches to specify length

        Returns
        -------
        self.n_batches : int
            Number of batches possible to iterate over
        """
        return self.n_batches

    def check_cached_stats(self):
        """Get standard deviations and means for all data features from cache
        files if available.

        Returns
        -------
        means : dict | none
            Dictionary of means for all features with keys: feature names and
            values: mean values. if None, this will be calculated. if norm is
            true these will be used for data normalization
        stds : dict | none
            dictionary of standard deviation values for all features with keys:
            feature names and values: standard deviations. if None, this will
            be calculated. if norm is true these will be used for data
            normalization
        """
        stdevs_check = (self.stdevs_file is not None
                        and not self.overwrite_stats)
        stdevs_check = stdevs_check and os.path.exists(self.stdevs_file)
        means_check = self.means_file is not None and not self.overwrite_stats
        means_check = means_check and os.path.exists(self.means_file)
        if stdevs_check and means_check:
            logger.info(f'Loading stdevs from {self.stdevs_file}')
            with open(self.stdevs_file) as fh:
                self.stds = json.load(fh)
            logger.info(f'Loading means from {self.means_file}')
            with open(self.means_file) as fh:
                self.means = json.load(fh)

            msg = ('The training features and cached statistics are '
                   'incompatible. Number of training features is '
                   f'{len(self.features)} and number of stats is'
                   f' {len(self.stds)}')
            check = len(self.means) == len(self.features)
            check = check and (len(self.stds) == len(self.features))
            assert check, msg
        return self.means, self.stds

    def cache_stats(self):
        """Saved stdevs and means to cache files if files are not None"""

        iter = ((self.means_file, self.means), (self.stdevs_file, self.stds))
        for fp, data in iter:
            if fp is not None:
                logger.info(f'Saving stats to {fp}')
                os.makedirs(os.path.dirname(fp), exist_ok=True)
                with open(fp, 'w') as fh:
                    # need to convert numpy float32 type to python float to be
                    # serializable in json
                    json.dump({k: float(v) for k, v in data.items()}, fh)

    def get_stats(self):
        """Get standard deviations and means for all data features"""

        self.means = {}
        self.stds = {}

        now = dt.now()
        logger.info('Calculating stdevs/means.')
        self._get_stats()
        logger.info(f'Finished calculating stats in {dt.now() - now}.')
        self.cache_stats()

    def _get_feature_means(self, feature):
        """Get mean for requested feature

        Parameters
        ----------
        feature : str
            Feature to get mean for
        """
        logger.debug(f'Calculating multi-handler mean for {feature}')
        for idh, dh in enumerate(self.data_handlers):
            self.means[feature] += (self.handler_weights[idh]
                                    * dh.means[feature])

        return self.means[feature]

    def _get_feature_stdev(self, feature):
        """Get stdev for requested feature

        NOTE: We compute the variance across all handlers as a pooled variance
        of the variances for each handler. We also assume that the number of
        samples in each handler is much greater than 1, so N - 1 ~ N.

        Parameters
        ----------
        feature : str
            Feature to get stdev for
        """

        logger.debug(f'Calculating multi-handler stdev for {feature}')
        for idh, dh in enumerate(self.data_handlers):
            variance = dh.stds[feature]**2
            self.stds[feature] += (variance * self.handler_weights[idh])

        self.stds[feature] = np.sqrt(self.stds[feature]).astype(np.float32)

        return self.stds[feature]

    def normalize(self, means=None, stds=None):
        """Compute means and stds for each feature across all datasets and
        normalize each data handler dataset.  Checks if input means and stds
        are different from stored means and stds and renormalizes if they are

        Parameters
        ----------
        means : dict | none
            Dictionary of means for all features with keys: feature names and
            values: mean values. if None, this will be calculated. if norm is
            true these will be used for data normalization
        stds : dict | none
            dictionary of standard deviation values for all features with keys:
            feature names and values: standard deviations. if None, this will
            be calculated. if norm is true these will be used for data
            normalization
        features : list | None
            Optional list of features used to index data array during
            normalization. If this is None self.features will be used.
        """
        if means is None or stds is None:
            self.get_stats()
        elif means is not None and stds is not None:
            means0, means1 = list(self.means.values()), list(means.values())
            stds0, stds1 = list(self.stds.values()), list(stds.values())
            if (not np.array_equal(means0, means1)
                    or not np.array_equal(stds0, stds1)):
                msg = (f'Normalization requested with new means/stdevs '
                       f'{means1}/{stds1} that '
                       f'dont match previous values: {means0}/{stds0}')
                logger.info(msg)
                raise ValueError(msg)
            else:
                self.means = means
                self.stds = stds

        now = dt.now()
        logger.info('Normalizing data in each data handler.')
        self._parallel_normalization()
        logger.info('Finished normalizing data in all data handlers in '
                    f'{dt.now() - now}.')

    def __iter__(self):
        self._i = 0
        return self

    def batch_next(self, high_res):
        """Assemble the next batch

        Parameters
        ----------
        high_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        batch : Batch
        """
        return self.BATCH_CLASS.get_coarse_batch(
                high_res=high_res,
                s_enhance=self.s_enhance,
                t_enhance=self.t_enhance,
                temporal_coarsening_method=self.temporal_coarsening_method,
                hr_features_ind=self.hr_features_ind,
                features=self.features,
                smoothing=self.smoothing,
                smoothing_ignore=self.smoothing_ignore)

    def __next__(self):
        """Get the next iterator output.

        Returns
        -------
        batch : Batch
            Batch object with batch.low_res and batch.high_res attributes
            with the appropriate coarsening.
        """
        self.current_batch_indices = []
        if self._i < self.n_batches:
            handler = self.get_rand_handler()
            hr_list = []
            for _ in range(self.batch_size):
                hr_sample = handler.get_next()
                hr_list.append(np.expand_dims(hr_sample, axis=0))
                self.current_batch_indices.append(handler.current_obs_index)

            high_res = np.concatenate(hr_list, axis=0)
            batch = self.batch_next(high_res)

            self._i += 1
            return batch
        else:
            raise StopIteration


class BatchHandlerCC(BatchHandler):
    """Batch handling class for climate change data with daily averages as the
    coarse dataset."""

    # Classes to use for handling an individual batch obj.
    VAL_CLASS = ValidationData
    BATCH_CLASS = Batch

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

        handler = self.get_rand_handler()

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

        if (self.hr_out_features is not None
                and 'clearsky_ratio' in self.hr_out_features):
            i_cs = self.hr_out_features.index('clearsky_ratio')
            if np.isnan(high_res[..., i_cs]).any():
                high_res[..., i_cs] = nn_fill_array(high_res[..., i_cs])

        if self.smoothing is not None:
            feat_iter = [
                j for j in range(low_res.shape[-1])
                if self.features[j] not in self.smoothing_ignore
            ]
            for i in range(low_res.shape[0]):
                for j in feat_iter:
                    low_res[i, ..., j] = gaussian_filter(low_res[i, ..., j],
                                                         self.smoothing,
                                                         mode='nearest')

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


class SpatialBatchHandlerCC(BatchHandler):
    """Batch handling class for climate change data with daily averages as the
    coarse dataset with only spatial samples, e.g. the batch tensor shape is
    (n_obs, spatial_1, spatial_2, features)
    """

    # Classes to use for handling an individual batch obj.
    VAL_CLASS = ValidationData
    BATCH_CLASS = Batch

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

        handler = self.get_rand_handler()

        high_res = None

        for i in range(self.batch_size):
            _, obs_daily_avg = handler.get_next()
            self.current_batch_indices.append(handler.current_obs_index)

            if high_res is None:
                hr_shape = (self.batch_size, *obs_daily_avg.shape)
                high_res = np.zeros(hr_shape, dtype=np.float32)

                msg = ('SpatialBatchHandlerCC can only use n_temporal==1 '
                       'but received HR shape {} with n_temporal={}.'.format(
                           hr_shape, hr_shape[3]))
                assert hr_shape[3] == 1, msg

            high_res[i] = obs_daily_avg

        low_res = spatial_coarsening(high_res, self.s_enhance)
        low_res = low_res[:, :, :, 0, :]
        high_res = high_res[:, :, :, 0, :]

        high_res = high_res[..., self.hr_features_ind]

        if (self.hr_out_features is not None
                and 'clearsky_ratio' in self.hr_out_features):
            i_cs = self.hr_out_features.index('clearsky_ratio')
            if np.isnan(high_res[..., i_cs]).any():
                high_res[..., i_cs] = nn_fill_array(high_res[..., i_cs])

        if self.smoothing is not None:
            feat_iter = [
                j for j in range(low_res.shape[-1])
                if self.features[j] not in self.smoothing_ignore
            ]
            for i in range(low_res.shape[0]):
                for j in feat_iter:
                    low_res[i, ..., j] = gaussian_filter(low_res[i, ..., j],
                                                         self.smoothing,
                                                         mode='nearest')

        batch = self.BATCH_CLASS(low_res, high_res)

        self._i += 1
        return batch


class SpatialBatchHandler(BatchHandler):
    """Sup3r spatial batch handling class"""

    def batch_next(self, high_res):
        """Assemble the next batch

        Parameters
        ----------
        high_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        batch : Batch
        """
        return self.BATCH_CLASS.get_coarse_batch(
                high_res,
                self.s_enhance,
                hr_features_ind=self.hr_features_ind,
                features=self.features,
                smoothing=self.smoothing,
                smoothing_ignore=self.smoothing_ignore)

    def __next__(self):
        if self._i < len(self):
            handler = self.get_rand_handler()

            hr_list = []
            for _ in range(self.batch_size):
                hr_sample = handler.get_next()[..., 0, :]
                hr_list.append(np.expand_dims(hr_sample, axis=0))
            high_res = np.concatenate(hr_list, axis=0)
            batch = self.batch_next(high_res)

            self._i += 1
            return batch
        else:
            raise StopIteration
