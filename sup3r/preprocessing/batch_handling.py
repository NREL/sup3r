# -*- coding: utf-8 -*-
"""
Sup3r batch_handling module.
@author: bbenton
"""
import logging
import numpy as np
from datetime import datetime as dt
import os
import pickle
from scipy.ndimage.filters import gaussian_filter

from concurrent.futures import (as_completed, ThreadPoolExecutor)

from rex.utilities import log_mem

from sup3r.utilities.utilities import (estimate_max_workers,
                                       weighted_time_sampler,
                                       weighted_box_sampler,
                                       spatial_coarsening,
                                       temporal_coarsening,
                                       smooth_data,
                                       nsrdb_reduce_daily_data,
                                       uniform_box_sampler,
                                       uniform_time_sampler,
                                       nn_fill_array)
from sup3r.preprocessing.data_handling import DataHandlerDCforH5

np.random.seed(42)

logger = logging.getLogger(__name__)


class Batch:
    """Batch of low_res and high_res data"""

    def __init__(self, low_res, high_res):
        """Stores low and high res data

        Parameters
        ----------
        low_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        high_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        """
        self._low_res = low_res
        self._high_res = high_res

    def __len__(self):
        """Get the number of observations in this batch."""
        return len(self._low_res)

    @property
    def shape(self):
        """Get the (low_res_shape, high_res_shape) shapes."""
        return (self._low_res.shape, self._high_res.shape)

    @property
    def low_res(self):
        """Get the low-resolution data for the batch."""
        return self._low_res

    @property
    def high_res(self):
        """Get the high-resolution data for the batch."""
        return self._high_res

    @staticmethod
    def reduce_features(high_res, output_features_ind=None):
        """Remove any feature channels that are only intended for the low-res
        training input.

        Parameters
        ----------
        high_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        output_features_ind : list | np.ndarray | None
            List/array of feature channel indices that are used for generative
            output, without any feature indices used only for training.
        """
        if output_features_ind is None:
            return high_res
        else:
            return high_res[..., output_features_ind]

    # pylint: disable=W0613
    @classmethod
    def get_coarse_batch(cls, high_res,
                         s_enhance, t_enhance=1,
                         temporal_coarsening_method='subsample',
                         output_features_ind=None,
                         output_features=None,
                         training_features=None,
                         smoothing=None,
                         smoothing_ignore=None):
        """Coarsen high res data and return Batch with high res and
        low res data

        Parameters
        ----------
        high_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        s_enhance : int
            Factor by which to coarsen spatial dimensions of the high
            resolution data
        t_enhance : int
            Factor by which to coarsen temporal dimension of the high
            resolution data
        temporal_coarsening_method : str
            Method to use for temporal coarsening. Can be subsample, average,
            min, max, or total
        output_features_ind : list | np.ndarray | None
            List/array of feature channel indices that are used for generative
            output, without any feature indices used only for training.
        output_features : list
            List of Generative model output feature names
        training_features : list | None
            Ordered list of training features input to the generative model
        smoothing : float | None
            Standard deviation to use for gaussian filtering of the coarse
            data. This can be tuned by matching the kinetic energy of a low
            resolution simulation with the kinetic energy of a coarsened and
            smoothed high resolution simulation. If None no smoothing is
            performed.
        smoothing_ignore : list | None
            List of features to ignore for the smoothing filter. None will
            smooth all features if smoothing kwarg is not None

        Returns
        -------
        Batch
            Batch instance with low and high res data
        """
        low_res = spatial_coarsening(high_res, s_enhance)

        if training_features is None:
            training_features = [None] * low_res.shape[-1]

        if smoothing_ignore is None:
            smoothing_ignore = []

        if t_enhance != 1:
            low_res = temporal_coarsening(low_res, t_enhance,
                                          temporal_coarsening_method)

        low_res = smooth_data(low_res, training_features, smoothing_ignore,
                              smoothing)
        high_res = cls.reduce_features(high_res, output_features_ind)
        batch = cls(low_res, high_res)

        return batch


class ValidationData:
    """Iterator for validation data"""

    # Classes to use for handling an individual batch obj.
    BATCH_CLASS = Batch

    def __init__(self, data_handlers, batch_size=8, s_enhance=3, t_enhance=1,
                 temporal_coarsening_method='subsample',
                 output_features_ind=None,
                 output_features=None,
                 smoothing=None, smoothing_ignore=None):
        """
        Parameters
        ----------
        handlers : list[DataHandler]
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
        output_features_ind : list | np.ndarray | None
            List/array of feature channel indices that are used for generative
            output, without any feature indices used only for training.
        output_features : list
            List of Generative model output feature names
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

        self.handlers = data_handlers
        self.batch_size = batch_size
        self.sample_shape = handler_shapes[0]
        self.val_indices = self._get_val_indices()
        self.max = np.ceil(
            len(self.val_indices) / (batch_size))
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        self._remaining_observations = len(self.val_indices)
        self.temporal_coarsening_method = temporal_coarsening_method
        self._i = 0
        self.output_features_ind = output_features_ind
        self.output_features = output_features
        self.smoothing = smoothing
        self.smoothing_ignore = smoothing_ignore

    def _get_val_indices(self):
        """List of dicts to index each validation data observation across all
        handlers

        Returns
        -------
        val_indices : list[dict]
            List of dicts with handler_index and tuple_index. The tuple index
            is used to get validation data observation with
            data[tuple_index]"""

        val_indices = []
        for i, h in enumerate(self.handlers):
            if h.val_data is not None:
                for _ in range(h.val_data.shape[2]):
                    spatial_slice = uniform_box_sampler(h.val_data,
                                                        self.sample_shape[:2])
                    temporal_slice = uniform_time_sampler(h.val_data,
                                                          self.sample_shape[2])
                    tuple_index = tuple(spatial_slice + [temporal_slice]
                                        + [np.arange(h.val_data.shape[-1])])
                    val_indices.append({'handler_index': i,
                                        'tuple_index': tuple_index})
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
        for h in self.handlers:
            time_steps += h.val_data.shape[2]
        return (self.handlers[0].val_data.shape[0],
                self.handlers[0].val_data.shape[1],
                time_steps,
                self.handlers[0].val_data.shape[3])

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
            high_res, self.s_enhance,
            t_enhance=self.t_enhance,
            temporal_coarsening_method=self.temporal_coarsening_method,
            output_features_ind=self.output_features_ind,
            smoothing=self.smoothing,
            smoothing_ignore=self.smoothing_ignore,
            output_features=self.output_features)

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
                high_res = np.zeros((self.batch_size, self.sample_shape[0],
                                     self.sample_shape[1],
                                     self.sample_shape[2],
                                     self.handlers[0].shape[-1]),
                                    dtype=np.float32)
            else:
                high_res = np.zeros((self._remaining_observations,
                                     self.sample_shape[0],
                                     self.sample_shape[1],
                                     self.sample_shape[2],
                                     self.handlers[0].shape[-1]),
                                    dtype=np.float32)
            for i in range(high_res.shape[0]):
                val_index = self.val_indices[self._i + i]
                high_res[i, ...] = self.handlers[
                    val_index['handler_index']].val_data[
                    val_index['tuple_index']]
                self._remaining_observations -= 1

            if self.sample_shape[2] == 1:
                high_res = high_res[..., 0, :]
            batch = self.batch_next(high_res)
            self._i += 1
            return batch
        else:
            raise StopIteration


class BatchHandler:
    """Sup3r base batch handling class"""

    # Classes to use for handling an individual batch obj.
    VAL_CLASS = ValidationData
    BATCH_CLASS = Batch
    DATA_HANDLER_CLASS = None

    def __init__(self, data_handlers, batch_size=8, s_enhance=3, t_enhance=1,
                 means=None, stds=None, norm=True, n_batches=10,
                 temporal_coarsening_method='subsample', stdevs_file=None,
                 means_file=None, overwrite_stats=False, smoothing=None,
                 smoothing_ignore=None, worker_kwargs=None):
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
        means : np.ndarray
            dimensions (features)
            array of means for all features with same ordering as data
            features. If not None and norm is True these will be used for
            normalization
        stds : np.ndarray
            dimensions (features)
            array of means for all features with same ordering as data
            features.  If not None and norm is True these will be used form
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
            Path to stdevs data or where to save data after calling get_stats
        means_file : str | None
            Path to means data or where to save data after calling get_stats
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
        self._stats_workers = worker_kwargs.get('stats_workers', stats_workers)
        self._norm_workers = worker_kwargs.get('norm_workers', norm_workers)
        self._load_workers = worker_kwargs.get('load_workers', load_workers)

        msg = ('All data handlers must have the same sample_shape')
        handler_shapes = np.array([d.sample_shape for d in data_handlers])
        assert np.all(handler_shapes[0] == handler_shapes), msg

        self.data_handlers = data_handlers
        self._i = 0
        self.low_res = None
        self.high_res = None
        self.batch_size = batch_size
        self._val_data = None
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        self.sample_shape = handler_shapes[0]
        self.means = means
        self.stds = stds
        self.n_batches = n_batches
        self.temporal_coarsening_method = temporal_coarsening_method
        self.current_batch_indices = None
        self.current_handler_index = None
        self.stdevs_file = stdevs_file
        self.means_file = means_file
        self.overwrite_stats = overwrite_stats
        self.smoothing = smoothing
        self.smoothing_ignore = smoothing_ignore or []
        self.smoothed_features = [f for f in self.training_features
                                  if f not in self.smoothing_ignore]

        logger.info(f'Initializing BatchHandler with smoothing={smoothing}. '
                    f'Using stats_workers={self.stats_workers}, '
                    f'norm_workers={self.norm_workers}, '
                    f'load_workers={self.load_workers}.')

        now = dt.now()
        self.parallel_load()
        logger.debug(f'Finished loading data of shape {self.shape} '
                     f'for BatchHandler in {dt.now() - now}.')
        log_mem(logger, log_level='INFO')

        if norm:
            self.means, self.stds = self.check_cached_stats()
            self.normalize(self.means, self.stds)

        logger.debug('Getting validation data for BatchHandler.')
        self.val_data = self.VAL_CLASS(
            data_handlers, batch_size=batch_size,
            s_enhance=s_enhance, t_enhance=t_enhance,
            temporal_coarsening_method=temporal_coarsening_method,
            output_features_ind=self.output_features_ind,
            output_features=self.output_features,
            smoothing=self.smoothing,
            smoothing_ignore=self.smoothing_ignore)

        logger.info('Finished initializing BatchHandler.')
        log_mem(logger, log_level='INFO')

    @property
    def feature_mem(self):
        """Get memory used by each feature in data handlers"""
        return self.data_handlers[0].feature_mem

    @property
    def stats_workers(self):
        """Get max workers for calculating stats based on memory usage"""
        proc_mem = self.feature_mem
        stats_workers = estimate_max_workers(self._stats_workers, proc_mem,
                                             len(self.data_handlers))
        return stats_workers

    @property
    def load_workers(self):
        """Get max workers for loading data handler based on memory usage"""
        proc_mem = len(self.data_handlers[0].features) * self.feature_mem
        max_workers = estimate_max_workers(self._load_workers, proc_mem,
                                           len(self.data_handlers))
        return max_workers

    @property
    def norm_workers(self):
        """Get max workers used for calculating and normalization across
        features"""
        proc_mem = 2 * self.feature_mem
        norm_workers = estimate_max_workers(self._norm_workers, proc_mem,
                                            len(self.training_features))
        return norm_workers

    @property
    def training_features(self):
        """Get the ordered list of feature names held in this object's
        data handlers"""
        return self.data_handlers[0].features

    @property
    def output_features(self):
        """Get the ordered list of feature names held in this object's
        data handlers"""
        return self.data_handlers[0].output_features

    @property
    def output_features_ind(self):
        """Get the feature channel indices that should be used for the
        generated output features"""
        if self.training_features == self.output_features:
            return None
        else:
            out = [i for i, feature in enumerate(self.training_features)
                   if feature in self.output_features]
            return out

    @property
    def shape(self):
        """Shape of full dataset across all handlers

        Returns
        -------
        shape : tuple
            (spatial_1, spatial_2, temporal, features)
            With temporal extent equal to the sum across all data handlers time
            dimension
        """
        time_steps = np.sum([h.shape[-2] for h in self.data_handlers])
        return (self.data_handlers[0].shape[0], self.data_handlers[0].shape[1],
                time_steps, self.data_handlers[0].shape[-1])

    def parallel_normalization(self):
        """Normalize data in all data handlers in parallel."""
        logger.info(f'Normalizing {len(self.data_handlers)} data handlers.')
        max_workers = self.load_workers
        if max_workers == 1:
            for d in self.data_handlers:
                d.normalize(self.means, self.stds)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                futures = {}
                now = dt.now()
                for i, d in enumerate(self.data_handlers):
                    future = exe.submit(d.normalize, self.means, self.stds)
                    futures[future] = i

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
                    logger.debug(f'{i+1} out of {len(futures)} data handlers'
                                 ' normalized.')

    def parallel_load(self):
        """Load data handler data in parallel"""
        logger.info(f'Loading {len(self.data_handlers)} data handlers')
        max_workers = self.load_workers
        if max_workers == 1:
            for d in self.data_handlers:
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
                    logger.debug(f'{i+1} out of {len(futures)} handlers '
                                 'loaded.')

    def parallel_stats(self):
        """Get standard deviations and means for training features in parallel
        """
        logger.info(f'Calculating stats for {len(self.training_features)} '
                    'features.')
        max_workers = self.norm_workers
        if max_workers == 1:
            for f in self.training_features:
                self.get_stats_for_feature(f)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                futures = {}
                now = dt.now()
                for i, f in enumerate(self.training_features):
                    future = exe.submit(self.get_stats_for_feature, f)
                    futures[future] = i

                logger.info('Started calculating stats for '
                            f'{len(self.training_features)} features in '
                            f'{dt.now() - now}.')

                for i, future in enumerate(as_completed(futures)):
                    try:
                        future.result()
                    except Exception as e:
                        msg = ('Error calculating stats for '
                               f'{self.training_features[futures[future]]}')
                        logger.exception(msg)
                        raise RuntimeError(msg) from e
                    logger.debug(f'{i+1} out of '
                                 f'{len(self.training_features)} stats '
                                 'calculated.')

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
        means : ndarray
            Array of means for each feature
        stds : ndarray
            Array of stdevs for each feature
        """
        stdevs_check = (self.stdevs_file is not None
                        and not self.overwrite_stats)
        stdevs_check = stdevs_check and os.path.exists(self.stdevs_file)
        means_check = (self.means_file is not None
                       and not self.overwrite_stats)
        means_check = means_check and os.path.exists(self.means_file)
        if stdevs_check and means_check:
            logger.info(f'Loading stdevs from {self.stdevs_file}')
            with open(self.stdevs_file, 'rb') as fh:
                self.stds = pickle.load(fh)
            logger.info(f'Loading means from {self.means_file}')
            with open(self.means_file, 'rb') as fh:
                self.means = pickle.load(fh)

            msg = ('The training features and cached statistics are '
                   'incompatible. Number of training features is '
                   f'{len(self.training_features)} and number of stats is'
                   f' {len(self.stds)}')
            check = len(self.means) == len(self.training_features)
            check = check and (len(self.stds) == len(self.training_features))
            assert check, msg
        return self.means, self.stds

    def cache_stats(self):
        """Saved stdevs and means to cache files if files are not None
        """

        if self.stdevs_file is not None:
            logger.info(f'Saving stdevs to {self.stdevs_file}')
            with open(self.stdevs_file, 'wb') as fh:
                pickle.dump(self.stds, fh)
        if self.means_file is not None:
            logger.info(f'Saving means to {self.means_file}')
            with open(self.means_file, 'wb') as fh:
                pickle.dump(self.means, fh)

    def get_stats(self):
        """Get standard deviations and means for all data features"""

        self.means = np.zeros((self.shape[-1]), dtype=np.float32)
        self.stds = np.zeros((self.shape[-1]), dtype=np.float32)

        now = dt.now()
        logger.info('Calculating stdevs/means.')
        self.parallel_stats()
        logger.info(f'Finished calculating stats in {dt.now() - now}.')
        self.cache_stats()

    def get_handler_mean(self, feature_idx, handler_idx):
        """Get feature mean for a given handler

        Parameters
        ----------
        feature_idx : int
            Index of feature to get mean for
        handler_idx : int
            Index of data handler to get mean for

        Returns
        -------
        float
            Feature mean
        """
        return np.nanmean(
            self.data_handlers[handler_idx].data[..., feature_idx])

    def get_handler_variance(self, feature_idx, handler_idx, mean):
        """Get feature variance for a given handler

        Parameters
        ----------
        feature_idx : int
            Index of feature to get variance for
        handler_idx : int
            Index of data handler to get variance for

        Returns
        -------
        float
            Feature variance
        """
        istd = self.data_handlers[handler_idx].data[..., feature_idx] - mean
        return np.nanmean(istd**2)

    def get_stats_for_feature(self, feature):
        """Get standard deviation and mean for requested feature

        Parameters
        ----------
        feature : str
            Feature to get stats for
        max_workers : int | None
            Max number of workers to use for parallel stats calculations. If
            None the max number of available workers will be used.
        """
        idx = self.training_features.index(feature)
        logger.debug(f'Calculating stdev/mean for {feature}')
        max_workers = self.stats_workers
        self.means[idx] = self.get_means_for_feature(feature, max_workers)
        self.stds[idx] = self.get_stdevs_for_feature(feature, max_workers)

    def get_means_for_feature(self, feature, max_workers=None):
        """Get mean for requested feature

        Parameters
        ----------
        feature : str
            Feature to get mean for
        max_workers : int | None
            Max number of workers to use for parallel stats calculations. If
            None the max number of available workers will be used.
        """
        idx = self.training_features.index(feature)
        logger.debug(f'Calculating mean for {feature}')
        if max_workers == 1:
            for didx, _ in enumerate(self.data_handlers):
                self.means[idx] += self.get_handler_mean(idx, didx)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                futures = {}
                now = dt.now()
                for didx, _ in enumerate(self.data_handlers):
                    future = exe.submit(self.get_handler_mean, idx, didx)
                    futures[future] = didx

                logger.info('Started calculating means for '
                            f'{len(self.data_handlers)} data_handlers in '
                            f'{dt.now() - now}.')

                for i, future in enumerate(as_completed(futures)):
                    self.means[idx] += future.result()
                    logger.debug(f'{i+1} out of {len(self.data_handlers)} '
                                 'means calculated.')
        self.means[idx] /= len(self.data_handlers)
        return self.means[idx]

    def get_stdevs_for_feature(self, feature, max_workers=None):
        """Get stdev for requested feature

        Parameters
        ----------
        feature : str
            Feature to get stdev for
        max_workers : int | None
            Max number of workers to use for parallel stats calculations. If
            None the max number of available workers will be used.
        """
        idx = self.training_features.index(feature)
        logger.debug(f'Calculating stdev for {feature}')
        if max_workers == 1:
            for didx, _ in enumerate(self.data_handlers):
                self.stds[idx] += self.get_handler_variance(idx, didx,
                                                            self.means[idx])
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                futures = {}
                now = dt.now()
                for didx, _ in enumerate(self.data_handlers):
                    future = exe.submit(self.get_handler_variance, idx, didx,
                                        self.means[idx])
                    futures[future] = didx

                logger.info('Started calculating stdevs for '
                            f'{len(self.data_handlers)} data_handlers in '
                            f'{dt.now() - now}.')

                for i, future in enumerate(as_completed(futures)):
                    self.stds[idx] += future.result()
                    logger.debug(f'{i+1} out of {len(self.data_handlers)} '
                                 'stdevs calculated.')
        self.stds[idx] /= len(self.data_handlers)
        self.stds[idx] = np.sqrt(self.stds[idx])
        return self.stds[idx]

    def normalize(self, means=None, stds=None):
        """Compute means and stds for each feature across all datasets and
        normalize each data handler dataset.  Checks if input means and stds
        are different from stored means and stds and renormalizes if they are

        Parameters
        ----------
        means : ndarray | None
            Array of means for each feature. If None it will be calculated.
        stds : ndarray | None
            Array of stdevs for each feature. If None it will be calculated.
        """
        if means is None or stds is None:
            self.get_stats()
        elif means is not None and stds is not None:
            if (not np.array_equal(means, self.means)
                    or not np.array_equal(stds, self.stds)):
                self.unnormalize()
            self.means = means
            self.stds = stds
        now = dt.now()
        logger.info('Normalizing data in each data handler.')
        self.parallel_normalization()
        logger.info('Finished normalizing data in all data handlers in '
                    f'{dt.now() - now}.')

    def unnormalize(self):
        """Remove normalization from stored means and stds"""
        for d in self.data_handlers:
            d.unnormalize(self.means, self.stds)

    def __iter__(self):
        self._i = 0
        return self

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
            handler_index = np.random.randint(0, len(self.data_handlers))
            self.current_handler_index = handler_index
            handler = self.data_handlers[handler_index]
            high_res = np.zeros((self.batch_size, self.sample_shape[0],
                                 self.sample_shape[1], self.sample_shape[2],
                                 self.shape[-1]), dtype=np.float32)

            for i in range(self.batch_size):
                high_res[i, ...] = handler.get_next()
                self.current_batch_indices.append(handler.current_obs_index)

            batch = self.BATCH_CLASS.get_coarse_batch(
                high_res, self.s_enhance, t_enhance=self.t_enhance,
                temporal_coarsening_method=self.temporal_coarsening_method,
                output_features_ind=self.output_features_ind,
                output_features=self.output_features,
                training_features=self.training_features,
                smoothing=self.smoothing,
                smoothing_ignore=self.smoothing_ignore)

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

        handler_index = np.random.randint(0, len(self.data_handlers))
        self.current_handler_index = handler_index
        handler = self.data_handlers[handler_index]

        low_res = None
        high_res = None

        for i in range(self.batch_size):
            obs_hourly, obs_daily_avg = handler.get_next()
            self.current_batch_indices.append(handler.current_obs_index)

            obs_hourly = self.BATCH_CLASS.reduce_features(
                obs_hourly, self.output_features_ind)

            if low_res is None:
                lr_shape = (self.batch_size,) + obs_daily_avg.shape
                hr_shape = (self.batch_size,) + obs_hourly.shape
                low_res = np.zeros(lr_shape, dtype=np.float32)
                high_res = np.zeros(hr_shape, dtype=np.float32)

            low_res[i] = obs_daily_avg
            high_res[i] = obs_hourly

        high_res = self.reduce_high_res_sub_daily(high_res)
        low_res = spatial_coarsening(low_res, self.s_enhance)

        if (self.output_features is not None
                and 'clearsky_ratio' in self.output_features):
            i_cs = self.output_features.index('clearsky_ratio')
            if np.isnan(high_res[..., i_cs]).any():
                high_res[..., i_cs] = nn_fill_array(high_res[..., i_cs])

        if self.smoothing is not None:
            feat_iter = [j for j in range(low_res.shape[-1])
                         if self.training_features[j]
                         not in self.smoothing_ignore]
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

        handler_index = np.random.randint(0, len(self.data_handlers))
        self.current_handler_index = handler_index
        handler = self.data_handlers[handler_index]

        high_res = None

        for i in range(self.batch_size):
            _, obs_daily_avg = handler.get_next()
            self.current_batch_indices.append(handler.current_obs_index)

            if high_res is None:
                hr_shape = (self.batch_size,) + obs_daily_avg.shape
                high_res = np.zeros(hr_shape, dtype=np.float32)

                msg = ('SpatialBatchHandlerCC can only use n_temporal==1 '
                       'but received HR shape {} with n_temporal={}.'
                       .format(hr_shape, hr_shape[3]))
                assert hr_shape[3] == 1, msg

            high_res[i] = obs_daily_avg

        low_res = spatial_coarsening(high_res, self.s_enhance)
        low_res = low_res[:, :, :, 0, :]
        high_res = high_res[:, :, :, 0, :]

        high_res = self.BATCH_CLASS.reduce_features(
            high_res, self.output_features_ind)

        if (self.output_features is not None
                and 'clearsky_ratio' in self.output_features):
            i_cs = self.output_features.index('clearsky_ratio')
            if np.isnan(high_res[..., i_cs]).any():
                high_res[..., i_cs] = nn_fill_array(high_res[..., i_cs])

        if self.smoothing is not None:
            feat_iter = [j for j in range(low_res.shape[-1])
                         if self.training_features[j]
                         not in self.smoothing_ignore]
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

    def __next__(self):
        if self._i < self.n_batches:
            handler_index = np.random.randint(
                0, len(self.data_handlers))
            handler = self.data_handlers[handler_index]
            high_res = np.zeros((self.batch_size, self.sample_shape[0],
                                 self.sample_shape[1], self.shape[-1]),
                                dtype=np.float32)
            for i in range(self.batch_size):
                high_res[i, ...] = handler.get_next()[..., 0, :]

            batch = self.BATCH_CLASS.get_coarse_batch(
                high_res, self.s_enhance,
                output_features_ind=self.output_features_ind,
                training_features=self.training_features,
                smoothing=self.smoothing,
                smoothing_ignore=self.smoothing_ignore)

            self._i += 1
            return batch
        else:
            raise StopIteration


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
            data[tuple_index]"""

        val_indices = {}
        for t in range(self.N_TIME_BINS):
            val_indices[t] = []
            h_idx = np.random.choice(np.arange(len(self.handlers)))
            h = self.handlers[h_idx]
            for _ in range(self.batch_size):
                spatial_slice = uniform_box_sampler(
                    h.data, self.sample_shape[:2])
                weights = np.zeros(self.N_TIME_BINS)
                weights[t] = 1
                temporal_slice = weighted_time_sampler(
                    h.data, self.sample_shape[2], weights)
                tuple_index = tuple(spatial_slice + [temporal_slice]
                                    + [np.arange(h.data.shape[-1])])
                val_indices[t].append({'handler_index': h_idx,
                                       'tuple_index': tuple_index})
        for s in range(self.N_SPACE_BINS):
            val_indices[s + self.N_TIME_BINS] = []
            h_idx = np.random.choice(np.arange(len(self.handlers)))
            h = self.handlers[h_idx]
            for _ in range(self.batch_size):
                weights = np.zeros(self.N_SPACE_BINS)
                weights[s] = 1
                spatial_slice = weighted_box_sampler(
                    h.data, self.sample_shape[:2], weights)
                temporal_slice = uniform_time_sampler(
                    h.data, self.sample_shape[2])
                tuple_index = tuple(spatial_slice + [temporal_slice]
                                    + [np.arange(h.data.shape[-1])])
                val_indices[s + self.N_TIME_BINS].append(
                    {'handler_index': h_idx, 'tuple_index': tuple_index})
        return val_indices

    def __next__(self):
        if self._i < len(self.val_indices.keys()):
            high_res = np.zeros((self.batch_size, self.sample_shape[0],
                                 self.sample_shape[1],
                                 self.sample_shape[2],
                                 self.handlers[0].shape[-1]),
                                dtype=np.float32)
            val_indices = self.val_indices[self._i]
            for i, idx in enumerate(val_indices):
                high_res[i, ...] = self.handlers[
                    idx['handler_index']].data[idx['tuple_index']]

            batch = self.BATCH_CLASS.get_coarse_batch(
                high_res, self.s_enhance, t_enhance=self.t_enhance,
                temporal_coarsening_method=self.temporal_coarsening_method,
                output_features_ind=self.output_features_ind,
                smoothing=self.smoothing,
                smoothing_ignore=self.smoothing_ignore,
                output_features=self.output_features)
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
            high_res = np.zeros((self.batch_size, self.sample_shape[0],
                                 self.sample_shape[1],
                                 self.handlers[0].shape[-1]),
                                dtype=np.float32)
            val_indices = self.val_indices[self._i]
            for i, idx in enumerate(val_indices):
                high_res[i, ...] = self.handlers[
                    idx['handler_index']].data[idx['tuple_index']][..., 0, :]

            batch = self.BATCH_CLASS.get_coarse_batch(
                high_res, self.s_enhance,
                output_features_ind=self.output_features_ind,
                smoothing=self.smoothing,
                smoothing_ignore=self.smoothing_ignore,
                output_features=self.output_features)
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
        bin_range -= (self.sample_shape[2] - 1)
        self.temporal_bins = np.array_split(np.arange(0, bin_range),
                                            self.val_data.N_TIME_BINS)
        self.temporal_bins = [b[0] for b in self.temporal_bins]

        logger.info('Using temporal weights: '
                    f'{[round(w, 3) for w in self.temporal_weights]}')
        self.temporal_sample_record = [0] * self.val_data.N_TIME_BINS
        self.norm_temporal_record = [0] * self.val_data.N_TIME_BINS

    def update_training_sample_record(self):
        """Keep track of number of observations from each temporal bin"""
        handler = self.data_handlers[self.current_handler_index]
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
            handler_index = np.random.randint(0, len(self.data_handlers))
            self.current_handler_index = handler_index
            handler = self.data_handlers[handler_index]
            high_res = np.zeros((self.batch_size, self.sample_shape[0],
                                 self.sample_shape[1], self.sample_shape[2],
                                 self.shape[-1]), dtype=np.float32)

            for i in range(self.batch_size):
                high_res[i, ...] = handler.get_next(
                    temporal_weights=self.temporal_weights)
                self.current_batch_indices.append(handler.current_obs_index)

                self.update_training_sample_record()

            batch = self.BATCH_CLASS.get_coarse_batch(
                high_res, self.s_enhance, t_enhance=self.t_enhance,
                temporal_coarsening_method=self.temporal_coarsening_method,
                output_features_ind=self.output_features_ind,
                output_features=self.output_features,
                training_features=self.training_features,
                smoothing=self.smoothing,
                smoothing_ignore=self.smoothing_ignore)

            self._i += 1
            return batch
        else:
            total_count = self.n_batches * self.batch_size
            self.norm_temporal_record = [c / total_count for c
                                         in self.temporal_sample_record.copy()]
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
        handler = self.data_handlers[self.current_handler_index]
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
            handler_index = np.random.randint(0, len(self.data_handlers))
            self.current_handler_index = handler_index
            handler = self.data_handlers[handler_index]
            high_res = np.zeros((self.batch_size, self.sample_shape[0],
                                 self.sample_shape[1], self.shape[-1]),
                                dtype=np.float32)

            for i in range(self.batch_size):
                high_res[i, ...] = handler.get_next(
                    spatial_weights=self.spatial_weights)[..., 0, :]
                self.current_batch_indices.append(handler.current_obs_index)

                self.update_training_sample_record()

            batch = self.BATCH_CLASS.get_coarse_batch(
                high_res, self.s_enhance,
                output_features_ind=self.output_features_ind,
                training_features=self.training_features,
                smoothing=self.smoothing,
                smoothing_ignore=self.smoothing_ignore)

            self._i += 1
            return batch
        else:
            total_count = self.n_batches * self.batch_size
            self.norm_spatial_record = [c / total_count
                                        for c in self.spatial_sample_record]
            self.old_spatial_weights = self.spatial_weights.copy()
            raise StopIteration
