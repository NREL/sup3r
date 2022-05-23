# -*- coding: utf-8 -*-
"""
Sup3r batch_handling module.

@author: bbenton
"""
import logging
import numpy as np
from datetime import datetime as dt
import threading
import os
import pickle

from rex.utilities import log_mem

from sup3r.utilities.utilities import (daily_time_sampler,
                                       weighted_time_sampler,
                                       spatial_coarsening,
                                       temporal_coarsening,
                                       uniform_box_sampler,
                                       uniform_time_sampler,
                                       nn_fill_array)
from sup3r.preprocessing.data_handling import DataHandlerDCforH5
from sup3r.preprocessing.data_handling import get_handler_class
from sup3r import __version__

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
                         output_features=None):
        """Coarsen high res data and return Batch with
        high res and low res data

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
            or total
        output_features_ind : list | np.ndarray | None
            List/array of feature channel indices that are used for generative
            output, without any feature indices used only for training.
        output_features : list
            List of Generative model output feature names

        Returns
        -------
        Batch
            Batch instance with low and high res data
        """
        low_res = spatial_coarsening(
            high_res, s_enhance)

        if t_enhance != 1:
            low_res = temporal_coarsening(
                low_res, t_enhance,
                temporal_coarsening_method)

        high_res = cls.reduce_features(high_res, output_features_ind)
        batch = cls(low_res, high_res)

        return batch


class NsrdbBatch(Batch):
    """Special batch handler for NSRDB data"""

    @classmethod
    def get_coarse_batch(cls, high_res,
                         s_enhance, t_enhance=1,
                         temporal_coarsening_method='subsample',
                         output_features_ind=None,
                         output_features=None):
        """Coarsen high res data and return Batch with high res and low res
        data

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
            or total
        output_features_ind : list | np.ndarray | None
            List/array of feature channel indices that are used for generative
            output, without any feature indices used only for training.
        output_features : list
            List of Generative model output feature names

        Returns
        -------
        Batch
            Batch instance with low and high res data
        """

        # for nsrdb, do temporal avg first so you dont have to do spatial agg
        # across NaNs
        low_res = temporal_coarsening(
            high_res, t_enhance,
            temporal_coarsening_method)

        low_res = spatial_coarsening(
            low_res, s_enhance)

        high_res = cls.reduce_features(high_res, output_features_ind)

        if (output_features is not None
                and 'clearsky_ratio' in output_features):
            cs_ind = output_features.index('clearsky_ratio')
            if np.isnan(high_res[..., cs_ind]).any():
                high_res[..., cs_ind] = nn_fill_array(high_res[..., cs_ind])

        batch = cls(low_res, high_res)

        return batch


class ValidationData:
    """Iterator for validation data"""

    # Classes to use for handling an individual batch obj.
    BATCH_CLASS = Batch

    def __init__(self, data_handlers, batch_size=8, s_enhance=3, t_enhance=1,
                 temporal_coarsening_method='subsample',
                 output_features_ind=None,
                 output_features=None):
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
            [subsample, average, total]
            Subsample will take every t_enhance-th time step, average will
            average over t_enhance time steps, total will sum over t_enhance
            time steps
        output_features_ind : list | np.ndarray | None
            List/array of feature channel indices that are used for generative
            output, without any feature indices used only for training.
        output_features : list
            List of Generative model output feature names
        """

        handler_shapes = np.array(
            [d.sample_shape for d in data_handlers])
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
            batch = self.BATCH_CLASS.get_coarse_batch(
                high_res, self.s_enhance,
                t_enhance=self.t_enhance,
                temporal_coarsening_method=self.temporal_coarsening_method,
                output_features_ind=self.output_features_ind,
                output_features=self.output_features)
            self._i += 1
            return batch
        else:
            raise StopIteration


class NsrdbValidationData(ValidationData):
    """Iterator for daily NSRDB validation data"""

    # Classes to use for handling an individual batch obj.
    BATCH_CLASS = NsrdbBatch

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
            for _ in range(h.val_data.shape[2]):

                spatial_slice = uniform_box_sampler(h.val_data,
                                                    self.sample_shape[:2])

                temporal_slice = daily_time_sampler(h.val_data,
                                                    self.sample_shape[2],
                                                    h.val_time_index)
                tuple_index = tuple(spatial_slice
                                    + [temporal_slice]
                                    + [np.arange(h.val_data.shape[-1])])

                val_indices.append({'handler_index': i,
                                    'tuple_index': tuple_index})

        return val_indices


class BatchHandler:
    """Sup3r base batch handling class"""

    # Classes to use for handling an individual batch obj.
    VAL_CLASS = ValidationData
    BATCH_CLASS = Batch
    DATA_HANDLER_CLASS = None

    def __init__(self, data_handlers, batch_size=8, s_enhance=3, t_enhance=2,
                 means=None, stds=None, norm=True, n_batches=10,
                 temporal_coarsening_method='subsample', stdevs_file=None,
                 means_file=None, n_features_per_thread=12):
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
            features.  If not None and norm is True these will be used for
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
            [subsample, average, total]
            Subsample will take every t_enhance-th time step, average will
            average over t_enhance time steps, total will sum over t_enhance
            time steps
        stdevs_file : str | None
            Path to stdevs data or where to save data after calling _get_stats
        means_file : str | None
            Path to means data or where to save data after calling _get_stats
        n_features_per_thread : int
            Number of features to load from cache in parallel. This number will
            tell the BatchHandler how to chunk the data handlers so that
            number_of_features_per_handler * number_of_handlers_per_chunk <=
            n_features_per_thread.
        """

        handler_shapes = np.array(
            [d.sample_shape for d in data_handlers])
        assert np.all(handler_shapes[0] == handler_shapes)

        n_feature_arrays = len(data_handlers[0].features) * len(data_handlers)
        n_chunks = int(np.ceil(n_feature_arrays / n_features_per_thread))
        n_chunks = min(n_chunks, len(data_handlers))
        handler_chunks = np.array_split(data_handlers, n_chunks)

        for j, handler_chunk in enumerate(handler_chunks):
            futures = {}
            now = dt.now()
            for i, d in enumerate(handler_chunk):
                if d.data is None:
                    future = threading.Thread(target=d.load_cached_data)
                    futures[future] = i
                    future.start()

            logger.info(
                f'Started loading all data handlers for handler_chunk {j + 1} '
                f'of {len(handler_chunks)} in {dt.now() - now}. ')

            for i, future in enumerate(futures.keys()):
                future.join()
                logger.debug(
                    f'{i + 1} out of {len(futures)} handlers for handler_chunk'
                    f' {j + 1} loaded.')

        self.data_handlers = data_handlers
        logger.debug(f'Finished loading data of shape {self.shape} '
                     'for BatchHandler.')
        log_mem(logger)

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

        if norm:
            logger.debug('Normalizing data for BatchHandler.')
            self.means, self.stds = self.check_cached_stats()
            self.normalize(self.means, self.stds)
            self.cache_stats()

        logger.debug('Getting validation data for BatchHandler.')
        self.val_data = self.VAL_CLASS(
            data_handlers, batch_size=batch_size,
            s_enhance=s_enhance, t_enhance=t_enhance,
            temporal_coarsening_method=temporal_coarsening_method,
            output_features_ind=self.output_features_ind,
            output_features=self.output_features)

        logger.info('Finished initializing BatchHandler.')

    def __len__(self):
        """Use user input of n_batches to specify length

        Returns
        -------
        self.n_batches : int
            Number of batches possible to iterate over
        """
        return self.n_batches

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
        stdevs_check = (self.stdevs_file is not None)
        stdevs_check = stdevs_check and os.path.exists(self.stdevs_file)
        means_check = (self.means_file is not None)
        means_check = means_check and os.path.exists(self.means_file)
        if stdevs_check and means_check:
            logger.info(f'Loading stdevs from {self.stdevs_file}')
            with open(self.stdevs_file, 'rb') as fh:
                self.stds = pickle.load(fh)
            logger.info(f'Loading means from {self.means_file}')
            with open(self.means_file, 'rb') as fh:
                self.means = pickle.load(fh)
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

    def _get_stats(self):
        """Get standard deviations and means for all data features"""

        self.means = np.zeros((self.shape[-1]), dtype=np.float32)
        self.stds = np.zeros((self.shape[-1]), dtype=np.float32)

        logger.info('Calculating stdevs / means.')
        n_elems = np.product(self.shape[:-1])
        for i in range(self.shape[-1]):
            for data_handler in self.data_handlers:
                self.means[i] += np.nansum(data_handler.data[..., i])
            self.means[i] = self.means[i] / n_elems
            for data_handler in self.data_handlers:
                self.stds[i] += np.nansum(
                    (data_handler.data[..., i] - self.means[i])**2)
            self.stds[i] = np.sqrt(self.stds[i] / n_elems)

    def normalize(self, means=None, stds=None):
        """Compute means and stds for each feature across all datasets and
        normalize each data handler dataset.  Checks if input means and stds
        are different from stored means and stds and renormalizes if they are
        new """
        if means is None or stds is None:
            self._get_stats()
        elif means is not None and stds is not None:
            if (not np.array_equal(means, self.means)
                    or not np.array_equal(stds, self.stds)):
                self.unnormalize()
            self.means = means
            self.stds = stds

        logger.info('Normalizing data in each data handler.')
        futures = {}
        now = dt.now()
        for i, d in enumerate(self.data_handlers):
            future = threading.Thread(target=d.normalize,
                                      args=(self.means, self.stds))
            futures[future] = i
            future.start()

        logger.info(
            f'Started normalizing {len(self.data_handlers)} data handlers '
            f'in {dt.now() - now}. ')

        for i, future in enumerate(futures.keys()):
            future.join()
            logger.debug(
                f'{i + 1} out of {len(futures)} data handlers normalized')
        logger.info(f'Finished normalizing data in all data handlers')

    def unnormalize(self):
        """Remove normalization from stored means and stds"""
        for d in self.data_handlers:
            d.unnormalize(self.means, self.stds)

    def __iter__(self):
        self._i = 0
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
                high_res[i, ...] = handler.get_next()
                self.current_batch_indices.append(handler.current_obs_index)

            batch = self.BATCH_CLASS.get_coarse_batch(
                high_res, self.s_enhance, t_enhance=self.t_enhance,
                temporal_coarsening_method=self.temporal_coarsening_method,
                output_features_ind=self.output_features_ind,
                output_features=self.output_features)

            self._i += 1
            return batch
        else:
            raise StopIteration


class NsrdbBatchHandler(BatchHandler):
    """Sup3r base batch handling class"""

    # Classes to use for handling an individual batch obj.
    VAL_CLASS = NsrdbValidationData
    BATCH_CLASS = NsrdbBatch


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
                output_features_ind=self.output_features_ind)

            self._i += 1
            return batch
        else:
            raise StopIteration


class ValidationDataDC(ValidationData):
    """Iterator for data-centric validation data"""

    N_TIME_BINS = 12

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
                output_features=self.output_features)
            self._i += 1
            return batch
        else:
            raise StopIteration


class BatchHandlerDC(BatchHandler):
    """Data-centric batch handler"""

    VAL_CLASS = ValidationDataDC
    BATCH_CLASS = Batch
    DATA_HANDLER_CLASS = DataHandlerDCforH5

    def __init__(self, data_handlers, batch_size=8, s_enhance=3, t_enhance=2,
                 means=None, stds=None, norm=True, n_batches=10,
                 temporal_coarsening_method='subsample', stdevs_file=None,
                 means_file=None, n_features_per_thread=12):
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
            features.  If not None and norm is True these will be used for
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
            [subsample, average, total]
            Subsample will take every t_enhance-th time step, average will
            average over t_enhance time steps, total will sum over t_enhance
            time steps
        stdevs_file : str | None
            Path to stdevs data or where to save data after calling _get_stats
        means_file : str | None
            Path to means data or where to save data after calling _get_stats
        n_features_per_thread : int
            Number of features to load from cache in parallel. This number will
            tell the BatchHandler how to chunk the data handlers so that
            number_of_features_per_handler * number_of_handlers_per_chunk <=
            n_features_per_thread.
        """

        super().__init__(data_handlers=data_handlers, batch_size=batch_size,
                         s_enhance=s_enhance, t_enhance=t_enhance,
                         means=means, stds=stds, norm=norm,
                         n_batches=n_batches,
                         temporal_coarsening_method=temporal_coarsening_method,
                         stdevs_file=stdevs_file, means_file=means_file,
                         n_features_per_thread=n_features_per_thread)

        self.temporal_weights = np.ones(self.val_data.N_TIME_BINS)
        self.temporal_weights /= np.sum(self.temporal_weights)
        self.training_sample_record = [0] * self.val_data.N_TIME_BINS
        bin_range = self.data_handlers[0].data.shape[2] - self.sample_shape[2]
        self.temporal_bins = np.array_split(np.arange(0, bin_range),
                                            self.val_data.N_TIME_BINS)
        self.temporal_bins = [b[0] for b in self.temporal_bins]

        logger.info(
            'Using temporal weights: '
            f'{[round(w, 3) for w in self.temporal_weights]}')

    def update_training_sample_record(self):
        """Keep track of number of observations from each temporal bin"""
        handler = self.data_handlers[self.current_handler_index]
        t_start = handler.current_obs_index[2].start
        bin_number = np.digitize(t_start, self.temporal_bins)
        self.training_sample_record[bin_number - 1] += 1

    def __iter__(self):
        self._i = 0
        self.training_sample_record = [0] * self.val_data.N_TIME_BINS
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
                high_res[i, ...] = handler.get_next(self.temporal_weights)
                self.current_batch_indices.append(handler.current_obs_index)

                self.update_training_sample_record()

            batch = self.BATCH_CLASS.get_coarse_batch(
                high_res, self.s_enhance, t_enhance=self.t_enhance,
                temporal_coarsening_method=self.temporal_coarsening_method,
                output_features_ind=self.output_features_ind,
                output_features=self.output_features)

            self._i += 1
            return batch
        else:
            raise StopIteration
