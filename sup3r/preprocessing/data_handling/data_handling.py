"""Base data handler with training methods classes.
@author: bbenton
"""

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime as dt
from fnmatch import fnmatch

import numpy as np

from sup3r.preprocessing.data_handling.base import BaseDataHandler
from sup3r.utilities.utilities import (
    estimate_max_workers,
    uniform_box_sampler,
    uniform_time_sampler,
    weighted_box_sampler,
    weighted_time_sampler,
)

np.random.seed(42)

logger = logging.getLogger(__name__)


class DataHandler(BaseDataHandler):
    """Sup3r data handling and extraction for low-res source data or for
    artificially coarsened high-res source data for training. Extends
    BaseDataHandler to include training related methods.

    The sup3r data handler class is based on a 4D numpy array of shape:
    (spatial_1, spatial_2, temporal, features)
    """

    # list of features / feature name patterns that are input to the generative
    # model but are not part of the synthetic output and are not sent to the
    # discriminator. These are case-insensitive and follow the Unix shell-style
    # wildcard format.
    TRAIN_ONLY_FEATURES = (
        'BVF*',
        'inversemoninobukhovlength_*',
        'RMOL',
        'topography',
    )

    def __init__(
        self,
        *args,
        shuffle_time=False,
        sample_shape=(10, 10, 1),
        train_only_features=None,
        val_split=0.0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        *args : tuple
            Sample arguments as BaseDataHandler
        **kwargs : dict
            Same keyword arguments as BaseDataHandler
        val_split : float32
            Fraction of data to store for validation
        shuffle_time : bool
            Whether to shuffle data along the time dimension prior to training
        sample_shape : tuple
            Size of spatial and temporal domain used in a single high-res
            observation for batching
        train_only_features : list | tuple | None
            List of feature names or patt*erns that should only be included in
            the training set and not the output. If None (default), this will
            default to the class TRAIN_ONLY_FEATURES attribute.
        """
        super().__init__(*args, **kwargs)

        self.val_time_index = None
        self.val_split = val_split
        self.sample_shape = sample_shape
        self.shuffle_time = shuffle_time
        self.current_obs_index = None
        self.val_data = None
        self._train_only_features = train_only_features
        self._norm_workers = self.worker_kwargs.get('norm_workers', None)
        self._worker_attrs = [
            '_ti_workers',
            '_norm_workers',
            '_compute_workers',
            '_extract_workers',
            '_load_workers',
        ]

        self._sample_shape_check()

        self._val_split_check()

    def _sample_shape_check(self):
        """Check conditions on sample_shape"""

        if len(self.sample_shape) == 2:
            logger.info(
                'Found 2D sample shape of {}. Adding temporal dim of 1'.format(
                    self.sample_shape
                )
            )
            self.sample_shape = (*self.sample_shape, 1)

        n_steps = self.n_tsteps
        msg = (
            f'Temporal slice step ({self.temporal_slice.step}) does not '
            f'evenly divide the number of time steps ({n_steps})'
        )
        check = self.temporal_slice.step is None
        check = check or n_steps % self.temporal_slice.step == 0
        if not check:
            logger.warning(msg)
            warnings.warn(msg)

        msg = (
            f'sample_shape[2] ({self.sample_shape[2]}) cannot be larger '
            'than the number of time steps in the raw data '
            f'({len(self.raw_time_index)}).'
        )
        if len(self.raw_time_index) < self.sample_shape[2]:
            logger.warning(msg)
            warnings.warn(msg)

        bad_shape = (
            self.sample_shape[0] > self.grid_shape[0]
            and self.sample_shape[1] > self.grid_shape[1]
        )
        if bad_shape:
            msg = (
                f'spatial_sample_shape {self.sample_shape[:2]} is '
                f'larger than the raster size {self.grid_shape}'
            )
            logger.warning(msg)
            warnings.warn(msg)

    def _val_split_check(self):
        """Check if val_split > 0 and split data into validation and training.
        Make sure validation data is larger than sample_shape"""

        if self.data is not None and self.val_split > 0.0:
            self.data, self.val_data = self.split_data(
                val_split=self.val_split, shuffle_time=self.shuffle_time
            )
            msg = (
                f'Validation data has shape={self.val_data.shape} '
                f'and sample_shape={self.sample_shape}. Use a smaller '
                'sample_shape and/or larger val_split.'
            )
            check = any(
                val_size < samp_size
                for val_size, samp_size in zip(
                    self.val_data.shape, self.sample_shape
                )
            )
            if check:
                logger.warning(msg)
                warnings.warn(msg)

    @property
    def train_only_features(self):
        """Features to use for training only and not output"""
        if self._train_only_features is None:
            self._train_only_features = self.TRAIN_ONLY_FEATURES
        return self._train_only_features

    @property
    def norm_workers(self):
        """Get upper bound on workers used for normalization."""
        if self.data is not None:
            norm_workers = estimate_max_workers(
                self._norm_workers, 2 * self.feature_mem, self.shape[-1]
            )
        else:
            norm_workers = self._norm_workers
        return norm_workers

    @property
    def output_features(self):
        """Get a list of features that should be output by the generative model
        corresponding to the features in the high res batch array."""
        out = []
        for feature in self.features:
            ignore = any(
                fnmatch(feature.lower(), pattern.lower())
                for pattern in self.train_only_features
            )
            if not ignore:
                out.append(feature)
        return out

    def unnormalize(self, means, stds):
        """Remove normalization from stored means and stds"""
        self.val_data = (self.val_data * stds) + means
        self.data = (self.data * stds) + means

    def normalize(self, means, stds):
        """Normalize all data features

        Parameters
        ----------
        means : np.ndarray
            dimensions (features)
            array of means for all features with same ordering as data features
        stds : np.ndarray
            dimensions (features)
            array of means for all features with same ordering as data features
        """
        logger.info(f'Normalizing {self.shape[-1]} features.')
        max_workers = self.norm_workers
        if max_workers == 1:
            for i in range(self.shape[-1]):
                self._normalize_data(i, means[i], stds[i])
        else:
            self.parallel_normalization(means, stds, max_workers=max_workers)

    def parallel_normalization(self, means, stds, max_workers=None):
        """Run normalization of features in parallel

        Parameters
        ----------
        means : np.ndarray
            dimensions (features)
            array of means for all features with same ordering as data features
        stds : np.ndarray
            dimensions (features)
            array of means for all features with same ordering as data features
        max_workers : int | None
            Max number of workers to use for normalizing features
        """

        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = {}
            now = dt.now()
            for i in range(self.shape[-1]):
                future = exe.submit(self._normalize_data, i, means[i], stds[i])
                futures[future] = i

            logger.info(
                f'Started normalizing {self.shape[-1]} features '
                f'in {dt.now() - now}.'
            )

            for i, future in enumerate(as_completed(futures)):
                try:
                    future.result()
                except Exception as e:
                    msg = (
                        'Error while normalizing future number '
                        f'{futures[future]}.'
                    )
                    logger.exception(msg)
                    raise RuntimeError(msg) from e
                logger.debug(
                    f'{i+1} out of {self.shape[-1]} features ' 'normalized.'
                )

    def _normalize_data(self, feature_index, mean, std):
        """Normalize data with initialized mean and standard deviation for a
        specific feature

        Parameters
        ----------
        feature_index : int
            index of feature to be normalized
        mean : float32
            specified mean of associated feature
        std : float32
            specificed standard deviation for associated feature
        """

        if self.val_data is not None:
            self.val_data[..., feature_index] -= mean
        self.data[..., feature_index] -= mean

        if std > 0:
            if self.val_data is not None:
                self.val_data[..., feature_index] /= std
            self.data[..., feature_index] /= std
        else:
            msg = (
                'Standard Deviation is zero for '
                f'{self.features[feature_index]}'
            )
            logger.warning(msg)
            warnings.warn(msg)

    def get_observation_index(self):
        """Randomly gets spatial sample and time sample

        Returns
        -------
        observation_index : tuple
            Tuple of sampled spatial grid, time slice, and features indices.
            Used to get single observation like self.data[observation_index]
        """
        spatial_slice = uniform_box_sampler(self.data, self.sample_shape[:2])
        temporal_slice = uniform_time_sampler(self.data, self.sample_shape[2])
        return tuple(
            [*spatial_slice, temporal_slice, np.arange(len(self.features))]
        )

    def get_next(self):
        """Get data for observation using random observation index. Loops
        repeatedly over randomized time index

        Returns
        -------
        observation : np.ndarray
            4D array
            (spatial_1, spatial_2, temporal, features)
        """
        self.current_obs_index = self.get_observation_index()
        observation = self.data[self.current_obs_index]
        return observation

    def split_data(self, data=None, val_split=0.0, shuffle_time=False):
        """Split time dimension into set of training indices and validation
        indices

        Parameters
        ----------
        data : np.ndarray
            4D array of high res data
            (spatial_1, spatial_2, temporal, features)
        val_split : float
            Fraction of data to separate for validation.
        shuffle_time : bool
            Whether to shuffle time or not.

        Returns
        -------
        data : np.ndarray
            (spatial_1, spatial_2, temporal, features)
            Training data fraction of initial data array. Initial data array is
            overwritten by this new data array.
        val_data : np.ndarray
            (spatial_1, spatial_2, temporal, features)
            Validation data fraction of initial data array.
        """
        self.data = data if data is not None else self.data

        n_observations = self.data.shape[2]
        all_indices = np.arange(n_observations)
        n_val_obs = int(val_split * n_observations)

        if shuffle_time:
            np.random.shuffle(all_indices)

        val_indices = all_indices[:n_val_obs]
        training_indices = all_indices[n_val_obs:]

        if not shuffle_time:
            [self.val_data, self.data] = np.split(
                self.data, [n_val_obs], axis=2
            )
        else:
            self.val_data = self.data[:, :, val_indices, :]
            self.data = self.data[:, :, training_indices, :]

        self.val_time_index = self.time_index[val_indices]
        self.time_index = self.time_index[training_indices]

        return self.data, self.val_data


# pylint: disable=W0223
class DataHandlerDC(DataHandler):
    """Data-centric data handler"""

    def get_observation_index(
        self, temporal_weights=None, spatial_weights=None
    ):
        """Randomly gets weighted spatial sample and time sample

        Parameters
        ----------
        temporal_weights : array
            Weights used to select time slice
            (n_time_chunks)
        spatial_weights : array
            Weights used to select spatial chunks
            (n_lat_chunks * n_lon_chunks)

        Returns
        -------
        observation_index : tuple
            Tuple of sampled spatial grid, time slice, and features indices.
            Used to get single observation like self.data[observation_index]
        """
        if spatial_weights is not None:
            spatial_slice = weighted_box_sampler(
                self.data, self.sample_shape[:2], weights=spatial_weights
            )
        else:
            spatial_slice = uniform_box_sampler(
                self.data, self.sample_shape[:2]
            )
        if temporal_weights is not None:
            temporal_slice = weighted_time_sampler(
                self.data, self.sample_shape[2], weights=temporal_weights
            )
        else:
            temporal_slice = uniform_time_sampler(
                self.data, self.sample_shape[2]
            )

        return tuple(
            [*spatial_slice, temporal_slice, np.arange(len(self.features))]
        )

    def get_next(self, temporal_weights=None, spatial_weights=None):
        """Get data for observation using weighted random observation index.
        Loops repeatedly over randomized time index.

        Parameters
        ----------
        temporal_weights : array
            Weights used to select time slice
            (n_time_chunks)
        spatial_weights : array
            Weights used to select spatial chunks
            (n_lat_chunks * n_lon_chunks)

        Returns
        -------
        observation : np.ndarray
            4D array
            (spatial_1, spatial_2, temporal, features)
        """
        self.current_obs_index = self.get_observation_index(
            temporal_weights=temporal_weights, spatial_weights=spatial_weights
        )
        observation = self.data[self.current_obs_index]
        return observation
