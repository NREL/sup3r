"""Abstract sampler objects. These are containers which also can sample from
the underlying data. These interface with Batchers so they also have additional
information about how different features are used by models."""

import logging
from fnmatch import fnmatch
from typing import Dict, Optional, Tuple, Union
from warnings import warn

import dask.array as da
import numpy as np

from sup3r.preprocessing.accessor import Sup3rX
from sup3r.preprocessing.base import Container, Sup3rDataset
from sup3r.preprocessing.samplers.utilities import (
    uniform_box_sampler,
    uniform_time_sampler,
)
from sup3r.preprocessing.utilities import compute_if_dask, lowered
from sup3r.typing import T_Array

logger = logging.getLogger(__name__)


class Sampler(Container):
    """Sampler class for iterating through samples of contained data."""

    def __init__(
        self,
        data: Union[Sup3rX, Sup3rDataset],
        sample_shape: Optional[tuple] = None,
        batch_size: int = 16,
        feature_sets: Optional[Dict] = None,
    ):
        """
        Parameters
        ----------
        data: Union[Sup3rX, Sup3rDataset],
            Object with data that will be sampled from. Usually the `.data`
            attribute of various :class:`Container` objects.  i.e.
            :class:`Loader`, :class:`Rasterizer`, :class:`Deriver`, as long as
            the spatial dimensions are not flattened.
        sample_shape : tuple
            Size of arrays to sample from the contained data.
        batch_size : int
            Number of samples to get to build a single batch. A sample of
            (sample_shape[0], sample_shape[1], batch_size * sample_shape[2])
            is first selected from underlying dataset and then reshaped into
            (batch_size, *sample_shape) to get a single batch. This is more
            efficient than getting N = batch_size samples and then stacking.
        feature_sets : Optional[dict]
            Optional dictionary describing how the full set of features is
            split between `lr_only_features` and `hr_exo_features`.

            features : list | tuple
                List of full set of features to use for sampling. If no entry
                is provided then all data_vars from data will be used.
            lr_only_features : list | tuple
                List of feature names or patt*erns that should only be
                included in the low-res training set and not the high-res
                observations.
            hr_exo_features : list | tuple
                List of feature names or patt*erns that should be included
                in the high-resolution observation but not expected to be
                output from the generative model. An example is high-res
                topography that is to be injected mid-network.
        """
        super().__init__(data=data)
        feature_sets = feature_sets or {}
        self.features = feature_sets.get('features', self.data.features)
        self._lr_only_features = feature_sets.get('lr_only_features', [])
        self._hr_exo_features = feature_sets.get('hr_exo_features', [])
        self.sample_shape = sample_shape or (10, 10, 1)
        self.batch_size = batch_size
        self.lr_features = self.features
        self.preflight()

    def get_sample_index(self, n_obs=None):
        """Randomly gets spatiotemporal sample index.

        Note
        ----
        If n_obs > 1 this will
        get a time slice with n_obs * self.sample_shape[2] time steps, which
        will then be reshaped into n_obs samples each with self.sample_shape[2]
        time steps. This is a much more efficient way of getting batches of
        samples but only works if there are enough continuous time steps to
        sample.

        Returns
        -------
        sample_index : tuple
            Tuple of latitude slice, longitude slice, time slice, and features.
            Used to get single observation like self.data[sample_index]
        """
        n_obs = n_obs or self.batch_size
        spatial_slice = uniform_box_sampler(self.shape, self.sample_shape[:2])
        time_slice = uniform_time_sampler(
            self.shape, self.sample_shape[2] * n_obs
        )
        return (*spatial_slice, time_slice, self.features)

    def preflight(self):
        """Check if the sample_shape is larger than the requested raster
        size"""
        good_shape = (
            self.sample_shape[0] <= self.data.shape[0]
            and self.sample_shape[1] <= self.data.shape[1]
        )
        msg = (
            f'spatial_sample_shape {self.sample_shape[:2]} is '
            f'larger than the raster size {self.data.shape[:2]}'
        )
        assert good_shape, msg

        msg = (
            f'sample_shape[2] ({self.sample_shape[2]}) cannot be larger '
            'than the number of time steps in the raw data '
            f'({self.data.shape[2]}).'
        )

        assert self.data.shape[2] >= self.sample_shape[2], msg

        msg = (
            f'sample_shape[2] * batch_size ({self.sample_shape[2]} * '
            f'{self.batch_size}) is larger than the number of time steps in '
            'the raw data. This prevents us from building batches from '
            'a single sample with n_time_steps = sample_shape[2] * batch_size '
            'which is far more performant than building batches n_samples = '
            'batch_size, each with n_time_steps = sample_shape[2].')
        if self.data.shape[2] < self.sample_shape[2] * self.batch_size:
            logger.warning(msg)
            warn(msg)

    @property
    def sample_shape(self) -> Tuple:
        """Shape of the data sample to select when `__next__()` is called."""
        return self._sample_shape

    @sample_shape.setter
    def sample_shape(self, sample_shape):
        """Set the shape of the data sample to select when `__next__()` is
        called."""
        self._sample_shape = sample_shape
        if len(self._sample_shape) == 2:
            logger.info(
                'Found 2D sample shape of {}. Adding temporal dim of 1'.format(
                    self._sample_shape
                )
            )
            self._sample_shape = (*self._sample_shape, 1)

    @property
    def hr_sample_shape(self) -> Tuple:
        """Shape of the data sample to select when `__next__()` is called. Same
        as sample_shape"""
        return self._sample_shape

    @hr_sample_shape.setter
    def hr_sample_shape(self, hr_sample_shape):
        """Set the sample shape to select when `__next__()` is called. Same
        as sample_shape"""
        self._sample_shape = hr_sample_shape

    def _reshape_samples(self, samples):
        """Reshape samples into batch shapes, with shape = (batch_size,
        *sample_shape, n_features). Samples start out with a time dimension of
        shape = batch_size * sample_shape[2] so we need to split this and
        reorder the dimensions."""
        new_shape = list(samples.shape)
        new_shape = [
            *new_shape[:2],
            self.batch_size,
            new_shape[2] // self.batch_size,
            new_shape[-1],
        ]
        out = samples.reshape(new_shape)
        return compute_if_dask(out.transpose((2, 0, 1, 3, 4)))

    def _stack_samples(self, samples):
        if isinstance(samples[0], tuple):
            lr = da.stack([s[0] for s in samples], axis=0)
            hr = da.stack([s[1] for s in samples], axis=0)
            return (lr, hr)
        return da.stack(samples, axis=0)

    def _fast_batch(self):
        """Get batch of samples with adjacent time slices."""
        out = self.data.sample(
            self.get_sample_index(n_obs=self.batch_size)
        )
        if isinstance(out, tuple):
            return tuple(self._reshape_samples(o) for o in out)
        return self._reshape_samples(out)

    def _slow_batch(self):
        """Get batch of samples with random time slices."""
        samples = [
            self.data.sample(self.get_sample_index(n_obs=1))
            for _ in range(self.batch_size)
        ]
        return self._stack_samples(samples)

    def _fast_batch_possible(self):
        return self.batch_size * self.sample_shape[2] <= self.data.shape[2]

    def __next__(self) -> Union[T_Array, Tuple[T_Array, T_Array]]:
        """Get next batch of samples. This retrieves n_samples = batch_size
        with shape = sample_shape from the `.data` (a xr.Dataset or
        Sup3rDataset) through the Sup3rX accessor."""
        if self._fast_batch_possible():
            return self._fast_batch()
        return self._slow_batch()

    def __iter__(self):
        self._counter = 0
        return self

    def __len__(self):
        return self._size

    def _parse_features(self, unparsed_feats):
        """Return a list of parsed feature names without wildcards."""
        if isinstance(unparsed_feats, str):
            parsed_feats = [unparsed_feats]
        elif isinstance(unparsed_feats, tuple):
            parsed_feats = list(unparsed_feats)
        elif unparsed_feats is None:
            parsed_feats = []
        else:
            parsed_feats = unparsed_feats

        if any('*' in fn for fn in parsed_feats):
            out = []
            for feature in self.features:
                match = any(
                    fnmatch(feature.lower(), pattern.lower())
                    for pattern in parsed_feats
                )
                if match:
                    out.append(feature)
            parsed_feats = out
        return lowered(parsed_feats)

    @property
    def lr_only_features(self):
        """List of feature names or patt*erns that should only be included in
        the low-res training set and not the high-res observations."""
        return self._parse_features(self._lr_only_features)

    @property
    def hr_exo_features(self):
        """Get a list of exogenous high-resolution features that are only used
        for training e.g., mid-network high-res topo injection. These must come
        at the end of the high-res feature set. These can also be input to the
        model as low-res features."""
        self._hr_exo_features = self._parse_features(self._hr_exo_features)

        if len(self._hr_exo_features) > 0:
            msg = (
                f'High-res train-only features "{self._hr_exo_features}" '
                f'do not come at the end of the full high-res feature set: '
                f'{self.features}'
            )
            last_feat = self.features[-len(self._hr_exo_features) :]
            assert list(self._hr_exo_features) == list(last_feat), msg

        return self._hr_exo_features

    @property
    def hr_out_features(self):
        """Get a list of high-resolution features that are intended to be
        output by the GAN. Does not include high-resolution exogenous
        features"""

        out = []
        for feature in self.features:
            lr_only = any(
                fnmatch(feature.lower(), pattern.lower())
                for pattern in self.lr_only_features
            )
            ignore = lr_only or feature in self.hr_exo_features
            if not ignore:
                out.append(feature)

        if len(out) == 0:
            msg = (
                f'It appears that all handler features "{self.features}" '
                'were specified as `hr_exo_features` or `lr_only_features` '
                'and therefore there are no output features!'
            )
            logger.error(msg)
            raise RuntimeError(msg)

        return lowered(out)

    @property
    def hr_features_ind(self):
        """Get the high-resolution feature channel indices that should be
        included for training. Any high-resolution features that are only
        included in the data handler to be coarsened for the low-res input are
        removed"""
        hr_features = list(self.hr_out_features) + list(self.hr_exo_features)
        if list(self.features) == hr_features:
            return np.arange(len(self.features))
        return [
            i
            for i, feature in enumerate(self.features)
            if feature in hr_features
        ]

    @property
    def hr_features(self):
        """Get the high-resolution features corresponding to
        `hr_features_ind`"""
        return [self.features[ind].lower() for ind in self.hr_features_ind]
