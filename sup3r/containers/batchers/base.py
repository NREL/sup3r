"""Base objects which generate, build, and operate on batches. Also can
interface with models."""

import logging
from typing import Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf

from sup3r.containers.batchers.abstract import (
    AbstractNormedBatchQueue,
)
from sup3r.containers.samplers import Sampler
from sup3r.utilities.utilities import (
    smooth_data,
    spatial_coarsening,
    temporal_coarsening,
)

logger = logging.getLogger(__name__)


AUTO = tf.data.experimental.AUTOTUNE
option_no_order = tf.data.Options()
option_no_order.experimental_deterministic = False

option_no_order.experimental_optimization.noop_elimination = True
option_no_order.experimental_optimization.apply_default_optimizations = True


class BatchQueue(AbstractNormedBatchQueue):
    """Base BatchQueue class for single data object containers."""

    def __init__(
        self,
        containers: List[Sampler],
        s_enhance,
        t_enhance,
        batch_size,
        n_batches,
        queue_cap,
        means: Union[Dict, str],
        stds: Union[Dict, str],
        max_workers=None,
        coarsen_kwargs=None,
    ):
        """
        Parameters
        ----------
        containers : List[Sampler]
            List of Sampler instances
        batch_size : int
            Number of observations / samples in a batch
        n_batches : int
            Number of batches in an epoch, this sets the iteration limit for
            this object.
        queue_cap : int
            Maximum number of batches the batch queue can store.
        means : Union[Dict, str]
            Either a .json path containing a dictionary or a dictionary of
            means which will be used to normalize batches as they are built.
        stds : Union[Dict, str]
            Either a .json path containing a dictionary or a dictionary of
            standard deviations which will be used to normalize batches as they
            are built.
        s_enhance : int
            Integer factor by which the spatial axes is to be enhanced.
        t_enhance : int
            Integer factor by which the temporal axes is to be enhanced.
        max_workers : int
            Number of workers / threads to use for getting samples used to
            build batches.
        coarsen_kwargs : Union[Dict, None]
            Dictionary of kwargs to be passed to `self.coarsen`.
        """
        super().__init__(
            containers,
            s_enhance,
            t_enhance,
            batch_size,
            n_batches,
            queue_cap,
            means,
            stds,
            max_workers,
        )
        self.coarsen_kwargs = coarsen_kwargs
        logger.info(
            f'Initialized {self.__class__.__name__} with '
            f'{len(self.containers)} samplers, s_enhance = {self.s_enhance}, '
            f't_enhance = {self.t_enhance}, batch_size = {self.batch_size}, '
            f'n_batches = {self.n_batches}, queue_cap = {self.queue_cap}, '
            f'means = {self.means}, stds = {self.stds}, '
            f'max_workers = {self.max_workers}, '
            f'coarsen_kwargs = {self.coarsen_kwargs}.')

    def get_output_signature(self):
        """Get tensorflow dataset output signature for single data object
        containers."""

        output_signature = tf.TensorSpec(
            (*self.sample_shape, len(self.features)),
            tf.float32,
            name='high_res',
        )
        return output_signature

    def batch_next(self, samples):
        """Returns wrapped collection of samples / observations."""
        lr, hr = self.coarsen(high_res=samples, **self.coarsen_kwargs)
        return self.BATCH_CLASS(
            low_res=lr, high_res=hr)

    def normalize(
        self, samples
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Normalize a low-res / high-res pair with the stored means and
        stdevs."""
        means = np.array([self.means[k] for k in self.features])
        stds = np.array([self.stds[k] for k in self.features])
        return self._normalize(samples, means, stds)

    def coarsen(
        self,
        high_res,
        smoothing=None,
        smoothing_ignore=None,
        temporal_coarsening_method='subsample',
    ):
        """Coarsen high res data to get corresponding low res batch.

        Parameters
        ----------
        high_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        smoothing : float | None
            Standard deviation to use for gaussian filtering of the coarse
            data. This can be tuned by matching the kinetic energy of a low
            resolution simulation with the kinetic energy of a coarsened and
            smoothed high resolution simulation. If None no smoothing is
            performed.
        smoothing_ignore : list | None
            List of features to ignore for the smoothing filter. None will
            smooth all features if smoothing kwarg is not None
        temporal_coarsening_method : str
            Method to use for temporal coarsening. Can be subsample, average,
            min, max, or total

        Returns
        -------
        low_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        """
        low_res = spatial_coarsening(high_res, self.s_enhance)
        low_res = (
            low_res
            if self.t_enhance == 1
            else temporal_coarsening(
                low_res, self.t_enhance, temporal_coarsening_method
            )
        )
        smoothing_ignore = (
            smoothing_ignore if smoothing_ignore is not None else []
        )
        low_res = smooth_data(
            low_res, self.features, smoothing_ignore, smoothing
        )
        high_res = high_res.numpy()[..., self.hr_features_ind]
        return low_res, high_res


class PairBatchQueue(AbstractNormedBatchQueue):
    """Base BatchQueue for SamplerPair containers."""

    def __init__(
        self,
        containers: List[Sampler],
        s_enhance,
        t_enhance,
        batch_size,
        n_batches,
        queue_cap,
        means: Union[Dict, str],
        stds: Union[Dict, str],
        max_workers=None,
    ):
        super().__init__(
            containers,
            s_enhance,
            t_enhance,
            batch_size,
            n_batches,
            queue_cap,
            means,
            stds,
            max_workers,
        )
        self.check_for_consistent_enhancement_factors()

        logger.info(
            f'Initialized {self.__class__.__name__} with '
            f'{len(self.containers)} samplers, s_enhance = {self.s_enhance}, '
            f't_enhance = {self.t_enhance}, batch_size = {self.batch_size}, '
            f'n_batches = {self.n_batches}, queue_cap = {self.queue_cap}, '
            f'means = {self.means}, stds = {self.stds}, '
            f'max_workers = {self.max_workers}.'
        )

    def check_for_consistent_enhancement_factors(self):
        """Make sure each SamplerPair has the same enhancment factors and that
        they match those provided to the BatchQueue."""
        s_factors = [c.s_enhance for c in self.containers]
        msg = (f'Recived s_enhance = {self.s_enhance} but not all '
               f'SamplerPairs in the collection have the same value.')
        assert all(self.s_enhance == s for s in s_factors), msg
        t_factors = [c.t_enhance for c in self.containers]
        msg = (f'Recived t_enhance = {self.t_enhance} but not all '
               f'SamplerPairs in the collection have the same value.')
        assert all(self.t_enhance == t for t in t_factors), msg

    def get_output_signature(self):
        """Get tensorflow dataset output signature. If we are sampling from
        container pairs then this is a tuple for low / high res batches.
        Otherwise we are just getting high res batches and coarsening to get
        the corresponding low res batches."""
        return (
            tf.TensorSpec(self.lr_shape, tf.float32, name='low_res'),
            tf.TensorSpec(self.hr_shape, tf.float32, name='high_res'),
        )

    def batch_next(self, samples):
        """Returns wrapped collection of samples / observations."""
        low_res, high_res = samples
        batch = self.BATCH_CLASS(low_res=low_res, high_res=high_res)
        return batch

    def normalize(
        self, samples
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Normalize a low-res / high-res pair with the stored means and
        stdevs."""
        lr, hr = samples
        out = (
            self._normalize(lr, self.lr_means, self.lr_stds),
            self._normalize(hr, self.hr_means, self.hr_stds),
        )
        return out

    @property
    def lr_means(self):
        """Means specific the low-res objects in the ContainerPairs."""
        return np.array([self.means[k] for k in self.lr_features])

    @property
    def hr_means(self):
        """Means specific the high-res objects in the ContainerPairs."""
        return np.array([self.means[k] for k in self.hr_features])

    @property
    def lr_stds(self):
        """Stdevs specific the low-res objects in the ContainerPairs."""
        return np.array([self.stds[k] for k in self.lr_features])

    @property
    def hr_stds(self):
        """Stdevs specific the high-res objects in the ContainerPairs."""
        return np.array([self.stds[k] for k in self.hr_features])
