"""Base objects which generate, build, and operate on batches. Also can
interface with models."""

import logging
from typing import Dict, List, Optional, Union

import tensorflow as tf

from sup3r.containers.batchers.abstract import (
    AbstractBatchQueue,
)
from sup3r.containers.samplers import Sampler
from sup3r.containers.samplers.pair import SamplerPair
from sup3r.utilities.utilities import (
    smooth_data,
    spatial_coarsening,
    temporal_coarsening,
)

logger = logging.getLogger(__name__)


option_no_order = tf.data.Options()
option_no_order.experimental_deterministic = False

option_no_order.experimental_optimization.noop_elimination = True
option_no_order.experimental_optimization.apply_default_optimizations = True


class SingleBatchQueue(AbstractBatchQueue):
    """Base BatchQueue class for single data object containers with no
    validation queue."""

    def __init__(
        self,
        containers: Union[List[Sampler], List[SamplerPair]],
        batch_size,
        n_batches,
        s_enhance,
        t_enhance,
        means: Union[Dict, str],
        stds: Union[Dict, str],
        queue_cap: Optional[int] = None,
        max_workers: Optional[int] = None,
        coarsen_kwargs: Optional[Dict] = None,
        default_device: Optional[str] = None,
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
        queue_cap : int
            Maximum number of batches the batch queue can store.
        max_workers : int
            Number of workers / threads to use for getting samples used to
            build batches.
        coarsen_kwargs : Union[Dict, None]
            Dictionary of kwargs to be passed to `self.coarsen`.
        default_device : str
            Default device to use for batch queue (e.g. /cpu:0, /gpu:0). If
            None this will use the first GPU if GPUs are available otherwise
            the CPU.
        """
        super().__init__(
            containers=containers,
            batch_size=batch_size,
            n_batches=n_batches,
            s_enhance=s_enhance,
            t_enhance=t_enhance,
            means=means,
            stds=stds,
            queue_cap=queue_cap,
            max_workers=max_workers,
            default_device=default_device,
        )
        self.coarsen_kwargs = coarsen_kwargs or {
            'smoothing_ignore': [],
            'smoothing': None,
        }

    def get_output_signature(self):
        """Get tensorflow dataset output signature for single data object
        containers."""
        return tf.TensorSpec(
            (*self.sample_shape, len(self.features)),
            tf.float32,
            name='high_res',
        )

    def batch_next(self, samples):
        """Coarsens high res samples, normalizes low / high res and returns
        wrapped collection of samples / observations."""
        lr, hr = self.coarsen(high_res=samples, **self.coarsen_kwargs)
        lr, hr = self.normalize(lr, hr)
        return self.BATCH_CLASS(low_res=lr, high_res=hr)

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


class BatchQueue(SingleBatchQueue):
    """BatchQueue object built from list of samplers containing training data
    and an optional list of samplers containing validation data.

    Notes
    -----
    These lists of samplers can sample from the same underlying data source
    (e.g. CONUS WTK) (by using `CroppedSampler(..., crop_slice=crop_slice)`
    with `crop_slice` selecting different time periods to prevent
    cross-contamination), or they can sample from completely different data
    sources (e.g. train on CONUS WTK while validating on Canada WTK)."""

    def __init__(
        self,
        train_containers: Union[List[Sampler], List[SamplerPair]],
        batch_size,
        n_batches,
        s_enhance,
        t_enhance,
        means: Union[Dict, str],
        stds: Union[Dict, str],
        val_containers: Optional[
            Union[List[Sampler], List[SamplerPair]]
        ] = None,
        queue_cap: Optional[int] = None,
        max_workers: Optional[int] = None,
        coarsen_kwargs: Optional[Dict] = None,
        default_device: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        train_containers : List[Sampler]
            List of Sampler instances containing training data
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
        val_containers : Optional[List[Sampler]]
            Optional list of Sampler instances containing validation data
        queue_cap : int
            Maximum number of batches the batch queue can store.
        max_workers : int
            Number of workers / threads to use for getting samples used to
            build batches.
        coarsen_kwargs : Union[Dict, None]
            Dictionary of kwargs to be passed to `self.coarsen`.
        default_device : str
            Default device to use for batch queue (e.g. /cpu:0, /gpu:0). If
            None this will use the first GPU if GPUs are available otherwise
            the CPU.
        """
        super().__init__(
            containers=train_containers,
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
        self.val_data = (
            []
            if val_containers is None
            else self.init_validation_queue(val_containers)
        )

    def init_validation_queue(self, val_containers):
        """Initialize validation batch queue if validation samplers are
        provided."""
        val_queue = SingleBatchQueue(
            containers=val_containers,
            batch_size=self.batch_size,
            n_batches=self.n_batches,
            s_enhance=self.s_enhance,
            t_enhance=self.t_enhance,
            means=self.means,
            stds=self.stds,
            queue_cap=self.queue_cap,
            max_workers=self.max_workers,
            coarsen_kwargs=self.coarsen_kwargs,
            default_device=self.default_device,
        )
        val_queue.queue._name = 'validation'
        return val_queue

    def start(self):
        """Start the val data batch queue in addition to the train batch
        queue."""
        if hasattr(self.val_data, 'start'):
            self.val_data.start()
        super().start()

    def stop(self):
        """Stop the val data batch queue in addition to the train batch
        queue."""
        if hasattr(self.val_data, 'stop'):
            self.val_data.stop()
        super().stop()
