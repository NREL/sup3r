"""BatchQueue objects with train and testing collections."""

import logging
from typing import Dict, List, Optional, Union

from sup3r.containers.batchers.base import BatchQueue
from sup3r.containers.samplers.base import Sampler

logger = logging.getLogger(__name__)


class BatchQueueWithValidation(BatchQueue):
    """BatchQueue object built from list of samplers containing training data
    and a list of samplers containing validation data.

    Notes
    -----
    These lists of samplers can sample from the same underlying data source
    (e.g. CONUS WTK) (by using `CroppedSampler(..., crop_slice=crop_slice)`
    with `crop_slice` selecting different time periods to prevent
    cross-contamination), or they can sample from completely different data
    sources (e.g. train on CONUS WTK while validating on Canada WTK)."""

    def __init__(
        self,
        train_containers: List[Sampler],
        val_containers: List[Sampler],
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
        train_containers : List[Sampler]
            List of Sampler instances containing training data
        val_containers : List[Sampler]
            List of Sampler instances containing validation data
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
            default_device=default_device
        )
        self.val_data = BatchQueue(
            containers=val_containers,
            batch_size=batch_size,
            n_batches=n_batches,
            s_enhance=s_enhance,
            t_enhance=t_enhance,
            means=means,
            stds=stds,
            queue_cap=queue_cap,
            max_workers=max_workers,
            coarsen_kwargs=coarsen_kwargs,
            default_device=default_device
        )
        self.val_data.queue._name = 'validation'

    def start(self):
        """Start the val data batch queue in addition to the train batch
        queue."""
        self.val_data.start()
        super().start()

    def stop(self):
        """Stop the val data batch queue in addition to the train batch
        queue."""
        self.val_data.stop()
        super().stop()
