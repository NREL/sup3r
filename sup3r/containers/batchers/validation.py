"""BatchQueue objects with train and testing collections."""

import logging
from typing import Dict, List, Optional, Union

from sup3r.containers.batchers.base import BatchQueue
from sup3r.containers.samplers.cropped import CroppedSampler

logger = logging.getLogger(__name__)


class BatchQueueWithValidation(BatchQueue):
    """BatchQueue object built from list of samplers containing training data
    and a list of samplers containing validation data. These list of samplers
    can sample from the same underlying data source (by using `CropSampler(...,
    crop_slice=crop_slice)` with `crop_slice` selecting different time periods
    to prevent cross-contamination), or they can sample from completely
    different data sources (e.g. train on CONUS while validating on
    Ukraine)."""

    def __init__(
        self,
        train_containers: List[CroppedSampler],
        val_containers: List[CroppedSampler],
        batch_size,
        n_batches,
        s_enhance,
        t_enhance,
        means: Union[Dict, str],
        stds: Union[Dict, str],
        queue_cap: Optional[int] = None,
        max_workers: Optional[int] = None,
        coarsen_kwargs: Optional[Dict] = None,
    ):
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
        )
        self.val_data.queue._name = 'validation'

    def start(self):
        """Start the test batch queue in addition to the train batch queue."""
        self.val_data.start()
        super().start()

    def stop(self):
        """Stop the test batch queue in addition to the train batch queue."""
        self.val_data.stop()
        super().stop()
