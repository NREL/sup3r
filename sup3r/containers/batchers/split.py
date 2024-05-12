"""BatchQueue objects with train and testing collections."""

import copy
import logging
from typing import Dict, List, Optional, Tuple, Union

from sup3r.containers.batchers.base import BatchQueue
from sup3r.containers.samplers.cropped import CroppedSampler

logger = logging.getLogger(__name__)


class SplitBatchQueue(BatchQueue):
    """BatchQueue object which contains a BatchQueue for training batches and
    a BatchQueue for validation batches. This takes a val_split value and
    crops the sampling regions for the training queue samplers and the testing
    queue samplers."""

    def __init__(
        self,
        containers: List[CroppedSampler],
        val_split,
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
            containers=containers,
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
            copy.deepcopy(containers),
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
        self.val_split = val_split
        self.update_cropped_samplers()

        logger.info(f'Initialized {self.__class__.__name__} with '
                    f'val_split = {self.val_split}.')

    def get_test_train_slices(self) -> List[Tuple[slice, slice]]:
        """Get time slices consistent with the val_split value for each
        container in the collection

        Returns
        -------
        List[Tuple[slice, slice]]
            List of tuples of slices with the tuples being slices for testing
            and training, respectively
        """
        t_steps = [c.shape[2] for c in self.containers]
        return [
            (
                slice(0, int(self.val_split * t)),
                slice(int(self.val_split * t), t),
            )
            for t in t_steps
        ]

    def start(self):
        """Start the test batch queue in addition to the train batch queue."""
        self.val_data.start()
        super().start()

    def stop(self):
        """Stop the test batch queue in addition to the train batch queue."""
        self.val_data.stop()
        super().stop()

    def update_cropped_samplers(self):
        """Update cropped sampler crop slices so that the sampling regions for
        each collection are restricted according to the given val_split."""
        slices = self.get_test_train_slices()
        for i, (test_slice, train_slice) in enumerate(slices):
            self.containers[i].crop_slice = train_slice
            self.val_data.containers[i].crop_slice = test_slice
