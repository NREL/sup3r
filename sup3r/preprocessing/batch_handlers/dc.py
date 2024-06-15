"""Data centric batch handlers. Sample contained data according to
spatiotemporal weights, which are derived from losses on validation data during
training and updated each epoch.

TODO: Easy to implement dual dc batch handler - Just need to use DualBatchQueue
and override SamplerDC get_sample_index method.
"""

import logging

from sup3r.preprocessing.batch_handlers.factory import BatchHandlerFactory
from sup3r.preprocessing.batch_queues.dc import BatchQueueDC, ValBatchQueueDC
from sup3r.preprocessing.samplers.dc import SamplerDC

logger = logging.getLogger(__name__)


BaseBatchHandlerDC = BatchHandlerFactory(
    BatchQueueDC, SamplerDC, ValBatchQueueDC, name='BatchHandlerDC'
)


class BatchHandlerDC(BaseBatchHandlerDC):
    """Add validation data requirement. Makes no sense to use this handler
    without validation data."""

    def __init__(self, train_containers, val_containers, *args, **kwargs):
        msg = (
            f'{self.__class__.__name__} requires validation data. If you '
            'do not plan to sample training data based on performance '
            'across validation data use another type of batch handler.'
        )
        assert val_containers is not None and val_containers != [], msg
        super().__init__(train_containers, val_containers, *args, **kwargs)
        max_space_bins = (self.data[0].shape[0] - self.sample_shape[0] + 2) * (
            self.data[0].shape[1] - self.sample_shape[1] + 2
        )
        max_time_bins = self.data[0].shape[2] - self.sample_shape[2] + 2
        msg = (
            f'The requested sample_shape {self.sample_shape} is too large '
            f'for the requested number of bins (space = {self.n_space_bins}, '
            f'time = {self.n_time_bins}) and the shape of the sample data '
            f'{self.data[0].shape[:3]}.'
        )
        assert self.n_space_bins <= max_space_bins, msg
        assert self.n_time_bins <= max_time_bins, msg
