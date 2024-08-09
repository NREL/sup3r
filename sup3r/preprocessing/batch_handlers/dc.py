"""Data centric batch handlers. Sample contained data according to
spatiotemporal weights, which are derived from losses on validation data during
training and updated each epoch.

TODO: Easy to implement dual dc batch handler - Just need to use DualBatchQueue
and override SamplerDC get_sample_index method.
"""

import logging

from ..batch_queues.dc import BatchQueueDC, ValBatchQueueDC
from ..samplers.dc import SamplerDC
from ..utilities import log_args
from .factory import BatchHandlerFactory

logger = logging.getLogger(__name__)


BaseDC = BatchHandlerFactory(
    BatchQueueDC, SamplerDC, ValBatchQueueDC, name='BaseDC'
)


class BatchHandlerDC(BaseDC):
    """Data-Centric BatchHandler. This is used to adaptively select data
    from lower performing spatiotemporal extents during training. To do this,
    validation data is required, as it is used to compute losses within fixed
    spatiotemporal bins which are then used as sampling probabilities for those
    same regions when building batches.

    See Also
    --------
    :class:`~sup3r.preprocessing.batch_queues.dc.BatchQueueDC`,
    :class:`~sup3r.preprocessing.batch_queues.dc.ValBatchQueueDC`,
    :class:`~sup3r.preprocessing.samplers.dc.SamplerDC`,
    :func:`~.factory.BatchHandlerFactory`
    """

    @log_args
    def __init__(self, train_containers, val_containers, **kwargs):
        msg = (
            f'{self.__class__.__name__} requires validation data. If you '
            'do not plan to sample training data based on performance '
            'across validation data use another type of batch handler.'
        )
        assert val_containers is not None and val_containers != [], msg
        super().__init__(
            train_containers=train_containers,
            val_containers=val_containers,
            **kwargs,
        )
        max_space_bins = (self.data[0].shape[0] - self.sample_shape[0] + 1) * (
            self.data[0].shape[1] - self.sample_shape[1] + 1
        )
        max_time_bins = self.data[0].shape[2] - self.sample_shape[2] + 1
        msg = (
            f'The requested sample_shape {self.sample_shape} is too large '
            f'for the requested number of bins (space = {self.n_space_bins}, '
            f'time = {self.n_time_bins}) and the shape of the sample data '
            f'{self.data[0].shape[:3]}.'
        )
        assert self.n_space_bins <= max_space_bins, msg
        assert self.n_time_bins <= max_time_bins, msg

    _signature_objs = (__init__, BaseDC)
    _skip_params = ('samplers', 'data', 'thread_name', 'kwargs')
