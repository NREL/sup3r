"""
Sup3r batch_handling module.
@author: bbenton
"""

import logging
from typing import Dict, List, Optional, Union

from sup3r.preprocessing.base import (
    Container,
)
from sup3r.preprocessing.batch_queues.base import SingleBatchQueue
from sup3r.preprocessing.batch_queues.conditional import (
    QueueMom1,
    QueueMom1SF,
    QueueMom2,
    QueueMom2Sep,
    QueueMom2SepSF,
    QueueMom2SF,
)
from sup3r.preprocessing.batch_queues.dc import BatchQueueDC, ValBatchQueueDC
from sup3r.preprocessing.batch_queues.dual import DualBatchQueue
from sup3r.preprocessing.collections.stats import StatsCollection
from sup3r.preprocessing.samplers.base import Sampler
from sup3r.preprocessing.samplers.cc import DualSamplerCC
from sup3r.preprocessing.samplers.dc import DataCentricSampler
from sup3r.preprocessing.samplers.dual import DualSampler
from sup3r.preprocessing.utilities import FactoryMeta, get_class_kwargs

logger = logging.getLogger(__name__)


def BatchHandlerFactory(
    MainQueueClass, SamplerClass, ValQueueClass=None, name='BatchHandler'
):
    """BatchHandler factory. Can build handlers from different queue classes
    and sampler classes. For example, to build a standard BatchHandler use
    :class:`BatchQueue` and :class:`Sampler`. To build a
    :class:`DualBatchHandler` use :class:`DualBatchQueue` and
    :class:`DualSampler`. To build a BatchHandlerCC use a
    :class:`BatchQueueDC`, :class:`ValBatchQueueDC` and
    :class:`DataCentricSampler`

    Note
    ----
    (1) BatchHandlers include a queue for training samples and a queue for
    validation samples.
    (2) There is no need to generate "Spatial" batch handlers. Using
    :class:`Sampler` objects with a single time step in the sample shape will
    produce batches without a time dimension.
    """

    class BatchHandler(MainQueueClass, metaclass=FactoryMeta):
        """BatchHandler object built from two lists of class:`Container`
        objects, one with training data and one with validation data. These
        lists will be used to initialize lists of class:`Sampler` objects that
        will then be used to build batches at run time.

        Note
        ----
        These lists of containers can contain data from the same underlying
        data source (e.g. CONUS WTK) (e.g. initialize train / val containers
        with different time period and / or regions.  , or they can be used to
        sample from completely different data sources (e.g. train on CONUS WTK
        while validating on Canada WTK).

        See Also
        --------
        :class:`Sampler` and :class:`AbstractBatchQueue` for a description of
        arguments
        """

        VAL_QUEUE = MainQueueClass if ValQueueClass is None else ValQueueClass
        SAMPLER = SamplerClass

        __name__ = name

        def __init__(
            self,
            train_containers: List[Container],
            val_containers: Optional[List[Container]] = None,
            batch_size: Optional[int] = 16,
            n_batches: Optional[int] = 64,
            s_enhance=1,
            t_enhance=1,
            means: Optional[Union[Dict, str]] = None,
            stds: Optional[Union[Dict, str]] = None,
            **kwargs,
        ):
            [sampler_kwargs, main_queue_kwargs, val_queue_kwargs] = (
                get_class_kwargs(
                    [SamplerClass, MainQueueClass, self.VAL_QUEUE],
                    {'s_enhance': s_enhance, 't_enhance': t_enhance, **kwargs},
                )
            )

            train_samplers, val_samplers = self.init_samplers(
                train_containers, val_containers, sampler_kwargs
            )

            stats = StatsCollection(
                train_samplers + val_samplers,
                means=means,
                stds=stds,
            )

            if not val_samplers:
                self.val_data: Union[List, SingleBatchQueue] = []
            else:
                self.val_data = self.VAL_QUEUE(
                    samplers=val_samplers,
                    batch_size=batch_size,
                    n_batches=n_batches,
                    means=stats.means,
                    stds=stats.stds,
                    thread_name='validation',
                    **val_queue_kwargs,
                )

            super().__init__(
                samplers=train_samplers,
                batch_size=batch_size,
                n_batches=n_batches,
                means=stats.means,
                stds=stats.stds,
                **main_queue_kwargs,
            )

        def init_samplers(
            self, train_containers, val_containers, sampler_kwargs
        ):
            """Initialize samplers from given data containers."""
            train_samplers = [
                self.SAMPLER(c.data, **sampler_kwargs)
                for c in train_containers
            ]

            val_samplers = (
                []
                if val_containers is None
                else [
                    self.SAMPLER(c.data, **sampler_kwargs)
                    for c in val_containers
                ]
            )
            return train_samplers, val_samplers

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

    return BatchHandler


BatchHandler = BatchHandlerFactory(
    SingleBatchQueue, Sampler, name='BatchHandler'
)
DualBatchHandler = BatchHandlerFactory(
    DualBatchQueue, DualSampler, name='DualBatchHandler'
)
BatchHandlerCC = BatchHandlerFactory(
    DualBatchQueue, DualSamplerCC, name='BatchHandlerCC'
)
BatchHandlerMom1 = BatchHandlerFactory(
    QueueMom1, Sampler, name='BatchHandlerMom1'
)
BatchHandlerMom1SF = BatchHandlerFactory(
    QueueMom1SF, Sampler, name='BatchHandlerMom1SF'
)
BatchHandlerMom2 = BatchHandlerFactory(
    QueueMom2, Sampler, name='BatchHandlerMom2'
)
BatchHandlerMom2Sep = BatchHandlerFactory(
    QueueMom2Sep, Sampler, name='BatchHandlerMom2Sep'
)
BatchHandlerMom2SF = BatchHandlerFactory(
    QueueMom2SF, Sampler, name='BatchHandlerMom2F'
)
BatchHandlerMom2SepSF = BatchHandlerFactory(
    QueueMom2SepSF, Sampler, name='BatchHandlerMom2SepSF'
)

BatchHandlerDC = BatchHandlerFactory(
    BatchQueueDC, DataCentricSampler, ValBatchQueueDC, name='BatchHandlerDC'
)
