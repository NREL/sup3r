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
from sup3r.preprocessing.batch_queues.dual import DualBatchQueue
from sup3r.preprocessing.collections.stats import StatsCollection
from sup3r.preprocessing.common import FactoryMeta
from sup3r.preprocessing.samplers.base import Sampler
from sup3r.preprocessing.samplers.cc import DualSamplerCC
from sup3r.preprocessing.samplers.dual import DualSampler
from sup3r.utilities.utilities import get_class_kwargs

logger = logging.getLogger(__name__)


def BatchHandlerFactory(QueueClass, SamplerClass, name='BatchHandler'):
    """BatchHandler factory. Can build handlers from different queue classes
    and sampler classes. For example, to build a standard BatchHandler use
    :class:`BatchQueue` and :class:`Sampler`. To build a
    :class:`DualBatchHandler` use :class:`DualBatchQueue` and
    :class:`DualSampler`.

    Notes
    -----
    There is no need to generate "Spatial" batch handlers. Using
    :class:`Sampler` objects with a single time step in the sample shape will
    produce batches without a time dimension.
    """

    class BatchHandler(QueueClass, metaclass=FactoryMeta):
        """BatchHandler object built from two lists of class:`Container`
        objects, one with training data and one with validation data. These
        lists will be used to initialize lists of class:`Sampler` objects that
        will then be used to build batches at run time.

        Notes
        -----
        These lists of containers can contain data from the same underlying
        data source (e.g. CONUS WTK) (e.g. initialize train / val containers
        with different time period and / or regions.  , or they can be used to
        sample from completely different data sources (e.g. train on CONUS WTK
        while validating on Canada WTK).

        `.start()` is called upon initialization. Maybe should remove this and
        require manual start.
        """

        SAMPLER = SamplerClass

        __name__ = name

        def __init__(
            self,
            train_containers: List[Container],
            val_containers: List[Container],
            batch_size,
            n_batches,
            s_enhance=1,
            t_enhance=1,
            means: Optional[Union[Dict, str]] = None,
            stds: Optional[Union[Dict, str]] = None,
            **kwargs,
        ):
            sampler_kwargs = get_class_kwargs(
                SamplerClass,
                {'s_enhance': s_enhance, 't_enhance': t_enhance, **kwargs},
            )
            queue_kwargs = get_class_kwargs(QueueClass, kwargs)

            train_samplers = [
                self.SAMPLER(c, **sampler_kwargs) for c in train_containers
            ]

            val_samplers = (
                None
                if val_containers is None
                else [
                    self.SAMPLER(c, **sampler_kwargs) for c in val_containers
                ]
            )

            stats = StatsCollection(
                [*train_containers, *val_containers],
                means=means,
                stds=stds,
            )

            if not val_samplers:
                self.val_data: Union[List, SingleBatchQueue] = []
            else:
                self.val_data = QueueClass(
                    samplers=val_samplers,
                    batch_size=batch_size,
                    n_batches=n_batches,
                    s_enhance=s_enhance,
                    t_enhance=t_enhance,
                    means=stats.means,
                    stds=stats.stds,
                    thread_name='validation',
                    **queue_kwargs,
                )

            super().__init__(
                samplers=train_samplers,
                batch_size=batch_size,
                n_batches=n_batches,
                s_enhance=s_enhance,
                t_enhance=t_enhance,
                means=stats.means,
                stds=stats.stds,
                **queue_kwargs,
            )
            self.start()

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

DualBatchHandlerCC = BatchHandlerFactory(
    DualBatchQueue, DualSamplerCC, name='DualBatchHandlerCC'
)
