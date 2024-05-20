"""
Sup3r batch_handling module.
@author: bbenton
"""

import logging
from typing import List, Union

import numpy as np

from sup3r.containers.base import (
    Container,
    DualContainer,
)
from sup3r.containers.batchers.base import BatchQueue
from sup3r.containers.batchers.dual import DualBatchQueue
from sup3r.containers.samplers.base import Sampler
from sup3r.containers.samplers.dual import DualSampler
from sup3r.utilities.utilities import _get_class_kwargs

np.random.seed(42)

logger = logging.getLogger(__name__)


def handler_factory(QueueClass, SamplerClass):
    """BatchHandler factory. Can build handlers from different queue classes
    and sampler classes. For example, to build a standard BatchHandler use
    :class:`BatchQueue` and :class:`Sampler`. To build a
    :class:`DualBatchHandler` use :class:`DualBatchQueue` and
    :class:`DualSampler`.
    """

    class Handler(QueueClass):
        """BatchHandler object built from two lists of class:`Container`
        objects, one with training data and one with validation data. These
        lists will be used to initialize lists of class:`Sampler` objects that
        will then be used to build batches at run time.

        Notes
        -----
        These lists of containers can contain data from the same underlying
        data source (e.g. CONUS WTK) (by using `CroppedSampler(...,
        crop_slice=crop_slice)` with `crop_slice` selecting different time
        periods to prevent cross-contamination), or they can be used to sample
        from completely different data sources (e.g. train on CONUS WTK while
        validating on Canada WTK)."""

        SAMPLER = SamplerClass

        def __init__(
            self,
            train_containers: Union[List[Container], List[DualContainer]],
            val_containers: Union[List[Container], List[DualContainer]],
            **kwargs,
        ):
            sampler_kwargs = _get_class_kwargs(SamplerClass, kwargs)
            queue_kwargs = _get_class_kwargs(QueueClass, kwargs)

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
            super().__init__(
                train_containers=train_samplers,
                val_containers=val_samplers,
                **queue_kwargs,
            )
    return Handler


BatchHandler = handler_factory(BatchQueue, Sampler)
DualBatchHandler = handler_factory(DualBatchQueue, DualSampler)
