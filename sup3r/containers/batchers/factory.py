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

    class BatchHandler(QueueClass):
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

        __name__ = name

        def __init__(
            self,
            train_samplers: Union[List[Container], List[DualContainer]],
            val_samplers: Union[List[Container], List[DualContainer]],
            **kwargs,
        ):
            sampler_kwargs = _get_class_kwargs(SamplerClass, kwargs)
            queue_kwargs = _get_class_kwargs(QueueClass, kwargs)

            train_samplers = [
                self.SAMPLER(c, **sampler_kwargs) for c in train_samplers
            ]

            val_samplers = (
                None
                if val_samplers is None
                else [
                    self.SAMPLER(c, **sampler_kwargs) for c in val_samplers
                ]
            )
            super().__init__(
                train_samplers=train_samplers,
                val_samplers=val_samplers,
                **queue_kwargs,
            )

    return BatchHandler


BatchHandler = BatchHandlerFactory(BatchQueue, Sampler, name='BatchHandler')
DualBatchHandler = BatchHandlerFactory(
    DualBatchQueue, DualSampler, name='DualBatchHandler'
)
