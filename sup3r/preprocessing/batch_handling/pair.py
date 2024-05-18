"""
Sup3r batch_handling module.
@author: bbenton
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np

from sup3r.containers import ContainerPair, PairBatchQueue, SamplerPair

np.random.seed(42)

logger = logging.getLogger(__name__)


class PairBatchHandler(PairBatchQueue):
    """Same as BatchHandler but using  :class:`ContainerPair` objects instead of
     :class:`Container` objects. The former are pairs of low / high res data
    instead of just high-res data that will be coarsened to create
    corresponding low-res samples. This means `coarsen_kwargs` is not an input
    here either."""

    SAMPLER = SamplerPair

    def __init__(
        self,
        train_containers: List[ContainerPair],
        batch_size,
        n_batches,
        s_enhance,
        t_enhance,
        means: Union[Dict, str],
        stds: Union[Dict, str],
        sample_shape,
        feature_sets,
        val_containers: Optional[List[ContainerPair]] = None,
        queue_cap: Optional[int] = None,
        max_workers: Optional[int] = None,
        default_device: Optional[str] = None):

        train_samplers = [
            self.SAMPLER(
                c,
                sample_shape,
                s_enhance=s_enhance,
                t_enhance=t_enhance,
                feature_sets=feature_sets,
            )
            for c in train_containers
        ]

        val_samplers = (
            None
            if val_containers is None
            else [
                self.SAMPLER(
                    c,
                    sample_shape,
                    s_enhance=s_enhance,
                    t_enhance=t_enhance,
                    feature_sets=feature_sets,
                )
                for c in val_containers
            ]
        )
        super().__init__(
            train_containers=train_samplers,
            batch_size=batch_size,
            n_batches=n_batches,
            s_enhance=s_enhance,
            t_enhance=t_enhance,
            means=means,
            stds=stds,
            val_containers=val_samplers,
            queue_cap=queue_cap,
            max_workers=max_workers,
            default_device=default_device,
        )
