"""
Sup3r batch_handling module.
@author: bbenton
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np

from sup3r.containers import (
    BatchQueue,
    Container,
    Sampler,
)

np.random.seed(42)

logger = logging.getLogger(__name__)


class BatchHandler(BatchQueue):
    """BatchHandler object built from two lists of class:`Container` objects,
    one with training data and one with validation data. These lists will be
    used to initialize lists of class:`Sampler` objects that will then be used
    to build batches at run time.

    Notes
    -----
    These lists of containers can contain data from the same underlying data
    source (e.g. CONUS WTK) (by using `CroppedSampler(...,
    crop_slice=crop_slice)` with `crop_slice` selecting different time periods
    to prevent cross-contamination), or they can be used to sample from
    completely different data sources (e.g. train on CONUS WTK while validating
    on Canada WTK)."""

    SAMPLER = Sampler

    def __init__(
        self,
        train_containers: List[Container],
        batch_size,
        n_batches,
        s_enhance,
        t_enhance,
        means: Union[Dict, str],
        stds: Union[Dict, str],
        sample_shape,
        feature_sets,
        val_containers: Optional[List[Container]] = None,
        queue_cap: Optional[int] = None,
        max_workers: Optional[int] = None,
        coarsen_kwargs: Optional[Dict] = None,
        default_device: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        train_containers : List[Container]
            List of Container instances containing training data
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
        sample_shape : tuple
            Shape of samples to select from containers to build batches.
            Batches will be of shape (batch_size, *sample_shape, len(features))
        feature_sets : dict
            Dictionary of feature sets. This must include a 'features' entry
            and optionally can include 'lr_only_features' and/or
            'hr_only_features'

            The allowed keys are:
                lr_only_features : list | tuple
                    List of feature names or patt*erns that should only be
                    included in the low-res training set and not the high-res
                    observations.
                hr_exo_features : list | tuple
                    List of feature names or patt*erns that should be included
                    in the high-resolution observation but not expected to be
                    output from the generative model. An example is high-res
                    topography that is to be injected mid-network.
        val_containers : List[Container]
            List of Container instances containing validation data
        queue_cap : int
            Maximum number of batches the batch queue can store.
        max_workers : int
            Number of workers / threads to use for getting samples used to
            build batches. This goes into a call to data.map(...,
            num_parallel_calls=max_workers) before prefetching samples from the
            tensorflow dataset generator.
        coarsen_kwargs : Union[Dict, None]
            Dictionary of kwargs to be passed to `self.coarsen`.
        default_device : str
            Default device to use for batch queue (e.g. /cpu:0, /gpu:0). If
            None this will use the first GPU if GPUs are available otherwise
            the CPU.
        """
        train_samplers = [
            self.SAMPLER(c, sample_shape, feature_sets)
            for c in train_containers
        ]

        val_samplers = (
            None
            if val_containers is None
            else [
                self.SAMPLER(c, sample_shape, feature_sets)
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
            coarsen_kwargs=coarsen_kwargs,
            default_device=default_device,
        )
