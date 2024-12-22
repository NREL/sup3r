"""Extended Sampler for sampling observation data in addition to standard
gridded training data."""

import logging
from typing import Dict, Optional

from sup3r.preprocessing.base import Sup3rDataset

from .dual import DualSampler

logger = logging.getLogger(__name__)


class DualSamplerWithObs(DualSampler):
    """Dual Sampler which also samples from extra observation data. The
    observation data is on the same grid as the high-resolution data but
    includes NaNs at points where observation data doesn't exist. This will
    be used in an additional content loss term."""

    def __init__(
        self,
        data: Sup3rDataset,
        sample_shape: Optional[tuple] = None,
        batch_size: int = 16,
        s_enhance: int = 1,
        t_enhance: int = 1,
        feature_sets: Optional[Dict] = None,
    ):
        """
        Parameters
        ----------
        data : Sup3rDataset
            A :class:`~sup3r.preprocessing.base.Sup3rDataset` instance with
            low-res and high-res data members
        sample_shape : tuple
            Size of arrays to sample from the high-res data. The sample shape
            for the low-res sampler will be determined from the enhancement
            factors.
        s_enhance : int
            Spatial enhancement factor
        t_enhance : int
            Temporal enhancement factor
        feature_sets : Optional[dict]
            Optional dictionary describing how the full set of features is
            split between `lr_only_features` and `hr_exo_features`.

            lr_only_features : list | tuple
                List of feature names or patt*erns that should only be
                included in the low-res training set and not the high-res
                observations.
            hr_exo_features : list | tuple
                List of feature names or patt*erns that should be included
                in the high-resolution observation but not expected to be
                output from the generative model. An example is high-res
                topography that is to be injected mid-network.
        """

        msg = (
            f'{self.__class__.__name__} requires a Sup3rDataset object '
            'with `.low_res`, `.high_res`, and `.obs` data members, in that '
            'order'
        )
        assert (
            hasattr(data, 'low_res')
            and hasattr(data, 'high_res')
            and hasattr(data, 'obs')
        ), msg
        assert (
            data.low_res == data[0]
            and data.high_res == data[1]
            and data.obs == data[2]
        ), msg
        super().__init__(
            data,
            sample_shape=sample_shape,
            batch_size=batch_size,
            s_enhance=s_enhance,
            t_enhance=t_enhance,
            feature_sets=feature_sets,
        )

    def get_sample_index(self, n_obs=None):
        """Get paired sample index, consisting of index for the low res sample
        and the index for the high res sample with the same spatiotemporal
        extent, with an additional index (same as the index for the high-res
        data) for the observation data"""
        lr_index, hr_index = super().get_sample_index(n_obs=n_obs)
        return (lr_index, hr_index, hr_index)
