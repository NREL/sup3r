"""Data handling for H5 files.
@author: bbenton
"""

import logging
from typing import Dict, Optional

import numpy as np

from sup3r.preprocessing.base import Sup3rDataset
from sup3r.preprocessing.common import Dimension
from sup3r.preprocessing.samplers.dual import DualSampler

np.random.seed(42)

logger = logging.getLogger(__name__)


class DualSamplerCC(DualSampler):
    """Special sampling of WTK or NSRDB data for climate change applications"""

    def __init__(
        self,
        data: Sup3rDataset,
        sample_shape,
        s_enhance=1,
        t_enhance=24,
        feature_sets: Optional[Dict] = None,
    ):
        """
        Parameters
        ----------
        data : Sup3rDataset
            A tuple of xr.Dataset instances wrapped in the
            :class:`Sup3rDataset` interface. The first must be daily and the
            second must be hourly data
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
            'with `.daily` and `.hourly` data members, in that order'
        )
        assert hasattr(data, 'daily') and hasattr(data, 'hourly'), msg
        lr, hr = data.daily, data.hourly
        assert lr == data[0] and hr == data[1], msg
        if t_enhance == 1:
            hr = data.daily
        if s_enhance > 1:
            lr = lr.coarsen(
                {
                    Dimension.SOUTH_NORTH: s_enhance,
                    Dimension.WEST_EAST: s_enhance,
                }
            ).mean()
        n_hours = data.hourly.sizes['time']
        n_days = data.daily.sizes['time']
        self.daily_data_slices = [
            slice(x[0], x[-1] + 1)
            for x in np.array_split(np.arange(n_hours), n_days)
        ]
        data = Sup3rDataset(low_res=lr, high_res=hr)
        super().__init__(
            data=data,
            sample_shape=sample_shape,
            t_enhance=t_enhance,
            s_enhance=s_enhance,
            feature_sets=feature_sets,
        )
        sample_shape = self.check_sample_shape(sample_shape)

    def check_sample_shape(self, sample_shape):
        """Make sure sample_shape is consistent with required number of time
        steps in the sample data."""
        t_shape = sample_shape[-1]
        if len(sample_shape) == 2:
            logger.info(
                'Found 2D sample shape of {}. Adding spatial dim of {}'.format(
                    sample_shape, self.t_enhance
                )
            )
            sample_shape = (*sample_shape, self.t_enhance)
            t_shape = sample_shape[-1]

        if self.t_enhance != 1 and t_shape % 24 != 0:
            msg = (
                'Climate Change Sampler can only work with temporal '
                'sample shapes that are one or more days of hourly data '
                '(e.g. 24, 48, 72...), or for spatial only models t_enhance = '
                '1. The requested temporal sample '
                'shape was: {}'.format(t_shape)
            )
            logger.error(msg)
            raise RuntimeError(msg)
        return sample_shape
