"""
Sup3r conditional moment batch_handling module.

TODO: Remove BatchMom classes - this functionality should be handled by the
BatchQueue. Validation classes can be removed - these are now just additional
queues given to BatchHandlers. Remove __next__ methods - these are handling by
samplers.
"""

import logging

import numpy as np

from sup3r.preprocessing.batch_handlers.factory import BatchHandlerFactory
from sup3r.preprocessing.batch_queues.conditional import ConditionalBatchQueue
from sup3r.preprocessing.samplers import Sampler
from sup3r.utilities.utilities import (
    spatial_simple_enhancing,
    temporal_simple_enhancing,
)

np.random.seed(42)

logger = logging.getLogger(__name__)


BaseConditionalBatchHandler = BatchHandlerFactory(
    Sampler, ConditionalBatchQueue
)


class BatchHandlerMom1(BaseConditionalBatchHandler):
    """Batch handling class for conditional estimation of first moment"""

    def make_output(self, samples):
        """For the 1st moment the output is simply the high_res"""
        _, hr = samples
        return hr


class BatchHandlerMom1SF(BaseConditionalBatchHandler):
    """Batch handling class for conditional estimation of first moment
    of subfilter velocity"""

    def make_output(self, samples):
        """
        Returns
        -------
        SF: T_Array
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
            SF is subfilter, HR is high-res and LR is low-res
            SF = HR - LR
        """
        # Remove LR from HR
        lr, hr = samples
        enhanced_lr = spatial_simple_enhancing(lr, s_enhance=self.s_enhance)
        enhanced_lr = temporal_simple_enhancing(
            enhanced_lr,
            t_enhance=self.t_enhance,
            mode=self.time_enhance_mode,
        )
        enhanced_lr = enhanced_lr[..., self.hr_features_ind]

        return hr - enhanced_lr


class BatchHandlerMom2(BaseConditionalBatchHandler):
    """Batch handling class for conditional estimation of second moment"""

    def make_output(self, samples):
        """
        Returns
        -------
        (HR - <HR|LR>)**2: T_Array
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
            HR is high-res and LR is low-res
        """
        # Remove first moment from HR and square it
        lr, hr = samples
        exo_data = self.model_mom1.get_high_res_exo_input(hr)
        out = self.model_mom1._tf_generate(lr, exo_data).numpy()
        out = self.model_mom1._combine_loss_input(hr, out)
        return (hr - out) ** 2


class BatchHandlerMom2Sep(BatchHandlerMom1):
    """Batch handling class for conditional estimation of second moment
    without subtraction of first moment"""

    def make_output(self, samples):
        """
        Returns
        -------
        HR**2: T_Array
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
            HR is high-res
        """
        return super().make_output(samples) ** 2


class BatchHandlerMom2SF(BaseConditionalBatchHandler):
    """Batch handling class for conditional estimation of second moment of
    subfilter velocity."""

    def make_output(self, samples):
        """
        Returns
        -------
        (SF - <SF|LR>)**2: T_Array
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
            SF is subfilter, HR is high-res and LR is low-res
            SF = HR - LR
        """
        # Remove LR and first moment from HR and square it
        lr, hr = samples
        exo_data = self.model_mom1.get_high_res_exo_input(hr)
        out = self.model_mom1._tf_generate(lr, exo_data).numpy()
        out = self.model_mom1._combine_loss_input(hr, out)
        enhanced_lr = spatial_simple_enhancing(lr, s_enhance=self.s_enhance)
        enhanced_lr = temporal_simple_enhancing(
            enhanced_lr, t_enhance=self.t_enhance, mode=self.time_enhance_mode
        )
        enhanced_lr = enhanced_lr[..., self.hr_features_ind]
        return (hr - enhanced_lr - out) ** 2


class BatchMom2SepSF(BatchHandlerMom1SF):
    """Batch of low_res, high_res and output data when learning second moment
    of subfilter vel separate from first moment"""

    def make_output(self, samples):
        """
        Returns
        -------
        SF**2: T_Array
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
            SF is subfilter, HR is high-res and LR is low-res
            SF = HR - LR
        """
        # Remove LR from HR and square it
        return super().make_output(samples) ** 2
