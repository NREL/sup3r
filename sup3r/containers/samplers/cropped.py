"""'Cropped' sampler classes. These are Sampler objects with an additional
constraint on where samples can come from. For example, if we want to split
samples into training and testing we would use cropped samplers to prevent
cross-contamination."""

import logging
from warnings import warn

import numpy as np

from sup3r.containers.samplers import Sampler
from sup3r.utilities.utilities import uniform_box_sampler, uniform_time_sampler

logger = logging.getLogger(__name__)


class CroppedSampler(Sampler):
    """Cropped Sampler class used to splitting samples into train / test."""

    def __init__(
        self,
        container,
        sample_shape,
        feature_sets=None,
        crop_slice=slice(None),
    ):
        super().__init__(
            container=container,
            sample_shape=sample_shape,
            feature_sets=feature_sets,
        )

        self.crop_slice = crop_slice

    @property
    def crop_slice(self):
        """Return the slice used to crop the time dimension of the sampling
        region."""
        return self._crop_slice

    @crop_slice.setter
    def crop_slice(self, crop_slice):
        self._crop_slice = crop_slice
        self.crop_check()

    def get_sample_index(self):
        """Crop time dimension to restrict sampling."""
        spatial_slice = uniform_box_sampler(self.shape, self.sample_shape[:2])
        time_slice = uniform_time_sampler(
            self.shape, self.sample_shape[2], crop_slice=self.crop_slice
        )
        return (*spatial_slice, time_slice, slice(None))

    def crop_check(self):
        """Check if crop_slice limits the sampling region to fewer time steps
        than sample_shape[2]"""
        cropped_indices = np.arange(self.shape[2])[self.crop_slice]
        msg = (
            f'Cropped region has {len(cropped_indices)} but requested '
            f'sample_shape is {self.sample_shape}. Use a smaller '
            'sample_shape[2] or larger crop_slice.'
        )
        if len(cropped_indices) < self.sample_shape[2]:
            logger.warning(msg)
            warn(msg)
