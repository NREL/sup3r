"""'Cropped' sampler classes. These are Sampler objects with an additional
constraint on where samples can come from. For example, if we want to split
samples into training and testing we would use cropped samplers to prevent
cross-contamination."""

from sup3r.containers.samplers import Sampler
from sup3r.utilities.utilities import uniform_box_sampler, uniform_time_sampler


class CroppedSampler(Sampler):
    """Cropped sampler class used to splitting samples into train / test."""

    def __init__(
        self,
        data,
        features,
        sample_shape,
        crop_slice,
        lr_only_features,
        hr_exo_features,
    ):
        super().__init__(
            data, features, sample_shape, lr_only_features, hr_exo_features
        )
        self.crop_slice = crop_slice

    def get_sample_index(self):
        """Crop time dimension to restrict sampling."""
        spatial_slice = uniform_box_sampler(self.shape, self.sample_shape[:2])
        temporal_slice = uniform_time_sampler(
            self.shape, self.sample_shape[2], crop_slice=self.crop_slice
        )
        return (*spatial_slice, temporal_slice, slice(None))
