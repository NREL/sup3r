"""DualBatchQueue with additional observation data on the same grid as the
high-res data. The observation data is sampled with the same index as the
high-res data during training."""

import logging

from scipy.ndimage import gaussian_filter

from .dual import DualBatchQueue

logger = logging.getLogger(__name__)


class DualBatchQueueWithObs(DualBatchQueue):
    """Base BatchQueue for use with
    :class:`~sup3r.preprocessing.samplers.DualSamplerWithObs` objects."""

    _signature_objs = (DualBatchQueue,)

    @property
    def queue_shape(self):
        """Shape of objects stored in the queue."""
        return [
            (self.batch_size, *self.lr_shape),
            (self.batch_size, *self.hr_shape),
            (self.batch_size, *self.hr_shape),
        ]

    def transform(self, samples, smoothing=None, smoothing_ignore=None):
        """Perform smoothing if requested.

        Note
        ----
        This does not include temporal or spatial coarsening like
        :class:`SingleBatchQueue`
        """
        low_res, high_res, obs = samples

        if smoothing is not None:
            feat_iter = [
                j
                for j in range(low_res.shape[-1])
                if self.features[j] not in smoothing_ignore
            ]
            for i in range(low_res.shape[0]):
                for j in feat_iter:
                    low_res[i, ..., j] = gaussian_filter(
                        low_res[i, ..., j], smoothing, mode='nearest'
                    )
        return low_res, high_res, obs
