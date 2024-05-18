"""Base loading classes. These are containers which also load data from
file_paths and include some sampling ability to interface with batcher
classes."""

import logging

import numpy as np
from rex import MultiFileWindX

from sup3r.containers.loaders import Loader

logger = logging.getLogger(__name__)


class LoaderH5(Loader):
    """Base H5 loader. "Loads" h5 files so that a `.data` attribute
    provides access to the data in the files. This object provides a
    `__getitem__` method that can be used by Sampler objects to build batches
    or by Wrangler objects to derive / extract specific features / regions /
    time_periods."""

    def _get_res(self):
        return MultiFileWindX(self.file_paths, **self._res_kwargs)

    def scale_factor(self, feature):
        """Get scale factor for given feature. Data is stored in scaled form to
        reduce memory."""
        feat = self.get(feature)
        return (
            1
            if not hasattr(feat, 'attrs')
            else feat.attrs.get('scale_factor', 1)
        )

    def get(self, feature):
        """Get feature from base resource"""
        if feature in self.res.h5:
            return self.res.h5[feature]
        if feature.lower() in self.res.h5:
            return self.res.h5[feature.lower()]
        if hasattr(self.res, 'meta') and feature in self.res.meta:
            return np.repeat(
                self.res.h5['meta'][feature][None],
                self.res.h5['time_index'].shape[0],
                axis=0,
            )
        msg = f'{feature} not found in {self.file_paths}.'
        logger.error(msg)
        raise RuntimeError(msg)
