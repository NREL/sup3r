"""Base loading classes. These are containers which also load data from
file_paths and include some sampling ability to interface with batcher
classes."""

import logging

import dask.array as da
import numpy as np
from rex import MultiFileWindX

from sup3r.containers.loaders import Loader

logger = logging.getLogger(__name__)


class LoaderH5(Loader):
    """Base H5 loader. "Loads" h5 files so that a `.data` attribute
    provides access to the data in the files. This object provides a
    `__getitem__` method that can be used by :class:`Sampler` objects to build
    batches or by :class:`Extracter` objects to derive / extract specific
    features / regions / time_periods."""

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

    def _get_features(self, features):
        """Get feature(s) from base resource"""
        if isinstance(features, (list, tuple)):
            data = [self._get_features(f) for f in features]

        elif features in self.res.h5:
            data = da.from_array(
                self.res.h5[features], chunks=self.chunks
            ) / self.scale_factor(features)

        elif features.lower() in self.res.h5:
            data = self._get_features(features.lower())

        elif hasattr(self.res, 'meta') and features in self.res.meta:
            data = da.from_array(
                np.repeat(
                    self.res.h5['meta'][features][None],
                    self.res.h5['time_index'].shape[0],
                    axis=0,
                )
            )
        else:
            msg = f'{features} not found in {self.file_paths}.'
            logger.error(msg)
            raise KeyError(msg)

        data = da.moveaxis(data, 0, -1)
        return data
