"""Base loading classes. These are containers which also load data from
file_paths and include some sampling ability to interface with batcher
classes."""

import logging

import dask
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

    def load(self) -> dask.array:
        """Dask array with features in last dimension. Either lazily loaded
        (mode = 'lazy') or loaded into memory right away (mode = 'eager').

        Returns
        -------
        dask.array.core.Array
            (spatial, time, features) or (spatial_1, spatial_2, time, features)
        """
        arrays = []
        for feat in self.features:
            if feat in self.res.h5 or feat.lower() in self.res.h5:
                scale = self.res.h5[feat].attrs.get('scale_factor', 1)
                entry = dask.array.from_array(
                    self.res.h5[feat], chunks=self.chunks
                ) / scale
            elif hasattr(self.res, 'meta') and feat in self.res.meta:
                entry = dask.array.from_array(
                    np.repeat(
                        self.res.h5['meta'][feat][None],
                        self.res.h5['time_index'].shape[0],
                        axis=0,
                    )
                )
            else:
                msg = f'{feat} not found in {self.file_paths}.'
                logger.error(msg)
                raise RuntimeError(msg)
            arrays.append(entry)

        data = dask.array.stack(arrays, axis=-1)
        data = dask.array.moveaxis(data, 0, -2)

        if self.mode == 'eager':
            data = data.compute()

        return data

    def _get_res(self):
        return MultiFileWindX(self.file_paths, **self._res_kwargs)
