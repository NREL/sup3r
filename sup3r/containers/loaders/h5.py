"""Base loading classes. These are containers which also load data from
file_paths and include some sampling ability to interface with batcher
classes."""

import logging
from typing import Dict, Tuple

import dask.array as da
import numpy as np
import xarray as xr
from rex import MultiFileWindX

from sup3r.containers.loaders import Loader

logger = logging.getLogger(__name__)


class LoaderH5(Loader):
    """Base H5 loader. "Loads" h5 files so that a `.data` attribute
    provides access to the data in the files. This object provides a
    `__getitem__` method that can be used by :class:`Sampler` objects to build
    batches or by :class:`Extracter` objects to derive / extract specific
    features / regions / time_periods."""

    BASE_LOADER = MultiFileWindX

    def _res_shape(self):
        """Get shape of H5 file. Flattened files are 2D but we have 3D H5 files
        available through caching."""
        return (
            len(self.res['time_index']),
            *self.res.h5['meta']['latitude'].shape,
        )

    def load(self) -> xr.Dataset:
        """Wrap data in xarray.Dataset(). Handle differences with flattened and
        cached h5."""
        data_vars: Dict[str, Tuple] = {}
        dims: Tuple[str, ...] = ('time', 'south_north', 'west_east')
        if len(self._res_shape()) == 2:
            dims = ('time', 'space')
            elev = da.expand_dims(self.res.meta['elevation'].values, axis=0)
            data_vars['elevation'] = (
                dims,
                da.repeat(
                    da.asarray(elev, dtype=np.float32),
                    len(self.res.h5['time_index']),
                    axis=0,
                ),
            )
        data_vars = {
            **data_vars,
            **{
                f: (
                    dims,
                    da.asarray(
                        self.res.h5[f], dtype=np.float32, chunks=self.chunks
                    )
                    / self.scale_factor(f),
                )
                for f in self.res.h5.datasets
                if f not in ('meta', 'time_index')
            },
        }
        coords = {
            'time': self.res['time_index'],
            'latitude': (
                dims[1:],
                da.from_array(self.res.h5['meta']['latitude']),
            ),
            'longitude': (
                dims[1:],
                da.from_array(self.res.h5['meta']['longitude']),
            ),
        }
        return xr.Dataset(coords=coords, data_vars=data_vars).astype(
            np.float32
        )

    def scale_factor(self, feature):
        """Get scale factor for given feature. Data is stored in scaled form to
        reduce memory."""
        feat = feature if feature in self.res else feature.lower()
        feat = self.res.h5[feat]
        return np.float32(
            1.0
            if not hasattr(feat, 'attrs')
            else feat.attrs.get('scale_factor', 1.0)
        )
