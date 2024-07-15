"""Base loading classes. These are containers which also load data from
file_paths and include some sampling ability to interface with batcher
classes."""

import logging
from typing import Dict, Tuple

import dask.array as da
import numpy as np
import xarray as xr
from rex import MultiFileWindX

from sup3r.preprocessing.utilities import Dimension

from .base import BaseLoader

logger = logging.getLogger(__name__)


class LoaderH5(BaseLoader):
    """Base H5 loader. "Loads" h5 files so that a `.data` attribute
    provides access to the data in the files. This object provides a
    `__getitem__` method that can be used by :class:`Sampler` objects to build
    batches or by :class:`Extracter` objects to derive / extract specific
    features / regions / time_periods."""

    BASE_LOADER = MultiFileWindX

    @property
    def _time_independent(self):
        return 'time_index' not in self.res

    def _meta_shape(self):
        """Get shape of spatial domain only."""
        return self.res.h5['meta']['latitude'].shape

    def _res_shape(self):
        """Get shape of H5 file. Flattened files are 2D but we have 3D H5 files
        available through caching."""
        return (
            self._meta_shape()
            if self._time_independent
            else (len(self.res['time_index']), *self._meta_shape())
        )

    def load(self) -> xr.Dataset:
        """Wrap data in xarray.Dataset(). Handle differences with flattened and
        cached h5."""
        data_vars: Dict[str, Tuple] = {}
        coords: Dict[str, Tuple] = {}
        if len(self._meta_shape()) == 2:
            dims: Tuple[str, ...] = (
                Dimension.SOUTH_NORTH,
                Dimension.WEST_EAST,
            )
        else:
            dims = (Dimension.FLATTENED_SPATIAL,)
        if not self._time_independent:
            dims = (Dimension.TIME, *dims)
            coords[Dimension.TIME] = self.res['time_index']

        if len(self._meta_shape()) == 1:
            elev = da.asarray(
                self.res.meta['elevation'].values, dtype=np.float32
            )
            if not self._time_independent:
                elev = da.repeat(
                    elev[None, ...], len(self.res['time_index']), axis=0
                )
            data_vars['elevation'] = (dims, elev)
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
                if f not in ('meta', 'time_index', 'coordinates')
            },
        }
        coords.update(
            {
                Dimension.LATITUDE: (
                    dims[-len(self._meta_shape()) :],
                    da.from_array(self.res.h5['meta']['latitude']),
                ),
                Dimension.LONGITUDE: (
                    dims[-len(self._meta_shape()) :],
                    da.from_array(self.res.h5['meta']['longitude']),
                ),
            }
        )
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
