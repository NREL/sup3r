"""Base loading classes. These are containers which also load data from
file_paths and include some sampling ability to interface with batcher
classes."""

import logging
from typing import Dict, Tuple
from warnings import warn

import dask.array as da
import numpy as np
import pandas as pd
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
        if 'latitude' in self.res.h5:
            return self.res.h5['latitude'].shape
        return self.res.h5['meta']['latitude'].shape

    def _res_shape(self):
        """Get shape of H5 file.

        Note
        ----
        Flattened files are 2D but we have 3D H5 files available through
        caching and bias correction factor calculations."""
        return (
            self._meta_shape()
            if self._time_independent
            else (len(self.res['time_index']), *self._meta_shape())
        )

    def _get_coords(self, dims):
        """Get coords dict for xr.Dataset construction."""
        coords: Dict[str, Tuple] = {}
        if not self._time_independent:
            coords[Dimension.TIME] = pd.DatetimeIndex(self.res['time_index'])
        coord_base = (
            self.res.h5 if 'latitude' in self.res.h5 else self.res.h5['meta']
        )
        coords.update(
            {
                Dimension.LATITUDE: (
                    dims[-len(self._meta_shape()) :],
                    da.from_array(coord_base['latitude']),
                ),
                Dimension.LONGITUDE: (
                    dims[-len(self._meta_shape()) :],
                    da.from_array(coord_base['longitude']),
                ),
            }
        )
        return coords

    def _get_dset_tuple(self, dset, dims, chunks):
        """Get tuple of (dims, array) for given dataset. Used in data_vars
        entries"""
        arr = da.asarray(
            self.res.h5[dset], dtype=np.float32, chunks=chunks
        ) / self.scale_factor(dset)
        if len(arr.shape) == 3 and self._time_independent:
            msg = (
                f'{dset} array is 3 dimensional but {self.file_paths} has '
                f'no time index. Assuming this is an array of bias correction '
                'factors.'
            )
            logger.warning(msg)
            warn(msg)
            if arr.shape[-1] == 1:
                arr_dims = (*Dimension.dims_2d(), Dimension.GLOBAL_TIME)
            else:
                arr_dims = Dimension.dims_3d()
        elif len(arr.shape) == 4:
            msg = (
                f'{dset} array is 4 dimensional. Assuming this is an array '
                'of spatiotemporal quantiles.'
            )
            logger.warning(msg)
            warn(msg)
            arr_dims = Dimension.dims_4d_bc()
        else:
            arr_dims = dims
        return (arr_dims, arr)

    def _get_data_vars(self, dims):
        """Define data_vars dict for xr.Dataset construction."""
        data_vars: Dict[str, Tuple] = {}
        logger.debug(f'Rechunking features with chunks: {self.chunks}')
        chunks = (
            tuple(self.chunks[d] for d in dims)
            if isinstance(self.chunks, dict)
            else self.chunks
        )
        if len(self._meta_shape()) == 1:
            elev = self.res.meta['elevation'].values
            if not self._time_independent:
                elev = np.repeat(
                    elev[None, ...], len(self.res['time_index']), axis=0
                )
            data_vars['elevation'] = (
                dims,
                da.asarray(elev, dtype=np.float32, chunks=chunks),
            )
        data_vars.update(
            {
                f: self._get_dset_tuple(dset=f, dims=dims, chunks=chunks)
                for f in set(self.res.h5.datasets)
                - {'meta', 'time_index', 'coordinates'}
            }
        )
        return data_vars

    def _get_dims(self):
        """Get tuple of named dims for dataset."""
        if len(self._meta_shape()) == 2:
            dims: Tuple[str, ...] = (
                Dimension.SOUTH_NORTH,
                Dimension.WEST_EAST,
            )
        else:
            dims = (Dimension.FLATTENED_SPATIAL,)
        if not self._time_independent:
            dims = (Dimension.TIME, *dims)
        return dims

    def load(self) -> xr.Dataset:
        """Wrap data in xarray.Dataset(). Handle differences with flattened and
        cached h5."""
        dims = self._get_dims()
        data_vars = self._get_data_vars(dims)
        coords = self._get_coords(dims)
        data_vars = {k: v for k, v in data_vars.items() if k not in coords}
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
