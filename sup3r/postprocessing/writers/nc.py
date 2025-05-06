"""Output handling"""

import logging
from datetime import datetime as dt

import numpy as np
import xarray as xr

from sup3r.preprocessing.cachers import Cacher
from sup3r.preprocessing.names import Dimension

from .base import OutputHandler

logger = logging.getLogger(__name__)


class OutputHandlerNC(OutputHandler):
    """Forward pass OutputHandler for NETCDF files"""

    @classmethod
    def _write_output(
        cls,
        data,
        features,
        lat_lon,
        times,
        out_file,
        meta_data=None,
        max_workers=None,
        invert_uv=None,
        nn_fill=False,
        gids=None,
    ):
        """Write forward pass output to NETCDF file

        Parameters
        ----------
        data : ndarray
            (spatial_1, spatial_2, temporal, features)
            High resolution forward pass output
        features : list
            List of feature names corresponding to the last dimension of data
        lat_lon : ndarray
            Array of high res lat/lon for output data.
            (spatial_1, spatial_2, 2)
            Last dimension has ordering (lat, lon)
        times : pd.Datetimeindex
            List of times for high res output data
        out_file : string
            Output file path
        meta_data : dict | None
            Dictionary of meta data from model
        max_workers : int | None
            Max workers to use for inverse transform.
        invert_uv : bool | None
            Whether to convert u and v wind components to windspeed and
            direction
        nn_fill : bool
            Whether to fill data outside of limits with nearest neighbour or
            cap to limits
        gids : list
            List of coordinate indices used to label each lat lon pair and to
            help with spatial chunk data collection
        """

        invert_uv = False if invert_uv is None else invert_uv
        data, features = cls._transform_output(
            data=data,
            features=features,
            lat_lon=lat_lon,
            invert_uv=invert_uv,
            nn_fill=nn_fill,
            max_workers=max_workers,
        )

        coords = {
            Dimension.TIME: times,
            Dimension.LATITUDE: (Dimension.dims_2d(), lat_lon[:, :, 0]),
            Dimension.LONGITUDE: (Dimension.dims_2d(), lat_lon[:, :, 1]),
        }
        data_vars = {}
        if gids is not None:
            data_vars = {'gids': (Dimension.dims_2d(), gids)}
        for i, f in enumerate(features):
            data_vars[f] = (
                (Dimension.TIME, *Dimension.dims_2d()),
                np.transpose(data[..., i], axes=(2, 0, 1)).astype(np.float32),
            )

        attrs = meta_data or {}
        now = dt.utcnow().isoformat()
        attrs['date_modified'] = now
        attrs['date_created'] = attrs.get('date_created', now)

        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        Cacher._write_single(
            out_file=out_file,
            data=ds,
            features=features,
            max_workers=max_workers,
        )
