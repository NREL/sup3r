"""Output handling

TODO: Remove redundant code re. Cachers
"""

import json
import logging
from datetime import datetime as dt

import numpy as np
import xarray as xr

from sup3r.preprocessing.utilities import Dimension

from .base import OutputHandler

logger = logging.getLogger(__name__)


class OutputHandlerNC(OutputHandler):
    """OutputHandler subclass for NETCDF files"""

    # pylint: disable=W0613
    @classmethod
    def _get_xr_dset(cls, data, features, lat_lon, times, meta_data=None):
        """Convert data to xarray Dataset() object.

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
        meta_data : dict | None
            Dictionary of meta data from model
        """
        coords = {
            Dimension.TIME: times,
            Dimension.LATITUDE: (
                Dimension.spatial_2d(),
                lat_lon[:, :, 0].astype(np.float32),
            ),
            Dimension.LONGITUDE: (
                Dimension.spatial_2d(),
                lat_lon[:, :, 1].astype(np.float32),
            ),
        }

        data_vars = {}
        for i, f in enumerate(features):
            data_vars[f] = (
                Dimension.dims_3d(),
                np.transpose(data[..., i], (2, 0, 1)),
            )

        attrs = {}
        if meta_data is not None:
            attrs = {
                k: v if isinstance(v, str) else json.dumps(v)
                for k, v in meta_data.items()
            }

        attrs['date_modified'] = dt.utcnow().isoformat()
        if 'date_created' not in attrs:
            attrs['date_created'] = attrs['date_modified']

        return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

    # pylint: disable=W0613
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
            Has no effect. For compliance with H5 output handler
        gids : list
            List of coordinate indices used to label each lat lon pair and to
            help with spatial chunk data collection
        """
        cls._get_xr_dset(
            data=data,
            lat_lon=lat_lon,
            features=features,
            times=times,
            meta_data=meta_data,
        ).to_netcdf(out_file)
        logger.info(f'Saved output of size {data.shape} to: {out_file}')

    @classmethod
    def combine_file(cls, files, outfile):
        """Combine all chunked output files from ForwardPass into a single file

        Parameters
        ----------
        files : list
            List of chunked output files from ForwardPass runs
        outfile : str
            Output file name for combined file
        """
        time_key = cls.get_time_dim_name(files[0])
        ds = xr.open_mfdataset(files, combine='nested', concat_dim=time_key)
        ds.to_netcdf(outfile)
        logger.info(f'Saved combined file: {outfile}')
