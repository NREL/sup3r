"""Output handling

TODO: Remove redundant code re. Cachers
"""

import logging

import numpy as np
import pandas as pd

from .base import OutputHandler

logger = logging.getLogger(__name__)


class OutputHandlerH5(OutputHandler):
    """Class to handle writing output to H5 file"""

    @classmethod
    def _write_output(
        cls,
        data,
        features,
        lat_lon,
        times,
        out_file,
        meta_data=None,
        invert_uv=None,
        max_workers=None,
        gids=None,
    ):
        """Write forward pass output to H5 file

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
        invert_uv : bool | None
            Whether to convert u and v wind components to windspeed and
            direction
        max_workers : int | None
            Max workers to use for inverse transform.
        gids : list
            List of coordinate indices used to label each lat lon pair and to
            help with spatial chunk data collection
        """
        msg = (
            f'Output data shape ({data.shape}) and lat_lon shape '
            f'({lat_lon.shape}) conflict.'
        )
        assert data.shape[:2] == lat_lon.shape[:-1], msg
        msg = (
            f'Output data shape ({data.shape}) and times shape '
            f'({len(times)}) conflict.'
        )
        assert data.shape[-2] == len(times), msg
        invert_uv = True if invert_uv is None else invert_uv
        data, features = cls._transform_output(
            data.copy(),
            features,
            lat_lon,
            max_workers=max_workers,
            invert_uv=invert_uv,
        )
        gids = (
            gids
            if gids is not None
            else np.arange(np.prod(lat_lon.shape[:-1]))
        )
        meta = pd.DataFrame(
            {
                'gid': gids.flatten(),
                'latitude': lat_lon[..., 0].flatten(),
                'longitude': lat_lon[..., 1].flatten(),
            }
        )
        data_list = []
        for i, _ in enumerate(features):
            flat_data = data[..., i].reshape((-1, len(times)))
            flat_data = np.transpose(flat_data, (1, 0))
            data_list.append(flat_data)
        cls.write_data(out_file, features, times, data_list, meta, meta_data)
