"""Output handling

TODO: Remove redundant code re. Cachers
"""

import logging
import re

import dask
import numpy as np
import pandas as pd

from sup3r.preprocessing.derivers.utilities import (
    invert_uv,
    parse_feature,
)

from .base import OutputHandler

logger = logging.getLogger(__name__)


class OutputHandlerH5(OutputHandler):
    """Class to handle writing output to H5 file"""

    @classmethod
    def get_renamed_features(cls, features):
        """Rename features based on transformation from u/v to
        windspeed/winddirection

        Parameters
        ----------
        features : list
            List of output features

        Returns
        -------
        list
            List of renamed features u/v -> windspeed/winddirection for each
            height
        """
        heights = [
            parse_feature(f).height
            for f in features
            if re.match('u_(.*?)m'.lower(), f.lower())
        ]
        renamed_features = features.copy()

        for height in heights:
            u_idx = features.index(f'u_{height}m')
            v_idx = features.index(f'v_{height}m')

            renamed_features[u_idx] = f'windspeed_{height}m'
            renamed_features[v_idx] = f'winddirection_{height}m'

        return renamed_features

    @classmethod
    def invert_uv_features(cls, data, features, lat_lon, max_workers=None):
        """Invert U/V to windspeed and winddirection. Performed in place.

        Parameters
        ----------
        data : ndarray
            High res data from forward pass
            (spatial_1, spatial_2, temporal, features)
        features : list
            List of output features. If this doesn't contain any names matching
            u_*m, this method will do nothing.
        lat_lon : ndarray
            High res lat/lon array
            (spatial_1, spatial_2, 2)
        max_workers : int | None
            Max workers to use for inverse transform. If None the maximum
            possible will be used
        """

        heights = [
            parse_feature(f).height
            for f in features
            if re.match('u_(.*?)m'.lower(), f.lower())
        ]

        if heights:
            logger.info(
                'Converting u/v to ws/wd for H5 output with max_workers=%s',
                max_workers,
            )
            logger.debug(
                'Found heights %s for output features %s', heights, features
            )

        tasks = []
        for height in heights:
            u_idx = features.index(f'u_{height}m')
            v_idx = features.index(f'v_{height}m')
            task = dask.delayed(cls.invert_uv_single_pair)(
                data, lat_lon, u_idx, v_idx
            )
            tasks.append(task)
            logger.info('Added %s futures to convert u/v to ws/wd', len(tasks))
        if max_workers == 1:
            dask.compute(*tasks, scheduler='single-threaded')
        else:
            dask.compute(*tasks, scheduler='threads', num_workers=max_workers)
        logger.info('Finished converting u/v to ws/wd')

    @staticmethod
    def invert_uv_single_pair(data, lat_lon, u_idx, v_idx):
        """Perform inverse transform in place on a single u/v pair.

        Parameters
        ----------
        data : ndarray
            High res data from forward pass
            (spatial_1, spatial_2, temporal, features)
        lat_lon : ndarray
            High res lat/lon array
            (spatial_1, spatial_2, 2)
        u_idx : int
            Index in data for U component to transform
        v_idx : int
            Index in data for V component to transform
        """
        ws, wd = invert_uv(data[..., u_idx], data[..., v_idx], lat_lon)
        data[..., u_idx] = ws
        data[..., v_idx] = wd

    @classmethod
    def _transform_output(cls, data, features, lat_lon, max_workers=None):
        """Transform output data before writing to H5 file

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
        max_workers : int | None
            Max workers to use for inverse transform. If None the max_workers
            will be estimated based on memory limits.
        """
        if any(
            re.match('u_(.*?)m'.lower(), f.lower())
            or re.match('v_(.*?)m'.lower(), f.lower())
            for f in features
        ):
            cls.invert_uv_features(
                data, features, lat_lon, max_workers=max_workers
            )
        features = cls.get_renamed_features(features)
        data = cls.enforce_limits(features=features, data=data)
        return data, features

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
        max_workers : int | None
            Max workers to use for inverse transform. If None the max_workers
            will be estimated based on memory limits.
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
        data, features = cls._transform_output(
            data.copy(), features, lat_lon, max_workers
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
