"""Output handling

author : @bbenton
"""
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import xarray as xr
import pandas as pd
import logging
from scipy.interpolate import RBFInterpolator
import re
from datetime import datetime as dt
import json

from sup3r.version import __version__
from sup3r.utilities import VERSION_RECORD
from sup3r.utilities.utilities import (invert_uv, estimate_max_workers,
                                       pd_date_range)
from sup3r.preprocessing.feature_handling import Feature

from rex.outputs import Outputs as BaseRexOutputs

logger = logging.getLogger(__name__)


H5_ATTRS = {'windspeed': {'scale_factor': 100.0,
                          'units': 'm s-1',
                          'dtype': 'uint16',
                          'chunks': (2000, 500),
                          'min': 0,
                          'max': 120},
            'winddirection': {'scale_factor': 100.0,
                              'units': 'degree',
                              'dtype': 'uint16',
                              'chunks': (2000, 500),
                              'min': 0,
                              'max': 360},
            'temperature': {'scale_factor': 100.0,
                            'units': 'C',
                            'dtype': 'int16',
                            'chunks': (2000, 500),
                            'min': -200,
                            'max': 100},
            'relativehumidity': {'scale_factor': 100.0,
                                 'units': 'percent',
                                 'dtype': 'uint16',
                                 'chunks': (2000, 500),
                                 'max': 100,
                                 'min': 0},
            'pressure': {'scale_factor': 0.1,
                         'units': 'Pa',
                         'dtype': 'uint16',
                         'chunks': (2000, 500),
                         'min': 0,
                         'max': 150000},
            'bvf_mo': {'scale_factor': 0.1,
                       'units': 'm s-2',
                       'dtype': 'uint16',
                       'chunks': (2000, 500)},
            'bvf2': {'scale_factor': 0.1,
                     'units': 's-2',
                     'dtype': 'int16',
                     'chunks': (2000, 500)},
            }


class RexOutputs(BaseRexOutputs):
    """Base class to handle NREL h5 formatted output data"""

    @property
    def full_version_record(self):
        """Get record of versions for dependencies

        Returns
        -------
        dict
            Dictionary of package versions for dependencies
        """
        versions = super().full_version_record
        versions.update(VERSION_RECORD)
        return versions

    def set_version_attr(self):
        """Set the version attribute to the h5 file."""
        self.h5.attrs['version'] = __version__
        self.h5.attrs['full_version_record'] = json.dumps(
            self.full_version_record)
        self.h5.attrs['package'] = 'sup3r'


class OutputHandler:
    """Class to handle forward pass output. This includes transforming features
    back to their original form and outputting to the correct file format.
    """

    @staticmethod
    def enforce_limits(features, data):
        """Enforce physical limits for feature data

        Parameters
        ----------
        features : list
            List of features with ordering corresponding to last channel of
            data array.
        data : ndarray
            Array of feature data

        Returns
        -------
        data : ndarray
            Array of feature data with physical limits enforced
        """
        for i, f in enumerate(features):
            attrs = H5_ATTRS[Feature.get_basename(f)]
            max = attrs.get('max', np.inf)
            min = attrs.get('min', -np.inf)
            data[..., i] = np.maximum(data[..., i], min)
            data[..., i] = np.minimum(data[..., i], max)
        return data

    @staticmethod
    def get_lat_lon(low_res_lat_lon, shape):
        """Get lat lon arrays for high res output file

        Parameters
        ----------
        low_res_lat_lon : ndarray
            Array of lat/lon for input data.
            (spatial_1, spatial_2, 2)
            Last dimension has ordering (lat, lon)
        shape : tuple
            (lons, lats) Shape of high res grid
        Returns
        -------
        lat_lon : ndarray
            Array of lat lons for high res output file
            (spatial_1, spatial_2, 2)
            Last dimension has ordering (lat, lon)
        """
        logger.debug('Getting high resolution lat / lon grid')
        s_enhance = shape[0] // low_res_lat_lon.shape[0]
        old_points = np.zeros((np.product(low_res_lat_lon.shape[:-1]), 2),
                              dtype=np.float32)
        new_points = np.zeros((np.product(shape), 2), dtype=np.float32)
        lats = low_res_lat_lon[..., 0].flatten()
        lons = low_res_lat_lon[..., 1].flatten()

        # This shifts the indices for the old points by the downsampling
        # fraction so that we can calculate the centers of the new points with
        # origin at zero. Obviously if the shapes are the same then there
        # should be no shift
        lat_shift = 1 - low_res_lat_lon.shape[0] / shape[0]
        lon_shift = 1 - low_res_lat_lon.shape[1] / shape[1]

        new_count = old_count = 0
        for i in range(shape[0]):
            for j in range(shape[1]):
                if i % s_enhance == 0 and j % s_enhance == 0:
                    old_points[old_count, 0] = i + lat_shift
                    old_points[old_count, 1] = j + lon_shift
                    old_count += 1
                new_points[new_count, 0] = i
                new_points[new_count, 1] = j
                new_count += 1
        lats = RBFInterpolator(old_points, lats, neighbors=20)(new_points)
        lons = RBFInterpolator(old_points, lons, neighbors=20)(new_points)
        lat_lon = np.dstack((lats.reshape(shape), lons.reshape(shape)))
        return lat_lon

    @staticmethod
    def get_times(low_res_times, shape):
        """Get array of times for high res output file

        Parameters
        ----------
        low_res_times : pd.Datetimeindex
            List of times for low res input data. If there is only a single low
            res timestep, it is assumed the data is daily.
        shape : int
            Number of time steps for high res time array

        Returns
        -------
        ndarray
            Array of times for high res output file.
        """
        logger.debug('Getting high resolution time indices')
        t_enhance = int(shape / len(low_res_times))

        if len(low_res_times) > 1:
            offset = (low_res_times[1] - low_res_times[0])
        else:
            offset = np.timedelta64(24, 'h')

        times = np.array([low_res_times[0] + i * offset / t_enhance
                          for i in range(shape)])
        freq = (times[1] - times[0]) / np.timedelta64(1, 's')
        freq = pd.tseries.offsets.DateOffset(seconds=freq)
        times = pd_date_range(times[0], times[-1], freq=freq)
        return times

    @classmethod
    @abstractmethod
    def _write_output(cls, data, features, lat_lon, times, out_file, meta_data,
                      max_workers=None):
        """Write output to file with specified times and lats/lons"""

    @classmethod
    def write_output(cls, data, features, low_res_lat_lon, low_res_times,
                     out_file, meta_data=None, max_workers=None):
        """Write forward pass output to file

        Parameters
        ----------
        data : ndarray
            (spatial_1, spatial_2, temporal, features)
            High resolution forward pass output
        features : list
            List of feature names corresponding to the last dimension of data
        low_res_lat_lon : ndarray
            Array of lat/lon for input data. (spatial_1, spatial_2, 2)
            Last dimension has ordering (lat, lon)
        low_res_times : pd.Datetimeindex
            List of times for low res source data
        out_file : string
            Output file path
        meta_data : dict | None
            Dictionary of meta data from model
        max_workers : int | None
            Max workers to use for inverse uv transform. If None the
            max_workers will be estimated based on memory limits.
        """
        lat_lon = cls.get_lat_lon(low_res_lat_lon, data.shape[:2])
        times = cls.get_times(low_res_times, data.shape[-2])
        cls._write_output(data, features, lat_lon, times, out_file,
                          meta_data, max_workers)


class OutputHandlerNC(OutputHandler):
    """OutputHandler subclass for NETCDF files"""

    # pylint: disable=W0613
    @classmethod
    def _write_output(cls, data, features, lat_lon, times, out_file,
                      meta_data=None, max_workers=None):
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
        """
        coords = {'Times': (['Time'], times),
                  'XLAT': (['south_north', 'east_west'], lat_lon[..., 0]),
                  'XLONG': (['south_north', 'east_west'], lat_lon[..., 1])}

        data_vars = {}
        for i, f in enumerate(features):
            data_vars[f] = (['Time', 'south_north', 'east_west'],
                            np.transpose(data[..., i], (2, 0, 1)))

        attrs = {'gan_meta': json.dumps(meta_data)}

        with xr.Dataset(data_vars=data_vars, coords=coords,
                        attrs=attrs) as ncfile:
            ncfile.to_netcdf(out_file)
        logger.info(f'Saved output of size {data.shape} to: {out_file}')

    @staticmethod
    def combine_file(files, outfile):
        """Combine all chunked output files from ForwardPass into a single file

        Parameters
        ----------
        files : list
            List of chunked output files from ForwardPass runs
        outfile : str
            Output file name for combined file
        """
        ds = xr.open_mfdataset(files, combine='nested', concat_dim='Time')
        ds.to_netcdf(outfile)
        logger.info(f'Saved combined file: {outfile}')


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
        heights = []
        renamed_features = features.copy()
        for f in features:
            if re.match('U_(.*?)m'.lower(), f.lower()):
                heights.append(Feature.get_height(f))

        for height in heights:
            u_idx = features.index(f'U_{height}m')
            v_idx = features.index(f'V_{height}m')

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
            List of output features. If this doesnt contain any names matching
            U_*m, this method will do nothing.
        lat_lon : ndarray
            High res lat/lon array
            (spatial_1, spatial_2, 2)
        max_workers : int | None
            Max workers to use for inverse transform. If None the max_workers
            will be estimated based on memory limits.
        """

        logger.info('Converting u/v to windspeed/winddirection for h5 output')
        heights = []
        for f in features:
            if re.match('U_(.*?)m'.lower(), f.lower()):
                heights.append(Feature.get_height(f))

        proc_mem = 4 * np.product(data.shape[:-1])
        n_procs = len(heights)
        max_workers = estimate_max_workers(max_workers, proc_mem, n_procs)

        futures = {}
        now = dt.now()
        if max_workers == 1:
            for height in heights:
                u_idx = features.index(f'U_{height}m')
                v_idx = features.index(f'V_{height}m')
                cls.invert_uv_single_pair(data, lat_lon, u_idx, v_idx)
                logger.info(f'U/V pair at height {height}m inverted.')
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                for height in heights:
                    u_idx = features.index(f'U_{height}m')
                    v_idx = features.index(f'V_{height}m')
                    future = exe.submit(cls.invert_uv_single_pair, data,
                                        lat_lon, u_idx, v_idx)
                    futures[future] = height

                logger.info(f'Started inverse transforms on {len(heights)} '
                            f'U/V pairs in {dt.now() - now}. ')

                for i, _ in enumerate(as_completed(futures)):
                    future.result()
                    logger.debug(f'{i+1} out of {len(futures)} inverse '
                                 'transforms completed.')

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
    def _write_output(cls, data, features, lat_lon, times, out_file,
                      meta_data=None, max_workers=None):
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
        """
        cls.invert_uv_features(data, features, lat_lon,
                               max_workers=max_workers)
        features = cls.get_renamed_features(features)
        meta = pd.DataFrame({'latitude': lat_lon[..., 0].flatten(),
                             'longitude': lat_lon[..., 1].flatten()})
        with RexOutputs(out_file, 'w') as fh:
            fh.meta = meta
            fh._set_time_index('time_index', times)

            data = cls.enforce_limits(features, data)

            for i, f in enumerate(features):
                attrs = H5_ATTRS[Feature.get_basename(f)]
                flat_data = data[..., i].reshape((-1, len(times)))
                flat_data = np.transpose(flat_data, (1, 0))
                fh.add_dataset(out_file, f, flat_data, dtype=attrs['dtype'],
                               attrs=attrs, chunks=attrs['chunks'])
                logger.debug(f'Added {f} to output file')

            if meta_data is not None:
                fh.run_attrs = {'gan_meta': json.dumps(meta_data)}
        logger.info(f'Saved output of size {data.shape} to: {out_file}')
