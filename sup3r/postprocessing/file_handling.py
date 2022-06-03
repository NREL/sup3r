"""Output handling

author : @bbenton
"""
from abc import abstractmethod
from attr import attr
import numpy as np
from datetime import datetime as dt
from netCDF4 import Dataset, date2num
import xarray as xr
import pandas as pd
import logging
from scipy.interpolate import RBFInterpolator
import warnings
import re

from sup3r.utilities.utilities import invert_uv
from sup3r.preprocessing.feature_handling import Feature

from reV.handlers.outputs import Outputs

logger = logging.getLogger(__name__)


def get_H5_attrs(feature):
    """Get attributes for feature being written to H5 file

    Parameters
    ----------
    feature : str
        Name of feature to write to H5 file

    Returns
    -------
    dict
        Dictionary of attributes for given feature
    """
    if 'windspeed' in feature.lower():
        attrs = {'fill_value': 65535, 'scale_factor': 100.0,
                 'units': 'm s-1'}
    if 'winddirection' in feature.lower():
        attrs = {'fill_value': 65535, 'scale_factor': 100.0,
                 'units': 'degree'}
    if 'temperature' in feature.lower():
        attrs = {'fill_value': 32767, 'scale_factor': 100.0,
                 'units': 'C'}
    if 'pressure' in feature.lower():
        attrs = {'fill_value': 65535, 'scale_factor': 0.1,
                 'units': 'Pa'}
    if 'bvfmo' in feature.lower():
        attrs = {'fill_value': 65535, 'scale_factor': 0.1,
                 'units': 'm s-2'}
    if 'bvf_squared' in feature.lower():
        attrs = {'fill_value': 65535, 'scale_factor': 0.1,
                 'units': 's-2'}
    return attrs


class OutputHandler:
    """Class to handle forward pass output. This includes transforming features
    back to their original form and outputting to the correct file format.
    """

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

        s_enhance = shape[0] // low_res_lat_lon.shape[0]
        old_points = np.zeros((np.product(low_res_lat_lon.shape[:-1]), 2))
        new_points = np.zeros((np.product(shape), 2))
        old_lats = low_res_lat_lon[..., 0].flatten()
        old_lons = low_res_lat_lon[..., 1].flatten()

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

        new_lats = RBFInterpolator(old_points, old_lats)(new_points)
        new_lons = RBFInterpolator(old_points, old_lons)(new_points)

        return np.concatenate([new_lats.reshape(shape)[..., np.newaxis],
                               new_lons.reshape(shape)[..., np.newaxis]],
                              axis=-1)

    @staticmethod
    def get_times(low_res_times, shape):
        """Get array of times for high res NETCDF output file

        Parameters
        ----------
        low_res_times : list
            List of np.datetime64 objects for input data.
        shape : int
            Number of time steps for high res time array
        Returns
        -------
        ndarray
            Array of times for high res NETCDF output file. In hours since
            1800-01-01.
        """
        t_enhance = int(shape / len(low_res_times))
        offset = (low_res_times[1] - low_res_times[0])
        low_res_offset = offset / np.timedelta64(1, 's')
        new_offset = offset / np.timedelta64(t_enhance, 's')

        msg = 'Found a difference of 0 seconds between successive file times'
        if new_offset == 0:
            logger.warning(msg)
            warnings.warn(msg)

        freq = pd.tseries.offsets.DateOffset(seconds=new_offset)
        end_time = low_res_times[-1] + np.timedelta64(int(low_res_offset), 's')
        time_index = pd.date_range(low_res_times[0], end_time, freq=freq,
                                   closed='left')
        return time_index

    @abstractmethod
    def write_output(self):
        """Write high res data with new coordinates and meta data to output"""


class OutputHandlerNC(OutputHandler):
    """OutputHandler subclass for NETCDF files"""

    @staticmethod
    def convert_times(time_index, time_description):
        """Convert times to NETCDF format

        Parameters
        ----------
        time_index : list
            List of np.datetime64 objects for high res data.
        time_description : string
            Description of time. e.g. minutes since 2016-01-30 00:00:00

        Returns
        -------
        ndarray
            Array of times for high res NETCDF output file. In hours since
            1800-01-01.
        """
        time_index = [np.datetime64(t) for t in time_index]
        t_ref = np.datetime64('1970-01-01T00:00:00')
        time_index = [(t - t_ref) / np.timedelta64(1, 's') for t in time_index]
        time_index = [date2num(dt.utcfromtimestamp(t), time_description,
                               has_year_zero=False, calendar='standard')
                      for t in time_index]
        return time_index

    @classmethod
    def write_output(cls, data, features, low_res_lat_lon,
                     low_res_times, time_description, out_file,
                     meta_data=None):
        """Write forward pass output to NETCDF file

        Parameters
        ----------
        data : ndarray
            (spatial_1, spatial_2, temporal, features)
            High resolution forward pass output
        features : list
            List of feature names corresponding to the last dimension of data
        low_res_lat_lon : ndarray
            Array of lat/lon for input data.
            (spatial_1, spatial_2, 2)
            Last dimension has ordering (lat, lon)
        low_res_times : list
            List of np.datetime64 objects for input data.
        time_description : string
            Description of time. e.g. minutes since 2016-01-30 00:00:00
        out_file : string
            Output file path
        meta_data : dict | None
            Dictionary of meta data from model
        """

        with Dataset(out_file, mode='w', format='NETCDF4_CLASSIC') as ncfile:
            ncfile.createDimension('south_north', data.shape[0])
            ncfile.createDimension('west_east', data.shape[1])
            ncfile.createDimension('Time', None)

            ncfile.title = "Forward pass output"
            if meta_data is not None:
                ncfile.description = str(meta_data)

            lat = ncfile.createVariable('XLAT', np.float32,
                                        ('south_north', 'west_east'))
            lat.units = 'degree_north'
            lat.long_name = 'latitude'
            lon = ncfile.createVariable('XLONG', np.float32,
                                        ('south_north', 'west_east'))
            lon.units = 'degree_east'
            lon.long_name = 'longitude'

            lat_lon = cls.get_lat_lon(low_res_lat_lon, data.shape[:2])
            lat[:, :] = lat_lon[..., 0]
            lon[:, :] = lat_lon[..., 1]

            time = ncfile.createVariable('XTIME', np.float32, ('Time',))
            time.description = time_description
            time.long_name = 'time'

            times = cls.get_times(low_res_times, data.shape[-2])
            time[:] = cls.convert_times(times, time_description)

            for i, f in enumerate(features):
                nc_feature = ncfile.createVariable(
                    f, np.float32, ('Time', 'south_north', 'west_east'))
                nc_feature[:, :, :] = np.transpose(data[..., i], (2, 0, 1))

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
    def invert_uv_features(cls, data, features, lat_lon):
        """Invert U/V to windspeed and winddirection

        Parameters
        ----------
        data : ndarray
            High res data from forward pass
            (spatial_1, spatial_2, temporal, features)
        features : list
            List of output features
        lat_lon : ndarray
            High res lat/lon array
            (spatial_1, spatial_2, 2)

        Returns
        -------
        ndarray
            High res data with u/v -> windspeed/winddirection for each height
        list
            List of renamed features u/v -> windspeed/winddirection for each
            height
        """

        heights = []
        renamed_features = features.copy()
        data_out = data.copy()
        for f in features:
            if re.match('U_(.*?)m'.lower(), f.lower()):
                heights.append(Feature.get_height(f))

        for height in heights:
            u_idx = features.index(f'U_{height}m')
            v_idx = features.index(f'V_{height}m')

            ws, wd = invert_uv(data[..., u_idx], data[..., v_idx], lat_lon)

            data_out[..., u_idx] = ws
            data_out[..., v_idx] = wd

            renamed_features[u_idx] = f'windspeed_{height}m'
            renamed_features[v_idx] = f'winddirection_{height}m'

        return data_out, renamed_features

    @classmethod
    def write_output(cls, data, features, low_res_lat_lon,
                     low_res_times, time_description, out_file):
        """Write forward pass output to NETCDF file

        Parameters
        ----------
        data : ndarray
            (spatial_1, spatial_2, temporal, features)
            High resolution forward pass output
        features : list
            List of feature names corresponding to the last dimension of data
        low_res_lat_lon : ndarray
            Array of lat/lon for input data.
            (spatial_1, spatial_2, 2)
            Last dimension has ordering (lat, lon)
        low_res_times : list
            List of np.datetime64 objects for input data.
        time_description : dict
            Description of time. e.g.
            {'description': 'minutes since 2016-01-30 00:00:00'}
        out_file : string
            Output file path
        """

        out_file = out_file.split('.')[0] + '.h5'
        lat_lon = cls.get_lat_lon(low_res_lat_lon, data.shape[:2])
        times = cls.get_times(low_res_times, data.shape[-2])
        data, renamed_features = cls.invert_uv_features(data, features,
                                                        lat_lon)
        meta = pd.DataFrame({'latitude': lat_lon[..., 0].flatten(),
                             'longitude': lat_lon[..., 1].flatten()})

        with Outputs(out_file, 'w') as fh:
            fh.time_index = times
            fh.meta = meta
            fh.attrs['time_index'] = time_description

            for i, f in enumerate(renamed_features):
                attrs = get_H5_attrs(f)
                if attrs['scale_factor'] != 1:
                    data_type = np.int
                else:
                    data_type = np.float32
                flat_data = data[..., i].reshape((-1, len(times)))
                flat_data = np.transpose(flat_data, (1, 0))
                Outputs.add_dataset(out_file, f, flat_data, dtype=data_type,
                                    attrs=attrs)
