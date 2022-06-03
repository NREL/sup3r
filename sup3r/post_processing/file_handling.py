"""Output handling

author : @bbenton
"""
from abc import abstractmethod
import numpy as np
from datetime import datetime as dt
from netCDF4 import Dataset, date2num
import xarray as xr
import pandas as pd
import logging
from scipy.interpolate import RBFInterpolator
import warnings

logger = logging.getLogger(__name__)


class OutputHandler:
    """Class to handle forward pass output. This includes transforming features
    back to their original form and outputting to the correct file format.
    """
    @staticmethod
    @abstractmethod
    def get_lat_lon():
        """Get lat lon arrays for writing to output file"""

    @staticmethod
    @abstractmethod
    def get_times():
        """Get time array for writing to output file"""

    @abstractmethod
    def write_output(self):
        """Write output in correct format with new times, grid, and meta data
        """


class OutputHandlerNC(OutputHandler):
    """OutputHandler subclass for NETCDF files"""

    @staticmethod
    def get_lat_lon(low_res_lats, low_res_lons, shape):
        """Get lat lon arrays for high res NETCDF output file

        Parameters
        ----------
        low_res_lats : ndarray
            Array of lats for input data.
        low_res_lons : ndarray
            Array of lons for input data.
        shape : tuple
            (lons, lats) Shape of high res grid

        Returns
        -------
        lons : ndarray
            Array of lons for high res NETCDF output file
        lats : ndarray
            Array of lats for high res NETCDF output file
        """

        s_enhance = shape[0] // low_res_lats.shape[0]

        old_points = np.zeros((np.product(low_res_lats.shape), 2), dtype=int)
        new_points = np.zeros((np.product(shape), 2), dtype=int)
        old_lats = low_res_lats.flatten()
        old_lons = low_res_lons.flatten()

        new_count = old_count = 0
        for i in range(shape[0]):
            for j in range(shape[1]):
                if i % s_enhance == 0 and j % s_enhance == 0:
                    old_points[old_count, 0] = i
                    old_points[old_count, 1] = j
                    old_count += 1
                new_points[new_count, 0] = i
                new_points[new_count, 1] = j
                new_count += 1

        new_lats = RBFInterpolator(old_points, old_lats)(new_points)
        new_lons = RBFInterpolator(old_points, old_lons)(new_points)

        return new_lats.reshape(shape), new_lons.reshape(shape)

    @staticmethod
    def get_times(low_res_times, time_description, shape):
        """Get array of times for high res NETCDF output file

        Parameters
        ----------
        low_res_times : list
            List of np.datetime64 objects for input data.
        time_description : string
            Description of time. e.g. minutes since 2016-01-30 00:00:00
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
        t_ref = np.datetime64('1970-01-01T00:00:00')
        time_index = [(t - t_ref) / np.timedelta64(1, 's') for t in time_index]
        time_index = [date2num(dt.utcfromtimestamp(t), time_description,
                               has_year_zero=False, calendar='standard')
                      for t in time_index]
        return time_index

    @classmethod
    def write_output(cls, data, features, low_res_lats, low_res_lons,
                     low_res_times, time_description, out_file):
        """Write forward pass output to NETCDF file

        Parameters
        ----------
        data : ndarray
            (spatial_1, spatial_2, temporal, features)
            High resolution forward pass output
        features : list
            List of feature names corresponding to the last dimension of data
        low_res_lats : ndarray
            Array of lats for input data.
        low_res_lons : ndarray
            Array of lons for input data.
        low_res_times : list
            List of np.datetime64 objects for input data.
        time_description : string
            Description of time. e.g. minutes since 2016-01-30 00:00:00
        out_file : string
            Output file path
        """
        with Dataset(out_file, mode='w', format='NETCDF4_CLASSIC') as ncfile:
            ncfile.createDimension('south_north', data.shape[0])
            ncfile.createDimension('west_east', data.shape[1])
            ncfile.createDimension('Time', None)

            ncfile.title = "Forward pass output"

            lat = ncfile.createVariable('XLAT', np.float32,
                                        ('south_north', 'west_east'))
            lat.units = 'degree_north'
            lat.long_name = 'latitude'
            lon = ncfile.createVariable('XLONG', np.float32,
                                        ('south_north', 'west_east'))
            lon.units = 'degree_east'
            lon.long_name = 'longitude'

            lat[:, :], lon[:, :] = cls.get_lat_lon(low_res_lats,
                                                   low_res_lons,
                                                   data.shape[:2])

            time = ncfile.createVariable('XTIME', np.float32, ('Time',))
            time.description = time_description
            time.long_name = 'time'

            time[:] = cls.get_times(low_res_times, time_description,
                                    data.shape[-2])

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
