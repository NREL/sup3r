"""Output handling"""
import json
import logging
import os
from abc import abstractmethod
from warnings import warn

import numpy as np
import pandas as pd
import xarray as xr
from rex.outputs import Outputs as BaseRexOutputs
from scipy.interpolate import griddata

from sup3r import __version__
from sup3r.preprocessing.derivers.utilities import parse_feature
from sup3r.utilities import VERSION_RECORD
from sup3r.utilities.utilities import pd_date_range

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
            'clearsky_ratio': {'scale_factor': 10000.0,
                               'units': 'ratio',
                               'dtype': 'uint16',
                               'chunks': (2000, 500),
                               'min': 0,
                               'max': 1},
            'dhi': {'scale_factor': 1.0,
                    'units': 'W/m2',
                    'dtype': 'uint16',
                    'chunks': (2000, 500),
                    'min': 0,
                    'max': 1350},
            'dni': {'scale_factor': 1.0,
                    'units': 'W/m2',
                    'dtype': 'uint16',
                    'chunks': (2000, 500),
                    'min': 0,
                    'max': 1350},
            'ghi': {'scale_factor': 1.0,
                    'units': 'W/m2',
                    'dtype': 'uint16',
                    'chunks': (2000, 500),
                    'min': 0,
                    'max': 1350},
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
            'pr': {'scale_factor': 1,
                   'units': 'kg m-2 s-1',
                   'dtype': 'float32',
                   'min': 0,
                   'chunks': (2000, 250)},
            'srl': {'scale_factor': 1,
                    'units': 'm',
                    'dtype': 'float32',
                    'min': 0,
                    'chunks': (2000, 250)}
            }


class OutputMixin:
    """Methods used by various Output and Collection classes"""

    @staticmethod
    def get_time_dim_name(filepath):
        """Get the name of the time dimension in the given file

        Parameters
        ----------
        filepath : str
            Path to the file

        Returns
        -------
        time_key : str
            Name of the time dimension in the given file
        """

        handle = xr.open_dataset(filepath)
        valid_vars = set(handle.dims)
        time_key = list({'time', 'Time'}.intersection(valid_vars))
        if len(time_key) > 0:
            return time_key[0]
        return 'time'

    @staticmethod
    def get_dset_attrs(feature):
        """Get attrributes for output feature

        Parameters
        ----------
        feature : str
            Name of feature to write

        Returns
        -------
        attrs : dict
            Dictionary of attributes for requested dset
        dtype : str
            Data type for requested dset. Defaults to float32
        """
        feat_base_name = parse_feature(feature).basename
        if feat_base_name in H5_ATTRS:
            attrs = H5_ATTRS[feat_base_name]
            dtype = attrs.get('dtype', 'float32')
        else:
            attrs = {}
            dtype = 'float32'
            msg = ('Could not find feature "{}" with base name "{}" in '
                   'H5_ATTRS global variable. Writing with float32 and no '
                   'chunking.'.format(feature, feat_base_name))
            logger.warning(msg)
            warn(msg)

        return attrs, dtype

    @staticmethod
    def _init_h5(out_file, time_index, meta, global_attrs):
        """Initialize the output h5 file to save data to.

        Parameters
        ----------
        out_file : str
            Output file path - must not yet exist.
        time_index : pd.datetimeindex
            Full datetime index of final output data.
        meta : pd.DataFrame
            Full meta dataframe for the final output data.
        global_attrs : dict
            Namespace of file-global attributes for the final output data.
        """

        with RexOutputs(out_file, mode='w-') as f:
            logger.info('Initializing output file: {}'
                        .format(out_file))
            logger.info('Initializing output file with shape {} '
                        'and meta data:\n{}'
                        .format((len(time_index), len(meta)), meta))
            f.time_index = time_index
            f.meta = meta
            f.run_attrs = global_attrs

    @classmethod
    def _ensure_dset_in_output(cls, out_file, dset, data=None):
        """Ensure that dset is initialized in out_file and initialize if not.

        Parameters
        ----------
        out_file : str
            Pre-existing H5 file output path
        dset : str
            Dataset name
        data : T_Array | None
            Optional data to write to dataset if initializing.
        """

        with RexOutputs(out_file, mode='a') as f:
            if dset not in f.dsets:
                attrs, dtype = cls.get_dset_attrs(dset)
                logger.info('Initializing dataset "{}" with shape {} and '
                            'dtype {}'.format(dset, f.shape, dtype))
                f._create_dset(dset, f.shape, dtype,
                               attrs=attrs, data=data,
                               chunks=attrs.get('chunks', None))

    @classmethod
    def write_data(cls, out_file, dsets, time_index, data_list, meta,
                   global_attrs=None):
        """Write list of datasets to out_file.

        Parameters
        ----------
        out_file : str
            Pre-existing H5 file output path
        dsets : list
            list of datasets to write to out_file
        time_index : pd.DatetimeIndex()
            Pandas datetime index to use for file time_index.
        data_list : list
            List of np.ndarray objects to write to out_file
        meta : pd.DataFrame
            Full meta dataframe for the final output data.
        global_attrs : dict
            Namespace of file-global attributes for the final output data.
        """
        tmp_file = out_file.replace('.h5', '.h5.tmp')
        with RexOutputs(tmp_file, 'w') as fh:
            fh.meta = meta
            fh.time_index = time_index

            for dset, data in zip(dsets, data_list):
                attrs, dtype = cls.get_dset_attrs(dset)
                fh.add_dataset(tmp_file, dset, data, dtype=dtype,
                               attrs=attrs, chunks=attrs['chunks'])
                logger.info(f'Added {dset} to output file {out_file}.')

            if global_attrs is not None:
                attrs = {k: v if isinstance(v, str) else json.dumps(v)
                         for k, v in global_attrs.items()}
                fh.run_attrs = attrs

        os.replace(tmp_file, out_file)
        msg = ('Saved output of size '
               f'{(len(data_list), *data_list[0].shape)} to: {out_file}')
        logger.info(msg)


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


class OutputHandler(OutputMixin):
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
        maxes = []
        mins = []
        for fidx, fn in enumerate(features):
            dset_name = parse_feature(fn).basename
            if dset_name not in H5_ATTRS:
                msg = f'Could not find "{dset_name}" in H5_ATTRS dict!'
                logger.error(msg)
                raise KeyError(msg)

            max = H5_ATTRS[dset_name].get('max', np.inf)
            min = H5_ATTRS[dset_name].get('min', -np.inf)
            logger.debug(f'Enforcing range of ({min}, {max} for "{fn}")')

            f_max = data[..., fidx].max()
            f_min = data[..., fidx].min()
            msg = f'{fn} has a max of {f_max} > {max}'
            if f_max > max:
                logger.warning(msg)
                warn(msg)
            msg = f'{fn} has a min of {f_min} > {min}'
            if f_min < min:
                logger.warning(msg)
                warn(msg)
            maxes.append(max)
            mins.append(min)

        data = np.maximum(data, mins)
        return np.minimum(data, maxes)

    @staticmethod
    def pad_lat_lon(lat_lon):
        """Pad lat lon grid with additional rows and columns to use for
        interpolation routine

        Parameters
        ----------
        lat_lon : ndarray
            Array of lat/lon for input data.
            (spatial_1, spatial_2, 2)
            Last dimension has ordering (lat, lon)
        shape : tuple
            (lons, lats) Shape of high res grid

        Returns
        -------
        lat_lon : ndarray
            Array of padded lat lons
            (spatial_1, spatial_2, 2)
            Last dimension has ordering (lat, lon)
        """

        # add row and column to boundaries
        padded_grid = np.zeros((2 + lat_lon.shape[0],
                                2 + lat_lon.shape[1], 2))

        # fill in interior values
        padded_grid[1:-1, 1:-1, :] = lat_lon

        # define edge spacing
        left_diffs = padded_grid[:, 2, 1] - padded_grid[:, 1, 1]
        right_diffs = padded_grid[:, -2, 1] - padded_grid[:, -3, 1]
        top_diffs = padded_grid[1, :, 0] - padded_grid[2, :, 0]
        bottom_diffs = padded_grid[-3, :, 0] - padded_grid[-2, :, 0]

        # use edge spacing to define new boundary values
        padded_grid[:, 0, 1] = padded_grid[:, 1, 1] - left_diffs
        padded_grid[:, 0, 0] = padded_grid[:, 1, 0]
        padded_grid[:, -1, 1] = padded_grid[:, -2, 1] + right_diffs
        padded_grid[:, -1, 0] = padded_grid[:, -2, 0]
        padded_grid[0, :, 0] = padded_grid[1, :, 0] + top_diffs
        padded_grid[0, :, 1] = padded_grid[1, :, 1]
        padded_grid[-1, :, 0] = padded_grid[-2, :, 0] - bottom_diffs
        padded_grid[-1, :, 1] = padded_grid[-2, :, 1]

        # use surrounding cells to define corner values
        # top left
        padded_grid[0, 0, 0] = padded_grid[0, 1, 0]
        padded_grid[0, 0, 1] = padded_grid[1, 0, 1]
        # top right
        padded_grid[0, -1, 0] = padded_grid[0, -2, 0]
        padded_grid[0, -1, 1] = padded_grid[1, -1, 1]
        # bottom left
        padded_grid[-1, 0, 0] = padded_grid[-1, 1, 0]
        padded_grid[-1, 0, 1] = padded_grid[-2, 0, 1]
        # bottom right
        padded_grid[-1, -1, 0] = padded_grid[-1, -2, 0]
        padded_grid[-1, -1, 1] = padded_grid[-2, -1, 1]

        return padded_grid

    @staticmethod
    def is_increasing_lons(lat_lon):
        """Check if longitudes are in increasing order. Need to check this
        for interpolation routine. This is primarily to identify whether the
        lons go through the 180 -> -180 boundary, which creates a
        discontinuity. For example, [130, 180, -130, -80]. If any lons go from
        positive to negative the lons need to be shifted to the range 0-360.

        Parameters
        ----------
        lat_lon : ndarray
            Array of lat/lon for input data.
            (spatial_1, spatial_2, 2)
            Last dimension has ordering (lat, lon)

        Returns
        -------
        bool
            Whether all lons are in increasing order or not

        """
        for i in range(lat_lon.shape[0]):
            if lat_lon[i, :, 1][-1] < lat_lon[i, :, 1][0]:
                return False
        return True

    @classmethod
    def get_lat_lon(cls, low_res_lat_lon, shape):
        """Get lat lon arrays for high res output file

        Parameters
        ----------
        low_res_lat_lon : ndarray
            Array of lat/lon for input data. Longitudes must be arranged in
            a counter-clockwise direction (when looking down from above the
            north pole). e.g. [-50, -25, 25, 50] or [130, 180, -130, -80]. The
            latter passes through the 180 -> -180 boundary and will be
            temporarily shifted to the 0-360 range before interpolation.
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

        # ensure lons are between -180 and 180
        low_res_lat_lon[..., 1] = (low_res_lat_lon[..., 1] + 180) % 360 - 180

        # check if lons go through the 180 -> -180 boundary.
        if not cls.is_increasing_lons(low_res_lat_lon):
            low_res_lat_lon[..., 1] = (low_res_lat_lon[..., 1] + 360) % 360

        # pad lat lon grid
        padded_grid = cls.pad_lat_lon(low_res_lat_lon)
        lats = padded_grid[..., 0].flatten()
        lons = padded_grid[..., 1].flatten()

        lr_y, lr_x = low_res_lat_lon.shape[:-1]
        hr_y, hr_x = shape

        # assume outer bounds of mesh (0, 10) w/ points on inside of that range
        y = np.arange(0, 10, 10 / lr_y) + 5 / lr_y
        x = np.arange(0, 10, 10 / lr_x) + 5 / lr_x

        # add values due to padding
        y = np.concatenate([[y[0] - 10 / lr_y], y, [y[-1] + 10 / lr_y]])
        x = np.concatenate([[x[0] - 10 / lr_x], x, [x[-1] + 10 / lr_x]])

        # remesh (0, 10) with high res spacing
        new_y = np.arange(0, 10, 10 / hr_y) + 5 / hr_y
        new_x = np.arange(0, 10, 10 / hr_x) + 5 / hr_x

        X, Y = np.meshgrid(x, y)
        old = np.array([Y.flatten(), X.flatten()]).T
        X, Y = np.meshgrid(new_x, new_y)
        new = np.array([Y.flatten(), X.flatten()]).T
        lons = griddata(old, lons, new)
        lats = griddata(old, lats, new)

        lons = (lons + 180) % 360 - 180
        lat_lon = np.dstack((lats.reshape(shape), lons.reshape(shape)))
        logger.debug('Finished getting high resolution lat / lon grid')

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
        logger.debug(f'Low res times: {low_res_times[0]} to '
                     f'{low_res_times[-1]}')
        t_enhance = int(shape / len(low_res_times))
        if len(low_res_times) > 1:
            offset = low_res_times[1] - low_res_times[0]
        else:
            offset = np.timedelta64(24, 'h')

        freq = offset / np.timedelta64(1, 's')
        freq = int(60 * np.round(freq / 60) / t_enhance)
        times = [low_res_times[0] + i * np.timedelta64(freq, 's')
                 for i in range(shape)]
        freq = pd.tseries.offsets.DateOffset(seconds=freq)
        times = pd_date_range(times[0], times[-1], freq=freq)
        logger.debug(f'High res times: {times[0]} to {times[-1]}')
        return times

    @classmethod
    @abstractmethod
    def _write_output(cls, data, features, lat_lon, times, out_file, meta_data,
                      max_workers=None, gids=None):
        """Write output to file with specified times and lats/lons"""

    @classmethod
    def write_output(cls, data, features, low_res_lat_lon, low_res_times,
                     out_file, meta_data=None, max_workers=None, gids=None):
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
        gids : list
            List of coordinate indices used to label each lat lon pair and to
            help with spatial chunk data collection
        """
        lat_lon = cls.get_lat_lon(low_res_lat_lon, data.shape[:2])
        times = cls.get_times(low_res_times, data.shape[-2])
        cls._write_output(data, features, lat_lon, times, out_file,
                          meta_data=meta_data, max_workers=max_workers,
                          gids=gids)