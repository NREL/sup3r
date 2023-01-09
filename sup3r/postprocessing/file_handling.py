"""Output handling

author : @bbenton
"""
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import xarray as xr
import pandas as pd
import logging
from scipy.interpolate import griddata
import re
from datetime import datetime as dt
import json
import os
from warnings import warn

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


class OutputMixIn:
    """MixIn class with methods used by various Output and Collection classes
    """

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
        feat_base_name = Feature.get_basename(feature)
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
        data : np.ndarray | None
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
               f'{(len(data_list),) + data_list[0].shape} to: {out_file}')
        logger.info(msg)


class OutputHandler(OutputMixIn):
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
        maxes = [H5_ATTRS[Feature.get_basename(f)].get('max', np.inf)
                 for f in features]
        mins = [H5_ATTRS[Feature.get_basename(f)].get('min', -np.inf)
                for f in features]
        data = np.maximum(data, mins)
        data = np.minimum(data, maxes)

        return data

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
            offset = (low_res_times[1] - low_res_times[0])
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


class OutputHandlerNC(OutputHandler):
    """OutputHandler subclass for NETCDF files"""

    # pylint: disable=W0613
    @classmethod
    def _write_output(cls, data, features, lat_lon, times, out_file,
                      meta_data=None, max_workers=None, gids=None):
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
        coords = {'Times': (['Time'], times),
                  'XLAT': (['south_north', 'east_west'], lat_lon[..., 0]),
                  'XLONG': (['south_north', 'east_west'], lat_lon[..., 1])}

        data_vars = {}
        for i, f in enumerate(features):
            data_vars[f] = (['Time', 'south_north', 'east_west'],
                            np.transpose(data[..., i], (2, 0, 1)))

        attrs = {}
        if meta_data is not None:
            attrs = {k: v if isinstance(v, str) else json.dumps(v)
                     for k, v in meta_data.items()}

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

        heights = []
        for f in features:
            if re.match('U_(.*?)m'.lower(), f.lower()):
                heights.append(Feature.get_height(f))
        if heights:
            logger.info('Converting u/v to windspeed/winddirection for h5'
                        ' output')
            logger.debug('Found heights {} for output features {}'
                         .format(heights, features))

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
                    try:
                        future.result()
                    except Exception as e:
                        msg = ('Failed to invert the U/V pair for for height '
                               f'{futures[future]}')
                        logger.exception(msg)
                        raise RuntimeError(msg) from e
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

        cls.invert_uv_features(data, features, lat_lon,
                               max_workers=max_workers)
        features = cls.get_renamed_features(features)
        data = cls.enforce_limits(features, data)
        return data, features

    @classmethod
    def _write_output(cls, data, features, lat_lon, times, out_file,
                      meta_data=None, max_workers=None, gids=None):
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
        msg = (f'Output data shape ({data.shape}) and lat_lon shape '
               f'({lat_lon.shape}) conflict.')
        assert data.shape[:2] == lat_lon.shape[:-1], msg
        msg = (f'Output data shape ({data.shape}) and times shape '
               f'({len(times)}) conflict.')
        assert data.shape[-2] == len(times), msg
        data, features = cls._transform_output(data.copy(), features, lat_lon,
                                               max_workers)
        gids = (gids if gids is not None
                else np.arange(np.product(lat_lon.shape[:-1])))
        meta = pd.DataFrame({'gid': gids.flatten(),
                             'latitude': lat_lon[..., 0].flatten(),
                             'longitude': lat_lon[..., 1].flatten()})
        data_list = []
        for i, f in enumerate(features):
            flat_data = data[..., i].reshape((-1, len(times)))
            flat_data = np.transpose(flat_data, (1, 0))
            data_list.append(flat_data)
        cls.write_data(out_file, features, times, data_list, meta, meta_data)
