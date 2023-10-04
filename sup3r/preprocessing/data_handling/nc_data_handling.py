"""Data handling for netcdf files.
@author: bbenton
"""

import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime as dt
from typing import ClassVar

import numpy as np
import pandas as pd
import xarray as xr
from rex import Resource
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import KDTree
from scipy.stats import mode

from sup3r.preprocessing.data_handling.base import DataHandler, DataHandlerDC
from sup3r.preprocessing.feature_handling import (
    BVFreqMon, BVFreqSquaredNC, ClearSkyRatioCC, Feature, InverseMonNC,
    LatLonNC, PotentialTempNC, PressureNC, Rews, Shear, Tas, TasMax, TasMin,
    TempNC, TempNCforCC, UWind, UWindPowerLaw, VWind, VWindPowerLaw,
    WinddirectionNC, WindspeedNC)
from sup3r.utilities.interpolation import Interpolator
from sup3r.utilities.utilities import (estimate_max_workers, get_time_dim_name,
                                       np_to_pd_times)

np.random.seed(42)

logger = logging.getLogger(__name__)


class DataHandlerNC(DataHandler):
    """Data Handler for NETCDF data"""

    FEATURE_REGISTRY: ClassVar[dict] = {
        'BVF2_(.*)': BVFreqSquaredNC,
        'BVF_MO_(.*)': BVFreqMon,
        'RMOL': InverseMonNC,
        'U_(.*)': UWind,
        'V_(.*)': VWind,
        'Windspeed_(.*)': WindspeedNC,
        'Winddirection_(.*)': WinddirectionNC,
        'lat_lon': LatLonNC,
        'Shear_(.*)': Shear,
        'REWS_(.*)': Rews,
        'Temperature_(.*)': TempNC,
        'Pressure_(.*)': PressureNC,
        'PotentialTemp_(.*)': PotentialTempNC,
        'PT_(.*)': PotentialTempNC,
        'topography': ['HGT', 'orog'],
    }

    CHUNKS: ClassVar[dict] = {
        'XTIME': 100,
        'XLAT': 150,
        'XLON': 150,
        'south_north': 150,
        'west_east': 150,
        'Time': 100,
    }
    """CHUNKS sets the chunk sizes to extract from the data in each dimension.
    Chunk sizes that approximately match the data volume being extracted
    typically results in the most efficient IO."""

    @property
    def extract_workers(self):
        """Get upper bound for extract workers based on memory limits. Used to
        extract data from source dataset"""
        # This large multiplier is due to the height interpolation allocating
        # multiple arrays with up to 60 vertical levels
        proc_mem = 6 * 64 * self.grid_mem * len(self.time_index)
        proc_mem /= len(self.time_chunks)
        n_procs = len(self.time_chunks) * len(self.extract_features)
        n_procs = int(np.ceil(n_procs))
        extract_workers = estimate_max_workers(self._extract_workers, proc_mem,
                                               n_procs)
        return extract_workers

    @classmethod
    def source_handler(cls, file_paths, **kwargs):
        """Xarray data handler

        Note that xarray appears to treat open file handlers as singletons
        within a threadpool, so its okay to open this source_handler without a
        context handler or a .close() statement.

        Parameters
        ----------
        file_paths : str | list
            paths to data files
        kwargs : dict
            kwargs passed to source handler for data extraction. e.g. This
            could be {'parallel': True,
                      'chunks': {'south_north': 120, 'west_east': 120}}
            which then gets passed to xr.open_mfdataset(file, **kwargs)

        Returns
        -------
        data : xarray.Dataset
        """
        time_key = get_time_dim_name(file_paths[0])
        default_kws = {
            'combine': 'nested',
            'concat_dim': time_key,
            'chunks': cls.CHUNKS,
        }
        default_kws.update(kwargs)
        return xr.open_mfdataset(file_paths, **default_kws)

    @classmethod
    def get_file_times(cls, file_paths, **kwargs):
        """Get time index from data files

        Parameters
        ----------
        file_paths : list
            path to data file
        kwargs : dict
            kwargs passed to source handler for data extraction. e.g. This
            could be {'parallel': True,
                      'chunks': {'south_north': 120, 'west_east': 120}}
            which then gets passed to xr.open_mfdataset(file, **kwargs)

        Returns
        -------
        time_index : pd.Datetimeindex
            List of times as a Datetimeindex
        """
        handle = cls.source_handler(file_paths, **kwargs)

        if hasattr(handle, 'Times'):
            time_index = np_to_pd_times(handle.Times.values)
        elif hasattr(handle, 'indexes') and 'time' in handle.indexes:
            time_index = handle.indexes['time']
            if not isinstance(time_index, pd.DatetimeIndex):
                time_index = time_index.to_datetimeindex()
        elif hasattr(handle, 'times'):
            time_index = np_to_pd_times(handle.times.values)
        else:
            msg = (f'Could not get time_index for {file_paths}. '
                   'Assuming time independence.')
            time_index = None
            logger.warning(msg)
            warnings.warn(msg)

        return time_index

    @classmethod
    def get_time_index(cls, file_paths, max_workers=None, **kwargs):
        """Get time index from data files

        Parameters
        ----------
        file_paths : list
            path to data file
        max_workers : int | None
            Max number of workers to use for parallel time index building
        kwargs : dict
            kwargs passed to source handler for data extraction. e.g. This
            could be {'parallel': True,
                      'chunks': {'south_north': 120, 'west_east': 120}}
            which then gets passed to xr.open_mfdataset(file, **kwargs)

        Returns
        -------
        time_index : pd.Datetimeindex
            List of times as a Datetimeindex
        """
        max_workers = (len(file_paths) if max_workers is None else np.min(
            (max_workers, len(file_paths))))
        if max_workers == 1:
            return cls.get_file_times(file_paths, **kwargs)
        ti = {}
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = {}
            now = dt.now()
            for i, f in enumerate(file_paths):
                future = exe.submit(cls.get_file_times, [f], **kwargs)
                futures[future] = {'idx': i, 'file': f}

            logger.info(f'Started building time index from {len(file_paths)} '
                        f'files in {dt.now() - now}.')

            for i, future in enumerate(as_completed(futures)):
                try:
                    val = future.result()
                    if val is not None:
                        ti[futures[future]['idx']] = list(val)
                except Exception as e:
                    msg = ('Error while getting time index from file '
                           f'{futures[future]["file"]}.')
                    logger.exception(msg)
                    raise RuntimeError(msg) from e
                logger.debug(f'Stored {i+1} out of {len(futures)} file times')
        times = np.concatenate(list(ti.values()))
        return pd.DatetimeIndex(sorted(set(times)))

    @classmethod
    def extract_feature(cls,
                        file_paths,
                        raster_index,
                        feature,
                        time_slice=slice(None),
                        **kwargs,
                        ):
        """Extract single feature from data source. The requested feature
        can match exactly to one found in the source data or can have a
        matching prefix with a suffix specifying the height or pressure level
        to interpolate to. e.g. feature=U_100m -> interpolate exact match U to
        100 meters.

        Parameters
        ----------
        file_paths : list
            path to data file
        raster_index : ndarray
            Raster index array
        feature : str
            Feature to extract from data
        time_slice : slice
            slice of time to extract
        kwargs : dict
            kwargs passed to source handler for data extraction. e.g. This
            could be {'parallel': True,
                      'chunks': {'south_north': 120, 'west_east': 120}}
            which then gets passed to xr.open_mfdataset(file, **kwargs)

        Returns
        -------
        ndarray
            Data array for extracted feature
            (spatial_1, spatial_2, temporal)
        """
        logger.debug(f'Extracting {feature} with time_slice={time_slice}, '
                     f'raster_index={raster_index}, kwargs={kwargs}.')
        handle = cls.source_handler(file_paths, **kwargs)
        f_info = Feature(feature, handle)
        interp_height = f_info.height
        interp_pressure = f_info.pressure
        basename = f_info.basename

        if cls.has_exact_feature(feature, handle):
            feat_key = feature if feature in handle else feature.lower()
            fdata = cls.direct_extract(handle, feat_key, raster_index,
                                       time_slice)

        elif interp_height is not None and (
                cls.has_multilevel_feature(feature, handle)
                or cls.has_surrounding_features(feature, handle)):
            fdata = Interpolator.interp_var_to_height(
                handle, feature, raster_index, np.float32(interp_height),
                time_slice)
        elif interp_pressure is not None and cls.has_multilevel_feature(
                feature, handle):
            fdata = Interpolator.interp_var_to_pressure(
                handle, basename, raster_index, np.float32(interp_pressure),
                time_slice)

        else:
            msg = f'{feature} cannot be extracted from source data.'
            logger.exception(msg)
            raise ValueError(msg)

        fdata = np.transpose(fdata, (1, 2, 0))
        return fdata.astype(np.float32)

    @classmethod
    def direct_extract(cls, handle, feature, raster_index, time_slice):
        """Extract requested feature directly from source data, rather than
        interpolating to a requested height or pressure level

        Parameters
        ----------
        handle : xarray
            netcdf data object
        feature : str
            Name of feature to extract directly from source handler
        raster_index : list
            List of slices for raster index of spatial domain
        time_slice : slice
            slice of time to extract

        Returns
        -------
        fdata : ndarray
            Data array for requested feature
        """
        # Sometimes xarray returns fields with (Times, time, lats, lons)
        # with a single entry in the 'time' dimension so we include this [0]
        if len(handle[feature].dims) == 4:
            idx = tuple([time_slice, 0, *raster_index])
        elif len(handle[feature].dims) == 3:
            idx = tuple([time_slice, *raster_index])
        else:
            idx = tuple(raster_index)
        fdata = np.array(handle[feature][idx], dtype=np.float32)
        if len(fdata.shape) == 2:
            fdata = np.expand_dims(fdata, axis=0)
        return fdata

    @classmethod
    def get_full_domain(cls, file_paths):
        """Get full shape and min available lat lon. To simplify processing
        of full domain without needing to specify target and shape.

        Parameters
        ----------
        file_paths : list
            List of data file paths

        Returns
        -------
        target : tuple
            (lat, lon) for lower left corner
        lat_lon : ndarray
            Raw lat/lon array for entire domain
        """
        return cls.get_lat_lon(file_paths, [slice(None), slice(None)])

    @classmethod
    def compute_raster_index(cls, file_paths, target, grid_shape):
        """Get raster index for a given target and shape

        Parameters
        ----------
        file_paths : list
            List of input data file paths
        target : tuple
            Target coordinate for lower left corner of extracted data
        grid_shape : tuple
            Shape out extracted data

        Returns
        -------
        list
            List of slices corresponding to extracted data region
        """
        lat_lon = cls.get_lat_lon(file_paths[:1],
                                  [slice(None), slice(None)],
                                  invert_lat=False)
        cls._check_grid_extent(target, grid_shape, lat_lon)

        row, col = cls.get_closest_lat_lon(lat_lon, target)

        closest = tuple(lat_lon[row, col])
        logger.debug(f'Found closest coordinate {closest} to target={target}')
        if np.hypot(closest[0] - target[0], closest[1] - target[1]) > 1:
            msg = 'Closest coordinate to target is more than 1 degree away'
            logger.warning(msg)
            warnings.warn(msg)

        if cls.lats_are_descending(lat_lon):
            row_end = row + 1
            row_start = row_end - grid_shape[0]
        else:
            row_end = row + grid_shape[0]
            row_start = row
        raster_index = [
            slice(row_start, row_end),
            slice(col, col + grid_shape[1]),
        ]
        cls._validate_raster_shape(target, grid_shape, lat_lon, raster_index)
        return raster_index

    @classmethod
    def _check_grid_extent(cls, target, grid_shape, lat_lon):
        """Make sure the requested target coordinate lies within the available
        lat/lon grid.

        Parameters
        ----------
        target : tuple
            Target coordinate for lower left corner of extracted data
        grid_shape : tuple
            Shape out extracted data
        lat_lon : ndarray
            Array of lat/lon coordinates for entire available grid. Used to
            check whether computed raster only includes coordinates within this
            grid.
        """
        min_lat = np.min(lat_lon[..., 0])
        min_lon = np.min(lat_lon[..., 1])
        max_lat = np.max(lat_lon[..., 0])
        max_lon = np.max(lat_lon[..., 1])
        logger.debug('Calculating raster index from WRF file '
                     f'for shape {grid_shape} and target {target}')
        logger.debug(f'lat/lon (min, max): {min_lat}/{min_lon}, '
                     f'{max_lat}/{max_lon}')
        msg = (f'target {target} out of bounds with min lat/lon '
               f'{min_lat}/{min_lon} and max lat/lon {max_lat}/{max_lon}')
        assert (min_lat <= target[0] <= max_lat
                and min_lon <= target[1] <= max_lon), msg

    @classmethod
    def _validate_raster_shape(cls, target, grid_shape, lat_lon, raster_index):
        """Make sure the computed raster_index only includes coordinates within
        the available grid

        Parameters
        ----------
        target : tuple
            Target coordinate for lower left corner of extracted data
        grid_shape : tuple
            Shape out extracted data
        lat_lon : ndarray
            Array of lat/lon coordinates for entire available grid. Used to
            check whether computed raster only includes coordinates within this
            grid.
        raster_index : list
            List of slices selecting region from entire available grid.
        """
        if (raster_index[0].stop > lat_lon.shape[0]
                or raster_index[1].stop > lat_lon.shape[1]
                or raster_index[0].start < 0 or raster_index[1].start < 0):
            msg = (f'Invalid target {target}, shape {grid_shape}, and raster '
                   f'{raster_index} for data domain of size '
                   f'{lat_lon.shape[:-1]} with lower left corner '
                   f'({np.min(lat_lon[..., 0])}, {np.min(lat_lon[..., 1])}) '
                   f' and upper right corner ({np.max(lat_lon[..., 0])}, '
                   f'{np.max(lat_lon[..., 1])}).')
            raise ValueError(msg)

    def get_raster_index(self):
        """Get raster index for file data. Here we assume the list of paths in
        file_paths all have data with the same spatial domain. We use the first
        file in the list to compute the raster.

        Returns
        -------
        raster_index : np.ndarray
            2D array of grid indices
        """
        self.raster_file = (self.raster_file if self.raster_file is None else
                            self.raster_file.replace('.txt', '.npy'))
        if self.raster_file is not None and os.path.exists(self.raster_file):
            logger.debug(f'Loading raster index: {self.raster_file} '
                         f'for {self.input_file_info}')
            raster_index = np.load(self.raster_file, allow_pickle=True)
            raster_index = list(raster_index)
        else:
            check = self.grid_shape is not None and self.target is not None
            msg = ('Must provide raster file or shape + target to get '
                   'raster index')
            assert check, msg
            raster_index = self.compute_raster_index(self.file_paths,
                                                     self.target,
                                                     self.grid_shape)
            logger.debug('Found raster index with row, col slices: {}'.format(
                raster_index))

            if self.raster_file is not None:
                basedir = os.path.dirname(self.raster_file)
                if not os.path.exists(basedir):
                    os.makedirs(basedir)
                logger.debug(f'Saving raster index: {self.raster_file}')
                np.save(self.raster_file.replace('.txt', '.npy'), raster_index)

        return raster_index


class DataHandlerNCforERA(DataHandlerNC):
    """Data Handler for NETCDF ERA5 data"""

    FEATURE_REGISTRY = DataHandlerNC.FEATURE_REGISTRY.copy()
    FEATURE_REGISTRY.update({'Pressure_(.*)m': 'level_(.*)'})


class DataHandlerNCforCC(DataHandlerNC):
    """Data Handler for NETCDF climate change data"""

    FEATURE_REGISTRY = DataHandlerNC.FEATURE_REGISTRY.copy()
    FEATURE_REGISTRY.update({
        'U_(.*)': 'ua_(.*)',
        'V_(.*)': 'va_(.*)',
        'relativehumidity_2m': 'hurs',
        'relativehumidity_min_2m': 'hursmin',
        'relativehumidity_max_2m': 'hursmax',
        'clearsky_ratio': ClearSkyRatioCC,
        'lat_lon': LatLonNC,
        'Pressure_(.*)': 'plev_(.*)',
        'Temperature_(.*)': TempNCforCC,
        'temperature_2m': Tas,
        'temperature_max_2m': TasMax,
        'temperature_min_2m': TasMin,
    })

    CHUNKS: ClassVar[dict] = {'time': 5, 'lat': 20, 'lon': 20}
    """CHUNKS sets the chunk sizes to extract from the data in each dimension.
    Chunk sizes that approximately match the data volume being extracted
    typically results in the most efficient IO."""

    def __init__(self,
                 *args,
                 nsrdb_source_fp=None,
                 nsrdb_agg=1,
                 nsrdb_smoothing=0,
                 **kwargs,
                 ):
        """Initialize NETCDF data handler for climate change data.

        Parameters
        ----------
        *args : list
            Same ordered required arguments as DataHandler parent class.
        nsrdb_source_fp : str | None
            Optional NSRDB source h5 file to retrieve clearsky_ghi from to
            calculate CC clearsky_ratio along with rsds (ghi) from the CC
            netcdf file.
        nsrdb_agg : int
            Optional number of NSRDB source pixels to aggregate clearsky_ghi
            from to a single climate change netcdf pixel. This can be used if
            the CC.nc data is at a much coarser resolution than the source
            nsrdb data.
        nsrdb_smoothing : float
            Optional gaussian filter smoothing factor to smooth out
            clearsky_ghi from high-resolution nsrdb source data. This is
            typically done because spatially aggregated nsrdb data is still
            usually rougher than CC irradiance data.
        **kwargs : list
            Same optional keyword arguments as DataHandler parent class.
        """
        self._nsrdb_source_fp = nsrdb_source_fp
        self._nsrdb_agg = nsrdb_agg
        self._nsrdb_smoothing = nsrdb_smoothing
        super().__init__(*args, **kwargs)

    @classmethod
    def source_handler(cls, file_paths, **kwargs):
        """Xarray data handler

        Note that xarray appears to treat open file handlers as singletons
        within a threadpool, so its okay to open this source_handler without a
        context handler or a .close() statement.

        Parameters
        ----------
        file_paths : str | list
            paths to data files
        kwargs : dict
            kwargs passed to source handler for data extraction. e.g. This
            could be {'parallel': True,
                      'chunks': {'south_north': 120, 'west_east': 120}}
            which then gets passed to xr.open_mfdataset(file, **kwargs)

        Returns
        -------
        data : xarray.Dataset
        """
        default_kws = {'chunks': cls.CHUNKS}
        default_kws.update(kwargs)
        return xr.open_mfdataset(file_paths, **default_kws)

    def run_data_extraction(self):
        """Run the raw dataset extraction process from disk to raw
        un-manipulated datasets.

        Includes a special method to extract clearsky_ghi from a exogenous
        NSRDB source h5 file (required to compute clearsky_ratio).
        """
        get_clearsky = False
        if 'clearsky_ghi' in self.raw_features:
            get_clearsky = True
            self._raw_features.remove('clearsky_ghi')

        super().run_data_extraction()

        if get_clearsky:
            cs_ghi = self.get_clearsky_ghi()

            # clearsky ghi is extracted at the proper starting time index so
            # the time chunks should start at 0
            tc0 = self.time_chunks[0].start
            cs_ghi_time_chunks = [
                slice(tc.start - tc0, tc.stop - tc0, tc.step)
                for tc in self.time_chunks
            ]
            for it, tslice in enumerate(cs_ghi_time_chunks):
                self._raw_data[it]['clearsky_ghi'] = cs_ghi[..., tslice]

            self._raw_features.append('clearsky_ghi')

    def get_clearsky_ghi(self):
        """Get clearsky ghi from an exogenous NSRDB source h5 file at the
        target CC meta data and time index.

        Returns
        -------
        cs_ghi : np.ndarray
            Clearsky ghi (W/m2) from the nsrdb_source_fp h5 source file. Data
            shape is (lat, lon, time) where time is daily average values.
        """

        msg = ('Need nsrdb_source_fp input arg as a valid filepath to '
               'retrieve clearsky_ghi (maybe for clearsky_ratio) but '
               'received: {}'.format(self._nsrdb_source_fp))
        assert self._nsrdb_source_fp is not None, msg
        assert os.path.exists(self._nsrdb_source_fp), msg

        msg = ('Can only handle source CC data in hourly frequency but '
               'received daily frequency of {}hrs (should be 24) '
               'with raw time index: {}'.format(self.time_freq_hours,
                                                self.raw_time_index))
        assert self.time_freq_hours == 24.0, msg

        msg = ('Can only handle source CC data with temporal_slice.step == 1 '
               'but received: {}'.format(self.temporal_slice.step))
        assert (self.temporal_slice.step is None) | (self.temporal_slice.step
                                                     == 1), msg

        with Resource(self._nsrdb_source_fp) as res:
            ti_nsrdb = res.time_index
            meta_nsrdb = res.meta

        ti_deltas = ti_nsrdb - np.roll(ti_nsrdb, 1)
        ti_deltas_hours = pd.Series(ti_deltas).dt.total_seconds()[1:-1] / 3600
        time_freq = float(mode(ti_deltas_hours).mode)
        t_start = self.temporal_slice.start or 0
        t_end_target = self.temporal_slice.stop or len(self.raw_time_index)
        t_start = int(t_start * 24 * (1 / time_freq))
        t_end = int(t_end_target * 24 * (1 / time_freq))
        t_end = np.minimum(t_end, len(ti_nsrdb))
        t_slice = slice(t_start, t_end)

        # pylint: disable=E1136
        lat = self.lat_lon[:, :, 0].flatten()
        lon = self.lat_lon[:, :, 1].flatten()
        cc_meta = np.vstack((lat, lon)).T

        tree = KDTree(meta_nsrdb[['latitude', 'longitude']])
        _, i = tree.query(cc_meta, k=self._nsrdb_agg)
        if len(i.shape) == 1:
            i = np.expand_dims(i, axis=1)

        logger.info('Extracting clearsky_ghi data from "{}" with time slice '
                    '{} and {} locations with agg factor {}.'.format(
                        os.path.basename(self._nsrdb_source_fp), t_slice,
                        i.shape[0], i.shape[1],
                    ))

        cs_shape = i.shape
        with Resource(self._nsrdb_source_fp) as res:
            cs_ghi = res['clearsky_ghi', t_slice, i.flatten()]

        cs_ghi = cs_ghi.reshape((len(cs_ghi), *cs_shape))
        cs_ghi = cs_ghi.mean(axis=-1)

        windows = np.array_split(np.arange(len(cs_ghi)),
                                 len(cs_ghi) // (24 // time_freq))
        cs_ghi = [cs_ghi[window].mean(axis=0) for window in windows]
        cs_ghi = np.vstack(cs_ghi)
        cs_ghi = cs_ghi.reshape((len(cs_ghi), *tuple(self.grid_shape)))
        cs_ghi = np.transpose(cs_ghi, axes=(1, 2, 0))

        if self.invert_lat:
            cs_ghi = cs_ghi[::-1]

        logger.info('Smoothing nsrdb clearsky ghi with a factor of {}'.format(
            self._nsrdb_smoothing))
        for iday in range(cs_ghi.shape[-1]):
            cs_ghi[..., iday] = gaussian_filter(cs_ghi[..., iday],
                                                self._nsrdb_smoothing,
                                                mode='nearest')

        if cs_ghi.shape[-1] < t_end_target:
            n = int(np.ceil(t_end_target / cs_ghi.shape[-1]))
            cs_ghi = np.repeat(cs_ghi, n, axis=2)
            cs_ghi = cs_ghi[..., :t_end_target]

        logger.info(
            'Reshaped clearsky_ghi data to final shape {} to '
            'correspond with CC daily average data over source '
            'temporal_slice {} with (lat, lon) grid shape of {}'.format(
                cs_ghi.shape, self.temporal_slice, self.grid_shape))

        return cs_ghi


class DataHandlerNCforCCwithPowerLaw(DataHandlerNCforCC):
    """Data Handler for NETCDF climate change data with power law based
    extrapolation for windspeeds"""

    FEATURE_REGISTRY = DataHandlerNCforCC.FEATURE_REGISTRY.copy()
    FEATURE_REGISTRY.update({'U_(.*)': UWindPowerLaw, 'V_(.*)': VWindPowerLaw})


class DataHandlerDCforNC(DataHandlerNC, DataHandlerDC):
    """Data centric data handler for NETCDF files"""
