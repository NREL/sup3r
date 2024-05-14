"""Basic container objects can perform transformations / extractions on the
contained data."""

import logging
import os
import pickle
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime as dt

import numpy as np
import pandas as pd
import psutil
import xarray as xr
from scipy.stats import mode

from sup3r.containers.wranglers.abstract import AbstractWrangler
from sup3r.containers.wranglers.derivers import FeatureDeriver
from sup3r.utilities.utilities import (
    get_chunk_slices,
    ignore_case_path_fetch,
)

np.random.seed(42)

logger = logging.getLogger(__name__)


class Wrangler(AbstractWrangler, FeatureDeriver, ABC):
    """Loader subclass with additional methods for wrangling data. e.g.
    Extracting specific spatiotemporal extents and features and deriving new
    features."""

    def __init__(self,
                 file_paths,
                 features,
                 target,
                 shape,
                 raster_file=None,
                 temporal_slice=slice(None, None, 1),
                 res_kwargs=None,
                 ):
        """
        Parameters
        ----------
        file_paths : str | pathlib.Path | list
            Globbable path str(s) or pathlib.Path for file locations.
        features : list
            List of feature names to extract from file_paths.
        target : tuple
            (lat, lon) lower left corner of raster. Either need target+shape or
            raster_file.
        shape : tuple
            (rows, cols) grid size. Either need target+shape or raster_file.
        raster_file : str | None
            File for raster_index array for the corresponding target and shape.
            If specified the raster_index will be loaded from the file if it
            exists or written to the file if it does not yet exist. If None and
            raster_index is not provided raster_index will be calculated
            directly. Either need target+shape, raster_file, or raster_index
            input.
        temporal_slice : slice
            Slice specifying extent and step of temporal extraction. e.g.
            slice(start, stop, time_pruning). If equal to slice(None, None, 1)
            the full time dimension is selected.
        res_kwargs : dict | None
            Dictionary of kwargs to pass to xarray.open_mfdataset.
        """
        self.res_kwargs = res_kwargs or {}
        self.raster_file = raster_file
        self.temporal_slice = temporal_slice
        self.target = target
        self.grid_shape = shape
        self.features = None
        self.cache_files = None
        self.overwrite_cache = None
        self.load_cached = None
        self.time_index = None
        self.data = None
        self.lat_lon = None
        self.max_workers = None
        self._noncached_features = None
        self._cache_pattern = None
        self._cache_files = None
        self._time_chunk_size = None
        self._raw_time_index = None
        self._raw_tsteps = None
        self._time_index = None
        self._file_paths = None
        self._single_ts_files = None
        self._invert_lat = None
        self._raw_lat_lon = None
        self._full_raw_lat_lon = None

    @abstractmethod
    def get_raster_index(self):
        """Get array of indices used to select the spatial region of
        interest."""

    @abstractmethod
    def get_time_index(self):
        """Get the time index for the time period of interest."""

    def to_netcdf(self, out_file, data=None, lat_lon=None, features=None):
        """Save data to netcdf file with appropriate lat/lon/time.

        Parameters
        ----------
        out_file : str
            Name of file to save data to. Should have .nc file extension.
        data : ndarray
            Array of data to write to netcdf. If None self.data will be used.
        lat_lon : ndarray
            Array of lat/lon to write to netcdf. If None self.lat_lon will be
            used.
        features : list
            List of features corresponding to last dimension of data. If None
            self.features will be used.
        """
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        data = data if data is not None else self.data
        lat_lon = lat_lon if lat_lon is not None else self.lat_lon
        features = features if features is not None else self.features
        data_vars = {
            f: (('time', 'south_north', 'west_east'),
                np.transpose(data[..., fidx], axes=(2, 0, 1)))
            for fidx, f in enumerate(features)}
        coords = {
            'latitude': (('south_north', 'west_east'), lat_lon[..., 0]),
            'longitude': (('south_north', 'west_east'), lat_lon[..., 1]),
            'time': self.time_index.values}
        out = xr.Dataset(data_vars=data_vars, coords=coords)
        out.to_netcdf(out_file)
        logger.info(f'Saved {features} to {out_file}.')

    @property
    def try_load(self):
        """Check if we should try to load cache"""
        return self._should_load_cache(self.cache_pattern, self.cache_files,
                                       self.overwrite_cache)

    @property
    def noncached_features(self):
        """Get list of features needing extraction or derivation"""
        if self._noncached_features is None:
            self._noncached_features = self.check_cached_features(
                self.features,
                cache_files=self.cache_files,
                overwrite_cache=self.overwrite_cache,
                load_cached=self.load_cached,
            )
        return self._noncached_features

    @property
    def cached_features(self):
        """List of features which have been requested but have been determined
        not to need extraction. Thus they have been cached already."""
        return [f for f in self.features if f not in self.noncached_features]

    def _get_timestamp_0(self, time_index):
        """Get a string timestamp for the first time index value with the
        format YYYYMMDDHHMMSS"""

        time_stamp = time_index[0]
        yyyy = str(time_stamp.year)
        mm = str(time_stamp.month).zfill(2)
        dd = str(time_stamp.day).zfill(2)
        hh = str(time_stamp.hour).zfill(2)
        min = str(time_stamp.minute).zfill(2)
        ss = str(time_stamp.second).zfill(2)
        return yyyy + mm + dd + hh + min + ss

    def _get_timestamp_1(self, time_index):
        """Get a string timestamp for the last time index value with the
        format YYYYMMDDHHMMSS"""

        time_stamp = time_index[-1]
        yyyy = str(time_stamp.year)
        mm = str(time_stamp.month).zfill(2)
        dd = str(time_stamp.day).zfill(2)
        hh = str(time_stamp.hour).zfill(2)
        min = str(time_stamp.minute).zfill(2)
        ss = str(time_stamp.second).zfill(2)
        return yyyy + mm + dd + hh + min + ss

    @property
    def cache_pattern(self):
        """Check for correct cache file pattern."""
        if self._cache_pattern is not None:
            msg = ('Cache pattern must have {feature} format key.')
            assert '{feature}' in self._cache_pattern, msg
        return self._cache_pattern

    @property
    def cache_files(self):
        """Cache files for storing extracted data"""
        if self.cache_pattern is not None:
            return [self.cache_pattern.format(feature=f)
                    for f in self.features]
        return None

    def _cache_data(self, data, features, cache_file_paths, overwrite=False):
        """Cache feature data to files

        Parameters
        ----------
        data : ndarray
            Array of feature data to save to cache files
        features : list
            List of feature names.
        cache_file_paths : str | None
            Path to file for saving feature data
        overwrite : bool
            Whether to overwrite exisiting files.
        """
        for i, fp in enumerate(cache_file_paths):
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            if not os.path.exists(fp) or overwrite:
                if overwrite and os.path.exists(fp):
                    logger.info(f'Overwriting {features[i]} with shape '
                                f'{data[..., i].shape} to {fp}')
                else:
                    logger.info(f'Saving {features[i]} with shape '
                                f'{data[..., i].shape} to {fp}')

                tmp_file = fp.replace('.pkl', '.pkl.tmp')
                with open(tmp_file, 'wb') as fh:
                    pickle.dump(data[..., i], fh, protocol=4)
                os.replace(tmp_file, fp)
            else:
                msg = (f'Called cache_data but {fp} already exists. Set to '
                       'overwrite_cache to True to overwrite.')
                logger.warning(msg)
                warnings.warn(msg)

    def _load_single_cached_feature(self, fp, cache_files, features,
                                    required_shape):
        """Load single feature from given file

        Parameters
        ----------
        fp : string
            File path for feature cache file
        cache_files : list
            List of cache files for each feature
        features : list
            List of requested features
        required_shape : tuple
            Required shape for full array of feature data

        Returns
        -------
        out : ndarray
            Array of data for given feature file.

        Raises
        ------
        RuntimeError
            Error raised if shape conflicts with requested shape
        """
        idx = cache_files.index(fp)
        msg = f'{features[idx].lower()} not found in {fp.lower()}.'
        assert features[idx].lower() in fp.lower(), msg
        fp = ignore_case_path_fetch(fp)
        mem = psutil.virtual_memory()
        logger.info(f'Loading {features[idx]} from {fp}. Current memory '
                    f'usage is {mem.used / 1e9:.3f} GB out of '
                    f'{mem.total / 1e9:.3f} GB total.')

        out = None
        with open(fp, 'rb') as fh:
            out = np.array(pickle.load(fh), dtype=np.float32)
            msg = ('Data loaded from from cache file "{}" '
                   'could not be written to feature channel {} '
                   'of full data array of shape {}. '
                   'The cached data has the wrong shape {}.'.format(
                       fp, idx, required_shape, out.shape))
            assert out.shape == required_shape, msg
        return out

    def _should_load_cache(self,
                           cache_pattern,
                           cache_files,
                           overwrite_cache=False):
        """Check if we should load cached data"""
        return (cache_pattern is not None and not overwrite_cache
                and all(os.path.exists(fp) for fp in cache_files))

    def parallel_load(self, data, cache_files, features, max_workers=None):
        """Load feature data in parallel

        Parameters
        ----------
        data : ndarray
            Array to fill with cached data
        cache_files : list
            List of cache files for each feature
        features : list
            List of requested features
        max_workers : int | None
            Max number of workers to use for parallel data loading. If None
            the max number of available workers will be used.
        """
        logger.info(f'Loading {len(cache_files)} cache files with '
                    f'max_workers={max_workers}.')
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = {}
            now = dt.now()
            for i, fp in enumerate(cache_files):
                future = exe.submit(self._load_single_cached_feature,
                                    fp=fp,
                                    cache_files=cache_files,
                                    features=features,
                                    required_shape=data.shape[:-1],
                                    )
                futures[future] = {'idx': i, 'fp': os.path.basename(fp)}

            logger.info(f'Started loading all {len(cache_files)} cache '
                        f'files in {dt.now() - now}.')

            for i, future in enumerate(as_completed(futures)):
                try:
                    data[..., futures[future]['idx']] = future.result()
                except Exception as e:
                    msg = ('Error while loading '
                           f'{cache_files[futures[future]["idx"]]}')
                    logger.exception(msg)
                    raise RuntimeError(msg) from e
                logger.debug(f'{i + 1} out of {len(futures)} cache files '
                             f'loaded: {futures[future]["fp"]}')

    def _load_cached_data(self, data, cache_files, features, max_workers=None):
        """Load cached data to provided array

        Parameters
        ----------
        data : ndarray
            Array to fill with cached data
        cache_files : list
            List of cache files for each feature
        features : list
            List of requested features
        required_shape : tuple
            Required shape for full array of feature data
        max_workers : int | None
            Max number of workers to use for parallel data loading. If None
            the max number of available workers will be used.
        """
        if max_workers == 1:
            for i, fp in enumerate(cache_files):
                out = self._load_single_cached_feature(fp, cache_files,
                                                       features,
                                                       data.shape[:-1])
                msg = ('Data loaded from from cache file "{}" '
                       'could not be written to feature channel {} '
                       'of full data array of shape {}. '
                       'The cached data has the wrong shape {}.'.format(
                           fp, i, data[..., i].shape, out.shape))
                assert data[..., i].shape == out.shape, msg
                data[..., i] = out

        else:
            self.parallel_load(data,
                               cache_files,
                               features,
                               max_workers=max_workers)

    @staticmethod
    def check_cached_features(features,
                              cache_files=None,
                              overwrite_cache=False,
                              load_cached=False):
        """Check which features have been cached and check flags to determine
        whether to load or extract this features again

        Parameters
        ----------
        features : list
            list of features to extract
        cache_files : list | None
            Path to files with saved feature data
        overwrite_cache : bool
            Whether to overwrite cached files
        load_cached : bool
            Whether to load data from cache files

        Returns
        -------
        list
            List of features to extract. Might not include features which have
            cache files.
        """
        extract_features = []
        # check if any features can be loaded from cache
        if cache_files is not None:
            for i, f in enumerate(features):
                check = (os.path.exists(cache_files[i])
                         and f.lower() in cache_files[i].lower())
                if check:
                    if not overwrite_cache:
                        if load_cached:
                            msg = (f'{f} found in cache file {cache_files[i]}.'
                                   ' Loading from cache instead of extracting '
                                   'from source files')
                            logger.info(msg)
                        else:
                            msg = (f'{f} found in cache file {cache_files[i]}.'
                                   ' Call load_cached_data() or use '
                                   'load_cached=True to load this data.')
                            logger.info(msg)
                    else:
                        msg = (f'{cache_files[i]} exists but overwrite_cache '
                               'is set to True. Proceeding with extraction.')
                        logger.info(msg)
                        extract_features.append(f)
                else:
                    extract_features.append(f)
        else:
            extract_features = features

        return extract_features

    @property
    def time_chunk_size(self):
        """Size of chunk to split the time dimension into for parallel
        extraction."""
        if self._time_chunk_size is None:
            self._time_chunk_size = self.n_tsteps
        return self._time_chunk_size

    @property
    def is_time_independent(self):
        """Get whether source data files are time independent"""
        return self.raw_time_index[0] is None

    @property
    def n_tsteps(self):
        """Get number of time steps to extract"""
        if self.is_time_independent:
            return 1
        return len(self.raw_time_index[self.temporal_slice])

    @property
    def time_chunks(self):
        """Get time chunks which will be extracted from source data

        Returns
        -------
        _time_chunks : list
            List of time chunks used to split up source data time dimension
            so that each chunk can be extracted individually
        """
        if self._time_chunks is None:
            if self.is_time_independent:
                self._time_chunks = [slice(None)]
            else:
                self._time_chunks = get_chunk_slices(len(self.raw_time_index),
                                                     self.time_chunk_size,
                                                     self.temporal_slice)
        return self._time_chunks

    @property
    def raw_tsteps(self):
        """Get number of time steps for all input files"""
        if self._raw_tsteps is None:
            if self.single_ts_files:
                self._raw_tsteps = len(self.file_paths)
            else:
                self._raw_tsteps = len(self.raw_time_index)
        return self._raw_tsteps

    @property
    def single_ts_files(self):
        """Check if there is a file for each time step, in which case we can
        send a subset of files to the data handler according to ti_pad_slice"""
        if self._single_ts_files is None:
            logger.debug('Checking if input files are single timestep.')
            t_steps = self.get_time_index(self.file_paths[:1])
            check = (len(self._file_paths) == len(self.raw_time_index)
                     and t_steps is not None and len(t_steps) == 1)
            self._single_ts_files = check
        return self._single_ts_files

    @property
    def temporal_slice(self):
        """Get temporal range to extract from full dataset"""
        if self._temporal_slice is None:
            self._temporal_slice = slice(None)
        msg = 'temporal_slice must be tuple, list, or slice'
        assert isinstance(self._temporal_slice, (tuple, list, slice)), msg
        if not isinstance(self._temporal_slice, slice):
            check = len(self._temporal_slice) <= 3
            msg = ('If providing list or tuple for temporal_slice length must '
                   'be <= 3')
            assert check, msg
            self._temporal_slice = slice(*self._temporal_slice)
        if self._temporal_slice.step is None:
            self._temporal_slice = slice(self._temporal_slice.start,
                                         self._temporal_slice.stop, 1)
        if self._temporal_slice.start is None:
            self._temporal_slice = slice(0, self._temporal_slice.stop,
                                         self._temporal_slice.step)
        return self._temporal_slice

    @property
    def raw_time_index(self):
        """Time index for input data without time pruning. This is the base
        time index for the raw input data."""

        if self._raw_time_index is None:
            self._raw_time_index = self.get_time_index(self.file_paths,
                                                       **self.res_kwargs)
        if self._single_ts_files:
            self.time_index_conflict_check()
        return self._raw_time_index

    def time_index_conflict_check(self):
        """Check if the number of input files and the length of the time index
        is the same"""
        msg = (f'Number of time steps ({len(self._raw_time_index)}) and files '
               f'({self.raw_tsteps}) conflict!')
        check = len(self._raw_time_index) == self.raw_tsteps
        assert check, msg

    @property
    def time_index(self):
        """Time index for input data with time pruning. This is the raw time
        index with a cropped range and time step applied."""
        return self.raw_time_index[self.temporal_slice]

    @property
    def time_freq_hours(self):
        """Get the time frequency in hours as a float"""
        ti_deltas = self.raw_time_index - np.roll(self.raw_time_index, 1)
        ti_deltas_hours = pd.Series(ti_deltas).dt.total_seconds()[1:-1] / 3600
        return float(mode(ti_deltas_hours).mode)

    @classmethod
    @abstractmethod
    def get_full_domain(cls, file_paths):
        """Get full lat/lon grid for when target + shape are not specified"""

    @classmethod
    @abstractmethod
    def get_lat_lon(cls, file_paths, raster_index, invert_lat=False):
        """Get lat/lon grid for requested target and shape"""

    @property
    def need_full_domain(self):
        """Check whether we need to get the full lat/lon grid to determine
        target and shape values"""
        no_raster_file = self.raster_file is None or not os.path.exists(
            self.raster_file)
        no_target_shape = self._target is None or self._grid_shape is None
        need_full = no_raster_file and no_target_shape

        if need_full:
            logger.info('Target + shape not specified. Getting full domain '
                        f'for {self.file_paths[0]}.')

        return need_full

    @property
    def full_raw_lat_lon(self):
        """Get the full lat/lon grid without doing any latitude inversion"""
        if self._full_raw_lat_lon is None and self.need_full_domain:
            self._full_raw_lat_lon = self.get_full_domain(self.file_paths[:1])
        return self._full_raw_lat_lon

    @property
    def raw_lat_lon(self):
        """Lat lon grid for data in format (spatial_1, spatial_2, 2) Lat/Lon
        array with same ordering in last dimension. This returns the gid
        without any lat inversion.

        Returns
        -------
        ndarray
        """
        raster_file_exists = self.raster_file is not None and os.path.exists(
            self.raster_file)

        if self.full_raw_lat_lon is not None and raster_file_exists:
            self._raw_lat_lon = self.full_raw_lat_lon[self.raster_index]

        elif self.full_raw_lat_lon is not None and not raster_file_exists:
            self._raw_lat_lon = self.full_raw_lat_lon

        if self._raw_lat_lon is None:
            self._raw_lat_lon = self.get_lat_lon(self.file_paths[0:1],
                                                 self.raster_index,
                                                 invert_lat=False)
        return self._raw_lat_lon

    @property
    def lat_lon(self):
        """Lat lon grid for data in format (spatial_1, spatial_2, 2) Lat/Lon
        array with same ordering in last dimension. This ensures that the
        lower left hand corner of the domain is given by lat_lon[-1, 0]

        Returns
        -------
        ndarray
        """
        if self._lat_lon is None:
            self._lat_lon = self.raw_lat_lon
            if self.invert_lat:
                self._lat_lon = self._lat_lon[::-1]
        return self._lat_lon

    @property
    def invert_lat(self):
        """Whether to invert the latitude axis during data extraction. This is
        to enforce a descending latitude ordering so that the lower left corner
        of the grid is at idx=(-1, 0) instead of idx=(0, 0)"""
        return (not self.lats_are_descending())

    @property
    def target(self):
        """Get lower left corner of raster

        Returns
        -------
        _target: tuple
            (lat, lon) lower left corner of raster.
        """
        if self._target is None:
            lat_lon = self.lat_lon
            if not self.lats_are_descending(lat_lon):
                self._target = tuple(lat_lon[0, 0, :])
            else:
                self._target = tuple(lat_lon[-1, 0, :])
        return self._target

    def lats_are_descending(self, lat_lon=None):
        """Check if latitudes are in descending order (i.e. the target
        coordinate is already at the bottom left corner)

        Parameters
        ----------
        lat_lon : np.ndarray
            Lat/Lon array with shape (n_lats, n_lons, 2)

        Returns
        -------
        bool
        """
        lat_lon = lat_lon if lat_lon is not None else self.raw_lat_lon
        return lat_lon[-1, 0, 0] < lat_lon[0, 0, 0]

    @property
    def grid_shape(self):
        """Get shape of raster

        Returns
        -------
        _grid_shape: tuple
            (rows, cols) grid size.
        """
        return self.lat_lon.shape[:-1]

    @property
    def domain_shape(self):
        """Get spatiotemporal domain shape

        Returns
        -------
        tuple
            (rows, cols, timesteps)
        """
        return (*self.grid_shape, len(self.time_index))
