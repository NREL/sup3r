"""MixIn classes for data handling.
@author: bbenton
"""

import logging
import os
import pickle
import warnings
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime as dt

import numpy as np
import pandas as pd
import psutil
from scipy.stats import mode

from sup3r.utilities.utilities import (
    estimate_max_workers,
    expand_paths,
    get_source_type,
    ignore_case_path_fetch,
    uniform_box_sampler,
    uniform_time_sampler,
)

np.random.seed(42)

logger = logging.getLogger(__name__)


class CacheHandlingMixIn:
    """Collection of methods for handling data caching and loading"""

    def __init__(self):
        """Initialize common attributes"""
        self._noncached_features = None
        self._cache_pattern = None
        self._cache_files = None
        self.features = None
        self.cache_files = None
        self.overwrite_cache = None
        self.load_cached = None
        self.time_index = None
        self.grid_shape = None
        self.target = None

    @property
    def cache_pattern(self):
        """Get correct cache file pattern for formatting.

        Returns
        -------
        _cache_pattern : str
            The cache file pattern with formatting keys included.
        """
        self._cache_pattern = self._get_cache_pattern(self._cache_pattern)
        return self._cache_pattern

    @cache_pattern.setter
    def cache_pattern(self, cache_pattern):
        """Update the cache file pattern"""
        self._cache_pattern = cache_pattern

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
        ts0 = yyyy + mm + dd + hh + min + ss
        return ts0

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
        ts1 = yyyy + mm + dd + hh + min + ss
        return ts1

    def _get_cache_pattern(self, cache_pattern):
        """Get correct cache file pattern for formatting.

        Returns
        -------
        cache_pattern : str
            The cache file pattern with formatting keys included.
        """
        if cache_pattern is not None:
            if '.pkl' not in cache_pattern:
                cache_pattern += '.pkl'
            if '{feature}' not in cache_pattern:
                cache_pattern = cache_pattern.replace('.pkl', '_{feature}.pkl')
        return cache_pattern

    def _get_cache_file_names(self, cache_pattern, grid_shape, time_index,
                              target, features,
                              ):
        """Get names of cache files from cache_pattern and feature names

        Parameters
        ----------
        cache_pattern : str
            Pattern to use for cache file names
        grid_shape : tuple
            Shape of grid to use for cache file naming
        time_index : list | pd.DatetimeIndex
            Time index to use for cache file naming
        target : tuple
            Target to use for cache file naming
        features : list
            List of features to use for cache file naming

        Returns
        -------
        list
            List of cache file names
        """
        cache_pattern = self._get_cache_pattern(cache_pattern)
        if cache_pattern is not None:
            if '{feature}' not in cache_pattern:
                cache_pattern = '{feature}_' + cache_pattern
            cache_files = [
                cache_pattern.replace('{feature}', f.lower()) for f in features
            ]
            for i, _ in enumerate(cache_files):
                f = cache_files[i]
                if '{shape}' in f:
                    shape = f'{grid_shape[0]}x{grid_shape[1]}'
                    shape += f'x{len(time_index)}'
                    f = f.replace('{shape}', shape)
                if '{target}' in f:
                    target_str = f'{target[0]:.2f}_{target[1]:.2f}'
                    f = f.replace('{target}', target_str)
                if '{times}' in f:
                    ts_0 = self._get_timestamp_0(time_index)
                    ts_1 = self._get_timestamp_1(time_index)
                    times = f'{ts_0}_{ts_1}'
                    f = f.replace('{times}', times)

                cache_files[i] = f

            for i, fp in enumerate(cache_files):
                fp_check = ignore_case_path_fetch(fp)
                if fp_check is not None:
                    cache_files[i] = fp_check
        else:
            cache_files = None

        return cache_files

    def get_cache_file_names(self,
                             cache_pattern,
                             grid_shape=None,
                             time_index=None,
                             target=None,
                             features=None):
        """Get names of cache files from cache_pattern and feature names

        Parameters
        ----------
        cache_pattern : str
            Pattern to use for cache file names
        grid_shape : tuple
            Shape of grid to use for cache file naming
        time_index : list | pd.DatetimeIndex
            Time index to use for cache file naming
        target : tuple
            Target to use for cache file naming
        features : list
            List of features to use for cache file naming

        Returns
        -------
        list
            List of cache file names
        """
        grid_shape = grid_shape if grid_shape is not None else self.grid_shape
        time_index = time_index if time_index is not None else self.time_index
        target = target if target is not None else self.target
        features = features if features is not None else self.features

        return self._get_cache_file_names(cache_pattern, grid_shape,
                                          time_index, target, features)

    @property
    def cache_files(self):
        """Cache files for storing extracted data"""
        if self._cache_files is None:
            self._cache_files = self.get_cache_file_names(self.cache_pattern)
        return self._cache_files

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
        try_load = (cache_pattern is not None and not overwrite_cache
                    and all(os.path.exists(fp) for fp in cache_files))
        return try_load

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
                logger.debug(f'{i+1} out of {len(futures)} cache files '
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


class InputMixIn(CacheHandlingMixIn):
    """MixIn class with properties and methods for handling the spatiotemporal
    data domain to extract from source data."""

    def __init__(self,
                 target,
                 shape,
                 raster_file=None,
                 raster_index=None,
                 temporal_slice=slice(None, None, 1),
                 res_kwargs=None,
                 ):
        """Provide properties of the spatiotemporal data domain

        Parameters
        ----------
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
        raster_index : list
            List of tuples or slices. Used as an alternative to computing the
            raster index from target+shape or loading the raster index from
            file
        temporal_slice : slice
            Slice specifying extent and step of temporal extraction. e.g.
            slice(start, stop, time_pruning). If equal to slice(None, None, 1)
            the full time dimension is selected.
        res_kwargs : dict | None
            Dictionary of kwargs to pass to xarray.open_mfdataset.
        """
        self.raster_file = raster_file
        self.target = target
        self.grid_shape = shape
        self.raster_index = raster_index
        self.temporal_slice = temporal_slice
        self.lat_lon = None
        self.overwrite_ti_cache = False
        self.max_workers = None
        self._ti_workers = None
        self._raw_time_index = None
        self._raw_tsteps = None
        self._time_index = None
        self._time_index_file = None
        self._file_paths = None
        self._cache_pattern = None
        self._invert_lat = None
        self._raw_lat_lon = None
        self._full_raw_lat_lon = None
        self._single_ts_files = None
        self._worker_attrs = ['ti_workers']
        self.res_kwargs = res_kwargs or {}

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
            t_steps = self.get_time_index(self.file_paths[:1], max_workers=1)
            check = (len(self._file_paths) == len(self.raw_time_index)
                     and t_steps is not None and len(t_steps) == 1)
            self._single_ts_files = check
        return self._single_ts_files

    @staticmethod
    def get_capped_workers(max_workers_cap, max_workers):
        """Get max number of workers for a given job. Capped to global max
        workers if specified

        Parameters
        ----------
        max_workers_cap : int | None
            Cap for job specific max_workers
        max_workers : int | None
            Job specific max_workers

        Returns
        -------
        max_workers : int | None
            job specific max_workers capped by max_workers_cap if provided
        """
        if max_workers is None and max_workers_cap is None:
            return max_workers
        elif max_workers_cap is not None and max_workers is None:
            return max_workers_cap
        elif max_workers is not None and max_workers_cap is None:
            return max_workers
        else:
            return np.min((max_workers_cap, max_workers))

    def cap_worker_args(self, max_workers):
        """Cap all workers args by max_workers"""
        for v in self._worker_attrs:
            capped_val = self.get_capped_workers(getattr(self, v), max_workers)
            setattr(self, v, capped_val)

    @classmethod
    @abstractmethod
    def get_full_domain(cls, file_paths):
        """Get full lat/lon grid for when target + shape are not specified"""

    @classmethod
    @abstractmethod
    def get_lat_lon(cls, file_paths, raster_index, invert_lat=False):
        """Get lat/lon grid for requested target and shape"""

    @abstractmethod
    def get_time_index(self, file_paths, max_workers=None, **kwargs):
        """Get raw time index for source data"""

    @property
    def input_file_info(self):
        """Method to provide info about files in log output. Since NETCDF files
        have single time slices printing out all the file paths is just a text
        dump without much info.

        Returns
        -------
        str
            message to append to log output that does not include a huge info
            dump of file paths
        """
        msg = (f'source files with dates from {self.raw_time_index[0]} to '
               f'{self.raw_time_index[-1]}')
        return msg

    @property
    def temporal_slice(self):
        """Get temporal range to extract from full dataset"""
        return self._temporal_slice

    @temporal_slice.setter
    def temporal_slice(self, temporal_slice):
        """Make sure temporal_slice is a slice. Need to do this because json
        cannot save slices so we can instead save as list and then convert.

        Parameters
        ----------
        temporal_slice : tuple | list | slice
            Time range to extract from input data. If a list or tuple it will
            be concerted to a slice. Tuple or list must have at least two
            elements and no more than three, corresponding to the inputs of
            slice()
        """
        if temporal_slice is None:
            temporal_slice = slice(None)
        msg = 'temporal_slice must be tuple, list, or slice'
        assert isinstance(temporal_slice, (tuple, list, slice)), msg
        if isinstance(temporal_slice, slice):
            self._temporal_slice = temporal_slice
        else:
            check = len(temporal_slice) <= 3
            msg = ('If providing list or tuple for temporal_slice length must '
                   'be <= 3')
            assert check, msg
            self._temporal_slice = slice(*temporal_slice)
        if self._temporal_slice.step is None:
            self._temporal_slice = slice(self._temporal_slice.start,
                                         self._temporal_slice.stop, 1)
        if self._temporal_slice.start is None:
            self._temporal_slice = slice(0, self._temporal_slice.stop,
                                         self._temporal_slice.step)

    @property
    def file_paths(self):
        """Get file paths for input data"""
        return self._file_paths

    @file_paths.setter
    def file_paths(self, file_paths):
        """Set file paths attr and do initial glob / sort

        Parameters
        ----------
        file_paths : str | list
            A list of files to extract raster data from. Each file must have
            the same number of timesteps. Can also pass a string or list of
            strings with a unix-style file path which will be passed through
            glob.glob
        """
        self._file_paths = expand_paths(file_paths)
        msg = ('No valid files provided to DataHandler. '
               f'Received file_paths={file_paths}. Aborting.')
        assert file_paths is not None and len(self._file_paths) > 0, msg

    @property
    def ti_workers(self):
        """Get max number of workers for computing time index"""
        if self._ti_workers is None:
            self._ti_workers = len(self._file_paths)
        return self._ti_workers

    @ti_workers.setter
    def ti_workers(self, val):
        """Set max number of workers for computing time index"""
        self._ti_workers = val

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
    def latitude(self):
        """Flattened list of latitudes"""
        return self.lat_lon[..., 0].flatten()

    @property
    def longitude(self):
        """Flattened list of longitudes"""
        return self.lat_lon[..., 1].flatten()

    @property
    def meta(self):
        """Meta dataframe with coordinates."""
        return pd.DataFrame({'latitude': self.latitude,
                             'longitude': self.longitude})

    @lat_lon.setter
    def lat_lon(self, lat_lon):
        """Update lat lon"""
        self._lat_lon = lat_lon

    @property
    def invert_lat(self):
        """Whether to invert the latitude axis during data extraction. This is
        to enforce a descending latitude ordering so that the lower left corner
        of the grid is at idx=(-1, 0) instead of idx=(0, 0)"""
        if self._invert_lat is None:
            lat_lon = self.raw_lat_lon
            self._invert_lat = not self.lats_are_descending(lat_lon)
        return self._invert_lat

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

    @target.setter
    def target(self, target):
        """Update target property"""
        self._target = target

    @classmethod
    def lats_are_descending(cls, lat_lon):
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
        return lat_lon[-1, 0, 0] < lat_lon[0, 0, 0]

    @property
    def grid_shape(self):
        """Get shape of raster

        Returns
        -------
        _grid_shape: tuple
            (rows, cols) grid size.
        """
        if self._grid_shape is None:
            self._grid_shape = self.lat_lon.shape[:-1]
        return self._grid_shape

    @grid_shape.setter
    def grid_shape(self, grid_shape):
        """Update grid_shape property"""
        self._grid_shape = grid_shape

    @property
    def source_type(self):
        """Get data type for source files. Either nc or h5"""
        return get_source_type(self.file_paths)

    @property
    def raw_time_index(self):
        """Time index for input data without time pruning. This is the base
        time index for the raw input data."""

        if self._raw_time_index is None:
            check = (self.time_index_file is not None
                     and os.path.exists(self.time_index_file)
                     and not self.overwrite_ti_cache)
            if check:
                logger.debug('Loading raw_time_index from '
                             f'{self.time_index_file}')
                with open(self.time_index_file, 'rb') as f:
                    self._raw_time_index = pd.DatetimeIndex(pickle.load(f))
            else:
                self._raw_time_index = self._build_and_cache_time_index()

            check = (self._raw_time_index is not None
                     and (self._raw_time_index.hour == 12).all())
            if check:
                self._raw_time_index -= pd.Timedelta(12, 'h')
            elif self._raw_time_index is None:
                self._raw_time_index = [None, None]

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
        if self._time_index is None:
            self._time_index = self.raw_time_index[self.temporal_slice]
        return self._time_index

    @time_index.setter
    def time_index(self, time_index):
        """Update time index"""
        self._time_index = time_index

    @property
    def time_freq_hours(self):
        """Get the time frequency in hours as a float"""
        ti_deltas = self.raw_time_index - np.roll(self.raw_time_index, 1)
        ti_deltas_hours = pd.Series(ti_deltas).dt.total_seconds()[1:-1] / 3600
        time_freq = float(mode(ti_deltas_hours).mode)
        return time_freq

    @property
    def time_index_file(self):
        """Get time index file path"""
        if self.source_type == 'h5':
            return None

        if self.cache_pattern is not None and self._time_index_file is None:
            basename = self.cache_pattern.replace('_{times}', '')
            basename = basename.replace('{times}', '')
            basename = basename.replace('{shape}', str(len(self.file_paths)))
            basename = basename.replace('_{target}', '')
            basename = basename.replace('{feature}', 'time_index')
            tmp = basename.split('_')
            if tmp[-2].isdigit() and tmp[-1].strip('.pkl').isdigit():
                basename = '_'.join(tmp[:-1]) + '.pkl'
            self._time_index_file = basename
        return self._time_index_file

    def _build_and_cache_time_index(self):
        """Build time index and cache if time_index_file is not None"""
        now = dt.now()
        logger.debug(f'Getting time index for {len(self.file_paths)} '
                     f'input files. Using ti_workers={self.ti_workers}'
                     f' and res_kwargs={self.res_kwargs}')
        self._raw_time_index = self.get_time_index(self.file_paths,
                                                   max_workers=self.ti_workers,
                                                   **self.res_kwargs)

        if self.time_index_file is not None:
            os.makedirs(os.path.dirname(self.time_index_file), exist_ok=True)
            logger.debug(f'Saving raw_time_index to {self.time_index_file}')
            with open(self.time_index_file, 'wb') as f:
                pickle.dump(self._raw_time_index, f)
        logger.debug(f'Built full time index in {dt.now() - now} seconds.')
        return self._raw_time_index


class TrainingPrepMixIn:
    """Collection of training related methods. e.g. Training + Validation
    splitting, normalization"""

    def __init__(self):
        """Initialize common attributes"""
        self.features = None
        self.data = None
        self.val_data = None
        self.feature_mem = None
        self.shape = None
        self._means = None
        self._stds = None
        self._is_normalized = False
        self._norm_workers = None

    @classmethod
    def _split_data_indices(cls,
                            data,
                            val_split=0.0,
                            n_val_obs=None,
                            shuffle_time=False):
        """Split time dimension into set of training indices and validation
        indices

        Parameters
        ----------
        data : np.ndarray
            4D array of high res data
            (spatial_1, spatial_2, temporal, features)
        val_split : float
            Fraction of data to separate for validation.
        n_val_obs : int | None
            Optional number of validation observations. If provided this
            overrides val_split
        shuffle_time : bool
            Whether to shuffle time or not.

        Returns
        -------
        training_indices : np.ndarray
            Array of timestep indices used to select training data. e.g.
            training_data = data[..., training_indices, :]
        val_indices : np.ndarray
            Array of timestep indices used to select validation data. e.g.
            val_data = data[..., val_indices, :]
        """
        n_observations = data.shape[2]
        all_indices = np.arange(n_observations)
        n_val_obs = (int(val_split
                         * n_observations) if n_val_obs is None else n_val_obs)

        if shuffle_time:
            np.random.shuffle(all_indices)

        val_indices = all_indices[:n_val_obs]
        training_indices = all_indices[n_val_obs:]

        return training_indices, val_indices

    def _get_observation_index(self, data, sample_shape):
        """Randomly gets spatial sample and time sample

        Parameters
        ----------
        data : ndarray
            Array of data to sample
            (spatial_1, spatial_2, temporal, n_features)
        sample_shape : tuple
            Size of observation to sample
            (n_lats, n_lons, n_timesteps)

        Returns
        -------
        observation_index : tuple
            Tuple of sampled spatial grid, time slice, and features indices.
            Used to get single observation like self.data[observation_index]
        """
        spatial_slice = uniform_box_sampler(data, sample_shape[:2])
        temporal_slice = uniform_time_sampler(data, sample_shape[2])
        return (*spatial_slice, temporal_slice, np.arange(data.shape[-1]))

    def _normalize_data(self, data, val_data, feature_index, mean, std):
        """Normalize data with initialized mean and standard deviation for a
        specific feature

        Parameters
        ----------
        data : np.ndarray
            Array of training data.
            (spatial_1, spatial_2, temporal, n_features)
        val_data : np.ndarray
            Array of validation data.
            (spatial_1, spatial_2, temporal, n_features)
        feature_index : int
            index of feature to be normalized
        mean : float32
            specified mean of associated feature
        std : float32
            specificed standard deviation for associated feature
        """

        if val_data is not None:
            val_data[..., feature_index] -= mean

        data[..., feature_index] -= mean

        if std > 0:
            if val_data is not None:
                val_data[..., feature_index] /= std
            data[..., feature_index] /= std
        else:
            msg = ('Standard Deviation is zero for '
                   f'{self.features[feature_index]}')
            logger.warning(msg)
            warnings.warn(msg)

        logger.debug(f'Finished normalizing {self.features[feature_index]} '
                     f'with mean {mean:.3e} and std {std:.3e}.')

    def _normalize(self, data, val_data, features=None, max_workers=None):
        """Normalize all data features

        Parameters
        ----------
        data : np.ndarray
            Array of training data.
            (spatial_1, spatial_2, temporal, n_features)
        val_data : np.ndarray
            Array of validation data.
            (spatial_1, spatial_2, temporal, n_features)
        features : list | None
            List of features used for indexing data array during normalization.
        max_workers : int | None
            Number of workers to use in thread pool for nomalization.
        """
        if features is None:
            features = self.features

        msg1 = (f'Not all feature names {features} were found in '
                f'self.means: {list(self.means.keys())}')
        msg2 = (f'Not all feature names {features} were found in '
                f'self.stds: {list(self.stds.keys())}')
        assert all(fn in self.means for fn in features), msg1
        assert all(fn in self.stds for fn in features), msg2

        logger.info(f'Normalizing {data.shape[-1]} features: {features}')

        if max_workers == 1:
            for idf, feature in enumerate(features):
                self._normalize_data(data, val_data, idf, self.means[feature],
                                     self.stds[feature])
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                futures = []
                for idf, feature in enumerate(features):
                    future = exe.submit(self._normalize_data,
                                        data, val_data, idf,
                                        self.means[feature],
                                        self.stds[feature])
                    futures.append(future)

                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        msg = ('Error while normalizing future number '
                               f'{futures[future]}.')
                        logger.exception(msg)
                        raise RuntimeError(msg) from e

    @property
    def means(self):
        """Get the mean values for each feature.

        Returns
        -------
        dict
        """
        self._get_stats()
        return self._means

    @property
    def stds(self):
        """Get the standard deviation values for each feature.

        Returns
        -------
        dict
        """
        self._get_stats()
        return self._stds

    def _get_stats(self, features=None):
        """Get the mean/stdev for each feature in the data handler."""
        if features is None:
            features = self.features
        if self._means is None or self._stds is None:
            msg = (f'DataHandler has {len(features)} features '
                   f'and mismatched shape of {self.shape}')
            assert len(features) == self.shape[-1], msg
            self._stds = {}
            self._means = {}
            for idf, fname in enumerate(features):
                self._means[fname] = np.nanmean(
                    self.data[..., idf].astype(np.float32))
                self._stds[fname] = np.nanstd(
                    self.data[..., idf].astype(np.float32))

    def normalize(self, means=None, stds=None, features=None,
                  max_workers=None):
        """Normalize all data features.

        Parameters
        ----------
        means : dict | none
            Dictionary of means for all features with keys: feature names and
            values: mean values. If this is None, the self.means attribute will
            be used. If this is not None, this DataHandler object means
            attribute will be updated.
        stds : dict | none
            dictionary of standard deviation values for all features with keys:
            feature names and values: standard deviations. If this is None, the
            self.stds attribute will be used. If this is not None, this
            DataHandler object stds attribute will be updated.
        features : list | None
            List of features used for indexing data array during normalization.
        max_workers : None | int
            Max workers to perform normalization. if None, self.norm_workers
            will be used
        """
        if means is not None:
            self._means = means
        if stds is not None:
            self._stds = stds

        if self._is_normalized:
            logger.info('Skipping DataHandler, already normalized')
        elif self.data is not None:
            self._normalize(self.data,
                            self.val_data,
                            features=features,
                            max_workers=max_workers)
            self._is_normalized = True

    @property
    def norm_workers(self):
        """Get upper bound on workers used for normalization."""
        if self.data is not None:
            norm_workers = estimate_max_workers(self._norm_workers,
                                                2 * self.feature_mem,
                                                self.shape[-1])
        else:
            norm_workers = self._norm_workers
        return norm_workers
