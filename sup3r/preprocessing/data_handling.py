# -*- coding: utf-8 -*-
"""
Sup3r preprocessing module.
@author: bbenton
"""

from abc import abstractmethod
import json
from fnmatch import fnmatch
import logging
import xarray as xr
import numpy as np
import os
from datetime import datetime as dt
import pickle
import warnings
import glob
from concurrent.futures import (as_completed, ThreadPoolExecutor)

from rex import MultiFileWindX, MultiFileNSRDBX
from rex.utilities import log_mem
from rex.utilities.fun_utils import get_fun_call_str

from sup3r.utilities.utilities import (estimate_max_workers,
                                       get_chunk_slices,
                                       interp_var_to_height,
                                       interp_var_to_pressure,
                                       uniform_box_sampler,
                                       uniform_time_sampler,
                                       weighted_time_sampler,
                                       get_raster_shape,
                                       ignore_case_path_fetch,
                                       daily_temporal_coarsening,
                                       spatial_coarsening,
                                       np_to_pd_times)
from sup3r.utilities import ModuleName
from sup3r.preprocessing.feature_handling import (FeatureHandler,
                                                  Feature,
                                                  BVFreqMon,
                                                  BVFreqSquaredH5,
                                                  BVFreqSquaredNC,
                                                  InverseMonNC,
                                                  LatLonNC,
                                                  LatLonNCforCC,
                                                  TempNC,
                                                  UWind,
                                                  VWind,
                                                  LatLonH5,
                                                  ClearSkyRatioH5,
                                                  CloudMaskH5,
                                                  WindspeedNC,
                                                  WinddirectionNC,
                                                  Shear,
                                                  Rews
                                                  )

np.random.seed(42)

logger = logging.getLogger(__name__)


class InputMixIn:
    """MixIn class for input handling methods and properties"""

    def __init__(self):
        self.raster_file = None
        self.raster_index = None
        self.lat_lon = None
        self._raw_time_index = None
        self._time_index = None
        self._temporal_slice = None
        self._file_paths = None
        self._cache_pattern = None
        self._target = None
        self._grid_shape = None

    @classmethod
    @abstractmethod
    def get_full_domain(cls, file_paths):
        """Get target and shape for largest domain possible when target + shape
        are not specified"""

    @classmethod
    @abstractmethod
    def get_time_index(cls, file_paths):
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
        msg = ('temporal_slice must be tuple, list, or slice')
        assert isinstance(temporal_slice, (tuple, list, slice)), msg
        if isinstance(temporal_slice, slice):
            self._temporal_slice = temporal_slice
        else:
            check = len(temporal_slice) <= 3
            msg = ('If providing list or tuple for temporal_slice length must '
                   'be <= 3')
            assert check, msg
            self._temporal_slice = slice(*temporal_slice)

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
            the same number of timesteps. Can also pass a string with a
            unix-style file path which will be passed through glob.glob
        """
        self._file_paths = file_paths
        if isinstance(self._file_paths, str):
            self._file_paths = glob.glob(self._file_paths)
        self._file_paths = sorted(self._file_paths)

    @property
    def cache_pattern(self):
        """Get correct cache file pattern for formatting.

        Returns
        -------
        _cache_pattern : str
            The cache file pattern with formatting keys included.
        """
        if self._cache_pattern is not None:
            if '.pkl' not in self._cache_pattern:
                self._cache_pattern += '.pkl'
            if '{feature}' not in self._cache_pattern:
                self._cache_pattern = self._cache_pattern.replace(
                    '.pkl', '_{feature}.pkl')

        return self._cache_pattern

    @cache_pattern.setter
    def cache_pattern(self, cache_pattern):
        """Update the cache file pattern"""
        self._cache_pattern = cache_pattern

    @property
    def full_domain(self):
        """Get target and shape for full domain if not specified and raster
        file is None or does not exist

        Returns
        -------
        _target: tuple
            (lat, lon) lower left corner of raster.
        _grid_shape: tuple
            (rows, cols) grid size.
        """
        check = (self.raster_file is None
                 or not os.path.exists(self.raster_file))
        check = check and (self._target is None or self._grid_shape is None)
        if check:
            new_target, new_shape = self.get_full_domain(self.file_paths)
            self._target = self._target or new_target
            self._grid_shape = self._grid_shape or new_shape
            logger.info('Target + shape not specified. Getting full domain '
                        f'with target={self._target} and '
                        f'shape={self._grid_shape}')
        return self._target, self._grid_shape

    @property
    def target(self):
        """Get lower left corner of raster

        Returns
        -------
        _target: tuple
            (lat, lon) lower left corner of raster.
        """
        if self._target is None:
            self._target = tuple(self.lat_lon[-1, 0, :])
        return self._target

    @target.setter
    def target(self, target):
        """Update target property"""
        self._target = target

    @property
    def grid_shape(self):
        """Get shape of raster

        Returns
        -------
        _grid_shape: tuple
            (rows, cols) grid size.
        """
        if self._grid_shape is None:
            check = (self.raster_file is not None
                     and os.path.exists(self.raster_file))
            if check:
                self._grid_shape = get_raster_shape(self.raster_index)
            else:
                self._target, self._grid_shape = self.full_domain
        return self._grid_shape

    @grid_shape.setter
    def grid_shape(self, grid_shape):
        """Update grid_shape property"""
        self._grid_shape = grid_shape

    @property
    def raw_time_index(self):
        """Time index for input data without time pruning. This is the base
        time index for the raw input data."""
        if self._raw_time_index is None:
            self._raw_time_index = self.get_time_index(self.file_paths)
        return self._raw_time_index

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
    def timestamp_0(self):
        """Get a string timestamp for the first time index value with the
        format YYYYMMDDHHMMSS"""

        time_stamp = self.time_index[0]
        yyyy = str(time_stamp.year)
        mm = str(time_stamp.month).zfill(2)
        dd = str(time_stamp.day).zfill(2)
        hh = str(time_stamp.hour).zfill(2)
        min = str(time_stamp.minute).zfill(2)
        ss = str(time_stamp.second).zfill(2)
        ts0 = yyyy + mm + dd + hh + min + ss
        return ts0

    @property
    def timestamp_1(self):
        """Get a string timestamp for the last time index value with the
        format YYYYMMDDHHMMSS"""

        time_stamp = self.time_index[-1]
        yyyy = str(time_stamp.year)
        mm = str(time_stamp.month).zfill(2)
        dd = str(time_stamp.day).zfill(2)
        hh = str(time_stamp.hour).zfill(2)
        min = str(time_stamp.minute).zfill(2)
        ss = str(time_stamp.second).zfill(2)
        ts1 = yyyy + mm + dd + hh + min + ss
        return ts1


class DataHandler(FeatureHandler, InputMixIn):
    """Sup3r data handling and extraction"""

    # list of features / feature name patterns that are input to the generative
    # model but are not part of the synthetic output and are not sent to the
    # discriminator. These are case-insensitive and follow the Unix shell-style
    # wildcard format.
    TRAIN_ONLY_FEATURES = ('BVF*', 'inversemoninobukhovlength_*', 'RMOL')

    def __init__(self, file_paths, features, target=None, shape=None,
                 max_delta=20,
                 temporal_slice=slice(None),
                 hr_spatial_coarsen=None,
                 time_roll=0,
                 val_split=0.05,
                 sample_shape=(10, 10, 1),
                 raster_file=None,
                 shuffle_time=False,
                 time_chunk_size=None,
                 cache_pattern=None,
                 overwrite_cache=False,
                 load_cached=False,
                 train_only_features=None,
                 max_workers=None,
                 extract_workers=None,
                 compute_workers=None,
                 load_workers=None,
                 norm_workers=None):
        """Data handling and extraction

        Parameters
        ----------
        file_paths : str | list
            A single source h5 wind file to extract raster data from or a list
            of netcdf files with identical grid. The string can be a unix-style
            file path which will be passed through glob.glob
        features : list
            list of features to extract
        target : tuple
            (lat, lon) lower left corner of raster. Either need target+shape or
            raster_file.
        shape : tuple
            (rows, cols) grid size. Either need target+shape or raster_file.
        max_delta : int, optional
            Optional maximum limit on the raster shape that is retrieved at
            once. If shape is (20, 20) and max_delta=10, the full raster will
            be retrieved in four chunks of (10, 10). This helps adapt to
            non-regular grids that curve over large distances, by default 20
        temporal_slice : slice
            Slice specifying extent and step of temporal extraction. e.g.
            slice(start, stop, time_pruning). If equal to slice(None, None, 1)
            the full time dimension is selected.
        hr_spatial_coarsen : int | None
            Optional input to coarsen the high-resolution spatial field. This
            can be used if (for example) you have 2km source data, but you want
            the final high res prediction target to be 4km resolution, then
            hr_spatial_coarsen would be 2 so that the GAN is trained on
            aggregated 4km high-res data.
        time_roll : int
            The number of places by which elements are shifted in the time
            axis. Can be used to convert data to different timezones. This is
            passed to np.roll(a, time_roll, axis=2) and happens AFTER the
            temporal_slice operation.
        val_split : float32
            Fraction of data to store for validation
        sample_shape : tuple
            Size of spatial and temporal domain used in a single high-res
            observation for batching
        raster_file : str | None
            File for raster_index array for the corresponding target and shape.
            If specified the raster_index will be loaded from the file if it
            exists or written to the file if it does not yet exist.  If None
            raster_index will be calculated directly. Either need target+shape
            or raster_file.
        shuffle_time : bool
            Whether to shuffle time indices before valiidation split
        time_chunk_size : int
            Size of chunks to split time dimension into for parallel data
            extraction. If running in serial this can be set to the size of the
            full time index for best performance.
        cache_pattern : str | None
            Pattern for files for saving feature data. e.g.
            file_path_{feature}.pkl. Each feature will be saved to a file with
            the feature name replaced in cache_pattern. If not None
            feature arrays will be saved here and not stored in self.data until
            load_cached_data is called. The cache_pattern can also include
            {shape}, {target}, {times} which will help ensure unique cache
            files for complex problems.
        overwrite_cache : bool
            Whether to overwrite any previously saved cache files.
        load_cached : bool
            Whether to load data from cache files
        train_only_features : list | tuple | None
            List of feature names or patt*erns that should only be included in
            the training set and not the output. If None (default), this will
            default to the class TRAIN_ONLY_FEATURES attribute.
        max_workers : int | None
            Providing a value for max workers will be used to set the value of
            extract_workers, compute_workers, load_workers, and norm_workers.
            If max_workers == 1 then all processes will be serialized. If None
            extract_workers, compute_workers, load_workers, and norm_workers
            will use their own provided values.
        extract_workers : int | None
            max number of workers to use for extracting features from source
            data. If None max workers will be estimated based on memory limits.
            If 1 processes will be serialized.
        compute_workers : int | None
            max number of workers to use for computing derived features from
            raw features in source data.
        load_workers : int | None
            max number of workers to use for loading cached feature data.
        norm_workers : int | None
            max number of workers to use for normalizing feature data.
        """
        if max_workers is not None:
            extract_workers = compute_workers = max_workers
            load_workers = norm_workers = max_workers

        self.file_paths = file_paths
        self.features = features
        self.val_time_index = None
        self.max_delta = max_delta
        self.raster_file = raster_file
        self.val_split = val_split
        self.sample_shape = sample_shape
        self.temporal_slice = temporal_slice
        self.hr_spatial_coarsen = hr_spatial_coarsen or 1
        self.time_roll = time_roll
        self.shuffle_time = shuffle_time
        self.current_obs_index = None
        self.overwrite_cache = overwrite_cache
        self.load_cached = load_cached
        self.data = None
        self.val_data = None
        self.target = target
        self.grid_shape = shape
        self._invert_lat = None
        self._cache_pattern = cache_pattern
        self._train_only_features = train_only_features
        self._extract_workers = extract_workers
        self._norm_workers = norm_workers
        self._load_workers = load_workers
        self._compute_workers = compute_workers
        self._time_chunk_size = time_chunk_size
        self._cache_files = None
        self._raw_time_index = None
        self._time_index = None
        self._lat_lon = None
        self._raster_index = None
        self._handle_features = None
        self._extract_features = None
        self._raw_features = None
        self._raw_data = None
        self._time_chunks = None

        msg = (f'Initializing DataHandler {self.input_file_info}. '
               f'Getting temporal range {str(self.time_index[0])} to '
               f'{str(self.time_index[-1])}')
        logger.info(msg)

        self.preflight()

        try_load = (cache_pattern is not None
                    and not self.overwrite_cache
                    and all(os.path.exists(fp) for fp in self.cache_files))

        overwrite = (self.overwrite_cache
                     and self.cache_files is not None
                     and all(os.path.exists(fp) for fp in self.cache_files))

        if try_load and self.load_cached:
            logger.info(f'All {self.cache_files} exist. Loading from cache '
                        f'instead of extracting from source files.')
            self.load_cached_data()

        elif try_load and not self.load_cached:
            self.clear_data()
            logger.info(f'All {self.cache_files} exist. Call '
                        'load_cached_data() or use load_cache=True to load '
                        'this data from cache files.')
        else:
            if overwrite:
                logger.info(f'{self.cache_files} exists but overwrite_cache '
                            'is set to True. Proceeding with extraction.')
            bad_shape = (sample_shape[0] > self.grid_shape[0]
                         and sample_shape[1] > self.grid_shape[1])
            if bad_shape:
                msg = (f'spatial_sample_shape {sample_shape[:2]} is larger '
                       f'than the raster size {self.grid_shape}')
                logger.warning(msg)
                warnings.warn(msg)

            self.data = self.extract_data()

            if cache_pattern is not None:
                self.cache_data(self.cache_files)
                self.data = None if not self.load_cached else self.data

            if self.data is not None:
                self.data, self.val_data = self.split_data()

        logger.info('Finished intializing DataHandler.')
        log_mem(logger, log_level='INFO')

    @classmethod
    @abstractmethod
    def get_full_domain(cls, file_paths):
        """Get target and shape for full domain"""

    def clear_data(self):
        """Free memory used for data arrays"""
        self.data = None
        self.val_data = None

    @classmethod
    @abstractmethod
    def source_handler(cls, file_paths, **kwargs):
        """Handler for source data. Can use xarray, ResourceX, etc."""

    @property
    def attrs(self):
        """Get atttributes of input data

        Returns
        -------
        dict
            Dictionary of attributes
        """
        with self.source_handler(self.file_paths) as handle:
            desc = handle.attrs
        return desc

    @property
    def invert_lat(self):
        """Whether to invert the latitude axis during data extraction. This is
        to enforce a descending latitude ordering so that the lower left corner
        of the grid is at idx=(-1, 0) instead of idx=(0, 0)"""
        if self._invert_lat is None:
            lat_lon = self.get_lat_lon(self.file_paths[:1],
                                       self.raster_index,
                                       invert_lat=False)
            self._invert_lat = (lat_lon[0, 0, 0] < lat_lon[-1, 0, 0])
        return self._invert_lat

    @property
    def train_only_features(self):
        """Features to use for training only and not output"""
        if self._train_only_features is None:
            self._train_only_features = self.TRAIN_ONLY_FEATURES
        return self._train_only_features

    @property
    def extract_workers(self):
        """Get upper bound for extract workers based on memory limits. Used to
        extract data from source dataset"""
        proc_mem = 4 * self.grid_mem * len(self.time_index)
        proc_mem /= len(self.time_chunks)
        n_procs = len(self.time_chunks) * len(self.raw_features)
        n_procs = int(np.ceil(n_procs))
        extract_workers = estimate_max_workers(self._extract_workers, proc_mem,
                                               n_procs)
        return extract_workers

    @property
    def compute_workers(self):
        """Get upper bound for compute workers based on memory limits. Used to
        compute derived features from source dataset."""
        proc_mem = np.int(np.ceil(len(self.raw_features)
                                  / len(self.extract_features)))
        proc_mem *= 4 * self.grid_mem * len(self.time_index)
        proc_mem /= len(self.time_chunks)
        n_procs = len(self.time_chunks) * len(self.derive_features)
        n_procs = int(np.ceil(n_procs))
        compute_workers = estimate_max_workers(self._compute_workers, proc_mem,
                                               n_procs)
        return compute_workers

    @property
    def load_workers(self):
        """Get upper bound on load workers based on memory limits. Used to load
        cached data."""
        proc_mem = 2 * self.feature_mem
        n_procs = 1
        if self.cache_files is not None:
            n_procs = len(self.cache_files)
        load_workers = estimate_max_workers(self._load_workers, proc_mem,
                                            n_procs)
        return load_workers

    @property
    def norm_workers(self):
        """Get upper bound on workers used for normalization."""
        norm_workers = estimate_max_workers(self._norm_workers,
                                            2 * self.feature_mem,
                                            self.shape[-1])
        return norm_workers

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
            self._time_chunks = get_chunk_slices(len(self.raw_time_index),
                                                 self.time_chunk_size,
                                                 self.temporal_slice)
        return self._time_chunks

    @property
    def n_tsteps(self):
        """Get number of time steps to extract"""
        return len(self.time_index)

    @property
    def time_chunk_size(self):
        """Get upper bound on time chunk size based on memory limits"""
        if self._time_chunk_size is None:
            step_mem = self.feature_mem * len(self.raw_features)
            step_mem /= len(self.time_index)
            self._time_chunk_size = np.min([np.int(1e9 / step_mem),
                                            self.n_tsteps])
            logger.info('time_chunk_size arg not specified. Using '
                        f'{self._time_chunk_size}.')
        return self._time_chunk_size

    @property
    def cache_files(self):
        """Cache files for storing extracted data"""
        if self._cache_files is None:
            self._cache_files = self.get_cache_file_names(self.cache_pattern)
        return self._cache_files

    @property
    def lat_lon(self):
        """lat lon grid for data"""
        if self._lat_lon is None:
            self._lat_lon = self.get_lat_lon(self.file_paths,
                                             self.raster_index,
                                             invert_lat=self.invert_lat)
        return self._lat_lon

    @lat_lon.setter
    def lat_lon(self, lat_lon):
        """Update lat lon"""
        self._lat_lon = lat_lon

    @property
    def raster_index(self):
        """Raster index property"""
        if self._raster_index is None:
            self._raster_index = self.get_raster_index()
        return self._raster_index

    @raster_index.setter
    def raster_index(self, raster_index):
        """Update raster index property"""
        self._raster_index = raster_index

    @property
    def handle_features(self):
        """All features available in raw input"""
        if self._handle_features is None:
            with self.source_handler(self.file_paths) as handle:
                self._handle_features = [Feature.get_basename(r)
                                         for r in handle]
        return self._handle_features

    @property
    def extract_features(self):
        """Get list of features needing extraction or derivation"""
        if self._extract_features is None:
            self._extract_features = self.check_cached_features(
                self.features, cache_files=self.cache_files,
                overwrite_cache=self.overwrite_cache,
                load_cached=self.load_cached)
        return self._extract_features

    @property
    def derive_features(self):
        """List of features which need to be derived from other features"""
        return [f for f in self.extract_features if f not in self.raw_features]

    @property
    def cached_features(self):
        """List of features which have been requested but have been determined
        not to need extraction. Thus they have been cached already."""
        return [f for f in self.features if f not in self.extract_features]

    @property
    def raw_features(self):
        """Get list of features needed for computations"""
        if self._raw_features is None:
            self._raw_features = self.get_raw_feature_list(
                self.extract_features, self.handle_features)
        return self._raw_features

    @property
    def output_features(self):
        """Get a list of features that should be output by the generative model
        corresponding to the features in the high res batch array."""
        out = []
        for feature in self.features:
            ignore = any(fnmatch(feature.lower(), pattern.lower())
                         for pattern in self.train_only_features)
            if not ignore:
                out.append(feature)
        return out

    @property
    def grid_mem(self):
        """Get memory used by a feature at a single time step

        Returns
        -------
        int
            Number of bytes for a single feature array at a single time step
        """
        grid_mem = np.product(self.grid_shape)
        # assuming feature arrays are float32 (4 bytes)
        return 4 * grid_mem

    @property
    def feature_mem(self):
        """Number of bytes for a single feature array. Used to estimate
        max_workers.

        Returns
        -------
        int
            Number of bytes for a single feature array
        """
        feature_mem = self.grid_mem * len(self.time_index)
        return feature_mem

    def preflight(self):
        """Run some preflight checks and verify that the inputs are valid"""
        if len(self.sample_shape) == 2:
            logger.info('Found 2D sample shape of {}. Adding temporal dim of 1'
                        .format(self.sample_shape))
            self.sample_shape = self.sample_shape + (1,)

        start = self.temporal_slice.start
        stop = self.temporal_slice.stop
        n_steps = self.raw_time_index[start:stop]
        n_steps = len(n_steps)
        msg = (f'Temporal slice step ({self.temporal_slice.step}) does not '
               f'evenly divide the number of time steps ({n_steps})')
        check = self.temporal_slice.step is None
        check = check or n_steps % self.temporal_slice.step == 0
        if not check:
            logger.warning(msg)
            warnings.warn(msg)

        msg = (f'sample_shape[2] ({self.sample_shape[2]}) cannot be larger '
               'than the number of time steps in the raw data '
               f'({len(self.raw_time_index)}).')
        assert len(self.raw_time_index) >= self.sample_shape[2], msg

        msg = (f'The requested time slice {self.temporal_slice} conflicts '
               f'with the number of time steps ({len(self.raw_time_index)}) '
               'in the raw data')
        t_slice_is_subset = (start is not None and stop is not None)
        good_subset = (t_slice_is_subset
                       and (stop - start <= len(self.raw_time_index))
                       and stop <= len(self.raw_time_index)
                       and start <= len(self.raw_time_index))
        if t_slice_is_subset and not good_subset:
            logger.error(msg)
            raise RuntimeError(msg)

    @classmethod
    def get_lat_lon(cls, file_paths, raster_index, invert_lat=False):
        """Store lat lon for future output

        Parameters
        ----------
        file_paths : list
            path to data file
        raster_index : ndarray | list
            Raster index array or list of slices
        invert_lat : bool
            Flag to invert data along the latitude axis. Wrf data tends to use
            an increasing ordering for latitude while wtk uses a decreasing
            ordering.

        Returns
        -------
        ndarray
            (spatial_1, spatial_2, 2) Lat/Lon array with same ordering in last
            dimension
        """
        lat_lon = cls.lookup('lat_lon', 'compute')(file_paths, raster_index)
        if invert_lat:
            lat_lon = lat_lon[::-1]
        # put angle betwen -180 and 180
        lat_lon[..., 1] = (lat_lon[..., 1] + 180) % 360 - 180
        return lat_lon

    @classmethod
    def get_node_cmd(cls, config):
        """Get a CLI call to initialize DataHandler and cache data.

        Parameters
        ----------
        config : dict
            sup3r data handler config with all necessary args and kwargs to
            initialize DataHandler and run data extraction.
        """

        import_str = ('from sup3r.preprocessing.data_handling '
                      f'import {cls.__name__};\n'
                      'import time;\n'
                      'from reV.pipeline.status import Status;\n'
                      'from rex import init_logger;\n')

        dh_init_str = get_fun_call_str(cls, config)

        log_file = config.get('log_file', None)
        log_level = config.get('log_level', 'INFO')
        log_arg_str = '\"sup3r\", '
        log_arg_str += f'log_file=\"{log_file}\", '
        log_arg_str += f'log_level=\"{log_level}\"'

        cache_check = config.get('cache_pattern', False)

        msg = ('No cache file prefix provided.')
        if not cache_check:
            logger.warning(msg)
            warnings.warn(msg)

        job_name = config.get('job_name', None)

        cmd = (f"python -c \'{import_str}\n"
               "t0 = time.time();\n"
               f"logger = init_logger({log_arg_str});\n"
               f"data_handler = {dh_init_str};\n"
               "t_elap = time.time() - t0;\n")

        if job_name is not None:
            status_dir = config.get('status_dir', None)
            status_file_arg_str = f'\"{status_dir}\", '
            status_file_arg_str += f'module=\"{ModuleName.DATA_EXTRACT}\", '
            status_file_arg_str += f'job_name=\"{job_name}\", '
            status_file_arg_str += 'attrs=job_attrs'

            cmd += ('job_attrs = {};\n'.format(json.dumps(config)
                                               .replace("null", "None")
                                               .replace("false", "False")
                                               .replace("true", "True")))
            cmd += 'job_attrs.update({"job_status": "successful"});\n'
            cmd += 'job_attrs.update({"time": t_elap});\n'
            cmd += (f"Status.make_job_file({status_file_arg_str})")

        cmd += (";\'\n")
        return cmd.replace('\\', '/')

    def get_cache_file_names(self, cache_pattern):
        """Get names of cache files from cache_pattern and feature names

        Parameters
        ----------
        cache_pattern : str
            Pattern to use for cache file names

        Returns
        -------
        list
            List of cache file names
        """
        if cache_pattern is not None:
            basedir = os.path.dirname(cache_pattern)
            if not os.path.exists(basedir):
                os.makedirs(basedir)
            cache_files = [cache_pattern.replace('{feature}', f.lower())
                           for f in self.features]

            for i, f in enumerate(cache_files):
                if '{shape}' in f:
                    shape = f'{self.grid_shape[0]}x{self.grid_shape[1]}'
                    f = f.replace('{shape}', shape)
                if '{target}' in f:
                    target = f'{self.target[0]:.2f}_{self.target[1]:.2f}'
                    f = f.replace('{target}', target)
                if '{times}' in f:
                    times = f'{self.timestamp_0}_{self.timestamp_1}'
                    f = f.replace('{times}', times)

                cache_files[i] = f

            for i, fp in enumerate(cache_files):
                fp_check = ignore_case_path_fetch(fp)
                if fp_check is not None:
                    cache_files[i] = fp_check
        else:
            cache_files = None

        return cache_files

    def unnormalize(self, means, stds):
        """Remove normalization from stored means and stds"""
        for i in range(self.shape[-1]):
            self.val_data[..., i] = self.val_data[..., i] * stds[i] + means[i]
            self.data[..., i] = self.data[..., i] * stds[i] + means[i]

    def normalize(self, means, stds, max_workers=None):
        """Normalize all data features

        Parameters
        ----------
        means : np.ndarray
            dimensions (features)
            array of means for all features with same ordering as data features
        stds : np.ndarray
            dimensions (features)
            array of means for all features with same ordering as data features
        """
        if max_workers == 1:
            for i in range(self.shape[-1]):
                self._normalize_data(i, means[i], stds[i])
        else:
            self.parallel_normalization(means, stds)

    def parallel_normalization(self, means, stds):
        """Run normalization of features in parallel

        Parameters
        ----------
        means : np.ndarray
            dimensions (features)
            array of means for all features with same ordering as data features
        stds : np.ndarray
            dimensions (features)
            array of means for all features with same ordering as data features
        """
        logger.info(f'Normalizing {self.shape[-1]} features.')
        max_workers = self.norm_workers
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = {}
            now = dt.now()
            for i in range(self.shape[-1]):
                future = exe.submit(self._normalize_data, i, means[i],
                                    stds[i])
                futures[future] = i

            logger.info(f'Started normalizing {self.shape[-1]} features '
                        f'in {dt.now() - now}.')

            for i, future in enumerate(as_completed(futures)):
                try:
                    future.result()
                except Exception as e:
                    msg = ('Error while normalizing future number '
                           f'{futures[future]}.')
                    logger.exception(msg)
                    raise RuntimeError(msg) from e
                logger.debug(f'{i+1} out of {self.shape[-1]} features '
                             'normalized.')

    def _normalize_data(self, feature_index, mean, std):
        """Normalize data with initialized mean and standard deviation for a
        specific feature

        Parameters
        ----------
        feature_index : int
            index of feature to be normalized
        mean : float32
            specified mean of associated feature
        std : float32
            specificed standard deviation for associated feature
        """

        self.val_data[..., feature_index] -= mean
        self.data[..., feature_index] -= mean

        if std > 0:
            self.val_data[..., feature_index] /= std
            self.data[..., feature_index] /= std
        else:
            msg = ('Standard Deviation is zero for '
                   f'{self.features[feature_index]}')
            logger.warning(msg)
            warnings.warn(msg)

    def get_observation_index(self):
        """Randomly gets spatial sample and time sample

        Returns
        -------
        observation_index : tuple
            Tuple of sampled spatial grid, time slice, and features indices.
            Used to get single observation like self.data[observation_index]
        """
        spatial_slice = uniform_box_sampler(self.data, self.sample_shape[:2])
        temporal_slice = uniform_time_sampler(self.data, self.sample_shape[2])
        return tuple(
            spatial_slice + [temporal_slice] + [np.arange(len(self.features))])

    def get_next(self):
        """Gets data for observation using random observation index. Loops
        repeatedly over randomized time index

        Returns
        -------
        observation : np.ndarray
            4D array
            (spatial_1, spatial_2, temporal, features)
        """
        self.current_obs_index = self.get_observation_index()
        observation = self.data[self.current_obs_index]
        return observation

    def split_data(self, data=None):
        """Splits time dimension into set of training indices and validation
        indices

        Parameters
        ----------
        data : np.ndarray
            4D array of high res data
            (spatial_1, spatial_2, temporal, features)

        Returns
        -------
        data : np.ndarray
            (spatial_1, spatial_2, temporal, features)
            Training data fraction of initial data array. Initial data array is
            overwritten by this new data array.
        val_data : np.ndarray
            (spatial_1, spatial_2, temporal, features)
            Validation data fraction of initial data array.
        """

        if data is not None:
            self.data = data

        n_observations = self.data.shape[2]
        all_indices = np.arange(n_observations)
        n_val_obs = int(self.val_split * n_observations)

        if self.shuffle_time:
            np.random.shuffle(all_indices)

        val_indices = all_indices[:n_val_obs]
        training_indices = all_indices[n_val_obs:]

        if not self.shuffle_time:
            [self.val_data, self.data] = np.split(self.data, [n_val_obs],
                                                  axis=2)
        else:
            self.val_data = self.data[:, :, val_indices, :]
            self.data = self.data[:, :, training_indices, :]

        self.val_time_index = self.time_index[val_indices]
        self.time_index = self.time_index[training_indices]

        return self.data, self.val_data

    @property
    def shape(self):
        """Full data shape

        Returns
        -------
        shape : tuple
            Full data shape
            (spatial_1, spatial_2, temporal, features)
        """
        return self.data.shape

    def cache_data(self, cache_file_paths):
        """Cache feature data to file and delete from memory

        Parameters
        ----------
        cache_file_paths : str | None
            Path to file for saving feature data
        """

        for i, fp in enumerate(cache_file_paths):
            if not os.path.exists(fp) or self.overwrite_cache:
                if self.overwrite_cache and os.path.exists(fp):
                    logger.info(f'Overwriting {self.features[i]} with shape '
                                f'{self.data[..., i].shape} to {fp}')
                else:
                    logger.info(f'Saving {self.features[i]} with shape '
                                f'{self.data[..., i].shape} to {fp}')

                with open(fp, 'wb') as fh:
                    pickle.dump(self.data[..., i], fh, protocol=4)
            else:
                msg = (f'Called cache_data but {fp} already exists. Set to '
                       'overwrite_cache to True to overwrite.')
                logger.warning(msg)
                warnings.warn(msg)

    def parallel_load(self, max_workers=None):
        """Load feature data in parallel

        Parameters
        ----------
        max_workers : int | None
            Max number of workers to use for parallel data loading. If None
            the max number of available workers will be used.
        """
        logger.info(f'Loading {len(self.cache_files)} cache files.')
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = {}
            now = dt.now()
            for i, fp in enumerate(self.cache_files):
                future = exe.submit(self.load_single_cached_feature, fp=fp)
                futures[future] = {'idx': i, 'fp': os.path.basename(fp)}

            logger.info(f'Started loading all {len(self.cache_files)} cache '
                        f'files in {dt.now() - now}.')

            for i, future in enumerate(as_completed(futures)):
                try:
                    future.result()
                except Exception as e:
                    msg = ('Error while loading '
                           f'{self.cache_files[futures[future]["idx"]]}')
                    logger.exception(msg)
                    raise RuntimeError(msg) from e
                logger.debug(f'{i+1} out of {len(futures)} cache files '
                             f'loaded: {futures[future]["fp"]}')

    def load_single_cached_feature(self, fp):
        """Load single feature from given file

        Parameters
        ----------
        fp : string
            File path for feature cache file

        Raises
        ------
        RuntimeError
            Error raised if shape conflicts with requested shape
        """
        idx = self.cache_files.index(fp)
        assert self.features[idx].lower() in fp.lower()
        fp = ignore_case_path_fetch(fp)
        logger.info(f'Loading {self.features[idx]} from '
                    f'{os.path.basename(fp)}')

        with open(fp, 'rb') as fh:
            try:
                self.data[..., idx] = np.array(pickle.load(fh),
                                               dtype=np.float32)
            except Exception as e:
                msg = ('Data loaded from from cache file "{}" '
                       'could not be written to feature channel {} '
                       'of full data array of shape {}. '
                       'Make sure the cached data has the '
                       'appropriate shape.'
                       .format(fp, idx, self.data.shape))
                raise RuntimeError(msg) from e

    def load_cached_data(self):
        """Load data from cache files and split into training and validation
        """
        if self.data is not None:
            msg = ('Called load_cached_data() but self.data is not None')
            logger.warning(msg)
            warnings.warn(msg)

        elif self.data is None:
            shape = get_raster_shape(self.raster_index)
            requested_shape = (shape[0] // self.hr_spatial_coarsen,
                               shape[1] // self.hr_spatial_coarsen,
                               len(self.time_index),
                               len(self.features))

            msg = ('Found {} cache files but need {} for features {}! '
                   'These are the cache files that were found: {}'
                   .format(len(self.cache_files), len(self.features),
                           self.features, self.cache_files))
            assert len(self.cache_files) == len(self.features), msg

            self.data = np.full(shape=requested_shape, fill_value=np.nan,
                                dtype=np.float32)

            logger.info(f'Loading cached data from: {self.cache_files}')
            max_workers = self.load_workers
            if max_workers == 1:
                for _, fp in enumerate(self.cache_files):
                    self.load_single_cached_feature(fp)
            else:
                self.parallel_load(max_workers=max_workers)

            nan_perc = (100 * np.isnan(self.data).sum() / self.data.size)
            if nan_perc > 0:
                msg = ('Data has {:.2f}% NaN values!'.format(nan_perc))
                logger.warning(msg)
                warnings.warn(msg)

            logger.debug('Splitting data into training / validation sets '
                         f'({1 - self.val_split}, {self.val_split}) '
                         f'for {self.input_file_info}')
            self.data, self.val_data = self.split_data()

    @classmethod
    def check_cached_features(cls, features, cache_files=None,
                              overwrite_cache=False, load_cached=False):
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

    def extract_data(self):
        """Building base 4D data array. Can handle multiple files but assumes
        each file has the same spatial domain

        Returns
        -------
        data : np.ndarray
            4D array of high res data
            (spatial_1, spatial_2, temporal, features)
        """
        now = dt.now()
        logger.debug(f'Loading data for raster of shape {self.grid_shape}')

        # get the file-native time index without pruning
        time_index = self.raw_time_index
        n_steps = len(time_index[self.temporal_slice])

        # split time dimension into smaller slices which can be
        # extracted in parallel
        time_chunks = self.time_chunks
        shifted_time_chunks = get_chunk_slices(n_steps, self.time_chunk_size)
        logger.info(f'Starting extraction of {self.raw_features} using '
                    f'{len(time_chunks)} time_chunks. ')
        self._raw_data = self.parallel_extract(self.file_paths,
                                               self.raster_index,
                                               time_chunks, self.raw_features,
                                               self.extract_workers,
                                               invert_lat=self.invert_lat)

        logger.info(f'Finished extracting {self.raw_features} for '
                    f'{self.input_file_info}')
        if self.derive_features:
            logger.info(f'Starting compution of {self.derive_features}')
            self._raw_data = self.parallel_compute(self._raw_data,
                                                   self.raster_index,
                                                   time_chunks,
                                                   self.derive_features,
                                                   self.extract_features,
                                                   self.handle_features,
                                                   self.compute_workers)
            logger.info(f'Finished computing {self.derive_features} for '
                        f'{self.input_file_info}')

        logger.info('Building final data array')
        self.parallel_data_fill(shifted_time_chunks, self.extract_workers)

        if self.time_roll != 0:
            logger.debug('Applying time roll to data array')
            self.data = np.roll(self.data, self.time_roll, axis=2)

        if self.hr_spatial_coarsen > 1:
            logger.debug('Applying hr spatial coarsening to data array')
            self.data = spatial_coarsening(self.data,
                                           s_enhance=self.hr_spatial_coarsen,
                                           obs_axis=False)
        if self.load_cached:
            for f in self.cached_features:
                f_index = self.features.index(f)
                logger.info(f'Loading {f} from {self.cache_files[f_index]}')
                with open(self.cache_files[f_index], 'rb') as fh:
                    self.data[..., f_index] = pickle.load(fh)

        logger.info('Finished extracting data for '
                    f'{self.input_file_info} in '
                    f'{dt.now() - now}')
        return self.data

    def data_fill(self, t, t_slice, f_index, f):
        """Place single extracted / computed chunk in final data array

        Parameters
        ----------
        t : int
            Index of time slice in extracted / computed raw data dictionary
        t_slice : slice
            Time slice corresponding to the location in the final data array
        f_index : int
            Index of feature in the final data array
        f : str
            Name of corresponding feature in the raw data dictionary
        """
        self.data[..., t_slice, f_index] = self._raw_data[t][f]

    def serial_data_fill(self, shifted_time_chunks):
        """Fill final data array in serial

        Parameters
        ----------
        shifted_time_chunks : list
            List of time slices corresponding to the appropriate location of
            extracted / computed chunks in the final data array
        """
        for t, ts in enumerate(shifted_time_chunks):
            for _, f in enumerate(self.extract_features):
                f_index = self.features.index(f)
                self.data[..., ts, f_index] = self._raw_data[t][f]
            interval = np.int(np.ceil(len(shifted_time_chunks) / 10))
            if interval > 0 and t % interval == 0:
                logger.info(f'Added {t + 1} of {len(shifted_time_chunks)} '
                            'chunks to final data array')
            self._raw_data.pop(t)

    def parallel_data_fill(self, shifted_time_chunks, max_workers=None):
        """Fill final data array with extracted / computed chunks

        Parameters
        ----------
        shifted_time_chunks : list
            List of time slices corresponding to the appropriate location of
            extracted / computed chunks in the final data array
        max_workers : int | None
            Max number of workers to use for building final data array. If None
            max available workers will be used. If 1 cached data will be loaded
            in serial
        """
        time_index = self.raw_time_index
        n_steps = len(time_index[self.temporal_slice])
        self.data = np.zeros((self.grid_shape[0], self.grid_shape[1],
                              n_steps, len(self.features)), dtype=np.float32)

        if max_workers == 1:
            self.serial_data_fill(shifted_time_chunks)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                futures = {}
                now = dt.now()
                for t, ts in enumerate(shifted_time_chunks):
                    for _, f in enumerate(self.extract_features):
                        f_index = self.features.index(f)
                        future = exe.submit(self.data_fill, t, ts, f_index, f)
                        futures[future] = {'t': t, 'fidx': f_index}

                logger.info(f'Started adding {len(futures)} chunks '
                            f'to data array in {dt.now() - now}.')

                interval = np.int(np.ceil(len(futures) / 10))
                for i, future in enumerate(as_completed(futures)):
                    try:
                        future.result()
                    except Exception as e:
                        msg = (f'Error adding ({futures[future]["t"]}, '
                               f'{futures[future]["fidx"]}) chunk to '
                               'final data array.')
                        logger.exception(msg)
                        raise RuntimeError(msg) from e
                    if interval > 0 and i % interval == 0:
                        logger.debug(f'Added {i+1} out of {len(futures)} '
                                     'chunks to final data array')
        logger.info('Finished building data array')

    @abstractmethod
    def get_raster_index(self):
        """Get raster index for file data. Here we assume the list of paths in
        file_paths all have data with the same spatial domain. We use the first
        file in the list to compute the raster

        Returns
        -------
        raster_index : np.ndarray
            2D array of grid indices for H5 or list of
            slices for NETCDF
        """


class DataHandlerNC(DataHandler):
    """Data Handler for NETCDF data"""

    @property
    def extract_workers(self):
        """Get upper bound for extract workers based on memory limits. Used to
        extract data from source dataset"""
        # This large multiplier is due to the height interpolation allocating
        # multiple arrays with up to 60 vertical levels
        proc_mem = 6 * 64 * self.grid_mem * len(self.time_index)
        proc_mem /= len(self.time_chunks)
        n_procs = len(self.time_chunks) * len(self.raw_features)
        n_procs = int(np.ceil(n_procs))
        extract_workers = estimate_max_workers(self._extract_workers, proc_mem,
                                               n_procs)
        return extract_workers

    @classmethod
    def source_handler(cls, file_paths, **kwargs):
        """Xarray data handler

        Parameters
        ----------
        file_paths : str | list
            paths to data files
        kwargs : dict
            Dictionary of keyword args passed to xarray.open_mfdataset()

        Returns
        -------
        data : xarray.Dataset
        """
        return xr.open_mfdataset(file_paths, combine='nested',
                                 concat_dim='Time', **kwargs)

    @classmethod
    def get_time_index(cls, file_paths):
        """Get time index from data files

        Parameters
        ----------
        file_paths : list
            path to data file

        Returns
        -------
        time_index : pd.Datetimeindex
            List of times as a Datetimeindex
        """
        with cls.source_handler(file_paths) as handle:
            if hasattr(handle, 'Times'):
                time_index = np_to_pd_times(handle.Times.values)
            elif hasattr(handle, 'indexes') and 'time' in handle.indexes:
                time_index = handle.indexes['time'].to_datetimeindex()
            elif hasattr(handle, 'times'):
                time_index = np_to_pd_times(handle.times.values)
            else:
                raise ValueError(f'Could not get time_index for {file_paths}')
        return time_index

    @classmethod
    def feature_registry(cls):
        """Registry of methods for computing features

        Returns
        -------
        dict
            Method registry
        """
        registry = {
            'BVF2_(.*)m': BVFreqSquaredNC,
            'BVF_MO_(.*)m': BVFreqMon,
            'RMOL': InverseMonNC,
            'U_(.*)': UWind,
            'V_(.*)': VWind,
            'Windspeed_(.*)m': WindspeedNC,
            'Winddirection_(.*)m': WinddirectionNC,
            'lat_lon': LatLonNC,
            'Shear_(.*)m': Shear,
            'REWS_(.*)m': Rews,
            'Temperature_(.*)m': TempNC,
            'Pressure_(.*)m': 'P_(.*)m'}
        return registry

    @classmethod
    def extract_feature(cls, file_paths, raster_index, feature,
                        time_slice=slice(None), invert_lat=True):
        """Extract single feature from data source

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
        invert_lat : bool
            Flag to invert data along the latitude axis. Wrf data tends to use
            an increasing ordering for latitude while wtk uses a decreasing
            ordering.

        Returns
        -------
        ndarray
            Data array for extracted feature
            (spatial_1, spatial_2, temporal)
        """
        logger.info(f'Extracting {feature}')
        with cls.source_handler(file_paths) as handle:
            f_info = Feature(feature, handle)
            interp_height = f_info.height
            interp_pressure = f_info.pressure
            basename = f_info.basename
            if feature == 'lat_lon':
                return cls.get_lat_lon(file_paths, raster_index,
                                       invert_lat=invert_lat)
            # Sometimes xarray returns fields with (Times, time, lats, lons)
            # with a single entry in the 'time' dimension
            if feature in handle:
                if len(handle[feature].dims) == 4:
                    idx = tuple([time_slice] + [0] + raster_index)
                else:
                    idx = tuple([time_slice] + raster_index)
                fdata = np.array(handle[feature][idx], dtype=np.float32)
            elif basename in handle:
                if interp_height is not None:
                    fdata = interp_var_to_height(handle, basename,
                                                 raster_index,
                                                 np.float32(interp_height),
                                                 time_slice)
                elif interp_pressure is not None:
                    fdata = interp_var_to_pressure(handle, basename,
                                                   raster_index,
                                                   np.float32(interp_pressure),
                                                   time_slice)
            else:
                msg = f'{feature} cannot be extracted from source data.'
                logger.exception(msg)
                raise ValueError(msg)

        fdata = np.transpose(fdata, (1, 2, 0))
        if invert_lat:
            fdata = fdata[::-1]
        return fdata.astype(np.float32)

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
        shape : tuple
            (n_rows, n_cols) grid size
        """
        lat_lon = cls.get_lat_lon(file_paths, [slice(None), slice(None)],
                                  slice(None))
        target = (np.min(lat_lon[..., 0]), np.min(lat_lon[..., 1]))
        shape = lat_lon.shape[:-1]
        return target, shape

    @staticmethod
    def get_closest_lat_lon(lat_lon, target):
        """Get closest indices to target lat lon to use for lower left corner
        of raster index

        Parameters
        ----------
        lat_lon : ndarray
            Array of lat/lon
            (spatial_1, spatial_2, 2)
            Last dimension in order of (lat, lon)
        target : tuple
            (lat, lon) for lower left corner

        Returns
        -------
        row : int
            row index for closest lat/lon to target lat/lon
        col : int
            col index for closest lat/lon to target lat/lon
        """
        lat_diff = lon_diff = np.inf
        row = col = -1

        for i in range(lat_lon.shape[0]):
            for j in range(lat_lon.shape[1]):
                lat = lat_lon[i, j, 0]
                lon = lat_lon[i, j, 1]
                tmp_lat_diff = lat - target[0]
                tmp_lon_diff = lon - target[1]
                check = (0 <= tmp_lat_diff < lat_diff
                         and 0 <= tmp_lon_diff < lon_diff)
                if check:
                    lat_diff = np.abs(lat - target[0])
                    lon_diff = np.abs(lon - target[1])
                    row = i
                    col = j
        return row, col

    def get_raster_index(self):
        """Get raster index for file data. Here we assume the list of paths in
        file_paths all have data with the same spatial domain. We use the first
        file in the list to compute the raster.

        Returns
        -------
        raster_index : np.ndarray
            2D array of grid indices
        """
        self.raster_file = (self.raster_file if self.raster_file is None
                            else self.raster_file.replace('.txt', '.npy'))
        if self.raster_file is not None and os.path.exists(self.raster_file):
            logger.debug(f'Loading raster index: {self.raster_file} '
                         f'for {self.input_file_info}')
            raster_index = np.load(self.raster_file, allow_pickle=True)
            raster_index = list(raster_index)
        else:
            check = (self.grid_shape is not None and self.target is not None)
            msg = ('Must provide raster file or shape + target to get '
                   'raster index')
            assert check, msg
            lat_lon = self.get_lat_lon(self.file_paths[:1],
                                       [slice(None), slice(None)],
                                       invert_lat=False)
            min_lat = np.min(lat_lon[..., 0])
            min_lon = np.min(lat_lon[..., 1])
            max_lat = np.max(lat_lon[..., 0])
            max_lon = np.max(lat_lon[..., 1])
            logger.debug('Calculating raster index from WRF file '
                         f'for shape {self.grid_shape} and target '
                         f'{self.target}')
            msg = (f'target {self.target} out of bounds with min lat/lon '
                   f'{min_lat}/{min_lon} and max lat/lon {max_lat}/{max_lon}')
            assert (min_lat <= self.target[0] <= max_lat
                    and min_lon <= self._target[1] <= max_lon), msg

            row, col = self.get_closest_lat_lon(lat_lon, self.target)
            raster_index = [slice(row, row + self.grid_shape[0]),
                            slice(col, col + self.grid_shape[1])]

            if (raster_index[0].stop > lat_lon.shape[0]
               or raster_index[1].stop > lat_lon.shape[1]):
                msg = (f'Invalid target {self.target}, shape '
                       f'{self.grid_shape}, and raster '
                       f'{raster_index} for data domain of size '
                       f'{lat_lon.shape[:-1]} with lower left corner '
                       f'({np.min(lat_lon[..., 0])}, '
                       f'{np.min(lat_lon[..., 1])}).')
                raise ValueError(msg)

            lat_lon = lat_lon[tuple(raster_index + [slice(None)])]
            mask = ((lat_lon[..., 0] >= self.target[0])
                    & (lat_lon[..., 1] >= self.target[1]))
            if mask.sum() != np.product(self.grid_shape):
                msg = (f'Found {mask.sum()} coordinates but should have found '
                       f'{self.grid_shape[0]} by {self.grid_shape[1]}')
                logger.warning(msg)
                warnings.warn(msg)

            if self.raster_file is not None:
                logger.debug(f'Saving raster index: {self.raster_file}')
                np.save(self.raster_file.replace('.txt', '.npy'), raster_index)
        return raster_index


class DataHandlerNCforCC(DataHandlerNC):
    """Data Handler for NETCDF climate change data"""

    @classmethod
    def feature_registry(cls):
        """Registry of methods for computing features or extracting renamed
        features

        Returns
        -------
        dict
            Method registry
        """
        registry = {
            'U_(.*)': 'ua_(.*)',
            'V_(.*)': 'va_(.*)',
            'temperature_2m': 'tas',
            'relativehumidity_2m': 'hurs',
            'lat_lon': LatLonNCforCC}
        return registry

    @classmethod
    def source_handler(cls, file_paths, **kwargs):
        """Xarray data handler

        Parameters
        ----------
        file_paths : str | list
            paths to data files

        Returns
        -------
        data : xarray.Dataset
        """
        return xr.open_mfdataset(file_paths, **kwargs)


class DataHandlerH5(DataHandler):
    """DataHandler for H5 Data"""

    # the handler from rex to open h5 data.
    REX_HANDLER = MultiFileWindX

    @classmethod
    def source_handler(cls, file_paths):
        """rex data handler

        Parameters
        ----------
        file_paths : str | list
            paths to data files

        Returns
        -------
        data : ResourceX
        """
        return cls.REX_HANDLER(file_paths)

    @classmethod
    def get_full_domain(cls, file_paths):
        """Get target and shape for largest domain possible"""
        msg = ('You must either provide the target+shape inputs or an '
               'existing raster_file input.')
        logger.error(msg)
        raise ValueError(msg)

    @classmethod
    def get_time_index(cls, file_paths):
        """Get time index from data files

        Parameters
        ----------
        file_paths : list
            path to data file

        Returns
        -------
        time_index : pd.DateTimeIndex
            Time index from h5 source file(s)
        """
        with cls.source_handler(file_paths) as handle:
            time_index = handle.time_index
        return time_index

    @classmethod
    def feature_registry(cls):
        """Registry of methods for computing features or extracting renamed
        features

        Returns
        -------
        dict
            Method registry
        """
        registry = {
            'BVF2_(.*)m': BVFreqSquaredH5,
            'BVF_MO_(.*)m': BVFreqMon,
            'U_(.*)m': UWind,
            'V_(.*)m': VWind,
            'lat_lon': LatLonH5,
            'REWS_(.*)m': Rews,
            'RMOL': 'inversemoninobukhovlength_2m',
            'P_(.*)m': 'pressure_(.*)m'}
        return registry

    @classmethod
    def extract_feature(cls, file_paths, raster_index, feature,
                        time_slice=slice(None), invert_lat=False):
        """Extract single feature from data source

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
        invert_lat : bool
            Flag to invert latitude axis to enforce descending ordering

        Returns
        -------
        ndarray
            Data array for extracted feature
            (spatial_1, spatial_2, temporal)
        """
        logger.info(f'Extracting {feature}')
        with cls.source_handler(file_paths) as handle:
            if feature == 'lat_lon':
                return cls.get_lat_lon(file_paths, raster_index,
                                       invert_lat=invert_lat)
            try:
                fdata = handle[(feature, time_slice,)
                               + tuple([raster_index.flatten()])]
            except ValueError as e:
                msg = f'{feature} cannot be extracted from source data'
                logger.exception(msg)
                raise ValueError(msg) from e

        fdata = fdata.reshape((-1, raster_index.shape[0],
                               raster_index.shape[1]))
        fdata = np.transpose(fdata, (1, 2, 0))
        if invert_lat:
            fdata = fdata[::-1]
        return fdata.astype(np.float32)

    def get_raster_index(self):
        """Get raster index for file data. Here we assume the list of paths in
        file_paths all have data with the same spatial domain. We use the first
        file in the list to compute the raster.

        Returns
        -------
        raster_index : np.ndarray
            2D array of grid indices
        """
        if self.raster_file is not None and os.path.exists(self.raster_file):
            logger.debug(f'Loading raster index: {self.raster_file} '
                         f'for {self.input_file_info}')
            raster_index = np.loadtxt(self.raster_file).astype(np.uint32)
        else:
            check = (self.grid_shape is not None and self.target is not None)
            msg = ('Must provide raster file or shape + target to get '
                   'raster index')
            assert check, msg
            logger.debug('Calculating raster index from WTK file '
                         f'for shape {self.grid_shape} and target '
                         f'{self.target}')
            with self.source_handler(self.file_paths[0]) as handle:
                raster_index = handle.get_raster_index(
                    self.target, self.grid_shape, max_delta=self.max_delta)
            if self.raster_file is not None:
                logger.debug(f'Saving raster index: {self.raster_file}')
                np.savetxt(self.raster_file, raster_index)
        return raster_index


class DataHandlerH5WindCC(DataHandlerH5):
    """Special data handling and batch sampling for h5 wtk or nsrdb data for
    climate change applications"""

    # the handler from rex to open h5 data.
    REX_HANDLER = MultiFileWindX

    # list of features / feature name patterns that are input to the generative
    # model but are not part of the synthetic output and are not sent to the
    # discriminator. These are case-insensitive and follow the Unix shell-style
    # wildcard format.
    TRAIN_ONLY_FEATURES = tuple()

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args : list
            Same positional args as DataHandlerH5
        **kwargs : dict
            Same keyword args as DataHandlerH5
        """
        sample_shape = kwargs.get('sample_shape', (10, 10, 24))
        t_shape = sample_shape[-1]

        if len(sample_shape) == 2:
            logger.info('Found 2D sample shape of {}. Adding spatial dim of 24'
                        .format(sample_shape))
            sample_shape = sample_shape + (24,)
            t_shape = sample_shape[-1]
            kwargs['sample_shape'] = sample_shape

        if t_shape < 24 or t_shape % 24 != 0:
            msg = ('Climate Change DataHandler can only work with temporal '
                   'sample shapes that are one or more days of hourly data '
                   '(e.g. 24, 48, 72...). The requested temporal sample '
                   'shape was: {}'.format(t_shape))
            logger.error(msg)
            raise RuntimeError(msg)

        # validation splits not enabled for solar CC model.
        kwargs['val_split'] = 0.0

        super().__init__(*args, **kwargs)

        self.daily_data = None
        self.daily_data_slices = None
        self.run_daily_averages()

    def run_daily_averages(self):
        """Calculate daily average data and store as attribute."""
        msg = ('Data needs to be hourly with at least 24 hours, but data '
               'shape is {}.'.format(self.data.shape))
        assert self.data.shape[2] % 24 == 0, msg
        assert self.data.shape[2] > 24, msg

        n_data_days = int(self.data.shape[2] / 24)
        daily_data_shape = (self.data.shape[0:2] + (n_data_days,)
                            + (self.data.shape[3],))

        logger.info('Calculating daily average datasets for {} training '
                    'data days.'.format(n_data_days))

        self.daily_data = np.zeros(daily_data_shape, dtype=np.float32)

        self.daily_data_slices = np.array_split(np.arange(self.data.shape[2]),
                                                n_data_days)
        self.daily_data_slices = [slice(x[0], x[-1] + 1)
                                  for x in self.daily_data_slices]
        for d, t_slice in enumerate(self.daily_data_slices):
            self.daily_data[:, :, d, :] = daily_temporal_coarsening(
                self.data[:, :, t_slice, :], temporal_axis=2)[:, :, 0, :]

        logger.info('Finished calculating daily average datasets for {} '
                    'training data days.'.format(n_data_days))

    def _normalize_data(self, feature_index, mean, std):
        """Normalize data with initialized mean and standard deviation for a
        specific feature

        Parameters
        ----------
        feature_index : int
            index of feature to be normalized
        mean : float32
            specified mean of associated feature
        std : float32
            specificed standard deviation for associated feature
        """
        super()._normalize_data(feature_index, mean, std)
        self.daily_data[..., feature_index] -= mean
        self.daily_data[..., feature_index] /= std

    @classmethod
    def feature_registry(cls):
        """Registry of methods for computing features

        Returns
        -------
        dict
            Method registry
        """
        registry = {'U_(.*)m': UWind,
                    'V_(.*)m': VWind,
                    'lat_lon': LatLonH5}
        return registry

    def get_observation_index(self):
        """Randomly gets spatial sample and time sample

        Returns
        -------
        obs_ind_hourly : tuple
            Tuple of sampled spatial grid, time slice, and features indices.
            Used to get single observation like self.data[observation_index].
            This is for hourly high-res data slicing.
        obs_ind_daily : tuple
            Same as obs_ind_hourly but the temporal index (i=2) is a slice of
            the daily data (self.daily_data) with day integers.
        """
        spatial_slice = uniform_box_sampler(self.data, self.sample_shape[:2])

        n_days = int(self.sample_shape[2] / 24)
        rand_day_ind = np.random.choice(len(self.daily_data_slices) - n_days)
        t_slice_0 = self.daily_data_slices[rand_day_ind]
        t_slice_1 = self.daily_data_slices[rand_day_ind + n_days - 1]
        t_slice_hourly = slice(t_slice_0.start, t_slice_1.stop)
        t_slice_daily = slice(rand_day_ind, rand_day_ind + n_days)

        obs_ind_hourly = tuple(spatial_slice
                               + [t_slice_hourly]
                               + [np.arange(len(self.features))])

        obs_ind_daily = tuple(spatial_slice
                              + [t_slice_daily]
                              + [np.arange(len(self.features))])

        return obs_ind_hourly, obs_ind_daily

    def get_next(self):
        """Gets data for observation using random observation index. Loops
        repeatedly over randomized time index

        Returns
        -------
        obs_hourly : np.ndarray
            4D array
            (spatial_1, spatial_2, temporal_hourly, features)
        obs_daily_avg : np.ndarray
            4D array but the temporal axis is temporal_hourly//24
            (spatial_1, spatial_2, temporal_daily, features)
        """
        obs_ind_hourly, obs_ind_daily = self.get_observation_index()
        self.current_obs_index = obs_ind_hourly
        obs_hourly = self.data[obs_ind_hourly]
        obs_daily_avg = self.daily_data[obs_ind_daily]
        return obs_hourly, obs_daily_avg

    def split_data(self, data=None):
        """Splits time dimension into set of training indices and validation
        indices. For NSRDB it makes sure that the splits happen at midnight.

        Parameters
        ----------
        data : np.ndarray
            4D array of high res data
            (spatial_1, spatial_2, temporal, features)

        Returns
        -------
        data : np.ndarray
            (spatial_1, spatial_2, temporal, features)
            Training data fraction of initial data array. Initial data array is
            overwritten by this new data array.
        val_data : np.ndarray
            (spatial_1, spatial_2, temporal, features)
            Validation data fraction of initial data array.
        """

        if data is not None:
            self.data = data

        midnight_ilocs = np.where((self.time_index.hour == 0)
                                  & (self.time_index.minute == 0)
                                  & (self.time_index.second == 0))[0]

        n_val_obs = int(np.ceil(self.val_split * len(midnight_ilocs)))
        val_split_index = midnight_ilocs[n_val_obs]

        self.val_data = self.data[:, :, slice(None, val_split_index), :]
        self.data = self.data[:, :, slice(val_split_index, None), :]

        self.val_time_index = self.time_index[slice(None, val_split_index)]
        self.time_index = self.time_index[slice(val_split_index, None)]

        return self.data, self.val_data


class DataHandlerH5SolarCC(DataHandlerH5WindCC):
    """Special data handling and batch sampling for h5 NSRDB solar data for
    climate change applications"""

    # the handler from rex to open h5 data.
    REX_HANDLER = MultiFileNSRDBX

    # list of features / feature name patterns that are input to the generative
    # model but are not part of the synthetic output and are not sent to the
    # discriminator. These are case-insensitive and follow the Unix shell-style
    # wildcard format.
    TRAIN_ONLY_FEATURES = ('U', 'V', 'air_temperature')

    @classmethod
    def feature_registry(cls):
        """Registry of methods for computing features

        Returns
        -------
        dict
            Method registry
        """
        registry = {
            'U': UWind,
            'V': VWind,
            'windspeed': 'wind_speed',
            'winddirection': 'wind_direction',
            'lat_lon': LatLonH5,
            'cloud_mask': CloudMaskH5,
            'clearsky_ratio': ClearSkyRatioH5}
        return registry


# pylint: disable=W0223
class DataHandlerDC(DataHandler):
    """Data-centric data handler"""

    def get_observation_index(self, temporal_weights):
        """Randomly gets spatial sample and time sample

        Returns
        -------
        observation_index : tuple
            Tuple of sampled spatial grid, time slice, and features indices.
            Used to get single observation like self.data[observation_index]
        temporal_focus : slice
            Slice used to select prefered temporal range from full extent
        """
        spatial_slice = uniform_box_sampler(self.data, self.sample_shape[:2])
        temporal_slice = weighted_time_sampler(self.data, self.sample_shape[2],
                                               weights=temporal_weights)
        return tuple(
            spatial_slice + [temporal_slice] + [np.arange(len(self.features))])

    def get_next(self, temporal_weights):
        """Gets data for observation using random observation index. Loops
        repeatedly over randomized time index

        Returns
        -------
        observation : np.ndarray
            4D array
            (spatial_1, spatial_2, temporal, features)
        temporal_focus : slice
            Slice used to select prefered temporal range from full extent
        """
        self.current_obs_index = self.get_observation_index(temporal_weights)
        observation = self.data[self.current_obs_index]
        return observation


class DataHandlerDCforNC(DataHandlerNC, DataHandlerDC):
    """Data centric data handler for NETCDF files"""


class DataHandlerDCforH5(DataHandlerH5, DataHandlerDC):
    """Data centric data handler for H5 files"""
