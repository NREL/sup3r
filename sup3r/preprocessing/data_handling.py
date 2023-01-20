# -*- coding: utf-8 -*-
"""
Sup3r preprocessing module.
@author: bbenton
"""

from abc import abstractmethod
import copy
from fnmatch import fnmatch
import logging
import xarray as xr
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import os
from datetime import datetime as dt
import pickle
import warnings
import glob
from scipy.stats import mode
from scipy.ndimage.filters import gaussian_filter
from concurrent.futures import (as_completed, ThreadPoolExecutor)

from rex import MultiFileWindX, MultiFileNSRDBX, Resource
from rex.utilities import log_mem
from rex.utilities.fun_utils import get_fun_call_str

from sup3r.utilities.utilities import (estimate_max_workers,
                                       get_chunk_slices,
                                       interp_var_to_height,
                                       interp_var_to_pressure,
                                       uniform_box_sampler,
                                       uniform_time_sampler,
                                       weighted_time_sampler,
                                       weighted_box_sampler,
                                       get_raster_shape,
                                       get_source_type,
                                       ignore_case_path_fetch,
                                       daily_temporal_coarsening,
                                       spatial_coarsening,
                                       np_to_pd_times)
from sup3r.utilities import ModuleName
from sup3r.utilities.cli import BaseCLI
from sup3r.preprocessing.feature_handling import (FeatureHandler,
                                                  Feature,
                                                  BVFreqMon,
                                                  BVFreqSquaredH5,
                                                  BVFreqSquaredNC,
                                                  InverseMonNC,
                                                  LatLonNC,
                                                  LatLonNCforCC,
                                                  TempNC,
                                                  TempNCforCC,
                                                  PotentialTempNC,
                                                  PressureNC,
                                                  UWind,
                                                  VWind,
                                                  LatLonH5,
                                                  ClearSkyRatioH5,
                                                  ClearSkyRatioCC,
                                                  CloudMaskH5,
                                                  WindspeedNC,
                                                  WinddirectionNC,
                                                  Shear,
                                                  Rews,
                                                  Tas,
                                                  TasMin,
                                                  TasMax,
                                                  TopoH5,
                                                  )

np.random.seed(42)

logger = logging.getLogger(__name__)


class InputMixIn:
    """MixIn class with properties and methods for handling the spatiotemporal
    data domain to extract from source data."""

    def __init__(self, target, shape, raster_file=None, raster_index=None,
                 temporal_slice=slice(None, None, 1)):
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
        self.res_kwargs = {}

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
            the same number of timesteps. Can also pass a string with a
            unix-style file path which will be passed through glob.glob
        """
        self._file_paths = file_paths
        if isinstance(self._file_paths, str):
            if '*' in self._file_paths:
                self._file_paths = glob.glob(self._file_paths)
            else:
                self._file_paths = [self._file_paths]

        self._file_paths = sorted(self._file_paths)

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
            basedir = os.path.dirname(self._cache_pattern)
            if not os.path.exists(basedir):
                os.makedirs(basedir, exist_ok=True)
        return self._cache_pattern

    @cache_pattern.setter
    def cache_pattern(self, cache_pattern):
        """Update the cache file pattern"""
        self._cache_pattern = cache_pattern

    @property
    def need_full_domain(self):
        """Check whether we need to get the full lat/lon grid to determine
        target and shape values"""
        no_raster_file = (self.raster_file is None
                          or not os.path.exists(self.raster_file))
        no_target_shape = (self._target is None or self._grid_shape is None)
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
        """lat lon grid for data in format (spatial_1, spatial_2, 2) Lat/Lon
        array with same ordering in last dimension. This returns the gid
        without any lat inversion.

        Returns
        -------
        ndarray
        """
        raster_file_exists = (self.raster_file is not None
                              and os.path.exists(self.raster_file))

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
        """lat lon grid for data in format (spatial_1, spatial_2, 2) Lat/Lon
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
            self._invert_lat = (lat_lon[0, 0, 0] < lat_lon[-1, 0, 0])
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
            if lat_lon[0, 0, 0] < lat_lon[-1, 0, 0]:
                self._target = tuple(lat_lon[0, 0, :])
            else:
                self._target = tuple(lat_lon[-1, 0, :])
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
    def time_index_file(self):
        """Get time index file path"""
        if self.source_type == 'h5':
            return None

        if self.cache_pattern is not None and self._time_index_file is None:
            basename = self.cache_pattern.replace('{times}', '')
            basename = basename.replace('{shape}', str(len(self.file_paths)))
            basename = basename.replace('_{target}', '')
            basename = basename.replace('{feature}', 'time_index')
            tmp = basename.split('_')
            if tmp[-2].isdigit() and tmp[-1].strip('.pkl').isdigit():
                basename = '_'.join(tmp[:-1]) + '.pkl'
            self._time_index_file = basename
        return self._time_index_file

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
            logger.debug(f'Saved raw_time_index to {self.time_index_file}')
            with open(self.time_index_file, 'wb') as f:
                pickle.dump(self._raw_time_index, f)
        logger.debug(f'Built full time index in {dt.now() - now} seconds.')
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
    def time_freq_hours(self):
        """Get the time frequency in hours as a float"""
        ti_deltas = self.raw_time_index - np.roll(self.raw_time_index, 1)
        ti_deltas_hours = ti_deltas.total_seconds()[1:-1] / 3600
        time_freq = float(mode(ti_deltas_hours).mode[0])
        return time_freq

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
    """Sup3r data handling and extraction for low-res source data or for
    artificially coarsened high-res source data for training.

    The sup3r data handler class is based on a 4D numpy array of shape:
    (spatial_1, spatial_2, temporal, features)
    """

    # list of features / feature name patterns that are input to the generative
    # model but are not part of the synthetic output and are not sent to the
    # discriminator. These are case-insensitive and follow the Unix shell-style
    # wildcard format.
    TRAIN_ONLY_FEATURES = ('BVF*', 'inversemoninobukhovlength_*', 'RMOL',
                           'topography')

    def __init__(self, file_paths, features, target=None, shape=None,
                 max_delta=20, temporal_slice=slice(None, None, 1),
                 hr_spatial_coarsen=None, time_roll=0, val_split=0.05,
                 sample_shape=(10, 10, 1), raster_file=None, raster_index=None,
                 shuffle_time=False, time_chunk_size=None, cache_pattern=None,
                 overwrite_cache=False, overwrite_ti_cache=False,
                 load_cached=False, train_only_features=None,
                 handle_features=None, single_ts_files=None, mask_nan=False,
                 worker_kwargs=None, res_kwargs=None):
        """
        Parameters
        ----------
        file_paths : str | list
            A single source h5 wind file to extract raster data from or a list
            of netcdf files with identical grid. The string can be a unix-style
            file path which will be passed through glob.glob
        features : list
            list of features to extract from the provided data
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
            exists or written to the file if it does not yet exist. If None and
            raster_index is not provided raster_index will be calculated
            directly. Either need target+shape, raster_file, or raster_index
            input.
        raster_index : list
            List of tuples or slices. Used as an alternative to computing the
            raster index from target+shape or loading the raster index from
            file
        shuffle_time : bool
            Whether to shuffle time indices before validation split
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
        overwrite_ti_cache : bool
            Whether to overwrite any previously saved time index cache files.
        overwrite_ti_cache : bool
            Whether to overwrite saved time index cache files.
        load_cached : bool
            Whether to load data from cache files
        train_only_features : list | tuple | None
            List of feature names or patt*erns that should only be included in
            the training set and not the output. If None (default), this will
            default to the class TRAIN_ONLY_FEATURES attribute.
        handle_features : list | None
            Optional list of features which are available in the provided data.
            Providing this eliminates the need for an initial search of
            available features prior to data extraction.
        single_ts_files : bool | None
            Whether input files are single time steps or not. If they are this
            enables some reduced computation. If None then this will be
            determined from file_paths directly.
        mask_nan : bool
            Flag to mask out (remove) any timesteps with NaN data from the
            source dataset. This is False by default because it can create
            discontinuities in the timeseries.
        worker_kwargs : dict | None
            Dictionary of worker values. Can include max_workers,
            extract_workers, compute_workers, load_workers, norm_workers,
            and ti_workers. Each argument needs to be an integer or None.

            The value of `max workers` will set the value of all other worker
            args. If max_workers == 1 then all processes will be serialized. If
            max_workers == None then other worker args will use their own
            provided values.

            `extract_workers` is the max number of workers to use for
            extracting features from source data. If None it will be estimated
            based on memory limits. If 1 processes will be serialized.
            `compute_workers` is the max number of workers to use for computing
            derived features from raw features in source data. `load_workers`
            is the max number of workers to use for loading cached feature
            data. `norm_workers` is the max number of workers to use for
            normalizing feature data. `ti_workers` is the max number of
            workers to use to get full time index. Useful when there are many
            input files each with a single time step. If this is greater than
            one, time indices for input files will be extracted in parallel
            and then concatenated to get the full time index. If input files
            do not all have time indices or if there are few input files this
            should be set to one.
        res_kwargs : dict | None
            kwargs passed to source handler for data extraction. e.g. This
            could be {'parallel': True,
                      'chunks': {'south_north': 120, 'west_east': 120}}
            which then gets passed to xr.open_mfdataset(file, **res_kwargs)
        """
        InputMixIn.__init__(self, target=target, shape=shape,
                            raster_file=raster_file,
                            raster_index=raster_index,
                            temporal_slice=temporal_slice)

        msg = 'No files provided to DataHandler. Aborting.'
        assert file_paths is not None and bool(file_paths), msg

        self.file_paths = file_paths
        self.features = features
        self.val_time_index = None
        self.max_delta = max_delta
        self.val_split = val_split
        self.sample_shape = sample_shape
        self.hr_spatial_coarsen = hr_spatial_coarsen or 1
        self.time_roll = time_roll
        self.shuffle_time = shuffle_time
        self.current_obs_index = None
        self.overwrite_cache = overwrite_cache
        self.overwrite_ti_cache = overwrite_ti_cache
        self.load_cached = load_cached
        self.data = None
        self.val_data = None
        self.res_kwargs = res_kwargs or {}
        self._single_ts_files = single_ts_files
        self._cache_pattern = cache_pattern
        self._train_only_features = train_only_features
        self._time_chunk_size = time_chunk_size
        self._handle_features = handle_features
        self._cache_files = None
        self._extract_features = None
        self._noncached_features = None
        self._raw_features = None
        self._raw_data = {}
        self._time_chunks = None
        worker_kwargs = worker_kwargs or {}
        self.max_workers = worker_kwargs.get('max_workers', None)
        self._ti_workers = worker_kwargs.get('ti_workers', None)
        self._extract_workers = worker_kwargs.get('extract_workers', None)
        self._norm_workers = worker_kwargs.get('norm_workers', None)
        self._load_workers = worker_kwargs.get('load_workers', None)
        self._compute_workers = worker_kwargs.get('compute_workers', None)
        self._worker_attrs = ['_ti_workers', '_norm_workers',
                              '_compute_workers', '_extract_workers',
                              '_load_workers']

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

            self._raster_size_check()
            self._run_data_init_if_needed()

            if cache_pattern is not None:
                self.cache_data(self.cache_files)
                self.data = None if not self.load_cached else self.data

            self._val_split_check()

        if mask_nan:
            nan_mask = np.isnan(self.data).any(axis=(0, 1, 3))
            logger.info('Removing {} out of {} timesteps due to NaNs'
                        .format(nan_mask.sum(), self.data.shape[2]))
            self.data = self.data[:, :, ~nan_mask, :]

        logger.info('Finished intializing DataHandler.')
        log_mem(logger, log_level='INFO')

    def _run_data_init_if_needed(self):
        """Check if any features need to be extracted and proceed with data
        extraction"""
        if any(self.features):
            self.data = self.run_all_data_init()
            nan_perc = (100 * np.isnan(self.data).sum() / self.data.size)
            if nan_perc > 0:
                msg = ('Data has {:.2f}% NaN values!'.format(nan_perc))
                logger.warning(msg)
                warnings.warn(msg)

    def _raster_size_check(self):
        """Check if the sample_shape is larger than the requested raster
        size"""
        bad_shape = (self.sample_shape[0] > self.grid_shape[0]
                     and self.sample_shape[1] > self.grid_shape[1])
        if bad_shape:
            msg = (f'spatial_sample_shape {self.sample_shape[:2]} is '
                   f'larger than the raster size {self.grid_shape}')
            logger.warning(msg)
            warnings.warn(msg)

    def _val_split_check(self):
        """Check if val_split > 0 and split data into validation and training.
        Make sure validation data is larger than sample_shape"""

        if self.data is not None and self.val_split > 0.0:
            self.data, self.val_data = self.split_data()
            msg = (f'Validation data has shape={self.val_data.shape} '
                   f'and sample_shape={self.sample_shape}. Use a smaller '
                   'sample_shape and/or larger val_split.')
            check = any(val_size < samp_size for val_size, samp_size
                        in zip(self.val_data.shape, self.sample_shape))
            if check:
                logger.warning(msg)
                warnings.warn(msg)

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
        """Handler for source data. Can use xarray, ResourceX, etc.

        Note that xarray appears to treat open file handlers as singletons
        within a threadpool, so its okay to open this source_handler without a
        context handler or a .close() statement.
        """

    @property
    def attrs(self):
        """Get atttributes of input data

        Returns
        -------
        dict
            Dictionary of attributes
        """
        handle = self.source_handler(self.file_paths)
        desc = handle.attrs
        return desc

    @property
    def train_only_features(self):
        """Features to use for training only and not output"""
        if self._train_only_features is None:
            self._train_only_features = self.TRAIN_ONLY_FEATURES
        return self._train_only_features

    @property
    def extract_workers(self):
        """Get upper bound for extract workers based on memory limits. Used to
        extract data from source dataset. The max number of extract workers
        is number of time chunks * number of features"""
        proc_mem = 4 * self.grid_mem * len(self.time_index)
        proc_mem /= len(self.time_chunks)
        n_procs = len(self.time_chunks) * len(self.extract_features)
        n_procs = int(np.ceil(n_procs))
        extract_workers = estimate_max_workers(self._extract_workers, proc_mem,
                                               n_procs)
        return extract_workers

    @property
    def compute_workers(self):
        """Get upper bound for compute workers based on memory limits. Used to
        compute derived features from source dataset."""
        proc_mem = int(np.ceil(len(self.extract_features)
                               / np.maximum(len(self.derive_features), 1)))
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
        if self.data is not None:
            norm_workers = estimate_max_workers(self._norm_workers,
                                                2 * self.feature_mem,
                                                self.shape[-1])
        else:
            norm_workers = self._norm_workers
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
            if self.is_time_independent:
                self._time_chunks = [slice(None)]
            else:
                self._time_chunks = get_chunk_slices(len(self.raw_time_index),
                                                     self.time_chunk_size,
                                                     self.temporal_slice)
        return self._time_chunks

    @property
    def is_time_independent(self):
        """Get whether source data files are time independent"""
        return self.raw_time_index[0] is None

    @property
    def n_tsteps(self):
        """Get number of time steps to extract"""
        if self.is_time_independent:
            return 1
        else:
            return len(self.raw_time_index[self.temporal_slice])

    @property
    def time_chunk_size(self):
        """Get upper bound on time chunk size based on memory limits"""
        if self._time_chunk_size is None:
            step_mem = self.feature_mem * len(self.extract_features)
            step_mem /= len(self.time_index)
            if step_mem == 0:
                self._time_chunk_size = self.n_tsteps
            else:
                self._time_chunk_size = np.min([int(1e9 / step_mem),
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
    def raster_index(self):
        """Raster index property"""
        if self._raster_index is None:
            self._raster_index = self.get_raster_index()
        return self._raster_index

    @raster_index.setter
    def raster_index(self, raster_index):
        """Update raster index property"""
        self._raster_index = raster_index

    @classmethod
    def get_handle_features(cls, file_paths):
        """Get all available features in input data

        Parameters
        ----------
        file_paths : list
            List of input file paths

        Returns
        -------
        handle_features : list
            List of available input features
        """
        handle_features = []
        for f in file_paths:
            handle = cls.source_handler([f])
            for r in handle:
                handle_features.append(Feature.get_basename(r))
        return list(set(handle_features))

    @property
    def handle_features(self):
        """All features available in raw input"""
        if self._handle_features is None:
            self._handle_features = self.get_handle_features(self.file_paths)
        return self._handle_features

    @property
    def noncached_features(self):
        """Get list of features needing extraction or derivation"""
        if self._noncached_features is None:
            self._noncached_features = self.check_cached_features(
                self.features, cache_files=self.cache_files,
                overwrite_cache=self.overwrite_cache,
                load_cached=self.load_cached)
        return self._noncached_features

    @property
    def extract_features(self):
        """Features to extract directly from the source handler"""
        return [f for f in self.raw_features
                if self.lookup(f, 'compute') is None
                or Feature.get_basename(f) in self.handle_features]

    @property
    def derive_features(self):
        """List of features which need to be derived from other features"""
        derive_features = [f for f in set(list(self.noncached_features)
                                          + list(self.extract_features))
                           if f not in self.extract_features]
        return derive_features

    @property
    def cached_features(self):
        """List of features which have been requested but have been determined
        not to need extraction. Thus they have been cached already."""
        return [f for f in self.features if f not in self.noncached_features]

    @property
    def raw_features(self):
        """Get list of features needed for computations"""
        if self._raw_features is None:
            self._raw_features = self.get_raw_feature_list(
                self.noncached_features, self.handle_features)
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

        self.cap_worker_args(self.max_workers)

        if len(self.sample_shape) == 2:
            logger.info('Found 2D sample shape of {}. Adding temporal dim of 1'
                        .format(self.sample_shape))
            self.sample_shape = self.sample_shape + (1,)

        start = self.temporal_slice.start
        stop = self.temporal_slice.stop
        n_steps = self.n_tsteps
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
        if len(self.raw_time_index) < self.sample_shape[2]:
            logger.warning(msg)
            warnings.warn(msg)

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

        msg = (f'Initializing DataHandler {self.input_file_info}. '
               f'Getting temporal range {str(self.time_index[0])} to '
               f'{str(self.time_index[-1])} (inclusive) '
               f'based on temporal_slice {self.temporal_slice}')
        logger.info(msg)

        logger.info(f'Using max_workers={self.max_workers}, '
                    f'norm_workers={self.norm_workers}, '
                    f'extract_workers={self.extract_workers}, '
                    f'compute_workers={self.compute_workers}, '
                    f'load_workers={self.load_workers}, '
                    f'ti_workers={self.ti_workers}')

    @classmethod
    def get_lat_lon(cls, file_paths, raster_index, invert_lat=False):
        """Get lat/lon grid for requested target and shape

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
        log_arg_str = (f'"sup3r", log_level="{log_level}"')
        if log_file is not None:
            log_arg_str += f', log_file="{log_file}"'

        cache_check = config.get('cache_pattern', False)

        msg = ('No cache file prefix provided.')
        if not cache_check:
            logger.warning(msg)
            warnings.warn(msg)

        cmd = (f"python -c \'{import_str}\n"
               "t0 = time.time();\n"
               f"logger = init_logger({log_arg_str});\n"
               f"data_handler = {dh_init_str};\n"
               "t_elap = time.time() - t0;\n")

        cmd = BaseCLI.add_status_cmd(config, ModuleName.DATA_EXTRACT, cmd)

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
            cache_files = [cache_pattern.replace('{feature}', f.lower())
                           for f in self.features]
            for i, f in enumerate(cache_files):
                if '{shape}' in f:
                    shape = f'{self.grid_shape[0]}x{self.grid_shape[1]}'
                    shape += f'x{len(self.time_index)}'
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
        self.val_data = (self.val_data * stds) + means
        self.data = (self.data * stds) + means

    def normalize(self, means, stds):
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
        logger.info(f'Normalizing {self.shape[-1]} features.')
        max_workers = self.norm_workers
        if max_workers == 1:
            for i in range(self.shape[-1]):
                self._normalize_data(i, means[i], stds[i])
        else:
            self.parallel_normalization(means, stds, max_workers=max_workers)

    def parallel_normalization(self, means, stds, max_workers=None):
        """Run normalization of features in parallel

        Parameters
        ----------
        means : np.ndarray
            dimensions (features)
            array of means for all features with same ordering as data features
        stds : np.ndarray
            dimensions (features)
            array of means for all features with same ordering as data features
        max_workers : int | None
            Max number of workers to use for normalizing features
        """

        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = {}
            now = dt.now()
            for i in range(self.shape[-1]):
                future = exe.submit(self._normalize_data, i, means[i], stds[i])
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

        if self.val_data is not None:
            self.val_data[..., feature_index] -= mean
        self.data[..., feature_index] -= mean

        if std > 0:
            if self.val_data is not None:
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

                tmp_file = fp.replace('.pkl', '.pkl.tmp')
                with open(tmp_file, 'wb') as fh:
                    pickle.dump(self.data[..., i], fh, protocol=4)
                os.replace(tmp_file, fp)
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
                       'The cached data has the wrong shape {}.'
                       .format(fp, idx, self.data.shape,
                               pickle.load(fh).shape))
                raise RuntimeError(msg) from e

    def load_cached_data(self):
        """Load data from cache files and split into training and validation
        """
        if self.data is not None:
            logger.info('Called load_cached_data() but self.data is not None')

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

    def run_all_data_init(self):
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
        if self.is_time_independent:
            n_steps = 1
            shifted_time_chunks = [slice(None)]
        else:
            n_steps = len(self.raw_time_index[self.temporal_slice])
            shifted_time_chunks = get_chunk_slices(n_steps,
                                                   self.time_chunk_size)

        self.run_data_extraction()
        self.run_data_compute()

        logger.info('Building final data array')
        self.parallel_data_fill(shifted_time_chunks, self.extract_workers)

        if self.invert_lat:
            self.data = self.data[::-1]

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

    def run_data_extraction(self):
        """Run the raw dataset extraction process from disk to raw
        un-manipulated datasets.
        """
        if self.extract_features:
            logger.info(f'Starting extraction of {self.extract_features} '
                        f'using {len(self.time_chunks)} time_chunks.')
            if self.extract_workers == 1:
                self._raw_data = self.serial_extract(self.file_paths,
                                                     self.raster_index,
                                                     self.time_chunks,
                                                     self.extract_features,
                                                     **self.res_kwargs)

            else:
                self._raw_data = self.parallel_extract(self.file_paths,
                                                       self.raster_index,
                                                       self.time_chunks,
                                                       self.extract_features,
                                                       self.extract_workers,
                                                       **self.res_kwargs)

            logger.info(f'Finished extracting {self.extract_features} for '
                        f'{self.input_file_info}')

    def run_data_compute(self):
        """Run the data computation / derivation from raw features to desired
        features.
        """
        if self.derive_features:
            logger.info(f'Starting computation of {self.derive_features}')

            if self.compute_workers == 1:
                self._raw_data = self.serial_compute(self._raw_data,
                                                     self.file_paths,
                                                     self.raster_index,
                                                     self.time_chunks,
                                                     self.derive_features,
                                                     self.noncached_features,
                                                     self.handle_features)

            elif self.compute_workers != 1:
                self._raw_data = self.parallel_compute(self._raw_data,
                                                       self.file_paths,
                                                       self.raster_index,
                                                       self.time_chunks,
                                                       self.derive_features,
                                                       self.noncached_features,
                                                       self.handle_features,
                                                       self.compute_workers)

            logger.info(f'Finished computing {self.derive_features} for '
                        f'{self.input_file_info}')

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
        tmp = self._raw_data[t][f]
        if len(tmp.shape) == 2:
            tmp = tmp[..., np.newaxis]
        self.data[..., t_slice, f_index] = tmp

    def serial_data_fill(self, shifted_time_chunks):
        """Fill final data array in serial

        Parameters
        ----------
        shifted_time_chunks : list
            List of time slices corresponding to the appropriate location of
            extracted / computed chunks in the final data array
        """
        for t, ts in enumerate(shifted_time_chunks):
            for _, f in enumerate(self.noncached_features):
                f_index = self.features.index(f)
                self.data_fill(t, ts, f_index, f)
            interval = int(np.ceil(len(shifted_time_chunks) / 10))
            if t % interval == 0:
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
        self.data = np.zeros((self.grid_shape[0], self.grid_shape[1],
                              self.n_tsteps, len(self.features)),
                             dtype=np.float32)

        if max_workers == 1:
            self.serial_data_fill(shifted_time_chunks)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                futures = {}
                now = dt.now()
                for t, ts in enumerate(shifted_time_chunks):
                    for _, f in enumerate(self.noncached_features):
                        f_index = self.features.index(f)
                        future = exe.submit(self.data_fill, t, ts, f_index, f)
                        futures[future] = {'t': t, 'fidx': f_index}

                logger.info(f'Started adding {len(futures)} chunks '
                            f'to data array in {dt.now() - now}.')

                interval = int(np.ceil(len(futures) / 10))
                for i, future in enumerate(as_completed(futures)):
                    try:
                        future.result()
                    except Exception as e:
                        msg = (f'Error adding ({futures[future]["t"]}, '
                               f'{futures[future]["fidx"]}) chunk to '
                               'final data array.')
                        logger.exception(msg)
                        raise RuntimeError(msg) from e
                    if i % interval == 0:
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

    def lin_bc(self, bc_files, threshold=0.1):
        """Bias correct the data in this DataHandler using linear bias
        correction factors from files output by MonthlyLinearCorrection or
        LinearCorrection from sup3r.bias.bias_calc

        Parameters
        ----------
        bc_files : list | tuple | str
            One or more filepaths to .h5 files output by
            MonthlyLinearCorrection or LinearCorrection. These should contain
            datasets named "{feature}_scalar" and "{feature}_adder" where
            {feature} is one of the features contained by this DataHandler and
            the data is a 3D array of shape (lat, lon, time) where time is
            length 1 for annual correction or 12 for monthly correction.
        threshold : float
            Nearest neighbor euclidian distance threshold. If the DataHandler
            coordinates are more than this value away from the bias correction
            lat/lon, an error is raised.
        """

        if isinstance(bc_files, str):
            bc_files = [bc_files]

        completed = []
        for idf, feature in enumerate(self.features):
            dset_scalar = f'{feature}_scalar'
            dset_adder = f'{feature}_adder'
            for fp in bc_files:
                with Resource(fp) as res:
                    lat = np.expand_dims(res['latitude'], axis=-1)
                    lon = np.expand_dims(res['longitude'], axis=-1)
                    lat_lon_bc = np.dstack((lat, lon))
                    lat_lon_0 = self.lat_lon[:1, :1]
                    diff = lat_lon_bc - lat_lon_0
                    diff = np.hypot(diff[..., 0], diff[..., 1])
                    idy, idx = np.where(diff == diff.min())
                    slice_y = slice(idy[0], idy[0] + self.shape[0])
                    slice_x = slice(idx[0], idx[0] + self.shape[1])

                    if diff.min() > threshold:
                        msg = ('The DataHandler top left coordinate of {} '
                               'appears to be {} away from the nearest '
                               'bias correction coordinate of {} from {}. '
                               'Cannot apply bias correction.'
                               .format(lat_lon_0, diff.min(),
                                       lat_lon_bc[idy, idx],
                                       os.path.basename(fp)))
                        logger.error(msg)
                        raise RuntimeError(msg)

                    check = (dset_scalar in res.dsets
                             and dset_adder in res.dsets
                             and feature not in completed)
                    if check:
                        scalar = res[dset_scalar, slice_y, slice_x]
                        adder = res[dset_adder, slice_y, slice_x]

                        if scalar.shape[-1] == 1:
                            scalar = np.repeat(scalar, self.shape[2], axis=2)
                            adder = np.repeat(adder, self.shape[2], axis=2)
                        elif scalar.shape[-1] == 12:
                            idm = self.time_index.month.values - 1
                            scalar = scalar[..., idm]
                            adder = adder[..., idm]
                        else:
                            msg = ('Can only accept bias correction factors '
                                   'with last dim equal to 1 or 12 but '
                                   'received bias correction factors with '
                                   'shape {}'.format(scalar.shape))
                            logger.error(msg)
                            raise RuntimeError(msg)

                        logger.info('Bias correcting "{}" with linear '
                                    'correction from "{}"'
                                    .format(feature, os.path.basename(fp)))
                        self.data[..., idf] *= scalar
                        self.data[..., idf] += adder
                        completed.append(feature)


class DataHandlerNC(DataHandler):
    """Data Handler for NETCDF data"""

    CHUNKS = {'XTIME': 100, 'XLAT': 150, 'XLON': 150,
              'south_north': 150, 'west_east': 150, 'Time': 100}
    """CHUNKS sets the chunk sizes to extract from the data in each dimension.
    Chunk sizes that approximately match the data volume being extracted
    typically results in the most efficient IO."""

    def __init__(self, *args, xr_chunks=None, **kwargs):
        """
        Parameters
        ----------
        *args : list
            Same ordered required arguments as DataHandler parent class.
        xr_chunks : int | "auto" | tuple | dict | None
            kwarg that goes to xr.DataArray.chunk(chunks=xr_chunks). Chunk
            sizes that approximately match the data volume being extracted
            typically results in the most efficient IO. If not provided, this
            defaults to the class CHUNKS attribute.
        **kwargs : list
            Same optional keyword arguments as DataHandler parent class.
        """
        if xr_chunks is not None:
            self.CHUNKS = xr_chunks

        super().__init__(*args, **kwargs)

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
        default_kws = {'combine': 'nested', 'concat_dim': 'Time',
                       'chunks': cls.CHUNKS}
        kwargs.update(default_kws)
        return xr.open_mfdataset(file_paths, **kwargs)

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
        max_workers = (len(file_paths) if max_workers is None
                       else np.min((max_workers, len(file_paths))))
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
                logger.debug(
                    f'Stored {i+1} out of {len(futures)} file times')
        times = np.concatenate(list(ti.values()))
        return pd.DatetimeIndex(sorted(set(times)))

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
            'Pressure_(.*)m': PressureNC,
            'PotentialTemp_(.*)m': PotentialTempNC,
            'PT_(.*)m': PotentialTempNC,
            'topography': 'HGT'}
        return registry

    @classmethod
    def extract_feature(cls, file_paths, raster_index, feature,
                        time_slice=slice(None), **kwargs):
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

        if feature in handle:
            fdata = cls.direct_extract(handle, feature, raster_index,
                                       time_slice)

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
        return fdata.astype(np.float32)

    @classmethod
    def direct_extract(cls, handle, feature, raster_index, time_slice):
        """Extract requested feature directly from source data, rather than
        interpolating to a requested height or pressure level

        Parameters
        ----------
        data : xarray
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
            idx = tuple([time_slice] + [0] + raster_index)
        elif len(handle[feature].dims) == 3:
            idx = tuple([time_slice] + raster_index)
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
        # shape of ll2 is (n, 2) where axis=1 is (lat, lon)
        ll2 = np.vstack((lat_lon[..., 0].flatten(),
                         lat_lon[..., 1].flatten())).T
        tree = KDTree(ll2)
        _, i = tree.query(np.array(target))
        row, col = np.where((lat_lon[..., 0] == ll2[i, 0])
                            & (lat_lon[..., 1] == ll2[i, 1]))
        row = row[0]
        col = col[0]
        return row, col

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
        lat_lon = cls.get_lat_lon(file_paths[:1], [slice(None), slice(None)],
                                  invert_lat=False)
        cls._check_grid_extent(target, grid_shape, lat_lon)

        row, col = cls.get_closest_lat_lon(lat_lon, target)

        closest = tuple(lat_lon[row, col])
        logger.debug(f'Found closest coordinate {closest} to target={target}')
        if np.hypot(closest[0] - target[0], closest[1] - target[1]) > 1:
            msg = 'Closest coordinate to target is more than 1 degree away'
            logger.warning(msg)
            warnings.warn(msg)

        raster_index = [slice(row, row + grid_shape[0]),
                        slice(col, col + grid_shape[1])]

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
           or raster_index[1].stop > lat_lon.shape[1]):
            msg = (f'Invalid target {target}, shape {grid_shape}, and raster '
                   f'{raster_index} for data domain of size '
                   f'{lat_lon.shape[:-1]} with lower left corner '
                   f'({np.min(lat_lon[..., 0])}, {np.min(lat_lon[..., 1])}).')
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
            raster_index = self.compute_raster_index(self.file_paths,
                                                     self.target,
                                                     self.grid_shape)
            logger.debug('Found raster index with row, col slices: {}'
                         .format(raster_index))

            if self.raster_file is not None:
                logger.debug(f'Saving raster index: {self.raster_file}')
                np.save(self.raster_file.replace('.txt', '.npy'), raster_index)

        return raster_index


class DataHandlerNCforCC(DataHandlerNC):
    """Data Handler for NETCDF climate change data"""

    CHUNKS = {'time': 5, 'lat': 20, 'lon': 20}
    """CHUNKS sets the chunk sizes to extract from the data in each dimension.
    Chunk sizes that approximately match the data volume being extracted
    typically results in the most efficient IO."""

    def __init__(self, *args, nsrdb_source_fp=None, nsrdb_agg=1,
                 nsrdb_smoothing=0, **kwargs):
        """
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
            'Windspeed_(.*)m': WindspeedNC,
            'Winddirection_(.*)m': WinddirectionNC,
            'topography': 'orog',
            'relativehumidity_2m': 'hurs',
            'relativehumidity_min_2m': 'hursmin',
            'relativehumidity_max_2m': 'hursmax',
            'clearsky_ratio': ClearSkyRatioCC,
            'lat_lon': LatLonNCforCC,
            'Pressure_(.*)': 'plev_(.*)',
            'Temperature_(.*)': TempNCforCC,
            'temperature_2m': Tas,
            'temperature_max_2m': TasMax,
            'temperature_min_2m': TasMin}
        return registry

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
        kwargs.update(default_kws)
        return xr.open_mfdataset(file_paths, **kwargs)

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
            cs_ghi_time_chunks = [slice(tc.start - tc0, tc.stop - tc0, tc.step)
                                  for tc in self.time_chunks]
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
               'with raw time index: {}'
               .format(self.time_freq_hours, self.raw_time_index))
        assert self.time_freq_hours == 24.0, msg

        msg = ('Can only handle source CC data with temporal_slice.step == 1 '
               'but received: {}'.format(self.temporal_slice.step))
        assert ((self.temporal_slice.step is None)
                | (self.temporal_slice.step == 1)), msg

        with Resource(self._nsrdb_source_fp) as res:
            ti_nsrdb = res.time_index
            meta_nsrdb = res.meta

        ti_deltas = ti_nsrdb - np.roll(ti_nsrdb, 1)
        ti_deltas_hours = ti_deltas.total_seconds()[1:-1] / 3600
        time_freq = float(mode(ti_deltas_hours).mode[0])
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
                    '{} and {} locations with agg factor {}.'
                    .format(os.path.basename(self._nsrdb_source_fp),
                            t_slice, i.shape[0], i.shape[1]))

        cs_shape = i.shape
        with Resource(self._nsrdb_source_fp) as res:
            cs_ghi = res['clearsky_ghi', t_slice, i.flatten()]

        cs_ghi = cs_ghi.reshape((len(cs_ghi),) + cs_shape)
        cs_ghi = cs_ghi.mean(axis=-1)

        windows = np.array_split(np.arange(len(cs_ghi)),
                                 len(cs_ghi) // (24 // time_freq))
        cs_ghi = [cs_ghi[window].mean(axis=0) for window in windows]
        cs_ghi = np.vstack(cs_ghi)
        cs_ghi = cs_ghi.reshape((len(cs_ghi),) + tuple(self.grid_shape))
        cs_ghi = np.transpose(cs_ghi, axes=(1, 2, 0))

        if self.invert_lat:
            cs_ghi = cs_ghi[::-1]

        logger.info('Smoothing nsrdb clearsky ghi with a factor of {}'
                    .format(self._nsrdb_smoothing))
        for iday in range(cs_ghi.shape[-1]):
            cs_ghi[..., iday] = gaussian_filter(cs_ghi[..., iday],
                                                self._nsrdb_smoothing,
                                                mode='nearest')

        if cs_ghi.shape[-1] < t_end_target:
            n = int(np.ceil(t_end_target / cs_ghi.shape[-1]))
            cs_ghi = np.repeat(cs_ghi, n, axis=2)
            cs_ghi = cs_ghi[..., :t_end_target]

        logger.info('Reshaped clearsky_ghi data to final shape {} to '
                    'correspond with CC daily average data over source '
                    'temporal_slice {} with (lat, lon) grid shape of {}'
                    .format(cs_ghi.shape, self.temporal_slice,
                            self.grid_shape))

        return cs_ghi


class DataHandlerH5(DataHandler):
    """DataHandler for H5 Data"""

    # the handler from rex to open h5 data.
    REX_HANDLER = MultiFileWindX

    @classmethod
    def source_handler(cls, file_paths):
        """rex data handler

        Note that xarray appears to treat open file handlers as singletons
        within a threadpool, so its okay to open this source_handler without a
        context handler or a .close() statement.

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
    def get_time_index(cls, file_paths, max_workers=None, **kwargs):
        """Get time index from data files

        Parameters
        ----------
        file_paths : list
            path to data file
        max_workers : int | None
            placeholder to match signature

        Returns
        -------
        time_index : pd.DateTimeIndex
            Time index from h5 source file(s)
        """
        handle = cls.source_handler(file_paths)
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
            'P_(.*)m': 'pressure_(.*)m',
            'topography': TopoH5,
            'cloud_mask': CloudMaskH5,
            'clearsky_ratio': ClearSkyRatioH5}
        return registry

    @classmethod
    def extract_feature(cls, file_paths, raster_index, feature,
                        time_slice=slice(None), **kwargs):
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

        Returns
        -------
        ndarray
            Data array for extracted feature
            (spatial_1, spatial_2, temporal)
        """
        logger.info(f'Extracting {feature} with kwargs={kwargs}')
        handle = cls.source_handler(file_paths, **kwargs)
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
            handle = self.source_handler(self.file_paths[0])
            raster_index = handle.get_raster_index(self.target,
                                                   self.grid_shape,
                                                   max_delta=self.max_delta)
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
    TRAIN_ONLY_FEATURES = ('temperature_max_*m',
                           'temperature_min_*m',
                           'relativehumidity_max_*m',
                           'relativehumidity_min_*m')

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
        for idf, fname in enumerate(self.features):
            for d, t_slice in enumerate(self.daily_data_slices):
                if '_max_' in fname:
                    tmp = np.max(self.data[:, :, t_slice, idf], axis=2)
                    self.daily_data[:, :, d, idf] = tmp[:, :]
                elif '_min_' in fname:
                    tmp = np.min(self.data[:, :, t_slice, idf], axis=2)
                    self.daily_data[:, :, d, idf] = tmp[:, :]
                else:
                    tmp = daily_temporal_coarsening(
                        self.data[:, :, t_slice, idf], temporal_axis=2)
                    self.daily_data[:, :, d, idf] = tmp[:, :, 0]

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
                    'lat_lon': LatLonH5,
                    'topography': TopoH5,
                    'temperature_max_(.*)m': 'temperature_(.*)m',
                    'temperature_min_(.*)m': 'temperature_(.*)m',
                    'relativehumidity_max_(.*)m': 'relativehumidity_(.*)m',
                    'relativehumidity_min_(.*)m': 'relativehumidity_(.*)m',
                    }
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
    TRAIN_ONLY_FEATURES = ('U*', 'V*', 'topography')

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args : list
            Same positional args as DataHandlerH5
        **kwargs : dict
            Same keyword args as DataHandlerH5
        """

        args = copy.deepcopy(args)  # safe copy for manipulation
        required = ['ghi', 'clearsky_ghi', 'clearsky_ratio']
        missing = [dset for dset in required if dset not in args[1]]
        if any(missing):
            msg = ('Cannot initialize DataHandlerH5SolarCC without required '
                   'features {}. All three are necessary to get the daily '
                   'average clearsky ratio (ghi sum / clearsky ghi sum), even '
                   'though only the clearsky ratio will be passed to the GAN.'
                   .format(required))
            logger.error(msg)
            raise KeyError(msg)

        super().__init__(*args, **kwargs)

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
            'clearsky_ratio': ClearSkyRatioH5,
            'topography': TopoH5}
        return registry

    def run_daily_averages(self):
        """Calculate daily average data and store as attribute.

        Note that the H5 clearsky ratio feature requires special logic to match
        the climate change dataset of daily average GHI / daily average CS_GHI.
        This target climate change dataset is not equivalent to the average of
        instantaneous hourly clearsky ratios
        """

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

        i_ghi = self.features.index('ghi')
        i_cs = self.features.index('clearsky_ghi')
        i_ratio = self.features.index('clearsky_ratio')

        for d, t_slice in enumerate(self.daily_data_slices):
            for idf in range(self.data.shape[-1]):
                self.daily_data[:, :, d, idf] = daily_temporal_coarsening(
                    self.data[:, :, t_slice, idf], temporal_axis=2)[:, :, 0]

            # note that this ratio of daily irradiance sums is not the same as
            # the average of hourly ratios.
            total_ghi = np.nansum(self.data[:, :, t_slice, i_ghi], axis=2)
            total_cs_ghi = np.nansum(self.data[:, :, t_slice, i_cs], axis=2)
            avg_cs_ratio = total_ghi / total_cs_ghi
            self.daily_data[:, :, d, i_ratio] = avg_cs_ratio

        # remove ghi and clearsky ghi from feature set. These shouldn't be used
        # downstream for solar cc and keeping them confuses the batch handler
        logger.info('Finished calculating daily average clearsky_ratio, '
                    'removing ghi and clearsky_ghi from the '
                    'DataHandlerH5SolarCC feature list.')
        ifeats = np.array([i for i in range(len(self.features))
                           if i not in (i_ghi, i_cs)])
        self.data = self.data[..., ifeats]
        self.daily_data = self.daily_data[..., ifeats]
        self.features.remove('ghi')
        self.features.remove('clearsky_ghi')

        logger.info('Finished calculating daily average datasets for {} '
                    'training data days.'.format(n_data_days))


# pylint: disable=W0223
class DataHandlerDC(DataHandler):
    """Data-centric data handler"""

    def get_observation_index(self, temporal_weights=None,
                              spatial_weights=None):
        """Randomly gets weighted spatial sample and time sample

        Parameters
        ----------
        temporal_weights : array
            Weights used to select time slice
            (n_time_chunks)
        spatial_weights : array
            Weights used to select spatial chunks
            (n_lat_chunks * n_lon_chunks)

        Returns
        -------
        observation_index : tuple
            Tuple of sampled spatial grid, time slice, and features indices.
            Used to get single observation like self.data[observation_index]
        """
        if spatial_weights is not None:
            spatial_slice = weighted_box_sampler(self.data,
                                                 self.sample_shape[:2],
                                                 weights=spatial_weights)
        else:
            spatial_slice = uniform_box_sampler(self.data,
                                                self.sample_shape[:2])
        if temporal_weights is not None:
            temporal_slice = weighted_time_sampler(self.data,
                                                   self.sample_shape[2],
                                                   weights=temporal_weights)
        else:
            temporal_slice = uniform_time_sampler(self.data,
                                                  self.sample_shape[2])

        return tuple(
            spatial_slice + [temporal_slice] + [np.arange(len(self.features))])

    def get_next(self, temporal_weights=None, spatial_weights=None):
        """Gets data for observation using weighted random observation index.
        Loops repeatedly over randomized time index.

        Parameters
        ----------
        temporal_weights : array
            Weights used to select time slice
            (n_time_chunks)
        spatial_weights : array
            Weights used to select spatial chunks
            (n_lat_chunks * n_lon_chunks)

        Returns
        -------
        observation : np.ndarray
            4D array
            (spatial_1, spatial_2, temporal, features)
        """
        self.current_obs_index = self.get_observation_index(
            temporal_weights=temporal_weights, spatial_weights=spatial_weights)
        observation = self.data[self.current_obs_index]
        return observation


class DataHandlerDCforNC(DataHandlerNC, DataHandlerDC):
    """Data centric data handler for NETCDF files"""


class DataHandlerDCforH5(DataHandlerH5, DataHandlerDC):
    """Data centric data handler for H5 files"""
