# -*- coding: utf-8 -*-
"""sup3r WindStats module."""
import pandas as pd
import numpy as np
import os
import pickle
import logging
import psutil
from abc import ABC, abstractmethod
from scipy.ndimage.filters import gaussian_filter
from rex.utilities.fun_utils import get_fun_call_str

from sup3r.utilities import ModuleName
from sup3r.utilities.cli import BaseCLI
from sup3r.utilities.utilities import (get_input_handler_class,
                                       get_source_type,
                                       temporal_coarsening,
                                       spatial_coarsening,
                                       st_interp, vorticity_calc)
from sup3r.qa.utilities import (time_derivative_dist, direct_dist,
                                gradient_dist, wavenumber_spectrum,
                                frequency_spectrum)
from sup3r.preprocessing.feature_handling import Feature


logger = logging.getLogger(__name__)


class Sup3rStatsBase(ABC):
    """Base stats class"""

    # Acceptable statistics to request
    _DIRECT = 'direct'
    _DY_DX = 'gradient'
    _DY_DT = 'time_derivative'
    _FFT_F = 'spectrum_f'
    _FFT_K = 'spectrum_k'
    _FLUCT_FFT_F = 'fluctuation_spectrum_f'
    _FLUCT_FFT_K = 'fluctuation_spectrum_k'

    def __init__(self):
        """Base stats class"""
        self.overwrite_stats = True

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

        if type is not None:
            raise

    @abstractmethod
    def close(self):
        """Close any open file handlers"""

    @classmethod
    def save_cache(cls, array, file_name):
        """Save data to cache file

        Parameters
        ----------
        array : ndarray
            Wind field data
        file_name : str
            Path to cache file
        """
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        logger.info(f'Saving data to {file_name}')
        with open(file_name, 'wb') as f:
            pickle.dump(array, f, protocol=4)

    @classmethod
    def load_cache(cls, file_name):
        """Load data from cache file

        Parameters
        ----------
        file_name : str
            Path to cache file

        Returns
        -------
        array : ndarray
            Wind field data
        """
        logger.info(f'Loading data from {file_name}')
        with open(file_name, 'rb') as f:
            arr = pickle.load(f)
        return arr

    def export(self, qa_fp, data):
        """Export stats dictionary to pkl file.

        Parameters
        ----------
        qa_fp : str | None
            Optional filepath to output QA file (only .h5 is supported)
        data : dict
            A dictionary with stats for low and high resolution wind fields
        overwrite_stats : bool
            Whether to overwrite saved stats or not
        """

        os.makedirs(os.path.dirname(qa_fp), exist_ok=True)
        if not os.path.exists(qa_fp) or self.overwrite_stats:
            logger.info('Saving sup3r stats output file: "{}"'.format(qa_fp))
            with open(qa_fp, 'wb') as f:
                pickle.dump(data, f, protocol=4)
        else:
            logger.info(f'{qa_fp} already exists. Delete file or run with '
                        'overwrite_stats=True.')

    @classmethod
    def get_node_cmd(cls, config):
        """Get a CLI call to initialize Sup3rStats and execute the
        Sup3rStats.run() method based on an input config

        Parameters
        ----------
        config : dict
            sup3r wind stats config with all necessary args and kwargs to
            initialize Sup3rStats and execute Sup3rStats.run()
        """
        import_str = 'import time;\n'
        import_str += 'from reV.pipeline.status import Status;\n'
        import_str += 'from rex import init_logger;\n'
        import_str += f'from sup3r.qa.stats import {cls.__name__};\n'

        qa_init_str = get_fun_call_str(cls, config)

        log_file = config.get('log_file', None)
        log_level = config.get('log_level', 'INFO')

        log_arg_str = (f'"sup3r", log_level="{log_level}"')
        if log_file is not None:
            log_arg_str += f', log_file="{log_file}"'

        cmd = (f"python -c \'{import_str}\n"
               "t0 = time.time();\n"
               f"logger = init_logger({log_arg_str});\n"
               f"qa = {qa_init_str};\n"
               "qa.run();\n"
               "t_elap = time.time() - t0;\n")

        cmd = BaseCLI.add_status_cmd(config, ModuleName.STATS, cmd)
        cmd += (";\'\n")

        return cmd.replace('\\', '/')


class Sup3rStatsCompute(Sup3rStatsBase):
    """Base class for computing stats on input data arrays"""

    def __init__(self, input_data=None, s_enhance=1, t_enhance=1,
                 compute_features=None, input_features=None,
                 cache_pattern=None, overwrite_cache=False,
                 overwrite_stats=True, get_interp=False,
                 include_stats=None, max_values=None, smoothing=None,
                 spatial_res=None, temporal_res=None, n_bins=40, qa_fp=None,
                 interp_dists=True, time_chunk_size=100):
        """
        Parameters
        ----------
        input_data : ndarray
            An array of feature data to use for computing statistics
            (spatial_1, spatial_2, temporal, features)
        s_enhance : int
            Factor by which the Sup3rGan model enhanced the spatial
            dimensions of the input data
        t_enhance : int
            Factor by which the Sup3rGan model enhanced the temporal dimension
            of the input data
        compute_features : list
            Features for which to compute wind stats. e.g. ['pressure_100m',
            'temperature_100m', 'windspeed_100m']
        input_features : list
            List of features available in input_data, with same order as the
            last channel of input_data.
        cache_pattern : str | None
            Pattern for files for saving feature data. e.g.
            file_path_{feature}.pkl Each feature will be saved to a file with
            the feature name replaced in cache_pattern. If not None
            feature arrays will be saved here and not stored in self.data until
            load_cached_data is called. The cache_pattern can also include
            {shape}, {target}, {times} which will help ensure unique cache
            files for complex problems.
        overwrite_cache : bool
            Whether to overwrite cache files storing the interpolated feature
            data
        get_interp : bool
            Whether to include interpolated baseline stats in output
        include_stats : list | None
            List of stats to include in output. e.g. ['time_derivative',
            'gradient', 'vorticity', 'avg_spectrum_k', 'avg_spectrum_f',
            'direct']. 'direct' means direct distribution, as opposed to a
            distribution of the gradient or time derivative.
        max_values : dict | None
            Dictionary of max values to keep for stats. e.g.
            {'time_derivative': 10, 'gradient': 14, 'vorticity': 7}
        smoothing : float | None
            Value passed to gaussian filter used for smoothing source data
        spatial_res : float | None
            Spatial resolution for source data in meters. e.g. 2000. This is
            used to determine the wavenumber range for spectra calculations and
            to scale spatial derivatives.
        temporal_res : float | None
            Temporal resolution for source data in seconds. e.g. 60. This is
            used to determine the frequency range for spectra calculations and
            to scale temporal derivatives.
        n_bins : int
            Number of bins to use for constructing probability distributions
        qa_fp : str
            File path for saving statistics. Only .pkl supported.
        interp_dists : bool
            Whether to interpolate distributions over bins with count=0.
        time_chunk_size : int
            Size of temporal chunks to interpolate. e.g. If time_chunk_size=10
            then the temporal axis of low_res will be split into chunks with 10
            time steps, each chunk interpolated, and then the interpolated
            chunks will be concatenated.
        """

        msg = 'Preparing to compute statistics.'
        if input_data is None:
            msg = ('Received empty input array. Skipping statistics '
                   'computations.')
        logger.info(msg)

        self.max_values = max_values or {}
        self.n_bins = n_bins
        self.direct_max = self.max_values.get(self._DIRECT, None)
        self.time_derivative_max = self.max_values.get(self._DY_DT, None)
        self.gradient_max = self.max_values.get(self._DY_DX, None)
        self.include_stats = include_stats or [self._DIRECT, self._DY_DX,
                                               self._DY_DT, self._FFT_K]
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        self._features = compute_features
        self._k_range = None
        self._f_range = None
        self.input_features = input_features
        self.smoothing = smoothing
        self.get_interp = get_interp
        self.cache_pattern = cache_pattern
        self.overwrite_cache = overwrite_cache
        self.overwrite_stats = overwrite_stats
        self.spatial_res = spatial_res or 1
        self.temporal_res = temporal_res or 1
        self.source_data = input_data
        self.qa_fp = qa_fp
        self.interp_dists = interp_dists
        self.time_chunk_size = time_chunk_size

    @property
    def k_range(self):
        """Get range of wavenumbers to use for wavenumber spectrum
        calculation"""
        if self.spatial_res is not None:
            domain_size = self.spatial_res * self.source_data.shape[1]
            self._k_range = [1 / domain_size, 1 / self.spatial_res]
        return self._k_range

    @property
    def f_range(self):
        """Get range of frequencies to use for frequency spectrum
        calculation"""
        if self.temporal_res is not None:
            domain_size = self.temporal_res * self.source_data.shape[2]
            self._f_range = [1 / domain_size, 1 / self.temporal_res]
        return self._f_range

    @property
    def features(self):
        """Get a list of requested feature names

        Returns
        -------
        list
        """
        return self._features

    def _compute_spectra_type(self, var, stat_type, interp=False):
        """Select the appropriate method and parameters for the given stat_type
        and compute that spectrum

        Parameters
        ----------
        var: ndarray
            Variable for which to compute given spectrum type.
            (lat, lon, temporal)
        stat_type: str
            Spectrum type to compute. e.g. avg_fluctuation_spectrum_k will
            compute the wavenumber spectrum of the difference between the var
            and mean var.
        interp : bool
            Whether or not this is interpolated data. If True then this means
            that the spatial_res and temporal_res is different than the input
            data and needs to be scaled to get accurate wavenumber/frequency
            ranges.

        Returns
        -------
        ndarray
            wavenumber/frequency values
        ndarray
            amplitudes corresponding to the wavenumber/frequency values
        """
        tmp = var.copy()
        if self._FFT_K in stat_type:
            method = wavenumber_spectrum
            x_range = [self.k_range[0], self.k_range[1]]
            if interp:
                x_range[1] = x_range[1] * self.s_enhance
            if stat_type == self._FLUCT_FFT_K:
                tmp = self.get_fluctuation(tmp)
            tmp = np.mean(tmp[..., :-1], axis=-1)
        elif self._FFT_F in stat_type:
            method = frequency_spectrum
            x_range = [self.f_range[0], self.f_range[1]]
            if interp:
                x_range[1] = x_range[1] * self.t_enhance
            if stat_type == self._FLUCT_FFT_F:
                tmp = tmp - np.mean(tmp)
        else:
            return None

        kwargs = dict(var=tmp, x_range=x_range)
        return method(**kwargs)

    @staticmethod
    def get_fluctuation(var):
        """Get difference between array and temporal average of the same array

        Parameters
        ----------
        var : ndarray
            Array of data to calculate flucation for
            (spatial_1, spatial_2, temporal)

        Returns
        -------
        dvar : ndarray
            Array with fluctuation data
            (spatial_1, spatial_2, temporal)
        """
        avg = np.mean(var, axis=-1)
        return var - np.repeat(np.expand_dims(avg, axis=-1), var.shape[-1],
                               axis=-1)

    def interpolate_data(self, feature, low_res):
        """Get interpolated low res field

        Parameters
        ----------
        feature : str
            Name of feature to interpolate
        low_res : ndarray
            Array of low resolution data to interpolate
            (spatial_1, spatial_2, temporal)

        Returns
        -------
        var_itp : ndarray
            Array of interpolated data
            (spatial_1, spatial_2, temporal)
        """
        var_itp, file_name = self.check_return_cache(feature, low_res.shape)
        if var_itp is None:
            logger.info(f'Interpolating low res {feature}.')

            chunks = []
            slices = np.arange(low_res.shape[-1])
            n_chunks = low_res.shape[-1] // self.time_chunk_size + 1
            slices = np.array_split(slices, n_chunks)
            slices = [slice(s[0], s[-1] + 1) for s in slices]

            for i, s in enumerate(slices):
                chunks.append(st_interp(low_res[..., s], self.s_enhance,
                                        self.t_enhance))
                mem = psutil.virtual_memory()
                logger.info(f'Finished interpolating {i+1} / {len(slices)} '
                            'chunks. Current memory usage is '
                            f'{mem.used / 1e9:.3f} GB out of '
                            f'{mem.total / 1e9:.3f} GB total.')
            var_itp = np.concatenate(chunks, axis=-1)

            if 'direction' in feature:
                var_itp = (var_itp + 360) % 360

            if file_name is not None:
                self.save_cache(var_itp, file_name)
        return var_itp

    def check_return_cache(self, feature, shape):
        """Check if interpolated data is cached and return data if it is.
        Returns cache file name if cache_pattern is not None

        Parameters
        ----------
        feature : str
            Name of interpolated feature to check for cache
        shape : tuple
            Shape of low resolution data. Used to define cache file_name.

        Returns
        -------
        var_itp : ndarray | None
            Array of interpolated data if data exists. Otherwise returns None
        file_name : str
            Name of cache file for interpolated data. If cache_pattern is None
            this returns None
        """

        var_itp = None
        file_name = None
        shape_str = f'{shape[0]}x{shape[1]}x{shape[2]}'
        if self.cache_pattern is not None:
            file_name = self.cache_pattern.replace('{shape}', f'{shape_str}')
            file_name = file_name.replace('{feature}',
                                          f'{feature.lower()}_interp')
        if file_name is not None and os.path.exists(file_name):
            var_itp = self.load_cache(file_name)
        return var_itp, file_name

    def _compute_dist_type(self, var, stat_type, interp=False, period=None):
        """Select the appropriate method and parameters for the given stat_type
        and compute that distribution

        Parameters
        ----------
        var: ndarray
            Variable for which to compute distribution.
            (lat, lon, temporal)
        stat_type: str
            Distribution type to compute. e.g. mean_gradient will compute the
            gradient distribution of the temporal mean of var
        interp : bool
            Whether or not this is interpolated data. If True then this means
            that the spatial_res and temporal_res is different than the input
            data and needs to be scaled to get accurate derivatives.
        period : float | None
            If variable is periodic this gives that period. e.g. If the
            variable is winddirection the period is 360 degrees and we need to
            account for 0 and 360 being close.

        Returns
        -------
        ndarray
            Distribution values at bin centers
        ndarray
            Distribution value counts
        float
            Normalization factor
        """
        tmp = var.copy()
        if 'mean' in stat_type:
            tmp = (np.mean(tmp, axis=-1) if 'time' not in stat_type
                   else np.mean(tmp, axis=(0, 1)))
        if self._DIRECT in stat_type:
            max_val = self.direct_max
            method = direct_dist
            scale = 1
        elif self._DY_DX in stat_type:
            max_val = self.gradient_max
            method = gradient_dist
            scale = (self.spatial_res if not interp
                     else self.spatial_res / self.s_enhance)
        elif self._DY_DT in stat_type:
            max_val = self.time_derivative_max
            method = time_derivative_dist
            scale = (self.temporal_res if not interp
                     else self.temporal_res / self.t_enhance)
        else:
            return None

        kwargs = dict(var=tmp, diff_max=max_val, bins=self.n_bins, scale=scale,
                      interpolate=self.interp_dists, period=period)
        return method(**kwargs)

    def get_stats(self, var, interp=False, period=None):
        """Get stats for wind fields

        Parameters
        ----------
        var: ndarray
            (lat, lon, temporal)
        interp : bool
            Whether or not this is interpolated data. If True then this means
            that the spatial_res and temporal_res is different than the input
            data and needs to be scaled to get accurate derivatives.
        period : float | None
            If variable is periodic this gives that period. e.g. If the
            variable is winddirection the period is 360 degrees and we need to
            account for 0 and 360 being close.

        Returns
        -------
        stats : dict
            Dictionary of stats for wind fields
        """
        stats_dict = {}
        for stat_type in self.include_stats:

            if 'spectrum' in stat_type:
                out = self._compute_spectra_type(var, stat_type, interp=interp)
            else:
                out = self._compute_dist_type(var, stat_type, interp=interp,
                                              period=period)
            if out is not None:
                mem = psutil.virtual_memory()
                logger.info(f'Computed {stat_type}. Current memory usage is '
                            f'{mem.used / 1e9:.3f} GB out of '
                            f'{mem.total / 1e9:.3f} GB total.')
                stats_dict[stat_type] = out

        return stats_dict

    def get_feature_data(self, feature):
        """Get data for requested feature

        Parameters
        ----------
        feature : str
            Name of feature to get stats for

        Returns
        -------
        ndarray
            Array of data for requested feature
        """
        if self.source_data is None:
            return None

        if 'vorticity' in feature:
            height = Feature.get_height(feature)
            lower_features = [f.lower() for f in self.input_features]
            uidx = lower_features.index(f'u_{height}m')
            vidx = lower_features.index(f'v_{height}m')
            out = vorticity_calc(self.source_data[..., uidx],
                                 self.source_data[..., vidx],
                                 scale=self.spatial_res)
        else:
            idx = self.input_features.index(feature)
            out = self.source_data[..., idx]
        return out

    def get_feature_stats(self, feature):
        """Get stats for high and low resolution fields

        Parameters
        ----------
        feature : str
            Name of feature to get stats for

        Returns
        -------
        source_stats : dict
            Dictionary of stats for input fields
        interp : dict
            Dictionary of stats for spatiotemporally interpolated fields
        """
        source_stats = {}
        period = None
        if 'direction' in feature:
            period = 360

        if self.source_data is not None:
            out = self.get_feature_data(feature)
            source_stats = self.get_stats(out, period=period)

        interp = {}
        if self.get_interp:
            logger.info(f'Getting interpolated baseline stats for {feature}')
            itp = self.interpolate_data(feature, out)
            interp = self.get_stats(itp, interp=True, period=period)
        return source_stats, interp

    def run(self):
        """Go through all requested features and get the dictionary of
        statistics.

        Returns
        -------
        stats : dict
            Dictionary of statistics, where keys are source/interp appended
            with the feature name. Values are dictionaries of statistics, such
            as gradient, avg_spectrum, time_derivative, etc
        """

        source_stats = {}
        interp_stats = {}
        for _, feature in enumerate(self.features):
            logger.info(f'Running Sup3rStats for {feature}')
            source, interp = self.get_feature_stats(feature)

            mem = psutil.virtual_memory()
            logger.info(f'Current memory usage is {mem.used / 1e9:.3f} '
                        f'GB out of {mem.total / 1e9:.3f} GB total.')

            if self.source_data is not None:
                source_stats[feature] = source
            if self.get_interp:
                interp_stats[feature] = interp

        stats = {'source': source_stats, 'interp': interp_stats}
        if self.qa_fp is not None:
            logger.info(f'Saving stats to {self.qa_fp}')
            self.export(self.qa_fp, stats)

        logger.info('Finished Sup3rStats run method.')

        return stats


class Sup3rStatsSingle(Sup3rStatsCompute):
    """Base class for doing statistical QA on single file set."""

    def __init__(self, source_file_paths=None,
                 s_enhance=1, t_enhance=1, features=None,
                 temporal_slice=slice(None), target=None, shape=None,
                 raster_file=None, time_chunk_size=None,
                 cache_pattern=None, overwrite_cache=False,
                 overwrite_stats=False, source_handler=None,
                 worker_kwargs=None, get_interp=False, include_stats=None,
                 max_values=None, smoothing=None, coarsen=False,
                 spatial_res=None, temporal_res=None, n_bins=40,
                 max_delta=10, qa_fp=None):
        """
        Parameters
        ----------
        source_file_paths : list | str
            A list of source files to compute statistics on. Either .nc or .h5
        s_enhance : int
            Factor by which the Sup3rGan model enhanced the spatial
            dimensions of low resolution data
        t_enhance : int
            Factor by which the Sup3rGan model enhanced temporal dimension
            of low resolution data
        features : list
            Features for which to compute wind stats. e.g. ['pressure_100m',
            'temperature_100m', 'windspeed_100m', 'vorticity_100m']
        temporal_slice : slice | tuple | list
            Slice defining size of full temporal domain. e.g. If we have 5
            files each with 5 time steps then temporal_slice = slice(None) will
            select all 25 time steps. This can also be a tuple / list with
            length 3 that will be interpreted as slice(*temporal_slice)
        target : tuple
            (lat, lon) lower left corner of raster. You should provide
            target+shape or raster_file, or if all three are None the full
            source domain will be used.
        shape : tuple
            (rows, cols) grid size. You should provide target+shape or
            raster_file, or if all three are None the full source domain will
            be used.
        raster_file : str | None
            File for raster_index array for the corresponding target and
            shape. If specified the raster_index will be loaded from the file
            if it exists or written to the file if it does not yet exist.
            If None raster_index will be calculated directly. You should
            provide target+shape or raster_file, or if all three are None the
            full source domain will be used.
        time_chunk_size : int
            Size of chunks to split time dimension into for parallel data
            extraction. If running in serial this can be set to the size
            of the full time index for best performance.
        cache_pattern : str | None
            Pattern for files for saving feature data. e.g.
            file_path_{feature}.pkl Each feature will be saved to a file with
            the feature name replaced in cache_pattern. If not None
            feature arrays will be saved here and not stored in self.data until
            load_cached_data is called. The cache_pattern can also include
            {shape}, {target}, {times} which will help ensure unique cache
            files for complex problems.
        overwrite_cache : bool
            Whether to overwrite cache files storing the computed/extracted
            feature data
        overwrite_stats : bool
            Whether to overwrite saved stats
        input_handler : str | None
            data handler class to use for input data. Provide a string name to
            match a class in data_handling.py. If None the correct handler will
            be guessed based on file type and time series properties.
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
        get_interp : bool
            Whether to include interpolated baseline stats in output
        include_stats : list | None
            List of stats to include in output. e.g. ['time_derivative',
            'gradient', 'vorticity', 'avg_spectrum_k', 'avg_spectrum_f',
            'direct']. 'direct' means direct distribution, as opposed to a
            distribution of the gradient or time derivative.
        max_values : dict | None
            Dictionary of max values to keep for stats. e.g.
            {'time_derivative': 10, 'gradient': 14, 'vorticity': 7}
        smoothing : float | None
            Value passed to gaussian filter used for smoothing source data
        spatial_res : float | None
            Spatial resolution for source data in meters. e.g. 2000. This is
            used to determine the wavenumber range for spectra calculations.
        temporal_res : float | None
            Temporal resolution for source data in seconds. e.g. 60. This is
            used to determine the frequency range for spectra calculations and
            to scale temporal derivatives.
        coarsen : bool
            Whether to coarsen data or not
        max_delta : int, optional
            Optional maximum limit on the raster shape that is retrieved at
            once. If shape is (20, 20) and max_delta=10, the full raster will
            be retrieved in four chunks of (10, 10). This helps adapt to
            non-regular grids that curve over large distances, by default 20
        n_bins : int
            Number of bins to use for constructing probability distributions
        qa_fp : str
            File path for saving statistics. Only .pkl supported.
        """

        logger.info('Initializing Sup3rStatsSingle and retrieving source data'
                    f' for features={features}.')

        worker_kwargs = worker_kwargs or {}
        max_workers = worker_kwargs.get('max_workers', None)
        extract_workers = compute_workers = load_workers = ti_workers = None
        if max_workers is not None:
            extract_workers = compute_workers = load_workers = max_workers
            ti_workers = max_workers
        extract_workers = worker_kwargs.get('extract_workers', extract_workers)
        compute_workers = worker_kwargs.get('compute_workers', compute_workers)
        load_workers = worker_kwargs.get('load_workers', load_workers)
        ti_workers = worker_kwargs.get('ti_workers', ti_workers)

        self.ti_workers = ti_workers
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        self.smoothing = smoothing
        self.coarsen = coarsen
        self.get_interp = get_interp
        self.cache_pattern = cache_pattern
        self.overwrite_cache = overwrite_cache
        self.overwrite_stats = overwrite_stats
        self.source_file_paths = source_file_paths
        self.spatial_res = spatial_res
        self.temporal_res = temporal_res
        self.temporal_slice = temporal_slice
        self._shape = shape
        self._target = target
        self._source_handler = None
        self._source_handler_class = source_handler
        self._features = features
        self._input_features = None
        self._k_range = None
        self._f_range = None

        source_handler_kwargs = dict(target=target,
                                     shape=shape,
                                     temporal_slice=temporal_slice,
                                     raster_file=raster_file,
                                     cache_pattern=cache_pattern,
                                     time_chunk_size=time_chunk_size,
                                     overwrite_cache=overwrite_cache,
                                     worker_kwargs=worker_kwargs,
                                     max_delta=max_delta)
        self.source_data = self.get_source_data(source_file_paths,
                                                source_handler_kwargs)

        super().__init__(self.source_data, s_enhance=s_enhance,
                         t_enhance=t_enhance,
                         compute_features=self.compute_features,
                         input_features=self.input_features,
                         cache_pattern=cache_pattern,
                         overwrite_cache=overwrite_cache,
                         overwrite_stats=overwrite_stats,
                         get_interp=get_interp, include_stats=include_stats,
                         max_values=max_values, smoothing=smoothing,
                         spatial_res=spatial_res,
                         temporal_res=self.temporal_res, n_bins=n_bins,
                         qa_fp=qa_fp)

    def close(self):
        """Close any open file handlers"""
        if hasattr(self.source_handler, 'close'):
            self.source_handler.close()

    @property
    def source_type(self):
        """Get output data type

        Returns
        -------
        output_type
            e.g. 'nc' or 'h5'
        """
        if self.source_file_paths is None:
            return None

        ftype = get_source_type(self.source_file_paths)
        if ftype not in ('nc', 'h5'):
            msg = ('Did not recognize source file type: '
                   f'{self.source_file_paths}')
            logger.error(msg)
            raise TypeError(msg)
        return ftype

    @property
    def source_handler_class(self):
        """Get source handler class"""
        HandlerClass = get_input_handler_class(self.source_file_paths,
                                               self._source_handler_class)
        return HandlerClass

    @property
    def source_handler(self):
        """Get source data handler"""
        return self._source_handler

    # pylint: disable=E1102
    def get_source_data(self, file_paths, handler_kwargs=None):
        """Get source data using provided source file paths

        Parameters
        ----------

        file_paths : list | str
            A list of source files to extract raster data from. Each file must
            have the same number of timesteps. Can also pass a string with a
            unix-style file path which will be passed through glob.glob
        handler_kwargs : dict
            Dictionary of keyword arguments passed to
            `sup3r.preprocessing.data_handling.DataHandler`

        Returns
        -------
        ndarray
            Array of data from source file paths
            (spatial_1, spatial_2, temporal, features)
        """
        if file_paths is None:
            return None

        self._source_handler = self.source_handler_class(file_paths,
                                                         self.input_features,
                                                         val_split=0.0,
                                                         **handler_kwargs)
        self._source_handler.load_cached_data()
        if self.coarsen:
            logger.info('Coarsening data with shape='
                        f'{self._source_handler.data.shape}')
            self._source_handler.data = self.coarsen_data(
                self._source_handler.data, smoothing=self.smoothing)
            logger.info(f'Coarsened shape={self._source_handler.data.shape}')
        return self._source_handler.data

    @property
    def shape(self):
        """Shape of source data"""
        return self._shape

    @property
    def lat_lon(self):
        """Get lat/lon for output data"""
        if self.source_type is None:
            return None

        return self.source_handler.lat_lon

    @property
    def meta(self):
        """Get the meta data corresponding to the flattened source low-res data

        Returns
        -------
        pd.DataFrame
        """
        meta = pd.DataFrame({'latitude': self.lat_lon[..., 0].flatten(),
                             'longitude': self.lat_lon[..., 1].flatten()})
        return meta

    @property
    def time_index(self):
        """Get the time index associated with the source data

        Returns
        -------
        pd.DatetimeIndex
        """
        return self.source_handler.time_index

    @property
    def input_features(self):
        """Get a list of requested feature names

        Returns
        -------
        list
        """
        self._input_features = [f for f in self.compute_features if 'vorticity'
                                not in f]
        for feature in self.compute_features:
            if 'vorticity' in feature:
                height = Feature.get_height(feature)
                uf = f'U_{height}m'
                vf = f'V_{height}m'
                if uf.lower() not in [f.lower() for f in self._input_features]:
                    self._input_features.append(f'U_{height}m')
                if vf.lower() not in [f.lower() for f in self._input_features]:
                    self._input_features.append(f'V_{height}m')
        return self._input_features

    @property
    def compute_features(self):
        """Get list of requested feature names"""
        return self._features

    @input_features.setter
    def input_features(self, input_features):
        """Set input features"""
        self._input_features = [f for f in input_features if 'vorticity'
                                not in f]
        for feature in input_features:
            if 'vorticity' in feature:
                height = Feature.get_height(feature)
                uf = f'U_{height}m'
                vf = f'V_{height}m'
                if uf.lower() not in [f.lower() for f in self._input_features]:
                    self._input_features.append(f'U_{height}m')
                if vf.lower() not in [f.lower() for f in self._input_features]:
                    self._input_features.append(f'V_{height}m')
        return self._input_features

    def coarsen_data(self, data, smoothing=None):
        """Re-coarsen a high-resolution synthetic output dataset

        Parameters
        ----------
        data : np.ndarray
            A copy of the high-resolution output data as a numpy
            array of shape (spatial_1, spatial_2, temporal)

        Returns
        -------
        data : np.ndarray
            A spatiotemporally coarsened copy of the input dataset, still with
            shape (spatial_1, spatial_2, temporal)
        """
        n_lats = self.s_enhance * (data.shape[0] // self.s_enhance)
        n_lons = self.s_enhance * (data.shape[1] // self.s_enhance)
        data = spatial_coarsening(data[:n_lats, :n_lons],
                                  s_enhance=self.s_enhance,
                                  obs_axis=False)

        # t_coarse needs shape to be 5D: (obs, s1, s2, t, f)
        data = np.expand_dims(data, axis=0)
        data = temporal_coarsening(data, t_enhance=self.t_enhance)
        data = data[0]

        if smoothing is not None:
            for i in range(data.shape[-1]):
                for t in range(data.shape[-2]):
                    data[..., t, i] = gaussian_filter(data[..., t, i],
                                                      smoothing,
                                                      mode='nearest')
        return data


class Sup3rStatsMulti(Sup3rStatsBase):
    """Class for doing statistical QA on multiple datasets. These datasets
    are low resolution input to sup3r, the synthetic output, and the true
    high resolution corresponding to the low resolution input. This class
    will provide statistics used to compare all these datasets."""

    def __init__(self, lr_file_paths=None, synth_file_paths=None,
                 hr_file_paths=None, s_enhance=1, t_enhance=1, features=None,
                 lr_t_slice=slice(None), synth_t_slice=slice(None),
                 hr_t_slice=slice(None), target=None, shape=None,
                 raster_file=None, qa_fp=None, time_chunk_size=None,
                 cache_pattern=None, overwrite_cache=False,
                 overwrite_synth_cache=False, overwrite_stats=False,
                 source_handler=None, output_handler=None, worker_kwargs=None,
                 get_interp=False, include_stats=None, max_values=None,
                 smoothing=None, spatial_res=None, temporal_res=None,
                 n_bins=40, max_delta=10, save_fig_data=False):
        """
        Parameters
        ----------
        lr_file_paths : list | str
            A list of low-resolution source files (either .nc or .h5)
            to extract raster data from.
        synth_file_paths : list | str
            Sup3r-resolved output files (either .nc or .h5) with
            high-resolution data corresponding to the
            lr_file_paths * s_enhance * t_enhance
        hr_file_paths : list | str
            A list of high-resolution source files (either .nc or .h5)
            corresponding to the low-resolution source files in
            lr_file_paths
        s_enhance : int
            Factor by which the Sup3rGan model will enhance the spatial
            dimensions of low resolution data
        t_enhance : int
            Factor by which the Sup3rGan model will enhance temporal dimension
            of low resolution data
        features : list
            Features for which to compute wind stats. e.g. ['pressure_100m',
            'temperature_100m', 'windspeed_100m', 'vorticity_100m']
        lr_t_slice : slice | tuple | list
            Slice defining size of temporal domain for the low resolution data.
        synth_t_slice : slice | tuple | list
            Slice defining size of temporal domain for the sythetic high
            resolution data.
        hr_t_slice : slice | tuple | list
            Slice defining size of temporal domain for the true high
            resolution data.
        target : tuple
            (lat, lon) lower left corner of raster. You should provide
            target+shape or raster_file, or if all three are None the full
            source domain will be used.
        shape : tuple
            Shape of the low resolution grid size. (rows, cols). You should
            provide target+shape or raster_file, or if all three are None the
            full source domain will be used.
        raster_file : str | None
            File for raster_index array for the corresponding target and
            shape. If specified the raster_index will be loaded from the file
            if it exists or written to the file if it does not yet exist.
            If None raster_index will be calculated directly. You should
            provide target+shape or raster_file, or if all three are None the
            full source domain will be used.
        qa_fp : str | None
            Optional filepath to output QA file when you call
            Sup3rStatsWind.run()
            (only .pkl is supported)
        time_chunk_size : int
            Size of chunks to split time dimension into for parallel data
            extraction. If running in serial this can be set to the size
            of the full time index for best performance.
        cache_pattern : str | None
            Pattern for files for saving feature data. e.g.
            file_path_{feature}.pkl Each feature will be saved to a file with
            the feature name replaced in cache_pattern. If not None
            feature arrays will be saved here and not stored in self.data until
            load_cached_data is called. The cache_pattern can also include
            {shape}, {target}, {times} which will help ensure unique cache
            files for complex problems.
        overwrite_cache : bool
            Whether to overwrite cache files storing the computed/extracted
            feature data for low-resolution and high-resolution data
        overwrite_synth_cache : bool
            Whether to overwrite cache files stored computed/extracted data
            for synthetic output.
        overwrite_stats : bool
            Whether to overwrite saved stats
        input_handler : str | None
            data handler class to use for input data. Provide a string name to
            match a class in data_handling.py. If None the correct handler will
            be guessed based on file type and time series properties.
        output_handler : str | None
            data handler class to use for output data. Provide a string name to
            match a class in data_handling.py. If None the correct handler will
            be guessed based on file type and time series properties.
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
        get_interp : bool
            Whether to include interpolated baseline stats in output
        include_stats : list | None
            List of stats to include in output. e.g. ['time_derivative',
            'gradient', 'vorticity', 'avg_spectrum_k', 'avg_spectrum_f',
            'direct']. 'direct' means direct distribution, as opposed to a
            distribution of the gradient or time derivative.
        max_values : dict | None
            Dictionary of max values to keep for stats. e.g.
            {'time_derivative': 10, 'gradient': 14, 'vorticity': 7}
        smoothing : float | None
            Value passed to gaussian filter used for smoothing source data
        spatial_res : float | None
            Spatial resolution for source data in meters. e.g. 2000. This is
            used to determine the wavenumber range for spectra calculations.
        temporal_res : float | None
            Temporal resolution for source data in seconds. e.g. 60. This is
            used to determine the frequency range for spectra calculations and
            to scale temporal derivatives.
        max_delta : int, optional
            Optional maximum limit on the raster shape that is retrieved at
            once. If shape is (20, 20) and max_delta=10, the full raster will
            be retrieved in four chunks of (10, 10). This helps adapt to
            non-regular grids that curve over large distances, by default 20
        n_bins : int
            Number of bins to use for constructing probability distributions
        """

        logger.info('Initializing Sup3rStatsMulti and retrieving source data'
                    f' for features={features}.')

        self.qa_fp = qa_fp
        self.overwrite_stats = overwrite_stats
        self.save_fig_data = save_fig_data
        self.features = features

        # get low res and interp stats
        logger.info('Retrieving source data for low-res and interp stats')
        kwargs = dict(source_file_paths=lr_file_paths,
                      s_enhance=s_enhance, t_enhance=t_enhance,
                      features=features, temporal_slice=lr_t_slice,
                      target=target, shape=shape,
                      time_chunk_size=time_chunk_size,
                      cache_pattern=cache_pattern,
                      overwrite_cache=overwrite_cache,
                      overwrite_stats=overwrite_stats,
                      source_handler=source_handler,
                      worker_kwargs=worker_kwargs,
                      get_interp=get_interp, include_stats=include_stats,
                      max_values=max_values, smoothing=None,
                      spatial_res=spatial_res, temporal_res=temporal_res,
                      n_bins=n_bins, max_delta=max_delta)
        self.lr_stats = Sup3rStatsSingle(**kwargs)

        if self.lr_stats.source_data is not None:
            self.lr_shape = self.lr_stats.source_handler.grid_shape
            target = self.lr_stats.source_handler.target
        else:
            self.lr_shape = shape

        # get high res stats
        shape = (self.lr_shape[0] * s_enhance, self.lr_shape[1] * s_enhance)
        logger.info('Retrieving source data for high-res stats with '
                    f'shape={shape}')
        tmp_raster = (raster_file if raster_file is None
                      else raster_file.replace('.txt', '_hr.txt'))
        tmp_cache = (cache_pattern if cache_pattern is None
                     else cache_pattern.replace('.pkl', '_hr.pkl'))
        hr_spatial_res = spatial_res or 1
        hr_spatial_res /= s_enhance
        hr_temporal_res = temporal_res or 1
        hr_temporal_res /= t_enhance
        kwargs_new = dict(source_file_paths=hr_file_paths,
                          s_enhance=1, t_enhance=1,
                          shape=shape, target=target,
                          spatial_res=hr_spatial_res,
                          temporal_res=hr_temporal_res,
                          get_interp=False, source_handler=source_handler,
                          cache_pattern=tmp_cache,
                          temporal_slice=hr_t_slice)
        kwargs_hr = kwargs.copy()
        kwargs_hr.update(kwargs_new)
        self.hr_stats = Sup3rStatsSingle(**kwargs_hr)

        # get synthetic stats
        shape = (self.lr_shape[0] * s_enhance, self.lr_shape[1] * s_enhance)
        logger.info('Retrieving source data for synthetic stats with '
                    f'shape={shape}')
        tmp_raster = (raster_file if raster_file is None
                      else raster_file.replace('.txt', '_synth.txt'))
        tmp_cache = (cache_pattern if cache_pattern is None
                     else cache_pattern.replace('.pkl', '_synth.pkl'))
        kwargs_new = dict(source_file_paths=synth_file_paths,
                          s_enhance=1, t_enhance=1,
                          shape=shape, target=target,
                          spatial_res=hr_spatial_res,
                          temporal_res=hr_temporal_res,
                          get_interp=False, source_handler=output_handler,
                          raster_file=tmp_raster, cache_pattern=tmp_cache,
                          overwrite_cache=(overwrite_synth_cache),
                          temporal_slice=synth_t_slice)
        kwargs_synth = kwargs.copy()
        kwargs_synth.update(kwargs_new)
        self.synth_stats = Sup3rStatsSingle(**kwargs_synth)

        # get coarse stats
        logger.info('Retrieving source data for coarse stats')
        tmp_raster = (raster_file if raster_file is None
                      else raster_file.replace('.txt', '_coarse.txt'))
        tmp_cache = (cache_pattern if cache_pattern is None
                     else cache_pattern.replace('.pkl', '_coarse.pkl'))
        kwargs_new = dict(source_file_paths=hr_file_paths,
                          spatial_res=spatial_res, temporal_res=temporal_res,
                          target=target, shape=shape, smoothing=smoothing,
                          coarsen=True, get_interp=False,
                          source_handler=output_handler,
                          cache_pattern=tmp_cache,
                          temporal_slice=hr_t_slice)
        kwargs_coarse = kwargs.copy()
        kwargs_coarse.update(kwargs_new)
        self.coarse_stats = Sup3rStatsSingle(**kwargs_coarse)

    def export_fig_data(self):
        """Save data fields for data viz comparison"""
        for feature in self.features:
            fig_data = {}
            if self.synth_stats.source_data is not None:
                fig_data.update(
                    {'time_index': self.synth_stats.time_index,
                     'synth': self.synth_stats.get_feature_data(feature),
                     'synth_grid': self.synth_stats.source_handler.lat_lon})
            if self.lr_stats.source_data is not None:
                fig_data.update(
                    {'low_res': self.lr_stats.get_feature_data(feature),
                     'low_res_grid': self.lr_stats.source_handler.lat_lon})
            if self.hr_stats.source_data is not None:
                fig_data.update(
                    {'high_res': self.hr_stats.get_feature_data(feature),
                     'high_res_grid': self.hr_stats.source_handler.lat_lon})
            if self.coarse_stats.source_data is not None:
                fig_data.update(
                    {'coarse': self.coarse_stats.get_feature_data(feature)})

            file_name = self.qa_fp.replace('.pkl', f'_{feature}_compare.pkl')
            with open(file_name, 'wb') as fp:
                pickle.dump(fig_data, fp, protocol=4)
            logger.info(f'Saved figure data for {feature} to {file_name}.')

    def close(self):
        """Close any open file handlers"""
        stats = [self.lr_stats, self.hr_stats, self.synth_stats,
                 self.coarse_stats]
        for s_handle in stats:
            s_handle.close()

    def run(self):
        """Go through all datasets and get the dictionary of statistics.

        Returns
        -------
        stats : dict
            Dictionary of statistics, where keys are lr/hr/interp appended with
            the feature name. Values are dictionaries of statistics, such as
            gradient, avg_spectrum, time_derivative, etc
        """

        stats = {}
        if self.lr_stats.source_data is not None:
            logger.info('Computing statistics on low-resolution dataset.')
            lr_stats = self.lr_stats.run()
            stats['low_res'] = lr_stats['source']
            if lr_stats['interp']:
                stats['interp'] = lr_stats['interp']
        if self.synth_stats.source_data is not None:
            logger.info('Computing statistics on synthetic high-resolution '
                        'dataset.')
            synth_stats = self.synth_stats.run()
            stats['synth'] = synth_stats['source']
        if self.coarse_stats.source_data is not None:
            logger.info('Computing statistics on coarsened low-resolution '
                        'dataset.')
            coarse_stats = self.coarse_stats.run()
            stats['coarse'] = coarse_stats['source']
        if self.hr_stats.source_data is not None:
            logger.info('Computing statistics on high-resolution dataset.')
            hr_stats = self.hr_stats.run()
            stats['high_res'] = hr_stats['source']

        if self.qa_fp is not None:
            self.export(self.qa_fp, stats)

        if self.save_fig_data:
            self.export_fig_data()

        logger.info('Finished Sup3rStats run method.')

        return stats
