"""Sup3r preprocessing module.
@author: bbenton
"""

import copy
import glob
import logging
import os
import pickle
import warnings
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime as dt
from fnmatch import fnmatch

import numpy as np
import pandas as pd
import xarray as xr
from rex import MultiFileNSRDBX, MultiFileWindX, Resource
from rex.utilities import log_mem
from rex.utilities.fun_utils import get_fun_call_str
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import KDTree
from scipy.stats import mode

from sup3r.preprocessing.feature_handling import (
    BVFreqMon,
    BVFreqSquaredH5,
    BVFreqSquaredNC,
    ClearSkyRatioCC,
    ClearSkyRatioH5,
    CloudMaskH5,
    Feature,
    FeatureHandler,
    InverseMonNC,
    LatLonH5,
    LatLonNC,
    PotentialTempNC,
    PressureNC,
    Rews,
    Shear,
    Tas,
    TasMax,
    TasMin,
    TempNC,
    TempNCforCC,
    TopoH5,
    UWind,
    VWind,
    WinddirectionNC,
    WindspeedNC,
)
from sup3r.utilities import ModuleName
from sup3r.utilities.cli import BaseCLI
from sup3r.utilities.interpolation import Interpolator
from sup3r.utilities.utilities import (
    daily_temporal_coarsening,
    estimate_max_workers,
    get_chunk_slices,
    get_raster_shape,
    get_source_type,
    get_time_dim_name,
    ignore_case_path_fetch,
    np_to_pd_times,
    spatial_coarsening,
    uniform_box_sampler,
    uniform_time_sampler,
    weighted_box_sampler,
    weighted_time_sampler,
    smooth_data,
    spatial_coarsening,
    temporal_coarsening,
    spatial_simple_enhancing,
    temporal_simple_enhancing,
)
from sup3r.preprocessing.data_handling import InputMixIn, DataHandler

np.random.seed(42)

logger = logging.getLogger(__name__)


class ELRDataHandler(DataHandler):
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
                 worker_kwargs=None, res_kwargs=None,
                 s_enhance=1, t_enhance=1,
                 temporal_coarsening_method='subsample',
                 temporal_enhancing_method='constant',
                 output_features_ind=None,
                 output_features=None,
                 training_features=None,
                 smoothing=None,
                 smoothing_ignore=None,
                 t_enhance_mode='constant'):
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
                      'concat_dim': 'Time',
                      'combine': 'nested',
                      'chunks': {'south_north': 120, 'west_east': 120}}
            which then gets passed to xr.open_mfdataset(file, **res_kwargs)
        """
        super().__init__(file_paths, features, target=target, shape=shape,
                 max_delta=max_delta, temporal_slice=temporal_slice,
                 hr_spatial_coarsen=hr_spatial_coarsen, time_roll=time_roll, val_split=val_split,
                 sample_shape=sample_shape, raster_file=raster_file, raster_index=raster_index,
                 shuffle_time=shuffle_time, time_chunk_size=time_chunk_size, cache_pattern=cache_pattern,
                 overwrite_cache=overwrite_cache, overwrite_ti_cache=overwrite_ti_cache,
                 load_cached=load_cached, train_only_features=train_only_features,
                 handle_features=handle_features, single_ts_files=single_ts_files, mask_nan=mask_nan,
                 worker_kwargs=worker_kwargs, res_kwargs=res_kwargs)

        self.s_enhance=s_enhance
        self.t_enhance=t_enhance
        self.temporal_coarsening_method=temporal_coarsening_method
        self.temporal_enhancing_method=temporal_enhancing_method
        self.output_features_ind=output_features_ind
        self.output_features=output_features
        selgf.training_features=training_features
        self.smoothing=smoothing
        self.smoothing_ignore=smoothing_ignore
        self.t_enhance_mode=t_enhance_mode

    

    @staticmethod
    def reduce_features(high_res, output_features_ind=None):
        """Remove any feature channels that are only intended for the low-res
        training input.

        Parameters
        ----------
        high_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        output_features_ind : list | np.ndarray | None
            List/array of feature channel indices that are used for generative
            output, without any feature indices used only for training.
        """
        if output_features_ind is None:
            return high_res
        else:
            return high_res[..., output_features_ind] 

    def get_lr_enhanced(self, high_res):

        low_res = spatial_coarsening(high_res, self.s_enhance)
        if self.training_features is None:
            self.training_features = [None] * low_res.shape[-1]
        if self.smoothing_ignore is None:
            self.smoothing_ignore = []
        if self.t_enhance != 1:
            low_res = temporal_coarsening(low_res, self.t_enhance,
                                          self.temporal_coarsening_method)
        low_res = smooth_data(low_res, self.training_features, self.smoothing_ignore,
                              self.smoothing)
        high_res = reduce_features(high_res, self.output_features_ind)

        enhanced_lr = spatial_simple_enhancing(low_res,
                                               s_enhance=self.s_enhance)
        enhanced_lr = temporal_simple_enhancing(enhanced_lr,
                                                t_enhance=self.t_enhance,
                                                mode=self.t_enhance_mode)
        enhanced_lr = reduce_features(enhanced_lr, self.output_features_ind)

        return enhanced_lr

    def run_all_data_init(self):
        """Build base 4D data array. Can handle multiple files but assumes
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
            
        self.data = self.get_lr_enhanced(self.data)

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


    def get_next_determ(self, obs_index):
        """Get data for observation using random observation index. Loops
        repeatedly over randomized time index

        Returns
        -------
        observation : np.ndarray
            4D array
            (spatial_1, spatial_2, temporal, features)
        """
        observation = self.data[obs_index]
        return observation
