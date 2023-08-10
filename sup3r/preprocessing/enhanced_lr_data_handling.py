"""Sup3r preprocessing module.
@author: bbenton
"""

import logging
import os
import pickle
from abc import abstractmethod
from datetime import datetime as dt

import numpy as np
from rex import MultiFileWindX

from sup3r.preprocessing.feature_handling import (
    BVFreqMon,
    BVFreqSquaredH5,
    ClearSkyRatioH5,
    CloudMaskH5,
    Feature,
    LatLonH5,
    Rews,
    TopoH5,
    UWind,
    VWind,
)
from sup3r.utilities.utilities import (
    get_chunk_slices,
    smooth_data,
    spatial_coarsening,
    temporal_coarsening,
    spatial_simple_enhancing,
    temporal_simple_enhancing,
)
from sup3r.preprocessing.data_handling import DataHandler

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
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        self.temporal_coarsening_method = temporal_coarsening_method
        self.temporal_enhancing_method = temporal_enhancing_method
        self.output_features_ind = output_features_ind
        self.training_features = training_features
        self.smoothing = smoothing
        self.smoothing_ignore = smoothing_ignore
        self.t_enhance_mode = t_enhance_mode

        DataHandler.__init__(self, file_paths=file_paths, features=features,
                             target=target, shape=shape, max_delta=max_delta,
                             temporal_slice=temporal_slice,
                             hr_spatial_coarsen=hr_spatial_coarsen,
                             time_roll=time_roll, val_split=val_split,
                             sample_shape=sample_shape,
                             raster_file=raster_file,
                             raster_index=raster_index,
                             shuffle_time=shuffle_time,
                             time_chunk_size=time_chunk_size,
                             cache_pattern=cache_pattern,
                             overwrite_cache=overwrite_cache,
                             overwrite_ti_cache=overwrite_ti_cache,
                             load_cached=load_cached,
                             train_only_features=train_only_features,
                             handle_features=handle_features,
                             single_ts_files=single_ts_files,
                             mask_nan=mask_nan,
                             worker_kwargs=worker_kwargs,
                             res_kwargs=res_kwargs)

    def reduce_features(self, high_res):
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
        if self.output_features_ind is None:
            return high_res
        else:
            return high_res[..., self.output_features_ind]

    def get_lr_enhanced(self, high_res):
        """Enhance low res to match high res shape

        Parameters
        ----------
        high_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        """

        low_res = spatial_coarsening(high_res[np.newaxis, :], self.s_enhance)
        if self.training_features is None:
            self.training_features = [None] * low_res.shape[-1]
        if self.smoothing_ignore is None:
            self.smoothing_ignore = []
        if self.t_enhance != 1:
            low_res = temporal_coarsening(low_res, self.t_enhance,
                                          self.temporal_coarsening_method)

        low_res = smooth_data(low_res, self.training_features,
                              self.smoothing_ignore,
                              self.smoothing)
        high_res = self.reduce_features(high_res)

        enhanced_lr = spatial_simple_enhancing(low_res,
                                               s_enhance=self.s_enhance)
        enhanced_lr = temporal_simple_enhancing(enhanced_lr,
                                                t_enhance=self.t_enhance,
                                                mode=self.t_enhance_mode)
        enhanced_lr = self.reduce_features(enhanced_lr)

        return enhanced_lr[0]

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
        kwargs : dict
            keyword arguments passed to source handler

        Returns
        -------
        ndarray
            Data array for extracted feature
            (spatial_1, spatial_2, temporal)
        """
        logger.info(f'Extracting {feature} with kwargs={kwargs}')
        handle = cls.source_handler(file_paths, **kwargs)
        try:
            fdata = handle[(feature, time_slice,
                            *tuple([raster_index.flatten()]))]
        except ValueError as e:
            msg = f'{feature} cannot be extracted from source data'
            logger.exception(msg)
            raise ValueError(msg) from e

        fdata = fdata.reshape((-1, raster_index.shape[0],
                               raster_index.shape[1]))
        fdata = np.transpose(fdata, (1, 2, 0))
        return fdata.astype(np.float32)

    @classmethod
    @abstractmethod
    def feature_registry(cls):
        """Registry of methods for computing features or extracting renamed
        features

        Returns
        -------
        dict
            Method registry
        """

    @classmethod
    @abstractmethod
    def get_full_domain(cls, file_paths):
        """Get target and shape for full domain"""

    @classmethod
    @abstractmethod
    def source_handler(cls, file_paths, **kwargs):
        """Handle for source data. Uses xarray, ResourceX, etc.
        NOTE: that xarray appears to treat open file handlers as singletons
        within a threadpool, so its okay to open this source_handler without a
        context handler or a .close() statement.
        """

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

    @property
    def extract_features(self):
        """Features to extract directly from the source handler"""
        lower_features = [f.lower() for f in self.handle_features]
        return [f for f in self.raw_features
                if self.lookup(f, 'compute') is None
                or Feature.get_basename(f.lower()) in lower_features]


class ELRDataHandlerH5(ELRDataHandler):
    """ELRDataHandler for H5 Data"""

    # the handler from rex to open h5 data.
    REX_HANDLER = MultiFileWindX

    @classmethod
    def source_handler(cls, file_paths, **kwargs):
        """Rex data handler

        Note that xarray appears to treat open file handlers as singletons
        within a threadpool, so its okay to open this source_handler without a
        context handler or a .close() statement.

        Parameters
        ----------
        file_paths : str | list
            paths to data files
        kwargs : dict
            keyword arguments passed to source handler

        Returns
        -------
        data : ResourceX
        """
        return cls.REX_HANDLER(file_paths, **kwargs)

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
        kwargs : dict
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
        kwargs : dict
            keyword arguments passed to source handler

        Returns
        -------
        ndarray
            Data array for extracted feature
            (spatial_1, spatial_2, temporal)
        """
        logger.info(f'Extracting {feature} with kwargs={kwargs}')
        handle = cls.source_handler(file_paths, **kwargs)
        try:
            fdata = handle[(feature, time_slice,
                            *tuple([raster_index.flatten()]))]
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
                basedir = os.path.dirname(self.raster_file)
                if not os.path.exists(basedir):
                    os.makedirs(basedir)
                logger.debug(f'Saving raster index: {self.raster_file}')
                np.savetxt(self.raster_file, raster_index)
        return raster_index
