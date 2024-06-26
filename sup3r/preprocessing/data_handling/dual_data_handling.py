"""Dual data handler class for using separate low_res and high_res datasets"""
import copy
import logging
import pickle
from warnings import warn

import numpy as np
import pandas as pd

from sup3r.preprocessing.data_handling.mixin import (
    CacheHandlingMixIn,
    TrainingPrepMixIn,
)
from sup3r.utilities.regridder import Regridder
from sup3r.utilities.utilities import nn_fill_array, spatial_coarsening

logger = logging.getLogger(__name__)


# pylint: disable=unsubscriptable-object
class DualDataHandler(CacheHandlingMixIn, TrainingPrepMixIn):
    """Batch handling class for h5 data as high res (usually WTK) and netcdf
    data as low res (usually ERA5)

    NOTE: When initializing the lr_handler it's important to pick a shape
    argument that will produce a low res domain that completely overlaps with
    the high res domain. When the high res data is not on a regular grid
    (WTK uses lambert) the low res shape is not simply the high res shape
    divided by s_enhance. It is easiest to not provide a shape argument at all
    for lr_handler and to get the full domain.
    """

    def __init__(self,
                 hr_handler,
                 lr_handler,
                 cache_pattern=None,
                 overwrite_cache=False,
                 regrid_workers=1,
                 load_cached=True,
                 shuffle_time=False,
                 regrid_lr=True,
                 s_enhance=1,
                 t_enhance=1,
                 val_split=0.0):
        """Initialize data handler using hr and lr data handlers for h5 data
        and nc data

        Parameters
        ----------
        hr_handler : DataHandler
            DataHandler for high_res data
        lr_handler : DataHandler
            DataHandler for low_res data
        cache_pattern : str
            Pattern for files to use for saving regridded ERA data.
        overwrite_cache : bool
            Whether to overwrite regrid cache
        regrid_workers : int | None
            Number of workers to use for regridding routine.
        load_cached : bool
            Whether to load cache to memory or wait until load_cached()
            is called.
        shuffle_time : bool
            Whether to shuffle time indices prior to training/validation split
        regrid_lr : bool
            Flag to regrid the low-res handler data to the high-res handler
            grid. This will take care of any minor inconsistencies in different
            projections. Disable this if the grids are known to be the same.
        s_enhance : int
            Spatial enhancement factor
        t_enhance : int
            Temporal enhancement factor
        val_split : float
            Percentage of data to reserve for validation.
        """
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        self.lr_dh = lr_handler
        self.hr_dh = hr_handler
        self.overwrite_cache = overwrite_cache
        self.val_split = val_split
        self.current_obs_index = None
        self.load_cached = load_cached
        self.regrid_workers = regrid_workers
        self.shuffle_time = shuffle_time
        self.hr_data = None
        self.lr_val_data = None
        self.hr_val_data = None
        self.lr_data = np.zeros(self.shape, dtype=np.float32)
        self.lr_time_index = lr_handler.time_index
        self.hr_time_index = hr_handler.time_index
        self.lr_val_time_index = lr_handler.val_time_index
        self.hr_val_time_index = hr_handler.val_time_index
        self._lr_lat_lon = None
        self._hr_lat_lon = None
        self._lr_input_data = None
        self._cache_pattern = cache_pattern
        self._cached_features = None
        self._noncached_features = None
        self._means = None
        self._stds = None
        self._is_normalized = False
        self._regrid_lr = regrid_lr
        self._norm_workers = self.lr_dh.norm_workers

        if self.try_load and self.load_cached:
            self.load_cached_data()

        if not self.try_load:
            self.get_data()

        self._run_pair_checks(hr_handler, lr_handler)

        self.check_clear_data()

        logger.info('Finished initializing DualDataHandler.')

    def get_data(self):
        """Check hr and lr shapes and trim hr data if needed to match required
        relationship to lr shape based on enhancement factors. Then regrid lr
        data and split hr and lr data into training and validation sets."""
        self._set_hr_data()
        self.get_lr_data()
        self._val_split_check()

    def _val_split_check(self):
        """Check if val_split > 0 and split data into validation and training.
        Make sure validation data is larger than sample_shape

        Note that if val split > 0.0, hr_data will no longer be a view of
        self.hr_dh.data and this could lead to lots of memory usage.
        """

        if self.hr_data is not None and self.val_split > 0.0:
            n_val_obs = self.hr_data.shape[2] * (1 - self.val_split)
            n_val_obs = int(self.t_enhance * (n_val_obs // self.t_enhance))
            train_indices, val_indices = self._split_data_indices(
                self.hr_data,
                n_val_obs=n_val_obs,
                shuffle_time=self.shuffle_time)
            self.hr_val_data = self.hr_data[:, :, val_indices, :]
            self.hr_data = self.hr_data[:, :, train_indices, :]
            self.hr_val_time_index = self.hr_time_index[val_indices]
            self.hr_time_index = self.hr_time_index[train_indices]
            msg = ('High res validation data has shape='
                   f'{self.hr_val_data.shape} and sample_shape='
                   f'{self.hr_sample_shape}. Use a smaller sample_shape '
                   'and/or larger val_split.')
            check = any(val_size < samp_size for val_size, samp_size in zip(
                self.hr_val_data.shape, self.hr_sample_shape))
            if check:
                logger.warning(msg)
                warn(msg)

            if self.lr_data is not None and self.val_split > 0.0:
                train_indices = list(set(train_indices // self.t_enhance))
                val_indices = list(set(val_indices // self.t_enhance))

                self.lr_val_data = self.lr_data[:, :, val_indices, :]
                self.lr_data = self.lr_data[:, :, train_indices, :]

                self.lr_val_time_index = self.lr_time_index[val_indices]
                self.lr_time_index = self.lr_time_index[train_indices]

                msg = ('Low res validation data has shape='
                       f'{self.lr_val_data.shape} and sample_shape='
                       f'{self.lr_sample_shape}. Use a smaller sample_shape '
                       'and/or larger val_split.')
                check = any(val_size < samp_size
                            for val_size, samp_size in zip(
                                self.lr_val_data.shape, self.lr_sample_shape))
                if check:
                    logger.warning(msg)
                    warn(msg)

    def _get_stats(self):
        """Get mean/stdev stats for HR and LR data handlers"""
        super()._get_stats(features=self.lr_dh.features)
        self.hr_dh._get_stats()

    @property
    def means(self):
        """Get the mean values for each feature. Mean values from the low-res
        data handler are prioritized because these are typically the "input"
        features

        Returns
        -------
        dict
        """

        if self.hr_dh.data is None:
            msg = ('High-res DataHandler object has DataHandler.data=None! '
                   'Try initializing the high-res handler with '
                   'load_cached=True')
            logger.error(msg)
            raise RuntimeError(msg)

        out = copy.deepcopy(self.hr_dh.means)
        out.update(super().means)
        return out

    @property
    def stds(self):
        """Get the standard deviation values for each feature. Mean values from
        the low-res data handler are prioritized because these are typically
        the "input" features

        Returns
        -------
        dict
        """

        if self.hr_dh.data is None:
            msg = ('High-res DataHandler object has DataHandler.data=None! '
                   'Try initializing the high-res handler with '
                   'load_cached=True')
            logger.error(msg)
            raise RuntimeError(msg)

        out = copy.deepcopy(self.hr_dh.stds)
        out.update(super().stds)
        return out

    # pylint: disable=unused-argument
    def normalize(self, means=None, stds=None, max_workers=None):
        """Normalize low_res and high_res data

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
        max_workers : None | int
            Has no effect. Used to match MixIn class signature.
        """

        if self.hr_dh.data is None:
            msg = ('High-res DataHandler object has DataHandler.data=None! '
                   'Try initializing the high-res handler with '
                   'load_cached=True')
            logger.error(msg)
            raise RuntimeError(msg)

        if means is None:
            means = self.means
        if stds is None:
            stds = self.stds

        self._normalize_lr(means, stds)
        self._normalize_hr(means, stds)

    def _normalize_lr(self, means, stds):
        """Normalize the low-resolution data features including in the
        low-res data handler

        Note that self.lr_data is usually a unique regridded array but if
        regridding was not performed then it is just a sliced *view* of
        self.lr_dh.data and the super().normalize() operation will have applied
        to that data already.

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
        """

        logger.info('Normalizing low resolution data features='
                    f'{self.lr_dh.features}')
        super().normalize(means=means, stds=stds,
                          features=self.lr_dh.features,
                          max_workers=self.lr_dh.norm_workers)

        if id(self.lr_dh.data) != id(self.lr_data.base):
            self.lr_dh.normalize(means=means, stds=stds,
                                 features=self.lr_dh.features,
                                 max_workers=self.lr_dh.norm_workers)
        else:
            self.lr_dh._is_normalized = True

    def _normalize_hr(self, means, stds):
        """Normalize the high-resolution data features including in the
        high-res data handler

        Note that self.hr_data is usually just a sliced *view* of
        self.hr_dh.data but if the *view* is broken then it will have to be
        normalized too

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
        """

        logger.info('Normalizing high resolution data features='
                    f'{self.hr_dh.features}')
        self.hr_dh.normalize(means=means, stds=stds,
                             features=self.hr_dh.features,
                             max_workers=self.hr_dh.norm_workers)

        if id(self.hr_data.base) != id(self.hr_dh.data):
            mean_arr = np.array([means[fn] for fn in self.hr_dh.features])
            std_arr = np.array([stds[fn] for fn in self.hr_dh.features])
            self.hr_data = (self.hr_data - mean_arr) / std_arr
            self.hr_data = self.hr_data.astype(np.float32)

    @property
    def features(self):
        """Get a list of data features including features from both the lr and
        hr data handlers"""
        out = list(copy.deepcopy(self.lr_dh.features))
        out += [fn for fn in self.hr_dh.features if fn not in out]
        return out

    @property
    def lr_only_features(self):
        """Features to use for training only and not output"""
        tof = [fn for fn in self.lr_dh.features
               if fn not in self.hr_out_features
               and fn not in self.hr_exo_features]
        return tof

    @property
    def lr_features(self):
        """Get a list of low-resolution features. All low-resolution features
        are used for training."""
        return self.lr_dh.lr_features

    @property
    def hr_exo_features(self):
        """Get a list of high-resolution features that are only used for
        training e.g., mid-network high-res topo injection. These must come at
        the end of the high-res feature set."""
        return self.hr_dh.hr_exo_features

    @property
    def hr_out_features(self):
        """Get a list of high-resolution features that are intended to be
        output by the GAN. Does not include high-resolution exogenous features
        """
        return self.hr_dh.hr_out_features

    def _set_hr_data(self):
        """Set the high resolution data attribute and check if hr_handler.shape
        is divisible by s_enhance. If not, take the largest shape that can
        be."""

        if self.hr_data is None:
            logger.info("Loading high resolution cache.")
            self.hr_dh.load_cached_data(with_split=False)

        msg = (f'hr_handler.shape {self.hr_dh.shape[:-1]} is not divisible '
               f'by s_enhance ({self.s_enhance}). Using shape = '
               f'{self.hr_required_shape} instead.')
        if self.hr_dh.shape[:-1] != self.hr_required_shape:
            logger.warning(msg)
            warn(msg)

        # Note that operations like normalization on self.hr_dh.data will also
        # happen to self.hr_data because hr_data is just a sliced view not a
        # copy. This is to save memory with big data volume
        self.hr_data = self.hr_dh.data[:self.hr_required_shape[0],
                                       :self.hr_required_shape[1],
                                       :self.hr_required_shape[2]]
        self.hr_time_index = self.hr_dh.time_index[:self.hr_required_shape[2]]
        self.lr_time_index = self.lr_dh.time_index[:self.lr_required_shape[2]]

        assert np.array_equal(self.hr_time_index[::self.t_enhance].values,
                              self.lr_time_index.values)

    def _run_pair_checks(self, hr_handler, lr_handler):
        """Run sanity checks on high_res and low_res pairs. The handler data
        shapes are restricted by enhancement factors."""
        msg = ('Validation split is done by DualDataHandler. '
               'hr_handler.val_split and lr_handler.val_split should both be '
               'zero.')
        assert hr_handler.val_split == 0 and lr_handler.val_split == 0, msg
        hr_shape = hr_handler.sample_shape
        lr_shape = [hr_shape[0] // self.s_enhance,
                    hr_shape[1] // self.s_enhance,
                    hr_shape[2] // self.t_enhance]
        msg = (f'hr_handler.sample_shape {hr_handler.sample_shape} and '
               f'lr_handler.sample_shape {lr_handler.sample_shape} are '
               f'incompatible. Must be {hr_shape} and {lr_shape}.')
        assert list(lr_handler.sample_shape) == lr_shape, msg

        if hr_handler.data is not None and lr_handler.data is not None:
            hr_shape = self.hr_data.shape[:-1]
            lr_shape = [hr_shape[0] // self.s_enhance,
                        hr_shape[1] // self.s_enhance,
                        hr_shape[2] // self.t_enhance]
            msg = (f'hr_data.shape {self.hr_data.shape} and '
                   f'lr_data.shape {self.lr_data.shape} are '
                   f'incompatible. Must be {hr_shape} and {lr_shape}.')
            assert list(self.lr_data.shape[:-1]) == lr_shape, msg

        if self.lr_val_data is not None and self.hr_val_data is not None:
            hr_shape = self.hr_val_data.shape
            lr_shape = [hr_shape[0] // self.s_enhance,
                        hr_shape[1] // self.s_enhance,
                        hr_shape[2] // self.t_enhance]
            msg = (f'hr_val_data.shape {self.hr_val_data.shape} '
                   f'and lr_val_data.shape {self.lr_val_data.shape}'
                   f' are incompatible. Must be {hr_shape} and {lr_shape}.')
            assert list(self.lr_val_data.shape[:-1]) == lr_shape, msg

        if self.val_split == 0.0:
            assert id(self.hr_data.base) == id(hr_handler.data)

    @property
    def grid_mem(self):
        """Get memory used by a feature at a single time step

        Returns
        -------
        int
            Number of bytes for a single feature array at a single time step
        """
        grid_mem = np.prod(self.lr_grid_shape)
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
        feature_mem = self.grid_mem * self.lr_data.shape[-2]
        return feature_mem

    @property
    def sample_shape(self):
        """Get lr sample shape"""
        return self.lr_dh.sample_shape

    @property
    def lr_sample_shape(self):
        """Get lr sample shape"""
        return self.lr_dh.sample_shape

    @property
    def hr_sample_shape(self):
        """Get hr sample shape"""
        return self.hr_dh.sample_shape

    @property
    def data(self):
        """Get low res data. Same as self.lr_data but used to match property
        used for computing means and stdevs"""
        return self.lr_data

    @property
    def val_data(self):
        """Get low res validation data. Same as self.lr_val_data but used to
        match property used by normalization routine."""
        return self.lr_val_data

    @property
    def lr_input_data(self):
        """Get low res data used as input to regridding routine"""
        if self._lr_input_data is None:
            if self.lr_dh.data is None:
                self.lr_dh.load_cached_data()
            self._lr_input_data = self.lr_dh.data[
                ..., :self.lr_required_shape[2], :]
        return self._lr_input_data

    @property
    def lr_required_shape(self):
        """Return required shape for regridded low_res data"""
        return (self.hr_dh.requested_shape[0] // self.s_enhance,
                self.hr_dh.requested_shape[1] // self.s_enhance,
                self.hr_dh.requested_shape[2] // self.t_enhance)

    @property
    def shape(self):
        """Get low_res shape"""
        return (*self.lr_required_shape, len(self.lr_dh.features))

    @property
    def size(self):
        """Get low_res size"""
        return np.prod(self.shape)

    @property
    def hr_required_shape(self):
        """Return required shape for high_res data"""
        return (self.s_enhance * self.lr_required_shape[0],
                self.s_enhance * self.lr_required_shape[1],
                self.t_enhance * self.lr_required_shape[2])

    @property
    def lr_grid_shape(self):
        """Return grid shape for regridded low_res data"""
        return (self.lr_required_shape[0], self.lr_required_shape[1])

    @property
    def lr_lat_lon(self):
        """Get low_res lat lon array"""
        if self._lr_lat_lon is None:
            self._lr_lat_lon = spatial_coarsening(self.hr_lat_lon,
                                                  s_enhance=self.s_enhance,
                                                  obs_axis=False)
        return self._lr_lat_lon

    @lr_lat_lon.setter
    def lr_lat_lon(self, lat_lon):
        """Set low_res lat lon array"""
        self._lr_lat_lon = lat_lon

    @property
    def hr_lat_lon(self):
        """Get high_res lat lon array"""
        if self._hr_lat_lon is None:
            self._hr_lat_lon = self.hr_dh.lat_lon[:self.hr_required_shape[0], :
                                                  self.hr_required_shape[1]]
        return self._hr_lat_lon

    @hr_lat_lon.setter
    def hr_lat_lon(self, lat_lon):
        """Set high_res lat lon array"""
        self._hr_lat_lon = lat_lon

    @property
    def cache_files(self):
        """Get file names of regridded cache data"""
        cache_files = self._get_cache_file_names(self.cache_pattern,
                                                 grid_shape=self.lr_grid_shape,
                                                 time_index=self.lr_time_index,
                                                 target=self.lr_dh.target,
                                                 features=self.lr_dh.features)
        return cache_files

    @property
    def noncached_features(self):
        """Get list of features needing extraction or derivation"""
        if self._noncached_features is None:
            self._noncached_features = self.check_cached_features(
                self.lr_dh.features,
                cache_files=self.cache_files,
                overwrite_cache=self.overwrite_cache,
                load_cached=self.load_cached,
            )
        return self._noncached_features

    @property
    def try_load(self):
        """Check if we should try to load cached data"""
        try_load = self._should_load_cache(self.cache_pattern,
                                           self.cache_files,
                                           self.overwrite_cache)
        return try_load

    def load_lr_cached_data(self):
        """Load low_res cache data"""

        logger.info(
            f'Loading cache with requested_shape={self.shape}.')
        self._load_cached_data(self.lr_data,
                               self.cache_files,
                               self.lr_dh.features,
                               max_workers=self.hr_dh.load_workers)

    def load_cached_data(self):
        """Load regridded low_res and high_res cache data"""
        self.load_lr_cached_data()
        self._set_hr_data()
        self._val_split_check()

    def check_clear_data(self):
        """Check if data was cached and free memory if load_cached is False"""
        if self.cache_pattern is not None and not self.load_cached:
            self.lr_data = None
            self.lr_val_data = None
            self.hr_dh.check_clear_data()

    def get_lr_data(self):
        """Check if era data is cached. If not then extract data and regrid.
        Save to cache if cache pattern provided."""

        if self.try_load:
            self.load_lr_cached_data()
        else:
            self.get_lr_regridded_data()

            if self.cache_pattern is not None:
                logger.info('Caching low resolution data with '
                            f'shape={self.lr_data.shape}.')
                self._cache_data(self.lr_data,
                                 features=self.lr_dh.features,
                                 cache_file_paths=self.cache_files,
                                 overwrite=self.overwrite_cache)

    def get_regridder(self):
        """Get regridder object"""
        input_meta = pd.DataFrame()
        input_meta['latitude'] = self.lr_dh.lat_lon[..., 0].flatten()
        input_meta['longitude'] = self.lr_dh.lat_lon[..., 1].flatten()
        target_meta = pd.DataFrame()
        target_meta['latitude'] = self.lr_lat_lon[..., 0].flatten()
        target_meta['longitude'] = self.lr_lat_lon[..., 1].flatten()
        return Regridder(input_meta,
                         target_meta,
                         max_workers=self.regrid_workers)

    def get_lr_regridded_data(self):
        """Regrid low_res data for all requested noncached features. Load
        cached features if available and overwrite=False"""

        if self._regrid_lr:
            logger.info('Regridding low resolution feature data.')
            regridder = self.get_regridder()

            fnames = set(self.noncached_features)
            fnames = fnames.intersection(set(self.lr_dh.features))
            for fname in fnames:
                fidx = self.lr_dh.features.index(fname)
                tmp = regridder(self.lr_input_data[..., fidx])
                tmp = tmp.reshape(self.lr_required_shape)
                self.lr_data[..., fidx] = tmp
        else:
            self.lr_data = self.lr_input_data

        if self.load_cached:
            fnames = set(self.cached_features)
            fnames = fnames.intersection(set(self.lr_dh.features))
            for fname in fnames:
                fidx = self.lr_dh.features.index(fname)
                logger.info(f'Loading {fname} from {self.cache_files[fidx]}')
                with open(self.cache_files[fidx], 'rb') as fh:
                    self.lr_data[..., fidx] = pickle.load(fh)

        for fidx in range(self.lr_data.shape[-1]):
            nan_perc = (100 * np.isnan(self.lr_data[..., fidx]).sum()
                        / self.lr_data[..., fidx].size)
            if nan_perc > 0:
                msg = (f'{self.lr_dh.features[fidx]} data has '
                       f'{nan_perc:.3f}% NaN values!')
                logger.warning(msg)
                warn(msg)
                msg = (f'Doing nn nan fill on low res '
                       f'{self.lr_dh.features[fidx]} data.')
                logger.info(msg)
                self.lr_data[..., fidx] = nn_fill_array(
                    self.lr_data[..., fidx])

    def get_next(self):
        """Get next high_res + low_res. Gets random spatiotemporal sample for
        h5 data and then uses enhancement factors to subsample
        interpolated/regridded low_res data for same spatiotemporal extent.

        Returns
        -------
        hr_data : ndarray
            Array of high resolution data with each feature equal in shape to
            hr_sample_shape
        lr_data : ndarray
            Array of low resolution data with each feature equal in shape to
            lr_sample_shape
        """
        lr_obs_idx = self._get_observation_index(self.lr_data,
                                                 self.lr_sample_shape)
        hr_obs_idx = [slice(s.start * self.s_enhance, s.stop * self.s_enhance)
                      for s in lr_obs_idx[:2]]
        hr_obs_idx += [slice(s.start * self.t_enhance, s.stop * self.t_enhance)
                       for s in lr_obs_idx[2:-1]]

        hr_obs_idx.append(np.arange(len(self.hr_dh.features)))
        hr_obs_idx = tuple(hr_obs_idx)
        self.current_obs_index = {
            'hr_index': hr_obs_idx,
            'lr_index': lr_obs_idx
        }
        return self.hr_data[hr_obs_idx], self.lr_data[lr_obs_idx]
