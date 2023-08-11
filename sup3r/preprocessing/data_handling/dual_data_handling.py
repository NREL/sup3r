"""Dual data handler class for using separate low_res and high_res datasets"""
import logging
from warnings import warn

import numpy as np
import xesmf as xe

from sup3r.utilities.utilities import spatial_coarsening

logger = logging.getLogger(__name__)


# pylint: disable=unsubscriptable-object
class DualDataHandler:
    """Batch handling class for h5 data as high res (usually WTK) and netcdf
    data as low res (usually ERA5)"""

    def __init__(
        self,
        hr_handler,
        lr_handler,
        regrid_cache_pattern=None,
        overwrite_regrid_cache=False,
        load_cached=True,
        s_enhance=15,
        t_enhance=1,
        val_split=0.0,
    ):
        """Initialize data handler using hr and lr data handlers for h5 data
        and nc data

        Parameters
        ----------
        hr_handler : DataHandler
            DataHandler for high_res data
        lr_handler : DataHandler
            DataHandler for low_res data
        regrid_cache_pattern : str
            Pattern for files to use for saving regridded ERA data.
        overwrite_regrid_cache : bool
            Whether to overwrite regrid cache
        load_cached : bool
            Whether to load cache to memory or wait until load_cached()
            is called.
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
        self.regrid_cache_pattern = regrid_cache_pattern
        self.overwrite_regrid_cache = overwrite_regrid_cache
        self.regridder = None
        self._lr_lat_lon = None
        self.val_split = val_split
        self.current_obs_index = None
        self.load_cached = load_cached

        self._run_handler_checks(hr_handler, lr_handler)

        if self.try_load and self.load_cached:
            self.load_cached_data()

        if not self.try_load:
            self.get_data()

        self.check_clear_data()

    def get_data(self):
        """Check hr and lr shapes and trim hr data if needed to match required
        relationship to lr shape based on enhancement factors. Then regrid lr
        data and split hr and lr data into training and validation sets."""
        self._shape_check()
        self.get_lr_data()
        self._val_split_check()

    def _val_split_check(self):
        """Check if val_split > 0 and split data into validation and training.
        Make sure validation data is larger than sample_shape"""

        if self.lr_data is not None and self.val_split > 0.0:
            self.lr_data, self.lr_val_data = self.lr_dh.split_data(
                data=self.lr_data, val_split=self.val_split, shuffle_time=False
            )
            msg = (
                f'Low res validation data has shape={self.lr_val_data.shape} '
                f'and sample_shape={self.lr_sample_shape}. Use a smaller '
                'sample_shape and/or larger val_split.'
            )
            check = any(
                val_size < samp_size
                for val_size, samp_size in zip(
                    self.lr_val_data.shape, self.lr_sample_shape
                )
            )
            if check:
                logger.warning(msg)
                warn(msg)

        if self.hr_data is not None and self.val_split > 0.0:
            self.hr_data, self.hr_val_data = self.hr_dh.split_data(
                data=self.hr_data, val_split=self.val_split, shuffle_time=False
            )
            msg = (
                f'High res validation data has shape={self.hr_val_data.shape} '
                f'and sample_shape={self.hr_sample_shape}. Use a smaller '
                'sample_shape and/or larger val_split.'
            )
            check = any(
                val_size < samp_size
                for val_size, samp_size in zip(
                    self.hr_val_data.shape, self.hr_sample_shape
                )
            )
            if check:
                logger.warning(msg)
                warn(msg)

    def normalize(self, means, stdevs):
        """Normalize low_res data

        Parameters
        ----------
        means : np.ndarray
            dimensions (features)
            array of means for all features with same ordering as data features
        stdevs : np.ndarray
            dimensions (features)
            array of means for all features with same ordering as data features
        """
        self.lr_dh.normalize(means, stdevs)

    @property
    def output_features(self):
        """Get list of output features. e.g. those that are returned by a
        GAN"""
        return self.hr_dh.output_features

    def _shape_check(self):
        """Check if hr_handler.shape is divisible by s_enhance. If not take
        the largest shape that can be."""

        if self.hr_data is None:
            self.hr_dh.load_cached_data()

        old_shape = self.hr_dh.shape[:-1]
        new_shape = (
            self.s_enhance * (old_shape[0] // self.s_enhance),
            self.s_enhance * (old_shape[1] // self.s_enhance),
            self.t_enhance * (old_shape[2] // self.t_enhance),
        )
        msg = (
            'hr_handler.shape is not divisible by s_enhance. Using '
            f'shape = {new_shape} instead.'
        )
        if old_shape != new_shape:
            logger.warning(msg)
            warn(msg)

        self.hr_data = self.hr_data[
            : new_shape[0], : new_shape[1], : new_shape[2]
        ]
        self.hr_dh.lat_lon = self.hr_dh.lat_lon[: new_shape[0], : new_shape[1]]
        self.hr_dh.time_index = self.hr_dh.time_index[: new_shape[2]]
        self.lr_dh.time_index = self.lr_dh.time_index[
            : new_shape[2] // self.t_enhance
        ]

    def _run_handler_checks(self, hr_handler, lr_handler):
        """Run sanity checks on high_res and low_res handlers. The handler data
        shapes are restricted by enhancement factors."""
        msg = (
            'Validation split is done by DualDataHandler. '
            'hr_handler.val_split and lr_handler.val_split should both be '
            'zero.'
        )
        assert hr_handler.val_split == 0 and lr_handler.val_split == 0, msg
        msg = (
            'Handlers have incompatible number of features. '
            f'({hr_handler.features} vs {lr_handler.features})'
        )
        assert hr_handler.features == lr_handler.features, msg
        hr_shape = hr_handler.sample_shape
        lr_shape = (
            hr_shape[0] // self.s_enhance,
            hr_shape[1] // self.s_enhance,
            hr_shape[2] // self.t_enhance,
        )
        msg = (
            f'hr_handler.sample_shape {hr_handler.sample_shape} and '
            f'lr_handler.sample_shape {lr_handler.sample_shape} are '
            f'incompatible. Must be {hr_shape} and {lr_shape}.'
        )
        assert lr_handler.sample_shape == lr_shape, msg

        if hr_handler.data is not None:
            hr_shape = hr_handler.data.shape
            lr_shape = (
                hr_shape[0] // self.s_enhance,
                hr_shape[1] // self.s_enhance,
                hr_shape[2] // self.t_enhance,
                hr_shape[3],
            )
            msg = (
                f'hr_handler.data.shape {hr_handler.data.shape} and '
                f'lr_handler.data.shape {lr_handler.data.shape} are '
                f'incompatible. Must be {hr_shape} and {lr_shape}.'
            )
            assert lr_handler.data.shape == lr_shape, msg

        if hr_handler.val_data is not None:
            hr_shape = hr_handler.val_data.shape
            lr_shape = (
                hr_shape[0] // self.s_enhance,
                hr_shape[1] // self.s_enhance,
                hr_shape[2] // self.t_enhance,
                hr_shape[3],
            )
            msg = (
                f'hr_handler.val_data.shape {hr_handler.val_data.shape} '
                f'and lr_handler.val_data.shape {lr_handler.val_data.shape}'
                f' are incompatible. Must be {hr_shape} and {lr_shape}.'
            )
            assert lr_handler.val_data.shape == lr_shape, msg

    @property
    def grid_mem(self):
        """Get memory used by a feature at a single time step

        Returns
        -------
        int
            Number of bytes for a single feature array at a single time step
        """
        grid_mem = np.product(self.lr_grid_shape)
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
        feature_mem = self.grid_mem * len(self.lr_time_index)
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
    def features(self):
        """Get list of features in each data handler"""
        return self.hr_dh.features

    @property
    def lr_time_index(self):
        """Return time index for low_res data"""
        return self.lr_dh.time_index

    @lr_time_index.setter
    def lr_time_index(self, time_index):
        """Set time index for low_res data"""
        self.lr_dh.time_index = time_index

    @property
    def hr_time_index(self):
        """Return time index for high_res data"""
        return self.hr_dh.time_index

    @hr_time_index.setter
    def hr_time_index(self, time_index):
        """Set time index for high_res data"""
        self.hr_dh.time_index = time_index

    @property
    def lr_data(self):
        """Get low_res data"""
        return self.lr_dh.data

    @lr_data.setter
    def lr_data(self, lr_data):
        """Set low_res data"""
        self.lr_dh.data = lr_data

    @property
    def data(self):
        """Get low_res data"""
        return self.lr_data

    @data.setter
    def data(self, data):
        """Set low_res data"""
        self.lr_data = data

    @property
    def shape(self):
        """Get low_res shape"""
        return self.lr_dh.shape

    @property
    def hr_data(self):
        """Get high_res data"""
        return self.hr_dh.data

    @hr_data.setter
    def hr_data(self, hr_data):
        """Set high_res data"""
        self.hr_dh.data = hr_data

    @property
    def lr_val_data(self):
        """Get low_res validation data"""
        return self.lr_dh.val_data

    @lr_val_data.setter
    def lr_val_data(self, lr_val_data):
        """Set low_res validation data"""
        self.lr_dh.val_data = lr_val_data

    @property
    def hr_val_data(self):
        """Get high_res validation data"""
        return self.hr_dh.val_data

    @hr_val_data.setter
    def hr_val_data(self, hr_val_data):
        """Set high_res validation data"""
        self.hr_dh.val_data = hr_val_data

    @property
    def lr_grid_shape(self):
        """Return grid shape for regridded low_res data"""
        return (
            self.hr_dh.requested_shape[0] // self.s_enhance,
            self.hr_dh.requested_shape[1] // self.s_enhance,
        )

    @property
    def lr_requested_shape(self):
        """Return requested shape for low_res data"""
        return (
            self.hr_dh.requested_shape[0] // self.s_enhance,
            self.hr_dh.requested_shape[1] // self.s_enhance,
            self.hr_dh.requested_shape[2] // self.t_enhance,
        )

    @property
    def lr_lat_lon(self):
        """Get low_res lat lon array"""
        if self._lr_lat_lon is None:
            self._lr_lat_lon = spatial_coarsening(
                self.hr_dh.lat_lon, s_enhance=self.s_enhance, obs_axis=False
            )
            self.lr_dh.lat_lon = self._lr_lat_lon
        return self._lr_lat_lon

    @property
    def regrid_cache_files(self):
        """Get file names of regridded cache data"""
        cache_files = self.lr_dh.get_cache_file_names(
            self.regrid_cache_pattern,
            grid_shape=self.lr_grid_shape,
            time_index=self.lr_time_index,
            target=self.hr_dh.target,
            features=self.hr_dh.features,
        )
        return cache_files

    @property
    def try_load(self):
        """Check if we should try to load cached data"""
        try_load = self.lr_dh._should_load_cache(
            self.regrid_cache_pattern,
            self.regrid_cache_files,
            self.overwrite_regrid_cache,
        )
        return try_load

    def load_lr_cached_data(self):
        """Load low_res cache data"""

        regridded_data = np.full(
            shape=self.lr_requested_shape, fill_value=np.nan, dtype=np.float32
        )

        self.lr_dh._load_cached_data(
            regridded_data,
            self.regrid_cache_files,
            self.hr_dh.features,
            self.lr_requested_shape,
            max_workers=self.hr_dh.load_workers,
        )

        self.lr_data = regridded_data

    def load_cached_data(self):
        """Load regridded low_res and high_res cache data"""
        self.load_lr_cached_data()
        self._shape_check()
        self._val_split_check()

    def check_clear_data(self):
        """Check if data was cached and free memory if load_cached is False"""
        if self.regrid_cache_pattern is not None:
            self.lr_data = None if not self.load_cached else self.lr_data
            self.lr_val_data = (
                None if self.lr_data is None else self.lr_val_data
            )
        self.hr_dh.check_clear_data()

    def get_lr_data(self):
        """Check if era data is cached. If not then extract data and regrid.
        Save to cache if cache pattern provided."""

        if self.try_load:
            self.load_lr_cached_data()
        else:
            regridded_data = self.regrid_lr_data()

            if self.regrid_cache_pattern is not None:
                self.lr_dh._cache_data(
                    regridded_data,
                    features=self.hr_dh.features,
                    cache_file_paths=self.regrid_cache_files,
                    overwrite_cache=self.overwrite_regrid_cache,
                )
        self.lr_data = regridded_data

    def regrid_feature(self, fidx):
        """Regrid low_res feature data to high_res data grid

        Parameters
        ----------
        fidx : int
            Feature index

        Returns
        -------
        out : ndarray
            Array of regridded low_res data
            (spatial_1, spatial_2, temporal)
        """
        out = np.concatenate(
            [
                self.regridder(self.lr_data[..., i, fidx])[..., np.newaxis]
                for i in range(len(self.lr_time_index))
            ],
            axis=-1,
        )
        return out

    def regrid_lr_data(self):
        """Regrid low_res data for all requested features

        Returns
        -------
        out : ndarray
            Array of regridded low_res data with all features
            (spatial_1, spatial_2, temporal, n_features)
        """
        old_grid = {
            'lat': self.lr_dh.lat_lon[..., 0],
            'lon': self.lr_dh.lat_lon[..., 1],
        }

        new_grid = {
            'lat': self.lr_lat_lon[..., 0],
            'lon': self.lr_lat_lon[..., 1],
        }

        self.regridder = xe.Regridder(old_grid, new_grid, method='bilinear')

        logger.info('Regridding low resolution feature data.')
        return np.concatenate(
            [
                self.regrid_feature(i)[..., np.newaxis]
                for i in range(len(self.features))
            ],
            axis=-1,
        )

    def get_next(self):
        """Get next high_res + low_res. Gets random spatiotemporal sample for
        h5 data and then uses enhancement factors to subsample
        interpolated/regridded low_res data for same spatiotemporal extent.

        Returns
        -------
        hr_data : ndarray
            Array of high resolution data with each feature equal in shape to
            sample_shape
        lr_data : ndarray
            Array of low resolution data with each feature equal in shape to
            sample_shape // (s_enhance or t_enhance)
        """
        hr_obs_idx = self.hr_dh.get_observation_index()
        lr_obs_idx = []
        for s in hr_obs_idx[:2]:
            lr_obs_idx.append(
                slice(s.start // self.s_enhance, s.stop // self.s_enhance)
            )
        for s in hr_obs_idx[2:-1]:
            lr_obs_idx.append(
                slice(s.start // self.t_enhance, s.stop // self.t_enhance)
            )
        lr_obs_idx.append(hr_obs_idx[-1])
        self.current_obs_index = [hr_obs_idx, lr_obs_idx]
        return self.hr_data[hr_obs_idx], self.lr_data[lr_obs_idx]
