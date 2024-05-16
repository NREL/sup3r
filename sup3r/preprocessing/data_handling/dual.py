"""Dual data handler class for using separate low_res and high_res datasets"""
import logging
import pickle
from warnings import warn

import numpy as np
import pandas as pd

from sup3r.utilities.regridder import Regridder
from sup3r.utilities.utilities import nn_fill_array, spatial_coarsening

logger = logging.getLogger(__name__)


# pylint: disable=unsubscriptable-object
class DualDataHandler:
    """Batch handling class for h5 data as high res (usually WTK) and netcdf
    data as low res (usually ERA5)

    Notes
    -----
    When initializing the lr_handler it's important to pick a shape argument
    that will produce a low res domain that completely overlaps with the high
    res domain. When the high res data is not on a regular grid (WTK uses
    lambert) the low res shape is not simply the high res shape divided by
    s_enhance. It is easiest to not provide a shape argument at all for
    lr_handler and to get the full domain.
    """

    def __init__(self,
                 hr_handler,
                 lr_handler,
                 regrid_workers=1,
                 regrid_lr=True,
                 s_enhance=1,
                 t_enhance=1):
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
        self.regrid_workers = regrid_workers
        self.hr_data = None
        self.lr_data = np.zeros(self.shape, dtype=np.float32)
        self.lr_time_index = lr_handler.time_index
        self.hr_time_index = hr_handler.time_index
        self._lr_lat_lon = None
        self._hr_lat_lon = None
        self._lr_input_data = None
        self._regrid_lr = regrid_lr
        self.get_data()

        logger.info('Finished initializing DualDataHandler.')

    def get_data(self):
        """Check hr and lr shapes and trim hr data if needed to match required
        relationship to lr shape based on enhancement factors. Then regrid lr
        data and split hr and lr data into training and validation sets."""
        self._set_hr_data()
        self.get_lr_data()

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

    @property
    def data(self):
        """Get low res data. Same as self.lr_data but used to match property
        used for computing means and stdevs"""
        return self.lr_data

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

    def to_netcdf(self, lr_file, hr_file):
        """Write lr_data and hr_data to netcdf files."""
        self.lr_dh.to_netcdf(lr_file, data=self.lr_data,
                             lat_lon=self.lr_lat_lon,
                             features=self.lr_dh.features)
        self.hr_dh.to_netcdf(hr_file, data=self.hr_data,
                             lat_lon=self.hr_lat_lon,
                             features=self.hr_dh.features)

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
