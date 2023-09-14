"""Data handling for H5 files.
@author: bbenton
"""

import copy
import logging
import os
from typing import ClassVar

import numpy as np
from rex import MultiFileNSRDBX, MultiFileWindX

from sup3r.preprocessing.data_handling.base import DataHandler, DataHandlerDC
from sup3r.preprocessing.feature_handling import (
    BVFreqMon,
    BVFreqSquaredH5,
    ClearSkyRatioH5,
    CloudMaskH5,
    LatLonH5,
    Rews,
    TopoH5,
    UWind,
    VWind,
)
from sup3r.utilities.utilities import (
    daily_temporal_coarsening,
    uniform_box_sampler,
)

np.random.seed(42)

logger = logging.getLogger(__name__)


class DataHandlerH5(DataHandler):
    """DataHandler for H5 Data"""

    FEATURE_REGISTRY: ClassVar[dict] = {
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
        'clearsky_ratio': ClearSkyRatioH5,
    }

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
    def extract_feature(cls,
                        file_paths,
                        raster_index,
                        feature,
                        time_slice=slice(None),
                        **kwargs,
                        ):
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

        fdata = fdata.reshape(
            (-1, raster_index.shape[0], raster_index.shape[1]))
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
            check = self.grid_shape is not None and self.target is not None
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


class DataHandlerH5WindCC(DataHandlerH5):
    """Special data handling and batch sampling for h5 wtk or nsrdb data for
    climate change applications"""

    FEATURE_REGISTRY = DataHandlerH5.FEATURE_REGISTRY.copy()
    FEATURE_REGISTRY.update({
        'temperature_max_(.*)m': 'temperature_(.*)m',
        'temperature_min_(.*)m': 'temperature_(.*)m',
        'relativehumidity_max_(.*)m': 'relativehumidity_(.*)m',
        'relativehumidity_min_(.*)m': 'relativehumidity_(.*)m'
    })

    # the handler from rex to open h5 data.
    REX_HANDLER = MultiFileWindX

    # list of features / feature name patterns that are input to the generative
    # model but are not part of the synthetic output and are not sent to the
    # discriminator. These are case-insensitive and follow the Unix shell-style
    # wildcard format.
    TRAIN_ONLY_FEATURES = ('temperature_max_*m', 'temperature_min_*m',
                           'relativehumidity_max_*m',
                           'relativehumidity_min_*m',
                           )

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
            logger.info(
                'Found 2D sample shape of {}. Adding spatial dim of 24'.format(
                    sample_shape))
            sample_shape = (*sample_shape, 24)
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
        daily_data_shape = (*self.data.shape[0:2], n_data_days,
                            self.data.shape[3])

        logger.info('Calculating daily average datasets for {} training '
                    'data days.'.format(n_data_days))

        self.daily_data = np.zeros(daily_data_shape, dtype=np.float32)

        self.daily_data_slices = np.array_split(np.arange(self.data.shape[2]),
                                                n_data_days)
        self.daily_data_slices = [
            slice(x[0], x[-1] + 1) for x in self.daily_data_slices
        ]
        for idf, fname in enumerate(self.features):
            for d, t_slice in enumerate(self.daily_data_slices):
                if '_max_' in fname:
                    tmp = np.max(self.data[:, :, t_slice, idf], axis=2)
                    self.daily_data[:, :, d, idf] = tmp[:, :]
                elif '_min_' in fname:
                    tmp = np.min(self.data[:, :, t_slice, idf], axis=2)
                    self.daily_data[:, :, d, idf] = tmp[:, :]
                else:
                    tmp = daily_temporal_coarsening(self.data[:, :, t_slice,
                                                              idf],
                                                    temporal_axis=2)
                    self.daily_data[:, :, d, idf] = tmp[:, :, 0]

        logger.info('Finished calculating daily average datasets for {} '
                    'training data days.'.format(n_data_days))

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
        super()._normalize_data(data, val_data, feature_index, mean, std)
        self.daily_data[..., feature_index] -= mean
        self.daily_data[..., feature_index] /= std

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

        obs_ind_hourly = tuple(
            [*spatial_slice, t_slice_hourly,
             np.arange(len(self.features))])

        obs_ind_daily = tuple(
            [*spatial_slice, t_slice_daily,
             np.arange(len(self.features))])

        return obs_ind_hourly, obs_ind_daily

    def get_next(self):
        """Get data for observation using random observation index. Loops
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

    def split_data(self, data=None, val_split=0.0, shuffle_time=False):
        """Split time dimension into set of training indices and validation
        indices. For NSRDB it makes sure that the splits happen at midnight.

        Parameters
        ----------
        data : np.ndarray
            4D array of high res data
            (spatial_1, spatial_2, temporal, features)
        val_split : float
            Fraction of data to separate for validation.
        shuffle_time : bool
            No effect. Used to fit base class function signature.

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

        n_val_obs = int(np.ceil(val_split * len(midnight_ilocs)))
        val_split_index = midnight_ilocs[n_val_obs]

        self.val_data = self.data[:, :, slice(None, val_split_index), :]
        self.data = self.data[:, :, slice(val_split_index, None), :]

        self.val_time_index = self.time_index[slice(None, val_split_index)]
        self.time_index = self.time_index[slice(val_split_index, None)]

        return self.data, self.val_data


class DataHandlerH5SolarCC(DataHandlerH5WindCC):
    """Special data handling and batch sampling for h5 NSRDB solar data for
    climate change applications"""

    FEATURE_REGISTRY = DataHandlerH5WindCC.FEATURE_REGISTRY.copy()
    FEATURE_REGISTRY.update({
        'windspeed': 'wind_speed',
        'winddirection': 'wind_direction',
        'U': UWind,
        'V': VWind,
    })

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
                   'average clearsky ratio (ghi sum / clearsky ghi sum), '
                   'even though only the clearsky ratio will be passed to the '
                   'GAN.'.format(required))
            logger.error(msg)
            raise KeyError(msg)

        super().__init__(*args, **kwargs)

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
        daily_data_shape = (*self.data.shape[0:2], n_data_days,
                            self.data.shape[3])

        logger.info('Calculating daily average datasets for {} training '
                    'data days.'.format(n_data_days))

        self.daily_data = np.zeros(daily_data_shape, dtype=np.float32)

        self.daily_data_slices = np.array_split(np.arange(self.data.shape[2]),
                                                n_data_days)
        self.daily_data_slices = [
            slice(x[0], x[-1] + 1) for x in self.daily_data_slices
        ]

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
        ifeats = np.array(
            [i for i in range(len(self.features)) if i not in (i_ghi, i_cs)])
        self.data = self.data[..., ifeats]
        self.daily_data = self.daily_data[..., ifeats]
        self.features.remove('ghi')
        self.features.remove('clearsky_ghi')

        logger.info('Finished calculating daily average datasets for {} '
                    'training data days.'.format(n_data_days))


class DataHandlerDCforH5(DataHandlerH5, DataHandlerDC):
    """Data centric data handler for H5 files"""
