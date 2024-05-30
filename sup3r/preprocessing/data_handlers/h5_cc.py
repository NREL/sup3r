"""Data handling for H5 files.
@author: bbenton
"""

import copy
import logging

import dask.array as da
import numpy as np
from rex import MultiFileNSRDBX

from sup3r.preprocessing.data_handlers.factory import (
    DataHandlerFactory,
)
from sup3r.preprocessing.derivers.methods import (
    RegistryH5SolarCC,
    RegistryH5WindCC,
)
from sup3r.preprocessing.extracters import BaseExtracterH5
from sup3r.preprocessing.loaders import LoaderH5
from sup3r.utilities.utilities import (
    daily_temporal_coarsening,
)

logger = logging.getLogger(__name__)


BaseH5WindCC = DataHandlerFactory(
    BaseExtracterH5, LoaderH5, FeatureRegistry=RegistryH5WindCC
)


def _base_loader(file_paths, **kwargs):
    return MultiFileNSRDBX(file_paths, **kwargs)


BaseH5SolarCC = DataHandlerFactory(
    BaseExtracterH5,
    LoaderH5,
    BaseLoader=_base_loader,
    FeatureRegistry=RegistryH5SolarCC,
)


class DataHandlerH5WindCC(BaseH5WindCC):
    """Special data handling and batch sampling for h5 wtk or nsrdb data for
    climate change applications"""

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args : list
            Same positional args as DataHandlerH5
        **kwargs : dict
            Same keyword args as DataHandlerH5
        """
        super().__init__(*args, **kwargs)

        self.daily_data = None
        self.daily_data_slices = None
        self.run_daily_averages()

    def run_daily_averages(self):
        """Calculate daily average data and store as attribute."""
        msg = (
            'Data needs to be hourly with at least 24 hours, but data '
            'shape is {}.'.format(self.data.shape)
        )
        assert self.data.shape[2] % 24 == 0, msg
        assert self.data.shape[2] > 24, msg

        n_data_days = int(self.data.shape[2] / 24)

        logger.info(
            'Calculating daily average datasets for {} training '
            'data days.'.format(n_data_days)
        )

        self.daily_data_slices = np.array_split(
            np.arange(self.data.shape[2]), n_data_days
        )
        self.daily_data_slices = [
            slice(x[0], x[-1] + 1) for x in self.daily_data_slices
        ]
        feature_arr_list = []
        for idf, fname in enumerate(self.features):
            daily_arr_list = []
            for t_slice in self.daily_data_slices:
                if '_max_' in fname:
                    tmp = np.max(self.data[:, :, t_slice, idf], axis=2)
                elif '_min_' in fname:
                    tmp = np.min(self.data[:, :, t_slice, idf], axis=2)
                else:
                    tmp = daily_temporal_coarsening(
                        self.data[:, :, t_slice, idf], temporal_axis=2
                    )[..., 0]
                daily_arr_list.append(tmp)
            feature_arr_list.append(da.stack(daily_arr_list), axis=-1)
        self.daily_data = da.stack(feature_arr_list, axis=-1)

        logger.info(
            'Finished calculating daily average datasets for {} '
            'training data days.'.format(n_data_days)
        )


class DataHandlerH5SolarCC(BaseH5WindCC):
    """Special data handling and batch sampling for h5 NSRDB solar data for
    climate change applications"""

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
            msg = (
                'Cannot initialize DataHandlerH5SolarCC without required '
                'features {}. All three are necessary to get the daily '
                'average clearsky ratio (ghi sum / clearsky ghi sum), '
                'even though only the clearsky ratio will be passed to the '
                'GAN.'.format(required)
            )
            logger.error(msg)
            raise KeyError(msg)

        super().__init__(*args, **kwargs)

    def run_daily_averages(self):
        """Calculate daily average data and store as attribute.

        Note that the H5 clearsky ratio feature requires special logic to match
        the climate change dataset of daily average GHI / daily average CS_GHI.
        This target climate change dataset is not equivalent to the average of
        instantaneous hourly clearsky ratios

        TODO: can probably remove the feature pop at the end of this. Also,
        maybe some combination of Wind / Solar handlers would work. Some
        overlapping logic.
        """

        msg = (
            'Data needs to be hourly with at least 24 hours, but data '
            'shape is {}.'.format(self.data.shape)
        )
        assert self.data.shape[2] % 24 == 0, msg
        assert self.data.shape[2] > 24, msg

        n_data_days = int(self.data.shape[2] / 24)

        logger.info(
            'Calculating daily average datasets for {} training '
            'data days.'.format(n_data_days)
        )

        self.daily_data_slices = np.array_split(
            np.arange(self.data.shape[2]), n_data_days
        )
        self.daily_data_slices = [
            slice(x[0], x[-1] + 1) for x in self.daily_data_slices
        ]

        i_ghi = self.features.index('ghi')
        i_cs = self.features.index('clearsky_ghi')
        i_ratio = self.features.index('clearsky_ratio')

        feature_arr_list = []
        for idf in range(self.data.shape[-1]):
            daily_arr_list = []
            for t_slice in self.daily_data_slices:

                daily_arr_list.append(daily_temporal_coarsening(
                    self.data[:, :, t_slice, idf], temporal_axis=2
                )[:, :, 0])
            feature_arr_list.append(da.stack(daily_arr_list, axis=-1))

        avg_cs_ratio_list = []
        for t_slice in self.daily_data_slices:
            # note that this ratio of daily irradiance sums is not the same as
            # the average of hourly ratios.
            total_ghi = np.nansum(self.data[:, :, t_slice, i_ghi], axis=2)
            total_cs_ghi = np.nansum(self.data[:, :, t_slice, i_cs], axis=2)
            avg_cs_ratio = total_ghi / total_cs_ghi
            avg_cs_ratio_list.append(avg_cs_ratio)
        avg_cs_ratio = da.stack(avg_cs_ratio_list, axis=-1)
        feature_arr_list.insert(i_ratio, avg_cs_ratio)

        self.daily_data = da.stack(feature_arr_list, axis=-1)

        # remove ghi and clearsky ghi from feature set. These shouldn't be used
        # downstream for solar cc and keeping them confuses the batch handler
        logger.info(
            'Finished calculating daily average clearsky_ratio, '
            'removing ghi and clearsky_ghi from the '
            'DataHandlerH5SolarCC feature list.'
        )
        ifeats = np.array(
            [i for i in range(len(self.features)) if i not in (i_ghi, i_cs)]
        )
        self.data = self.data[..., ifeats]
        self.daily_data = self.daily_data[..., ifeats]
        self.features.remove('ghi')
        self.features.remove('clearsky_ghi')

        logger.info(
            'Finished calculating daily average datasets for {} '
            'training data days.'.format(n_data_days)
        )
