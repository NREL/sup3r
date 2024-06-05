"""Paired extracter class for matching separate low_res and high_res
datasets"""

import logging
from typing import Tuple
from warnings import warn

import numpy as np
import pandas as pd
import xarray as xr

from sup3r.preprocessing.base import Container, Sup3rDataset
from sup3r.preprocessing.cachers import Cacher
from sup3r.preprocessing.common import Dimension
from sup3r.utilities.regridder import Regridder
from sup3r.utilities.utilities import nn_fill_array, spatial_coarsening

logger = logging.getLogger(__name__)


class DualExtracter(Container):
    """Object containing xr.Dataset instances for low and high-res data.
    (Usually ERA5 and WTK, respectively). This essentially just regrids the
    low-res data to the coarsened high-res grid.  This is useful for caching
    data which then can go directly to a :class:`DualSampler` object for a
    :class:`DualBatchQueue`.

    Note
    ----
    When first extracting the low_res data make sure to extract a region that
    completely overlaps the high_res region.  It is easiest to load the full
    low_res domain and let :class:`DualExtracter` select the appropriate region
    through regridding.
    """

    def __init__(
        self,
        data: Sup3rDataset | Tuple[xr.Dataset, xr.Dataset],
        regrid_workers=1,
        regrid_lr=True,
        s_enhance=1,
        t_enhance=1,
        lr_cache_kwargs=None,
        hr_cache_kwargs=None,
    ):
        """Initialize data container lr and hr :class:`Data` instances.
        Typically lr = ERA5 data and hr = WTK data.

        Parameters
        ----------
        data : Sup3rDataset | Tuple[xr.Dataset, xr.Dataset]
            A tuple of xr.Dataset instances. The first must be low-res
            and the second must be high-res data
        regrid_workers : int | None
            Number of workers to use for regridding routine.
        regrid_lr : bool
            Flag to regrid the low-res data to the high-res grid. This will
            take care of any minor inconsistencies in different projections.
            Disable this if the grids are known to be the same.
        s_enhance : int
            Spatial enhancement factor
        t_enhance : int
            Temporal enhancement factor
        lr_cache_kwargs : dict
            Cache kwargs for the call to lr_data.cache_data(cache_kwargs).
            Must include 'cache_pattern' key if not None, and can also include
            dictionary of chunk tuples with feature keys
        hr_cache_kwargs : dict
            Cache kwargs for the call to hr_data.cache_data(cache_kwargs).
            Must include 'cache_pattern' key if not None, and can also include
            dictionary of chunk tuples with feature keys
        """
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        msg = (
            'The DualExtracter requires a data tuple with two members, low '
            'and high resolution in that order. Received inconsistent data '
            'argument.'
        )
        data = data if isinstance(data, Sup3rDataset) else Sup3rDataset(data)
        assert isinstance(data, tuple) and len(data) == 2, msg
        self.lr_data, self.hr_data = data.low_res, data.high_res
        self.regrid_workers = regrid_workers
        self.lr_time_index = self.lr_data.indexes['time']
        self.hr_time_index = self.hr_data.indexes['time']
        self.lr_required_shape = (
            self.hr_data.shape[0] // self.s_enhance,
            self.hr_data.shape[1] // self.s_enhance,
            self.hr_data.shape[2] // self.t_enhance,
        )
        self.hr_required_shape = (
            self.s_enhance * self.lr_required_shape[0],
            self.s_enhance * self.lr_required_shape[1],
            self.t_enhance * self.lr_required_shape[2],
        )

        msg = (
            f'The required low-res shape {self.lr_required_shape} is '
            'inconsistent with the shape of the raw data '
            f'{self.lr_data.shape}'
        )
        assert all(
            req_s <= true_s
            for req_s, true_s in zip(
                self.lr_required_shape, self.lr_data.shape
            )
        ), msg

        self.hr_lat_lon = self.hr_data.lat_lon[
            *map(slice, self.hr_required_shape[:2])
        ]
        self.lr_lat_lon = spatial_coarsening(
            self.hr_lat_lon, s_enhance=self.s_enhance, obs_axis=False
        )
        self._regrid_lr = regrid_lr

        self.update_lr_data()
        self.update_hr_data()

        self.check_regridded_lr_data()

        if lr_cache_kwargs is not None:
            Cacher(self.lr_data, lr_cache_kwargs)

        if hr_cache_kwargs is not None:
            Cacher(self.hr_data, hr_cache_kwargs)

        super().__init__(data=(self.lr_data, self.hr_data))

    def update_hr_data(self):
        """Set the high resolution data attribute and check if
        hr_data.shape is divisible by s_enhance. If not, take the largest
        shape that can be."""
        msg = (
            f'hr_data.shape {self.hr_data.shape[:3]} is not '
            f'divisible by s_enhance ({self.s_enhance}). Using shape = '
            f'{self.hr_required_shape} instead.'
        )
        if self.hr_data.shape[:3] != self.hr_required_shape[:3]:
            logger.warning(msg)
            warn(msg)

        hr_data_new = {
            f: self.hr_data[f, *map(slice, self.hr_required_shape)]
            for f in self.hr_data.data_vars
        }
        hr_coords_new = {
            Dimension.LATITUDE: self.hr_lat_lon[..., 0],
            Dimension.LONGITUDE: self.hr_lat_lon[..., 1],
            Dimension.TIME: self.hr_data.indexes['time'][
                : self.hr_required_shape[2]
            ],
        }
        self.hr_data = self.hr_data.update({**hr_coords_new, **hr_data_new})

    def get_regridder(self):
        """Get regridder object"""
        input_meta = pd.DataFrame.from_dict(
            {
                Dimension.LATITUDE: self.lr_data.lat_lon[..., 0].flatten(),
                Dimension.LONGITUDE: self.lr_data.lat_lon[..., 1].flatten(),
            }
        )
        target_meta = pd.DataFrame.from_dict(
            {
                Dimension.LATITUDE: self.lr_lat_lon[..., 0].flatten(),
                Dimension.LONGITUDE: self.lr_lat_lon[..., 1].flatten(),
            }
        )
        return Regridder(
            input_meta, target_meta, max_workers=self.regrid_workers
        )

    def update_lr_data(self):
        """Regrid low_res data for all requested noncached features. Load
        cached features if available and overwrite=False"""

        if self._regrid_lr:
            logger.info('Regridding low resolution feature data.')
            regridder = self.get_regridder()

            lr_data_new = {
                f: regridder(
                    self.lr_data[f, ..., :self.lr_required_shape[2]]
                ).reshape(self.lr_required_shape)
                for f in self.lr_data.data_vars
            }
            lr_coords_new = {
                Dimension.LATITUDE: self.lr_lat_lon[..., 0],
                Dimension.LONGITUDE: self.lr_lat_lon[..., 1],
                Dimension.TIME: self.lr_data.indexes['time'][
                    : self.lr_required_shape[2]
                ],
            }
            self.lr_data = self.lr_data.update(
                {**lr_coords_new, **lr_data_new}
            )

    def check_regridded_lr_data(self):
        """Check for NaNs after regridding and do NN fill if needed."""
        for f in self.lr_data.data_vars:
            nan_perc = (
                100 * np.isnan(self.lr_data[f]).sum() / self.lr_data[f].size
            )
            if nan_perc > 0:
                msg = f'{f} data has {nan_perc:.3f}% NaN values!'
                logger.warning(msg)
                warn(msg)
                msg = f'Doing nn nan fill on low res {f} data.'
                logger.info(msg)
                self.lr_data[f] = nn_fill_array(self.lr_data[f])
