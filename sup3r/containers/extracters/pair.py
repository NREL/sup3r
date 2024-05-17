"""Paired extracter class for matching separate low_res and high_res
datasets"""

import logging
from warnings import warn

import dask.array as da
import numpy as np
import pandas as pd

from sup3r.containers.base import ContainerPair
from sup3r.containers.extracters import Extracter
from sup3r.utilities.regridder import Regridder
from sup3r.utilities.utilities import nn_fill_array, spatial_coarsening

logger = logging.getLogger(__name__)


class ExtracterPair(ContainerPair):
    """Object containing Extracter objects for low and high-res containers.
    (Usually ERA5 and WTK, respectively). This essentially just regrids the
    low-res data to the coarsened high-res grid.  This is useful for caching
    data which then can go directly to a class:`PairSampler` object for a
    class:`PairBatchQueue`.

    Notes
    -----
    When initializing the lr_container it's important to pick a shape argument
    that will produce a low res domain that completely overlaps with the high
    res domain. When the high res data is not on a regular grid (WTK uses
    lambert) the low res shape is not simply the high res shape divided by
    s_enhance. It is easiest to not provide a shape argument at all for
    lr_container and to get the full domain.
    """

    def __init__(
        self,
        lr_container: Extracter,
        hr_container: Extracter,
        regrid_workers=1,
        regrid_lr=True,
        s_enhance=1,
        t_enhance=1,
        lr_cache_kwargs=None,
        hr_cache_kwargs=None
    ):
        """Initialize data container using hr and lr data containers for h5
        data and nc data

        Parameters
        ----------
        hr_container : Wrangler | Container
            Wrangler for high_res data. Needs to have `.cache_data` method if
            you want to cache the regridded data.
        lr_container : Wrangler | Container
            Wrangler for low_res data. Needs to have `.cache_data` method if
            you want to cache the regridded data.
        regrid_workers : int | None
            Number of workers to use for regridding routine.
        regrid_lr : bool
            Flag to regrid the low-res container data to the high-res container
            grid. This will take care of any minor inconsistencies in different
            projections. Disable this if the grids are known to be the same.
        s_enhance : int
            Spatial enhancement factor
        t_enhance : int
            Temporal enhancement factor
        lr_cache_kwargs : dict
            Cache kwargs for the call to lr_container.cache_data(cache_kwargs).
            Must include 'cache_pattern' key if not None, and can also include
            dictionary of chunk tuples with feature keys
        hr_cache_kwargs : dict
            Cache kwargs for the call to hr_container.cache_data(cache_kwargs).
            Must include 'cache_pattern' key if not None, and can also include
            dictionary of chunk tuples with feature keys
        """
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        self.lr_container = lr_container
        self.hr_container = hr_container
        self.regrid_workers = regrid_workers
        self.lr_time_index = lr_container.time_index
        self.hr_time_index = hr_container.time_index
        self._lr_lat_lon = None
        self._hr_lat_lon = None
        self._lr_input_data = None
        self._regrid_lr = regrid_lr

        self.update_lr_container()
        self.update_hr_container()

        self.lr_container.cache_data(lr_cache_kwargs)
        self.hr_container.cache_data(hr_cache_kwargs)

    def update_hr_container(self):
        """Set the high resolution data attribute and check if
        hr_container.shape is divisible by s_enhance. If not, take the largest
        shape that can be."""
        msg = (
            f'hr_container.shape {self.hr_container.shape[:-1]} is not '
            f'divisible by s_enhance ({self.s_enhance}). Using shape = '
            f'{self.hr_required_shape} instead.'
        )
        if self.hr_container.shape[:-1] != self.hr_required_shape:
            logger.warning(msg)
            warn(msg)

        self.hr_container.data = self.hr_container.data[
            : self.hr_required_shape[0],
            : self.hr_required_shape[1],
            : self.hr_required_shape[2],
        ]
        self.hr_container.lat_lon = self.hr_lat_lon

        self.hr_container.time_index = self.hr_container.time_index[
            : self.hr_required_shape[2]
        ]

    @property
    def lr_input_data(self):
        """Get low res data used as input to regridding routine"""
        if self._lr_input_data is None:
            self._lr_input_data = self.lr_container.data[
                ..., : self.lr_required_shape[2], :
            ]
        return self._lr_input_data

    @property
    def lr_required_shape(self):
        """Return required shape for regridded low_res data"""
        return (
            self.hr_container.shape[0] // self.s_enhance,
            self.hr_container.shape[1] // self.s_enhance,
            self.hr_container.shape[2] // self.t_enhance,
        )

    @property
    def shape(self):
        """Get low_res shape"""
        return (*self.lr_required_shape, len(self.lr_container.features))

    @property
    def hr_required_shape(self):
        """Return required shape for high_res data"""
        return (
            self.s_enhance * self.lr_required_shape[0],
            self.s_enhance * self.lr_required_shape[1],
            self.t_enhance * self.lr_required_shape[2],
        )

    @property
    def lr_grid_shape(self):
        """Return grid shape for regridded low_res data"""
        return (self.lr_required_shape[0], self.lr_required_shape[1])

    @property
    def lr_lat_lon(self):
        """Get low_res lat lon array"""
        if self._lr_lat_lon is None:
            self._lr_lat_lon = spatial_coarsening(
                self.hr_lat_lon, s_enhance=self.s_enhance, obs_axis=False
            )
        return self._lr_lat_lon

    @lr_lat_lon.setter
    def lr_lat_lon(self, lat_lon):
        """Set low_res lat lon array"""
        self._lr_lat_lon = lat_lon

    @property
    def hr_lat_lon(self):
        """Get high_res lat lon array"""
        if self._hr_lat_lon is None:
            self._hr_lat_lon = self.hr_container.lat_lon[
                : self.hr_required_shape[0], : self.hr_required_shape[1]
            ]
        return self._hr_lat_lon

    @hr_lat_lon.setter
    def hr_lat_lon(self, lat_lon):
        """Set high_res lat lon array"""
        self._hr_lat_lon = lat_lon

    def get_regridder(self):
        """Get regridder object"""
        input_meta = pd.DataFrame()
        input_meta['latitude'] = self.lr_container.lat_lon[..., 0].flatten()
        input_meta['longitude'] = self.lr_container.lat_lon[..., 1].flatten()
        target_meta = pd.DataFrame()
        target_meta['latitude'] = self.lr_lat_lon[..., 0].flatten()
        target_meta['longitude'] = self.lr_lat_lon[..., 1].flatten()
        return Regridder(
            input_meta, target_meta, max_workers=self.regrid_workers
        )

    def update_lr_container(self):
        """Regrid low_res data for all requested noncached features. Load
        cached features if available and overwrite=False"""

        if self._regrid_lr:
            logger.info('Regridding low resolution feature data.')
            regridder = self.get_regridder()

            lr_list = []
            for fname in self.lr_container.features:
                fidx = self.lr_container.features.index(fname)
                tmp = regridder(self.lr_input_data[..., fidx])
                lr_list.append(tmp.reshape(self.lr_required_shape)[..., None])

            self.lr_container.data = da.stack(lr_list, axis=-1)
            self.lr_container.lat_lon = self.lr_lat_lon
            self.lr_container.time_index = self.lr_container.time_index[
                : self.lr_required_shape[2]]

        for fidx in range(self.lr_container.data.shape[-1]):
            nan_perc = (
                100
                * np.isnan(self.lr_container.data[..., fidx]).sum()
                / self.lr_container.data[..., fidx].size
            )
            if nan_perc > 0:
                msg = (
                    f'{self.lr_container.features[fidx]} data has '
                    f'{nan_perc:.3f}% NaN values!'
                )
                logger.warning(msg)
                warn(msg)
                msg = (
                    f'Doing nn nan fill on low res '
                    f'{self.lr_container.features[fidx]} data.'
                )
                logger.info(msg)
                self.lr_container.data[..., fidx] = nn_fill_array(
                    self.lr_container.data[..., fidx]
                )
