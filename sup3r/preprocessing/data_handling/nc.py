"""Data handling for netcdf files.
@author: bbenton
"""

import logging
import os

import numpy as np
import pandas as pd
from rex import Resource
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from scipy.stats import mode

from sup3r.containers import LoaderNC, WranglerNC

np.random.seed(42)

logger = logging.getLogger(__name__)


class DataHandlerNC(WranglerNC):
    """DataHandler for NETCDF Data"""

    def __init__(
        self,
        file_paths,
        features,
        res_kwargs=None,
        chunks='auto',
        mode='lazy',
        target=None,
        shape=None,
        time_slice=None,
        transform=None,
        cache_kwargs=None,
    ):
        loader = LoaderNC(
            file_paths,
            features,
            res_kwargs=res_kwargs,
            chunks=chunks,
            mode=mode,
        )
        super().__init__(
            loader,
            features,
            target=target,
            shape=shape,
            time_slice=time_slice,
            transform=transform,
            cache_kwargs=cache_kwargs,
        )


class DataHandlerNCforCC(DataHandlerNC):
    """Data Handler for NETCDF climate change data"""

    def __init__(self,
                 *args,
                 nsrdb_source_fp=None,
                 nsrdb_agg=1,
                 nsrdb_smoothing=0,
                 **kwargs,
                 ):
        """Initialize NETCDF data handler for climate change data.

        Parameters
        ----------
        *args : list
            Same ordered required arguments as DataHandler parent class.
        nsrdb_source_fp : str | None
            Optional NSRDB source h5 file to retrieve clearsky_ghi from to
            calculate CC clearsky_ratio along with rsds (ghi) from the CC
            netcdf file.
        nsrdb_agg : int
            Optional number of NSRDB source pixels to aggregate clearsky_ghi
            from to a single climate change netcdf pixel. This can be used if
            the CC.nc data is at a much coarser resolution than the source
            nsrdb data.
        nsrdb_smoothing : float
            Optional gaussian filter smoothing factor to smooth out
            clearsky_ghi from high-resolution nsrdb source data. This is
            typically done because spatially aggregated nsrdb data is still
            usually rougher than CC irradiance data.
        **kwargs : list
            Same optional keyword arguments as DataHandler parent class.
        """
        self._nsrdb_source_fp = nsrdb_source_fp
        self._nsrdb_agg = nsrdb_agg
        self._nsrdb_smoothing = nsrdb_smoothing
        super().__init__(*args, **kwargs)

    def run_data_extraction(self):
        """Run the raw dataset extraction process from disk to raw
        un-manipulated datasets.

        Includes a special method to extract clearsky_ghi from a exogenous
        NSRDB source h5 file (required to compute clearsky_ratio).
        """
        get_clearsky = False
        if 'clearsky_ghi' in self.features:
            get_clearsky = True
            self._features.remove('clearsky_ghi')

        super().run_data_extraction()

        if get_clearsky:
            cs_ghi = self.get_clearsky_ghi()

            # clearsky ghi is extracted at the proper starting time index so
            # the time chunks should start at 0
            tc0 = self.time_chunks[0].start
            cs_ghi_time_chunks = [
                slice(tc.start - tc0, tc.stop - tc0, tc.step)
                for tc in self.time_chunks
            ]
            for it, tslice in enumerate(cs_ghi_time_chunks):
                self._raw_data[it]['clearsky_ghi'] = cs_ghi[..., tslice]

            self._raw_features.append('clearsky_ghi')

    def get_clearsky_ghi(self):
        """Get clearsky ghi from an exogenous NSRDB source h5 file at the
        target CC meta data and time index.

        Returns
        -------
        cs_ghi : np.ndarray
            Clearsky ghi (W/m2) from the nsrdb_source_fp h5 source file. Data
            shape is (lat, lon, time) where time is daily average values.
        """

        msg = ('Need nsrdb_source_fp input arg as a valid filepath to '
               'retrieve clearsky_ghi (maybe for clearsky_ratio) but '
               'received: {}'.format(self._nsrdb_source_fp))
        assert self._nsrdb_source_fp is not None, msg
        assert os.path.exists(self._nsrdb_source_fp), msg

        msg = ('Can only handle source CC data in hourly frequency but '
               'received daily frequency of {}hrs (should be 24) '
               'with raw time index: {}'.format(self.time_freq_hours,
                                                self.raw_time_index))
        assert self.time_freq_hours == 24.0, msg

        msg = ('Can only handle source CC data with time_slice.step == 1 '
               'but received: {}'.format(self.time_slice.step))
        assert (self.time_slice.step is None) | (self.time_slice.step
                                                     == 1), msg

        with Resource(self._nsrdb_source_fp) as res:
            ti_nsrdb = res.time_index
            meta_nsrdb = res.meta

        ti_deltas = ti_nsrdb - np.roll(ti_nsrdb, 1)
        ti_deltas_hours = pd.Series(ti_deltas).dt.total_seconds()[1:-1] / 3600
        time_freq = float(mode(ti_deltas_hours).mode)
        t_start = np.where((self.time_index[0].month == ti_nsrdb.month)
                           & (self.time_index[0].day == ti_nsrdb.day))[0][0]
        t_end = 1 + np.where(
            (self.time_index[-1].month == ti_nsrdb.month)
            & (self.time_index[-1].day == ti_nsrdb.day))[0][-1]
        t_slice = slice(t_start, t_end)

        # pylint: disable=E1136
        lat = self.lat_lon[:, :, 0].flatten()
        lon = self.lat_lon[:, :, 1].flatten()
        cc_meta = np.vstack((lat, lon)).T

        tree = KDTree(meta_nsrdb[['latitude', 'longitude']])
        _, i = tree.query(cc_meta, k=self._nsrdb_agg)
        if len(i.shape) == 1:
            i = np.expand_dims(i, axis=1)

        logger.info('Extracting clearsky_ghi data from "{}" with time slice '
                    '{} and {} locations with agg factor {}.'.format(
                        os.path.basename(self._nsrdb_source_fp), t_slice,
                        i.shape[0], i.shape[1],
                    ))

        cs_shape = i.shape
        with Resource(self._nsrdb_source_fp) as res:
            cs_ghi = res['clearsky_ghi', t_slice, i.flatten()]

        cs_ghi = cs_ghi.reshape((len(cs_ghi), *cs_shape))
        cs_ghi = cs_ghi.mean(axis=-1)

        windows = np.array_split(np.arange(len(cs_ghi)),
                                 len(cs_ghi) // (24 // time_freq))
        cs_ghi = [cs_ghi[window].mean(axis=0) for window in windows]
        cs_ghi = np.vstack(cs_ghi)
        cs_ghi = cs_ghi.reshape((len(cs_ghi), *tuple(self.grid_shape)))
        cs_ghi = np.transpose(cs_ghi, axes=(1, 2, 0))

        if self.invert_lat:
            cs_ghi = cs_ghi[::-1]

        logger.info('Smoothing nsrdb clearsky ghi with a factor of {}'.format(
            self._nsrdb_smoothing))
        for iday in range(cs_ghi.shape[-1]):
            cs_ghi[..., iday] = gaussian_filter(cs_ghi[..., iday],
                                                self._nsrdb_smoothing,
                                                mode='nearest')

        if cs_ghi.shape[-1] < len(self.time_index):
            n = int(np.ceil(len(self.time_index) / cs_ghi.shape[-1]))
            cs_ghi = np.repeat(cs_ghi, n, axis=2)

        cs_ghi = cs_ghi[..., :len(self.time_index)]

        logger.info(
            'Reshaped clearsky_ghi data to final shape {} to '
            'correspond with CC daily average data over source '
            'time_slice {} with (lat, lon) grid shape of {}'.format(
                cs_ghi.shape, self.time_slice, self.grid_shape))
        msg = ('nsrdb clearsky GHI time dimension {} '
               'does not match the GCM time dimension {}'
               .format(cs_ghi.shape[2], len(self.time_index)))
        assert cs_ghi.shape[2] == len(self.time_index), msg

        return cs_ghi
