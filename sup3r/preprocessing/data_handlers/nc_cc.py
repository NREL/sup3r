"""Data handling for netcdf files.
@author: bbenton
"""

import logging
import os

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.stats import mode

from sup3r.preprocessing.data_handlers.factory import (
    DataHandlerFactory,
)
from sup3r.preprocessing.derivers.methods import (
    RegistryNCforCC,
    RegistryNCforCCwithPowerLaw,
)
from sup3r.preprocessing.extracters import (
    BaseExtracterNC,
)
from sup3r.preprocessing.loaders import LoaderH5, LoaderNC

logger = logging.getLogger(__name__)


BaseNCforCC = DataHandlerFactory(
    BaseExtracterNC,
    LoaderNC,
    FeatureRegistry=RegistryNCforCC,
    name='BaseNCforCC',
)

logger = logging.getLogger(__name__)


class DataHandlerNCforCC(BaseNCforCC):
    """Extended NETCDF data handler. This implements an extracter hook to add
    "clearsky_ghi" to the extracted data if "clearsky_ghi" is requested."""

    def __init__(
        self,
        file_paths,
        features='all',
        nsrdb_source_fp=None,
        nsrdb_agg=1,
        nsrdb_smoothing=0,
        **kwargs,
    ):
        """Initialize NETCDF extracter for climate change data.

        Parameters
        ----------
        file_paths : str | list | pathlib.Path
            file_paths input to :class:`Extracter`
        features : list
            Features to derive from loaded data.
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
            Same optional keyword arguments as parent class.
        """
        self._nsrdb_source_fp = nsrdb_source_fp
        self._nsrdb_agg = nsrdb_agg
        self._nsrdb_smoothing = nsrdb_smoothing
        self._features = features
        super().__init__(file_paths, features=features, **kwargs)

    def _extracter_hook(self):
        """Extracter hook implementation to add 'clearsky_ghi' data to
        extracted data, which will then be used when the :class:`Deriver` is
        called."""
        if any(
            f in self._features
            for f in ('clearsky_ratio', 'clearsky_ghi', 'all')
        ):
            self.extracter.data['clearsky_ghi'] = self.get_clearsky_ghi()

    def run_input_checks(self):
        """Run checks on the files provided for extracting clearksky_ghi."""

        msg = (
            'Need nsrdb_source_fp input arg as a valid filepath to '
            'retrieve clearsky_ghi (maybe for clearsky_ratio) but '
            'received: {}'.format(self._nsrdb_source_fp)
        )
        assert os.path.exists(self._nsrdb_source_fp), msg

        ti_deltas = self.loader.time_index - np.roll(self.loader.time_index, 1)
        ti_deltas_hours = pd.Series(ti_deltas).dt.total_seconds()[1:-1] / 3600
        time_freq_hours = float(mode(ti_deltas_hours).mode)

        msg = (
            'Can only handle source CC data in hourly frequency but '
            'received daily frequency of {}hrs (should be 24) '
            'with raw time index: {}'.format(
                time_freq_hours, self.loader.time_index
            )
        )
        assert time_freq_hours == 24.0, msg

        msg = (
            'Can only handle source CC data with time_slice.step == 1 '
            'but received: {}'.format(self.extracter.time_slice.step)
        )
        assert (self.self.extracter.time_slice.step is None) | (
            self.extracter.time_slice.step == 1
        ), msg

    def run_wrap_checks(self, cs_ghi):
        """Run check on extracted data from clearsky_ghi source."""
        logger.info(
            'Reshaped clearsky_ghi data to final shape {} to '
            'correspond with CC daily average data over source '
            'time_slice {} with (lat, lon) grid shape of {}'.format(
                cs_ghi.shape,
                self.extracter.time_slice,
                self.extracter.grid_shape,
            )
        )
        msg = (
            'nsrdb clearsky GHI time dimension {} '
            'does not match the GCM time dimension {}'.format(
                cs_ghi.shape[2], len(self.extracter.time_index)
            )
        )
        assert cs_ghi.shape[2] == len(self.extracter.time_index), msg

    def get_time_slice(self, ti_nsrdb):
        """Get nsrdb data time slice consistent with self.time_index."""
        t_start = np.where(
            (self.extracter.time_index[0].month == ti_nsrdb.month)
            & (self.extracter.time_index[0].day == ti_nsrdb.day)
        )[0][0]
        t_end = (
            1
            + np.where(
                (self.extracter.time_index[-1].month == ti_nsrdb.month)
                & (self.extracter.time_index[-1].day == ti_nsrdb.day)
            )[0][-1]
        )
        t_slice = slice(t_start, t_end)
        return t_slice

    def get_clearsky_ghi(self):
        """Get clearsky ghi from an exogenous NSRDB source h5 file at the
        target CC meta data and time index.

        TODO: Replace some of this with call to Regridder? Perform daily
        means with self.loader.coarsen?

        Returns
        -------
        cs_ghi : T_Array
            Clearsky ghi (W/m2) from the nsrdb_source_fp h5 source file. Data
            shape is (lat, lon, time) where time is daily average values.
        """
        self.run_input_checks()

        res = LoaderH5(self._nsrdb_source_fp)
        ti_nsrdb = res.time_index
        t_slice = self.get_time_slice(ti_nsrdb)
        cc_meta = self.lat_lon.reshape((-1, 2))

        tree = KDTree(res.lat_lon)
        _, i = tree.query(cc_meta, k=self._nsrdb_agg)
        i = np.expand_dims(i, axis=1) if len(i.shape) == 1 else i

        logger.info(
            'Extracting clearsky_ghi data from "{}" with time slice '
            '{} and {} locations with agg factor {}.'.format(
                os.path.basename(self._nsrdb_source_fp),
                t_slice,
                i.shape[0],
                i.shape[1],
            )
        )

        cs_shape = i.shape
        cs_ghi = res['clearsky_ghi'][i.flatten(), t_slice].T

        cs_ghi = cs_ghi.reshape((len(cs_ghi), *cs_shape))
        cs_ghi = cs_ghi.mean(axis=-1)

        ti_deltas = ti_nsrdb - np.roll(ti_nsrdb, 1)
        ti_deltas_hours = pd.Series(ti_deltas).dt.total_seconds()[1:-1] / 3600
        time_freq = float(mode(ti_deltas_hours).mode)

        windows = np.array_split(
            np.arange(len(cs_ghi)), len(cs_ghi) // (24 // time_freq)
        )
        cs_ghi = [cs_ghi[window].mean(axis=0) for window in windows]
        cs_ghi = np.vstack(cs_ghi)
        cs_ghi = cs_ghi.reshape(
            (len(cs_ghi), *tuple(self.extracter.grid_shape))
        )
        cs_ghi = np.transpose(cs_ghi, axes=(1, 2, 0))

        if cs_ghi.shape[-1] < len(self.extracter.time_index):
            n = int(np.ceil(len(self.extracter.time_index) / cs_ghi.shape[-1]))
            cs_ghi = np.repeat(cs_ghi, n, axis=2)

        cs_ghi = cs_ghi[..., : len(self.extracter.time_index)]

        self.run_wrap_checks(cs_ghi)

        return cs_ghi


class DataHandlerNCforCCwithPowerLaw(DataHandlerNCforCC):
    """Add power law wind methods to feature registry."""

    FEATURE_REGISTRY = RegistryNCforCCwithPowerLaw
