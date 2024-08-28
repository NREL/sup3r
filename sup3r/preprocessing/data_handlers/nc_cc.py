"""NETCDF DataHandler for climate change applications."""

import logging
import os

import dask.array as da
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.stats import mode

from sup3r.preprocessing.derivers.methods import (
    RegistryNCforCC,
    RegistryNCforCCwithPowerLaw,
)
from sup3r.preprocessing.loaders import Loader
from sup3r.preprocessing.names import Dimension
from sup3r.preprocessing.utilities import log_args

from .factory import DataHandler, DataHandlerFactory

logger = logging.getLogger(__name__)


BaseNCforCC = DataHandlerFactory(DataHandler, FeatureRegistry=RegistryNCforCC)


class DataHandlerNCforCC(BaseNCforCC):
    """Extended NETCDF data handler. This implements a rasterizer hook to add
    "clearsky_ghi" to the rasterized data if "clearsky_ghi" is requested."""

    @log_args
    def __init__(
        self,
        file_paths,
        features='all',
        nsrdb_source_fp=None,
        nsrdb_agg=1,
        nsrdb_smoothing=0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        file_paths : str | list | pathlib.Path
            file_paths input to
            :class:`~sup3r.preprocessing.rasterizers.Rasterizer`
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
        kwargs : list
            Same optional keyword arguments as parent class.
        """
        self._nsrdb_source_fp = nsrdb_source_fp
        self._nsrdb_agg = nsrdb_agg
        self._nsrdb_smoothing = nsrdb_smoothing
        self._features = features
        super().__init__(file_paths=file_paths, features=features, **kwargs)

    _signature_objs = (__init__, BaseNCforCC)
    _skip_params = ('name', 'FeatureRegistry')

    def _rasterizer_hook(self):
        """Rasterizer hook implementation to add 'clearsky_ghi' data to
        rasterized data, which will then be used when the :class:`Deriver` is
        called."""
        cs_feats = ['clearsky_ratio', 'clearsky_ghi']
        need_ghi = any(
            f in self._features and f not in self.rasterizer for f in cs_feats
        )
        if need_ghi:
            self.rasterizer.data['clearsky_ghi'] = self.get_clearsky_ghi()

    def run_input_checks(self):
        """Run checks on the files provided for extracting clearsky_ghi. Make
        sure the loaded data is daily data and the step size is one day."""

        msg = (
            'Need nsrdb_source_fp input arg as a valid filepath to '
            'retrieve clearsky_ghi (maybe for clearsky_ratio) but '
            'received: {}'.format(self._nsrdb_source_fp)
        )
        assert self._nsrdb_source_fp is not None and os.path.exists(
            self._nsrdb_source_fp
        ), msg

        msg = (
            'Can only handle source CC data in hourly frequency but '
            'received daily frequency of {}hrs (should be 24) '
            'with raw time index: {}'.format(
                self.loader.time_step / 3600, self.rasterizer.time_index
            )
        )
        assert self.loader.time_step / 3600 == 24.0, msg

        msg = (
            'Can only handle source CC data with time_slice.step == 1 '
            'but received: {}'.format(self.rasterizer.time_slice.step)
        )
        assert (self.rasterizer.time_slice.step is None) | (
            self.rasterizer.time_slice.step == 1
        ), msg

    def run_wrap_checks(self, cs_ghi):
        """Run check on rasterized data from clearsky_ghi source."""
        logger.info(
            'Reshaped clearsky_ghi data to final shape {} to '
            'correspond with CC daily average data over source '
            'time_slice {} with (lat, lon) grid shape of {}'.format(
                cs_ghi.shape,
                self.rasterizer.time_slice,
                self.rasterizer.grid_shape,
            )
        )
        msg = (
            'nsrdb clearsky GHI time dimension {} '
            'does not match the GCM time dimension {}'.format(
                cs_ghi.shape[2], len(self.rasterizer.time_index)
            )
        )
        assert cs_ghi.shape[2] == len(self.rasterizer.time_index), msg

    def get_time_slice(self, ti_nsrdb):
        """Get nsrdb data time slice consistent with self.time_index."""
        t_start = np.where(
            (self.rasterizer.time_index[0].month == ti_nsrdb.month)
            & (self.rasterizer.time_index[0].day == ti_nsrdb.day)
        )[0][0]
        t_end = (
            1
            + np.where(
                (self.rasterizer.time_index[-1].month == ti_nsrdb.month)
                & (self.rasterizer.time_index[-1].day == ti_nsrdb.day)
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
        cs_ghi : Union[np.ndarray, da.core.Array]
            Clearsky ghi (W/m2) from the nsrdb_source_fp h5 source file. Data
            shape is (lat, lon, time) where time is daily average values.
        """
        self.run_input_checks()

        res = Loader(self._nsrdb_source_fp)
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

        cs_ghi = (
            res.data[['clearsky_ghi']]
            .isel(
                {
                    Dimension.FLATTENED_SPATIAL: i.flatten(),
                    Dimension.TIME: t_slice,
                }
            )
            .coarsen({Dimension.FLATTENED_SPATIAL: self._nsrdb_agg})
            .mean()
        )
        time_freq = float(
            mode(
                (ti_nsrdb[1:] - ti_nsrdb[:-1]).seconds / 3600, keepdims=False
            ).mode
        )

        cs_ghi = cs_ghi.coarsen({Dimension.TIME: int(24 // time_freq)}).mean()
        lat_idx, lon_idx = (
            np.arange(self.rasterizer.grid_shape[0]),
            np.arange(self.rasterizer.grid_shape[1]),
        )
        ind = pd.MultiIndex.from_product(
            (lat_idx, lon_idx), names=Dimension.dims_2d()
        )
        cs_ghi = cs_ghi.assign({Dimension.FLATTENED_SPATIAL: ind}).unstack(
            Dimension.FLATTENED_SPATIAL
        )

        cs_ghi = cs_ghi.transpose(*Dimension.dims_3d())

        cs_ghi = cs_ghi['clearsky_ghi'].data
        if cs_ghi.shape[-1] < len(self.rasterizer.time_index):
            n = int(
                da.ceil(len(self.rasterizer.time_index) / cs_ghi.shape[-1])
            )
            cs_ghi = da.repeat(cs_ghi, n, axis=2)

        cs_ghi = cs_ghi[..., : len(self.rasterizer.time_index)]

        self.run_wrap_checks(cs_ghi)

        return cs_ghi


class DataHandlerNCforCCwithPowerLaw(DataHandlerNCforCC):
    """Add power law wind methods to feature registry."""

    FEATURE_REGISTRY = RegistryNCforCCwithPowerLaw
