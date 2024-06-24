"""Exo data extracters for topography and sza

TODO: ExoDataHandler is pretty similar to ExoExtracter. Maybe a mixin or
subclass refactor here."""

import logging
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from warnings import warn

import dask.array as da
import numpy as np
import pandas as pd
from rex.utilities.solar_position import SolarPosition
from scipy.spatial import KDTree

from sup3r.postprocessing.file_handling import OutputHandler
from sup3r.preprocessing.cachers import Cacher
from sup3r.preprocessing.loaders import (
    LoaderH5,
    LoaderNC,
)
from sup3r.preprocessing.utilities import (
    Dimension,
    _compute_if_dask,
    get_input_handler_class,
    get_possible_class_args,
    log_args,
)
from sup3r.utilities.utilities import (
    generate_random_string,
    nn_fill_array,
)

logger = logging.getLogger(__name__)


@dataclass
class ExoExtracter(ABC):
    """Class to extract high-res (4km+) data rasters for new
    spatially-enhanced datasets (e.g. GCM files after spatial enhancement)
    using nearest neighbor mapping and aggregation from NREL datasets
    (e.g. WTK or NSRDB)


    Parameters
    ----------
    file_paths : str | list
        A single source h5 file to extract raster data from or a list
        of netcdf files with identical grid. The string can be a unix-style
        file path which will be passed through glob.glob. This is
        typically low-res WRF output or GCM netcdf data files that is
        source low-resolution data intended to be sup3r resolved.
    source_file : str
        Filepath to source data file to get hi-res exogenous data from which
        will be mapped to the enhanced grid of the file_paths input.  Pixels
        from this source_file will be mapped to their nearest low-res pixel in
        the file_paths input. Accordingly, source_file should be a
        significantly higher resolution than file_paths. Warnings will be
        raised if the low-resolution pixels in file_paths do not have unique
        nearest pixels from source_file. File format can be .h5 for
        TopoExtracterH5 or .nc for TopoExtracterNC
    s_enhance : int
        Factor by which the Sup3rGan model will enhance the spatial
        dimensions of low resolution data from file_paths input. For
        example, if getting topography data, file_paths has 100km data, and
        s_enhance is 4, this class will output a topography raster
        corresponding to the file_paths grid enhanced 4x to ~25km
    t_enhance : int
        Factor by which the Sup3rGan model will enhance the temporal
        dimension of low resolution data from file_paths input. For
        example, if getting sza data, file_paths has hourly data, and
        t_enhance is 4, this class will output a sza raster
        corresponding to the file_paths temporally enhanced 4x to 15 min
    input_handler_name : str
        data handler class to use for input data. Provide a string name to
        match a :class:`Extracter`. If None the correct handler will
        be guessed based on file type and time series properties.
    input_handler_kwargs : dict | None
        Any kwargs for initializing the `input_handler_name` class.
    cache_dir : str
        Directory for storing cache data. Default is './exo_cache'
    distance_upper_bound : float | None
        Maximum distance to map high-resolution data from source_file to the
        low-resolution file_paths input. None (default) will calculate this
        based on the median distance between points in source_file
    """

    file_paths: str
    source_file: str
    s_enhance: int
    t_enhance: int
    input_handler_name: Optional[str] = None
    input_handler_kwargs: Optional[dict] = None
    cache_dir: str = './exo_cache/'
    distance_upper_bound: Optional[int] = None

    @log_args
    def __post_init__(self):
        self._source_data = None
        self._tree = None
        self._hr_lat_lon = None
        self._source_lat_lon = None
        self._hr_time_index = None
        self._source_handler = None
        self.input_handler_kwargs = self.input_handler_kwargs or {}
        InputHandler = get_input_handler_class(
            self.file_paths, self.input_handler_name
        )
        params = get_possible_class_args(InputHandler)
        kwargs = {
            k: v for k, v in self.input_handler_kwargs.items() if k in params
        }
        self.input_handler = InputHandler(self.file_paths, **kwargs)

    @property
    @abstractmethod
    def source_data(self):
        """Get the 1D array of source data from the source_file_h5"""

    def get_cache_file(self, feature):
        """Get cache file name

        Parameters
        ----------
        feature : str
            Name of feature to get cache file for

        Returns
        -------
        cache_fp : str
            Name of cache file. This is a netcdf files which will be saved with
            :class:`Cacher` and loaded with :class:`LoaderNC`
        """
        fn = f'exo_{feature}_{"_".join(map(str, self.input_handler.target))}_'
        fn += f'{"x".join(map(str, self.input_handler.grid_shape))}_'
        fn += f'{self.s_enhance}x_{self.t_enhance}x.nc'
        cache_fp = os.path.join(self.cache_dir, fn)
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
        return cache_fp

    @property
    def source_lat_lon(self):
        """Get the 2D array (n, 2) of lat, lon data from the source_file_h5"""
        if self._source_lat_lon is None:
            with LoaderH5(self.source_file) as res:
                self._source_lat_lon = res.lat_lon
        return self._source_lat_lon

    @property
    def lr_shape(self):
        """Get the low-resolution spatial shape tuple"""
        return (
            *self.input_handler.lat_lon.shape[:2],
            len(self.input_handler.time_index),
        )

    @property
    def hr_shape(self):
        """Get the high-resolution spatial shape tuple"""
        return (
            self.s_enhance * self.input_handler.lat_lon.shape[0],
            self.s_enhance * self.input_handler.lat_lon.shape[1],
            self.t_enhance * len(self.input_handler.time_index),
        )

    @property
    def hr_lat_lon(self):
        """Lat lon grid for data in format (spatial_1, spatial_2, 2) Lat/Lon
        array with same ordering in last dimension. This corresponds to the
        enhanced meta data from the file_paths input * s_enhance.

        Returns
        -------
        ndarray
        """
        if self._hr_lat_lon is None:
            self._hr_lat_lon = (
                OutputHandler.get_lat_lon(
                    self.input_handler.lat_lon, self.hr_shape[:-1]
                )
                if self.s_enhance > 1
                else self.input_handler.lat_lon
            )
        return self._hr_lat_lon

    @property
    def hr_time_index(self):
        """Get the full time index for aggregated source data"""
        if self._hr_time_index is None:
            self._hr_time_index = (
                OutputHandler.get_times(
                    self.input_handler.time_index, self.hr_shape[-1]
                )
                if self.t_enhance > 1
                else self.input_handler.time_index
            )
        return self._hr_time_index

    def get_distance_upper_bound(self):
        """Maximum distance (float) to map high-resolution data from
        source_file to the low-resolution file_paths input."""
        if self.distance_upper_bound is None:
            diff = da.diff(self.source_lat_lon, axis=0)
            diff = da.median(diff, axis=0).max()
            self.distance_upper_bound = diff
            logger.info(
                'Set distance upper bound to {:.4f}'.format(
                    _compute_if_dask(self.distance_upper_bound)
                )
            )
        return self.distance_upper_bound

    @property
    def tree(self):
        """Get the KDTree built on the target lat lon data from the file_paths
        input with s_enhance"""
        if self._tree is None:
            self._tree = KDTree(self.hr_lat_lon.reshape((-1, 2)))
        return self._tree

    @property
    def nn(self):
        """Get the nearest neighbor indices. This uses a single neighbor by
        default"""
        _, nn = self.tree.query(
            self.source_lat_lon,
            distance_upper_bound=self.get_distance_upper_bound(),
        )
        return nn

    def cache_data(self, data, dset_name, cache_fp):
        """Save extracted data to cache file."""
        tmp_fp = cache_fp + f'.{generate_random_string(10)}.tmp'
        coords = {
            Dimension.LATITUDE: (
                (Dimension.SOUTH_NORTH, Dimension.WEST_EAST),
                self.hr_lat_lon[..., 0],
            ),
            Dimension.LONGITUDE: (
                (Dimension.SOUTH_NORTH, Dimension.WEST_EAST),
                self.hr_lat_lon[..., 1],
            ),
            Dimension.TIME: self.hr_time_index.values,
        }
        Cacher.write_netcdf(
            tmp_fp,
            feature=dset_name,
            data=da.broadcast_to(data, self.hr_shape),
            coords=coords,
        )
        shutil.move(tmp_fp, cache_fp)

    @property
    def data(self):
        """Get a raster of source values corresponding to the
        high-resolution grid (the file_paths input grid * s_enhance *
        t_enhance). The shape is (lats, lons, temporal, 1)

        TODO: Get actual feature name for cache file? Write attributes to cache
        here?
        """
        dset_name = self.__class__.__name__.lower()
        cache_fp = self.get_cache_file(feature=dset_name)

        if os.path.exists(cache_fp):
            data = LoaderNC(cache_fp)[dset_name, ...]
        else:
            data = self.get_data()

        if self.cache_dir is not None and not os.path.exists(cache_fp):
            self.cache_data(data=data, dset_name=dset_name, cache_fp=cache_fp)

        if data.shape[-1] != self.hr_shape[-1]:
            data = da.broadcast_to(data, self.hr_shape)

        # add trailing dimension for feature channel
        return data[..., None]

    @abstractmethod
    def get_data(self):
        """Get a raster of source values corresponding to the high-res grid
        (the file_paths input grid * s_enhance * t_enhance). The shape is
        (lats, lons, temporal)"""


class TopoExtracterH5(ExoExtracter):
    """TopoExtracter for H5 files"""

    @property
    def source_data(self):
        """Get the 1D array of elevation data from the source_file_h5"""
        if self._source_data is None:
            with LoaderH5(self.source_file) as res:
                self._source_data = res['topography', ..., None]
        return self._source_data

    def get_data(self):
        """Get a raster of source values corresponding to the
        high-resolution grid (the file_paths input grid * s_enhance *
        t_enhance). The shape is (lats, lons, 1)
        """
        assert (
            len(self.source_data.shape) == 2 and self.source_data.shape[1] == 1
        )

        df = pd.DataFrame(
            {'topo': self.source_data.flatten(), 'gid_target': self.nn}
        )
        n_target = np.prod(self.hr_shape[:-1])
        df = df[df['gid_target'] != n_target]
        df = df.sort_values('gid_target')
        df = df.groupby('gid_target').mean()

        missing = set(np.arange(n_target)) - set(df.index)
        if any(missing):
            msg = (
                f'{len(missing)} target pixels did not have unique '
                'high-resolution source data to map from. If there are a '
                'lot of target pixels missing source data this probably '
                'means the source data is not high enough resolution. '
                'Filling raster with NN.'
            )
            logger.warning(msg)
            warn(msg)
            temp_df = pd.DataFrame({'topo': np.nan}, index=sorted(missing))
            df = pd.concat((df, temp_df)).sort_index()

        hr_data = df['topo'].values.reshape(self.hr_shape[:-1])
        if np.isnan(hr_data).any():
            hr_data = nn_fill_array(hr_data)

        logger.info('Finished mapping raster from {}'.format(self.source_file))

        return da.from_array(hr_data[..., None])


class TopoExtracterNC(TopoExtracterH5):
    """TopoExtracter for netCDF files"""

    @property
    def source_handler(self):
        """Get the LoaderNC object that handles the .nc source topography
        data file."""
        if self._source_handler is None:
            logger.info(
                'Getting topography for full domain from '
                f'{self.source_file}'
            )
            self._source_handler = LoaderNC(
                self.source_file,
                features=['topography'],
            )
        return self._source_handler

    @property
    def source_data(self):
        """Get the 1D array of elevation data from the source_file_nc"""
        return self.source_handler['topography'].data.flatten()[..., None]

    @property
    def source_lat_lon(self):
        """Get the 2D array (n, 2) of lat, lon data from the source_file_nc"""
        source_lat_lon = self.source_handler.lat_lon.reshape((-1, 2))
        return source_lat_lon


class SzaExtracter(ExoExtracter):
    """SzaExtracter for H5 files"""

    @property
    def source_data(self):
        """Get the 1D array of sza data from the source_file_h5"""
        return SolarPosition(
            self.hr_time_index, self.hr_lat_lon.reshape((-1, 2))
        ).zenith.T

    def get_data(self):
        """Get a raster of source values corresponding to the high-res grid
        (the file_paths input grid * s_enhance * t_enhance). The shape is
        (lats, lons, temporal)
        """
        hr_data = self.source_data.reshape(self.hr_shape)
        logger.info('Finished computing SZA data')
        return hr_data.astype(np.float32)
