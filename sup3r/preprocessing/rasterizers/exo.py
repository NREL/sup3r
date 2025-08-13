"""Exo data rasterizers for topography, srl, sza, observation data, etc.

TODO: ExoDataHandler is pretty similar to ExoRasterizer and ExoRasterizer has
a lot of the same logic as DualRasterizer. Maybe a mixin or subclass refactor
here."""

import logging
import os
from abc import ABC
from dataclasses import dataclass
from typing import Optional, Union
from warnings import warn

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import KDTree

from sup3r.postprocessing.writers.base import OutputHandler
from sup3r.preprocessing.accessor import Sup3rX
from sup3r.preprocessing.base import Sup3rMeta
from sup3r.preprocessing.cachers import Cacher
from sup3r.preprocessing.derivers.utilities import SolarZenith
from sup3r.preprocessing.loaders import Loader
from sup3r.preprocessing.names import Dimension
from sup3r.preprocessing.utilities import compute_if_dask
from sup3r.utilities.utilities import nn_fill_array

from ..utilities import get_input_handler_class, log_args

logger = logging.getLogger(__name__)


@dataclass
class BaseExoRasterizer(ABC):
    """Class to extract high-res (4km+) data rasters for new spatially-enhanced
    datasets (e.g. GCM files after spatial enhancement) using nearest neighbor
    mapping and aggregation from high-res datasets (e.g. WTK or NSRDB)

    Parameters
    ----------
    feature : str
        Name of exogenous feature to rasterize.
    file_paths : str | list
        A single source h5 file to extract raster data from or a list of netcdf
        files with identical grid. The string can be a unix-style file path
        which will be passed through glob.glob. This is typically low-res WRF
        output or GCM netcdf data files that is source low-resolution data
        intended to be sup3r resolved.
    source_files : str | list | None
        Filepath(s) to source data file(s) to get hi-res exogenous data, which
        will be mapped to the enhanced grid of the file_paths input. Pixels
        from these files will be mapped to their nearest low-res pixel in the
        file_paths input. Accordingly, source_files should be a significantly
        higher resolution than file_paths. Warnings will be raised if the
        low-resolution pixels in file_paths do not have unique nearest pixels
        from source_files. File format can be .h5 or .nc
    s_enhance : int
        Factor by which the Sup3rGan model will enhance the spatial dimensions
        of low resolution data from file_paths input. For example, if getting
        topography data, file_paths has 100km data, and s_enhance is 4, this
        class will output a topography raster corresponding to the file_paths
        grid enhanced 4x to ~25km. This parameter is calculated automatically
        when running the forward pass with a config file.
    t_enhance : int
        Factor by which the Sup3rGan model will enhance the temporal dimension
        of low resolution data from file_paths input. For example, if getting
        "sza" data, file_paths has hourly data, and t_enhance is 4, this class
        will output an "sza" raster corresponding to ``file_paths``, temporally
        enhanced 4x to 15 min. This parameter is calculated automatically
        when running the forward pass with a config file.
    input_handler_name : str
        data handler class to use for input data. Provide a string name to
        match a ``data_handler`` or ``rasterizer`` imported into
        ``~sup3r.preprocessing``. If None the correct handler will be guessed
        based on file type and time series properties.
    input_handler_kwargs : dict | None
        Any kwargs for initializing the ``input_handler_name`` class.
    source_handler_kwargs : dict | None
        Any kwargs for initializing the source handler
        (:class:`~sup3r.preprocessing.Loader`).
    cache_dir : str | None
        Directory to use for caching rasterized data. If None (default) then no
        data will be cached. If a string is provided then this will be created
        if it does not exist and the rasterized data will be saved to this
        directory. This is useful for speeding up forward passes on large
        domains since the rasterized data will be cached once and then used for
        all forward passes on chunks of the full domain. Files will be saved to
        this directory with the name defined in ``.cache_file`` property.
        define
    chunks : str | dict
        Dictionary of dimension chunk sizes for returned exo data. e.g.
        {'time': 100, 'south_north': 100, 'west_east': 100}. This can also just
        be "auto". This is passed to ``.chunk()`` before returning exo data
        through ``.data`` attribute
    distance_upper_bound : float | None
        Maximum distance to map high-resolution data from source_files to the
        low-resolution file_paths input. None (default) will calculate this
        based on the median distance between points in source_files
    fill_nans : bool
        Whether to fill nans in the output data. This should probably be True
        for all cases except for sparse observation data.
    scale_factor : float
        Scale factor to apply to the raw data from the source_files. This is
        useful for scaling observation data which might systematically under or
        over estimate the true value. For example, MADIS data is negatively
        biased compared to 10m WTK data.
    max_workers : int
        Number of workers used for writing data to cache files. Gets passed to
        ``Cacher._write_single.``
    verbose : bool
        Whether to log output as each chunk is written to cache file.
    """

    feature: Optional[str] = None
    file_paths: Optional[str] = None
    source_files: Optional[str] = None
    source_handler_kwargs: Optional[dict] = None
    s_enhance: int = 1
    t_enhance: int = 1
    input_handler_name: Optional[str] = None
    input_handler_kwargs: Optional[dict] = None
    cache_dir: Optional[str] = None
    chunks: Optional[Union[str, dict]] = 'auto'
    distance_upper_bound: Optional[int] = None
    fill_nans: bool = True
    scale_factor: float = 1.0
    max_workers: int = 1
    verbose: bool = False

    # These sometimes have a time dimension but we don't need the time in
    # the cache file
    STATIC_FEATURES = ('topography', 'srl')

    @log_args
    def __post_init__(self):
        self._source_data = None
        self._source_handler = None
        self._tree = None
        self._hr_lat_lon = None
        self._source_lat_lon = None
        self._hr_time_index = None
        self.input_handler_kwargs = self.input_handler_kwargs or {}
        self.source_handler_kwargs = self.source_handler_kwargs or {}
        InputHandler = get_input_handler_class(self.input_handler_name)
        self.input_handler = InputHandler(
            self.file_paths, **self.input_handler_kwargs
        )

    @property
    def source_handler(self):
        """Get the Loader object that handles the exogenous data file."""
        assert (
            self.source_files is not None
        ), 'source_files must be provided to BaseExoRasterizer'
        if self._source_handler is None:
            self._source_handler = Loader(
                self.source_files,
                features=[self.feature],
                **self.source_handler_kwargs,
            )
        return self._source_handler

    @property
    def source_data(self):
        """Get the array of exogenous data from the source_files"""
        if self._source_data is None:
            self._source_data = self.source_handler[self.feature].data
            if 'time' not in self.source_handler[self.feature].dims:
                self._source_data = self._source_data[..., None]
            elif self.feature in self.STATIC_FEATURES:
                self._source_data = self._source_data[..., :1]
            shape = (-1, self._source_data.shape[-1])
            self._source_data = self._source_data.reshape(shape)
        return self._source_data

    @property
    def cache_file(self):
        """Get cache file name

        Returns
        -------
        cache_fp : str
            Name of cache file. This is a netcdf file which will be saved with
            :class:`~sup3r.preprocessing.cachers.Cacher` and loaded with
            :class:`~sup3r.preprocessing.loaders.Loader`
        """
        fn = f'exo_{self.feature}_'
        fn += f'{"_".join(map(str, self.input_handler.target))}_'
        fn += f'{"x".join(map(str, self.input_handler.grid_shape))}_'

        # add the time index to the filename if data is time dependent
        if self.source_data.shape[-1] > 1:
            start = str(self.hr_time_index[0])
            start = start.replace(':', '').replace('-', '').replace(' ', '')
            end = str(self.hr_time_index[-1])
            end = end.replace(':', '').replace('-', '').replace(' ', '')
            fn += f'{start}_{end}_'

        fn += f'{self.s_enhance}x_{self.t_enhance}x.nc'
        cache_fp = None
        if self.cache_dir is not None:
            cache_fp = os.path.join(self.cache_dir, fn)
            os.makedirs(self.cache_dir, exist_ok=True)
        return cache_fp

    @property
    def coords(self):
        """Get coords dictionary for initializing xr.Dataset."""
        coords = {
            coord: (Dimension.dims_2d(), self.hr_lat_lon[..., i])
            for i, coord in enumerate(Dimension.coords_2d())
        }
        if self.source_data.shape[1] > 1:
            coords['time'] = self.hr_time_index
        return coords

    @property
    def source_lat_lon(self):
        """Get the 2D array (n, 2) of lat, lon data from the source_files"""
        if self._source_lat_lon is None:
            self._source_lat_lon = self.source_handler.lat_lon.reshape((-1, 2))
            self._source_lat_lon = self._source_lat_lon.astype(np.float32)
        return self._source_lat_lon

    @property
    def lr_shape(self):
        """Get the low-resolution spatiotemporal shape"""
        return (
            *self.input_handler.lat_lon.shape[:2],
            len(self.input_handler.time_index),
        )

    @property
    def hr_shape(self):
        """Get the high-resolution spatiotemporal shape"""
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
        source_files to the low-resolution file_paths input."""
        if self.distance_upper_bound is None:
            diff = da.diff(self.hr_lat_lon, axis=0)
            diff = da.abs(da.median(diff, axis=0)).max()
            self.distance_upper_bound = np.asarray(diff)
            logger.info(
                'Set distance upper bound to {:.4f}'.format(
                    self.distance_upper_bound
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

    @property
    def data(self):
        """Get a raster of source values corresponding to the
        high-resolution grid (the file_paths input grid * s_enhance *
        t_enhance). The shape is (lats, lons, temporal, 1)"""

        cache_fp = self.cache_file
        if cache_fp is not None and os.path.exists(cache_fp):
            logger.info(
                'Loading cached data for {} from {}'.format(
                    self.feature, cache_fp
                )
            )
            data = Loader(cache_fp)
        else:
            data = self.get_data()

        if cache_fp is not None and not os.path.exists(cache_fp):
            Cacher._write_single(
                out_file=cache_fp,
                data=data,
                max_workers=self.max_workers,
                chunks=self.chunks,
                verbose=self.verbose,
            )

        return Sup3rX(data.chunk(self.chunks))

    def get_data(self):
        """Get a raster of source values corresponding to the
        high-resolution grid (the file_paths input grid * s_enhance *
        t_enhance). The shape is (lats, lons, temporal)
        """
        assert len(self.source_data.shape) == 2

        if self.source_data.shape[1] == 1:
            hr_data = self._get_data_2d()
            dims = Dimension.dims_2d()

        else:
            hr_data = self._get_data_3d()
            dims = Dimension.dims_3d()

        hr_data *= self.scale_factor

        if np.isnan(hr_data).any() and self.fill_nans:
            msg = (
                f'{np.isnan(hr_data).sum()} target pixels did not have unique '
                f'high-resolution {self.feature} source data to map from. If '
                'there are a lot of target pixels missing source data this '
                'probably means the source data is not high enough '
                'resolution. Filling raster with NN.'
            )
            logger.warning(msg)
            warn(msg)
            hr_data = nn_fill_array(hr_data)

        logger.info(
            f'Finished mapping raster from {self.source_files} for '
            f'"{self.feature}"',
        )
        data_vars = (dims, da.asarray(hr_data, dtype=np.float32))
        data_vars = {self.feature: data_vars}
        return Sup3rX(xr.Dataset(coords=self.coords, data_vars=data_vars))

    def _get_data_2d(self):
        """Get a raster of source values corresponding to the high-resolution
        grid (the file_paths input grid * s_enhance). This is used for time
        independent exogenous data like topography. The shape is (lats, lons,
        1)

        Returns
        -------
        out : np.ndarray
            3D array of source data with shape (lats, lons, 1).
        """
        assert (
            len(self.source_data.shape) == 2 and self.source_data.shape[1] == 1
        )

        df = pd.DataFrame({
            self.feature: self.source_data.flatten(),
            'gid_target': self.nn,
        })
        n_target = np.prod(self.hr_shape[:-1])
        out = np.full((n_target, 1), np.nan, dtype=np.float32)
        df = df[df['gid_target'] != n_target]
        df = df.sort_values('gid_target')
        df = df.groupby('gid_target').mean()
        inds = df.index.values[df.index.values != n_target]
        out[inds, 0] = df[self.feature].values
        return out.reshape(self.hr_shape[:-1])

    def _get_data_3d(self):
        """Get a raster of source values for spatiotemporal exogeneous data
        corresponding to the high-resolution grid (the file_paths input grid *
        s_enhance) and high-resolution time index (file paths input time index
        * t_enhance). The shape is (lats, lons, time)

        TODO: This does not currently perform any time aggregation of the
        source data, it just uses exact matches for time steps.

        Returns
        -------
        out : np.ndarray
            3D array of source data with shape (lats, lons, time).
        """
        assert (
            len(self.source_data.shape) == 2 and self.source_data.shape[1] > 1
        )
        target_tmask = self.hr_time_index.isin(self.source_handler.time_index)
        source_tmask = self.source_handler.time_index.isin(self.hr_time_index)
        data = self.source_data[:, source_tmask]
        rows = pd.MultiIndex.from_product(
            [self.nn, range(data.shape[-1])], names=['gid_target', 'time']
        )
        df = pd.DataFrame({self.feature: data.flatten()}, index=rows)
        n_target = np.prod(self.hr_shape[:-1])
        out = np.full(
            (n_target, len(self.hr_time_index)), np.nan, dtype=np.float32
        )
        gids = df.index.get_level_values(0)
        df = df[gids != n_target]
        df = df.sort_values('gid_target')
        df = df.groupby(['gid_target', 'time']).mean()
        inds = gids.unique().values[gids.unique() != n_target][:, None]
        vals = df[self.feature].values.reshape((-1, data.shape[-1]))
        out[inds, target_tmask] = vals
        out = out.reshape((*self.hr_shape[:-1], -1))
        return out


class ObsRasterizer(BaseExoRasterizer):
    """Rasterizer for sparse spatiotemporal observation data. This is used in
    the same way as other rasterizers (provide netcdf or flattened h5 data)
    but it does not aggregate and leaves NaNs in the output data if there are
    no observations within the ``distance_upper_bound`` of the target pixel.

    Note
    ----
    When setting up a forward pass config file you have to specifiy how to
    access exogenous features. To automatically select this rasterizer instead
    of the ``BaseExoRasterizer`` name the exogenous feature with an '_obs'
    suffix. For example, to use this rasterizer with u_10m data, set the
    feature to 'u_10m_obs'.

    See Also
    --------
    ExoRasterizer
        Type agnostic class that returns the correct rasterizer based on the
        feature name.
    """

    @property
    def source_handler(self):
        """Get the Loader object that handles the exogenous data file."""
        feat = self.feature.replace('_obs', '')
        if self._source_handler is None:
            self._source_handler = Loader(
                self.source_files,
                features=[feat],
                **self.source_handler_kwargs,
            )
        return self._source_handler

    @property
    def source_data(self):
        """Get the flattened observation data from the source_files"""
        if self._source_data is None:
            feat = self.feature.replace('_obs', '')
            src = self.source_handler[feat].data
            self._source_data = src.reshape((-1, src.shape[-1]))
        return self._source_data

    def _get_data_3d(self):
        """Get a raster of source observation values corresponding to the
        high-resolution grid (the file_paths input grid * s_enhance *
        t_enhance). The shape is (lats, lons, time)
        """
        hr_data = super()._get_data_3d()
        gid_mask = self.nn != np.prod(self.hr_shape[:-1])
        logger.info(
            'Found {} of {} observations within {:4f} degrees of high-'
            'resolution grid points.'.format(
                gid_mask.sum(), len(gid_mask), self.distance_upper_bound
            )
        )
        cover_frac = (~np.isnan(hr_data)).sum() / hr_data.size
        if cover_frac == 0:
            msg = (
                'No observations found within the distance upper bound of '
                f'{self.distance_upper_bound} degrees. '
            )
            warn(msg)
            logger.warning(msg)
        else:
            msg = 'Observations cover {:.4e} of the high-res domain.'.format(
                compute_if_dask(cover_frac)
            )
            logger.info(msg)
        self.fill_nans = False  # override parent attribute to not fill nans
        return hr_data


class SzaRasterizer(BaseExoRasterizer):
    """SzaRasterizer for H5 files"""

    @property
    def source_data(self):
        """Get the 1D array of sza data from the source_file_h5"""
        return SolarZenith.get_zenith(self.hr_time_index, self.hr_lat_lon)

    def get_data(self):
        """Get a raster of source values corresponding to the high-res grid
        (the file_paths input grid * s_enhance * t_enhance). The shape is
        (lats, lons, temporal)
        """
        logger.info(f'Finished computing {self.feature} data')
        data_vars = {self.feature: (Dimension.dims_3d(), self.source_data)}
        ds = xr.Dataset(coords=self.coords, data_vars=data_vars)
        return Sup3rX(ds)


class ExoRasterizer(BaseExoRasterizer, metaclass=Sup3rMeta):
    """Type agnostic `ExoRasterizer` class."""

    def __new__(cls, feature, file_paths, source_files=None, **kwargs):
        """Override parent class to return type specific class based on
        `source_files`"""
        if feature.lower() == 'sza':
            ExoClass = SzaRasterizer
        elif feature.lower().endswith('_obs'):
            ExoClass = ObsRasterizer
        else:
            ExoClass = BaseExoRasterizer

        kwargs = {
            'file_paths': file_paths,
            'source_files': source_files,
            'feature': feature,
            **kwargs,
        }

        return ExoClass(**kwargs)

    _signature_objs = (BaseExoRasterizer,)
    __doc__ = BaseExoRasterizer.__doc__
