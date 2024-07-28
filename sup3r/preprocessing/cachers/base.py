"""Basic objects that can cache extracted / derived data."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

import dask.array as da
import h5py
import xarray as xr

from sup3r.preprocessing.base import Container
from sup3r.preprocessing.names import Dimension
from sup3r.preprocessing.utilities import _mem_check
from sup3r.typing import T_Dataset

from .utilities import _check_for_cache

logger = logging.getLogger(__name__)


class Cacher(Container):
    """Base cacher object. Simply writes given data to H5 or NETCDF files."""

    def __init__(
        self,
        data: T_Dataset,
        cache_kwargs: Optional[Dict] = None,
    ):
        """
        Parameters
        ----------
        data : T_Dataset
            Data to write to file
        cache_kwargs : dict
            Dictionary with kwargs for caching wrangled data. This should at
            minimum include a 'cache_pattern' key, value. This pattern must
            have a {feature} format key and either a h5 or nc file extension,
            based on desired output type.

            Can also include a 'max_workers' key and a 'chunks' key, value with
            a dictionary of tuples for each feature. e.g. {'cache_pattern':
            ..., 'chunks': {'windspeed_100m': (20, 100, 100)}} where the chunks
            ordering is (time, lats, lons)

            Note: This is only for saving cached data. If you want to reload
            the cached files load them with a Loader object.
        """
        super().__init__(data=data)
        if (
            cache_kwargs is not None
            and cache_kwargs.get('cache_pattern') is not None
        ):
            self.out_files = self.cache_data(cache_kwargs)

    def _write_single(self, feature, out_file, chunks):
        """Write single NETCDF or H5 cache file."""
        if os.path.exists(out_file):
            logger.info(
                f'{out_file} already exists. Delete if you want to overwrite.'
            )
        else:
            _, ext = os.path.splitext(out_file)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            tmp_file = out_file + '.tmp'
            logger.info(
                'Writing %s to %s. %s', feature, tmp_file, _mem_check()
            )
            data = self[feature, ...]
            if ext == '.h5':
                func = self.write_h5
                if len(data.shape) == 3:
                    data = da.transpose(data, axes=(2, 0, 1))
            elif ext == '.nc':
                func = self.write_netcdf
            else:
                msg = (
                    'cache_pattern must have either h5 or nc extension. '
                    f'Received {ext}.'
                )
                logger.error(msg)
                raise ValueError(msg)
            func(
                tmp_file,
                feature,
                data,
                self.coords,
                chunks=chunks,
                attrs=self.attrs,
            )
            os.replace(tmp_file, out_file)
            logger.info('Moved %s to %s', tmp_file, out_file)

    def cache_data(self, kwargs):
        """Cache data to file with file type based on user provided
        cache_pattern.

        Parameters
        ----------
        cache_kwargs : dict
            Can include 'cache_pattern', 'chunks', and 'max_workers'. 'chunks'
            is a dictionary of tuples (time, lats, lons) for each feature
            specifying the chunks for h5 writes. 'cache_pattern' must have a
            {feature} format key.
        """
        cache_pattern = kwargs.get('cache_pattern', None)
        max_workers = kwargs.get('max_workers', 1)
        chunks = kwargs.get('chunks', None)
        msg = 'cache_pattern must have {feature} format key.'
        assert '{feature}' in cache_pattern, msg

        cached_files, _, missing_files, missing_features = _check_for_cache(
            self.features, kwargs
        )

        if any(cached_files):
            logger.info(
                'Cache files %s already exist. Delete to overwrite.',
                cached_files,
            )

        if any(missing_files):
            if max_workers == 1:
                for feature, out_file in zip(missing_features, missing_files):
                    self._write_single(
                        feature=feature, out_file=out_file, chunks=chunks
                    )
            else:
                futures = {}
                with ThreadPoolExecutor(max_workers=max_workers) as exe:
                    for feature, out_file in zip(
                        missing_features, missing_files
                    ):
                        future = exe.submit(
                            self._write_single,
                            feature=feature,
                            out_file=out_file,
                            chunks=chunks,
                        )
                        futures[future] = (feature, out_file)
                    logger.info(
                        f'Submitted cacher futures for {self.features}.'
                    )
                for i, future in enumerate(as_completed(futures)):
                    _ = future.result()
                    feature, out_file = futures[future]
                    logger.info(
                        'Finished writing %s. (%s of %s files).',
                        out_file,
                        i + 1,
                        len(futures),
                    )
            logger.info('Finished writing %s', missing_files)
        return missing_files + cached_files

    @classmethod
    def write_h5(
        cls, out_file, feature, data, coords, chunks=None, attrs=None
    ):
        """Cache data to h5 file using user provided chunks value.

        Parameters
        ----------
        out_file : str
            Name of file to write. Must have a .h5 extension.
        feature : str
            Name of feature to write to file.
        data : T_Array | xr.Dataset
            Data to write to file
        coords : dict
            Dictionary of coordinate variables
        chunks : dict | None
            Chunk sizes for coordinate dimensions. e.g. {'windspeed': (100,
            100, 10)}
        attrs : dict | None
            Optional attributes to write to file
        """
        chunks = chunks or {}
        attrs = attrs or {}
        with h5py.File(out_file, 'w') as f:
            lats = coords[Dimension.LATITUDE].data
            lons = coords[Dimension.LONGITUDE].data
            times = coords[Dimension.TIME].astype(int)
            for k, v in attrs.items():
                f.attrs[k] = v
            data_dict = dict(
                zip(
                    [
                        'time_index',
                        Dimension.LATITUDE,
                        Dimension.LONGITUDE,
                        feature,
                    ],
                    [da.from_array(times), lats, lons, data],
                )
            )
            for dset, vals in data_dict.items():
                if dset in (Dimension.LATITUDE, Dimension.LONGITUDE):
                    dset = f'meta/{dset}'
                d = f.require_dataset(
                    f'/{dset}',
                    dtype=vals.dtype,
                    shape=vals.shape,
                    chunks=chunks.get(dset, None),
                )
                da.store(vals, d)
                logger.debug(f'Added {dset} to {out_file}.')

    @classmethod
    def write_netcdf(
        cls, out_file, feature, data, coords, chunks=None, attrs=None
    ):
        """Cache data to a netcdf file.

        Parameters
        ----------
        out_file : str
            Name of file to write. Must have a .nc extension.
        feature : str
            Name of feature to write to file.
        data : T_Array | xr.Dataset
            Data to write to file
        coords : dict | xr.Dataset.coords
            Dictionary of coordinate variables or xr.Dataset coords attribute.
        chunks : dict | None
            Chunk sizes for coordinate dimensions. e.g. {'windspeed':
            {'south_north': 100, 'west_east': 100, 'time': 10}}
        attrs : dict | None
            Optional attributes to write to file
        """
        chunks = chunks or {}
        attrs = attrs or {}
        if isinstance(coords, dict):
            flattened = (
                Dimension.FLATTENED_SPATIAL in coords[Dimension.LATITUDE][0]
            )
        else:
            flattened = (
                Dimension.FLATTENED_SPATIAL in coords[Dimension.LATITUDE].dims
            )
        dims = (
            Dimension.flat_2d()
            if flattened
            else Dimension.order()[1 : len(data.shape) + 1]
        )
        out = xr.Dataset(
            data_vars={feature: (dims, data)}, coords=coords, attrs=attrs
        )
        out = out.chunk(chunks.get(feature, 'auto'))
        out.to_netcdf(out_file)
        out.close()
