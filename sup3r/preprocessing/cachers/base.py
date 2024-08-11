"""Basic objects that can cache rasterized / derived data."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Union
from warnings import warn

import dask.array as da
import h5py
import xarray as xr

from sup3r.preprocessing.accessor import Sup3rX
from sup3r.preprocessing.base import Container, Sup3rDataset
from sup3r.preprocessing.names import Dimension
from sup3r.preprocessing.utilities import _mem_check
from sup3r.utilities.utilities import safe_serialize

from .utilities import _check_for_cache

logger = logging.getLogger(__name__)


class Cacher(Container):
    """Base cacher object. Simply writes given data to H5 or NETCDF files."""

    def __init__(
        self,
        data: Union[Sup3rX, Sup3rDataset],
        cache_kwargs: Optional[Dict] = None,
    ):
        """
        Parameters
        ----------
        data : Union[Sup3rX, Sup3rDataset]
            Data to write to file
        cache_kwargs : dict
            Dictionary with kwargs for caching wrangled data. This should at
            minimum include a 'cache_pattern' key, value. This pattern must
            have a {feature} format key and either a h5 or nc file extension,
            based on desired output type.

            Can also include a 'max_workers' key and a 'chunks' key, value with
            a dictionary of tuples for each feature. e.g.
            ``{'cache_pattern': ...,
               'chunks': {
                 'u_10m': {'time': 20, 'south_north': 100, 'west_east': 100}}
              }``

        Note
        ----
        This is only for saving cached data. If you want to reload the
        cached files load them with a ``Loader`` object. ``DataHandler``
        objects can cache and reload from cache automatically.
        """
        super().__init__(data=data)
        if (
            cache_kwargs is not None
            and cache_kwargs.get('cache_pattern', None) is not None
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
            if ext == '.h5':
                func = self.write_h5
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
                data=self.data,
                chunks=chunks,
                attrs={k: safe_serialize(v) for k, v in self.attrs.items()},
            )
            os.replace(tmp_file, out_file)
            logger.info('Moved %s to %s', tmp_file, out_file)

    def cache_data(self, cache_kwargs):
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
        cache_pattern = cache_kwargs.get('cache_pattern', None)
        max_workers = cache_kwargs.get('max_workers', 1)
        chunks = cache_kwargs.get('chunks', None)
        msg = 'cache_pattern must have {feature} format key.'
        assert '{feature}' in cache_pattern, msg

        cached_files, _, missing_files, missing_features = _check_for_cache(
            features=self.features, cache_kwargs=cache_kwargs
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

    @staticmethod
    def parse_chunks(feature, chunks, shape):
        """Parse chunks input to Cacher. Needs to be a dictionary of dimensions
        and chunk values but parsed to a tuple for H5 caching."""
        if any(d in chunks for d in Dimension.coords_3d()):
            fchunks = chunks.copy()
        else:
            fchunks = chunks.get(feature, {})
        if isinstance(fchunks, tuple):
            msg = (
                'chunks value should be a dictionary with dimension names '
                'as keys and values as dimension chunksizes. Will try '
                'to use this %s for (time, lats, lons)'
            )
            logger.warning(msg, fchunks)
            warn(msg % fchunks)
            return fchunks
        fchunks = {} if fchunks == 'auto' else fchunks
        out = (
            fchunks.get(Dimension.TIME, None),
            fchunks.get(Dimension.SOUTH_NORTH, None),
            fchunks.get(Dimension.WEST_EAST, None),
        )
        if len(shape) == 2:
            out = out[1:]
        if len(shape) == 1:
            out = (out[0],)
        if any(o is None for o in out):
            return None
        return out

    @classmethod
    def write_h5(cls, out_file, feature, data, chunks=None, attrs=None):
        """Cache data to h5 file using user provided chunks value.

        Parameters
        ----------
        out_file : str
            Name of file to write. Must have a .h5 extension.
        feature : str
            Name of feature to write to file.
        data : Sup3rDataset
            Data to write to file. Comes from ``self.data``, so a Sup3rDataset
            with coords attributes
        chunks : dict | None
            Chunk sizes for coordinate dimensions. e.g.
            ``{'u_10m': {'time': 10, 'south_north': 100, 'west_east': 100}}``
        attrs : dict | None
            Optional attributes to write to file
        """
        chunks = chunks or {}
        attrs = attrs or {}
        coords = data.coords
        data = data[feature].data
        if len(data.shape) == 3:
            data = da.transpose(data, axes=(2, 0, 1))

        dsets = [f'/meta/{d}' for d in Dimension.coords_2d()]
        dsets += ['time_index', feature]
        vals = [
            coords[Dimension.LATITUDE].data,
            coords[Dimension.LONGITUDE].data,
        ]
        vals += [da.asarray(coords[Dimension.TIME].astype(int)), data]

        with h5py.File(out_file, 'w') as f:
            for k, v in attrs.items():
                f.attrs[k] = v
            for dset, val in zip(dsets, vals):
                fchunk = cls.parse_chunks(dset, chunks, val.shape)
                logger.debug(
                    'Adding %s to %s with chunks=%s', dset, out_file, fchunk
                )
                d = f.create_dataset(
                    f'/{dset}',
                    dtype=val.dtype,
                    shape=val.shape,
                    chunks=fchunk,
                )
                da.store(val, d)

    @classmethod
    def write_netcdf(cls, out_file, feature, data, chunks=None, attrs=None):
        """Cache data to a netcdf file.

        Parameters
        ----------
        out_file : str
            Name of file to write. Must have a .nc extension.
        feature : str
            Name of feature to write to file.
        data : Sup3rDataset
            Data to write to file. Comes from ``self.data``, so a Sup3rDataset
            with coords attributes
        chunks : dict | None
            Chunk sizes for coordinate dimensions. e.g. ``{'windspeed':
            {'south_north': 100, 'west_east': 100, 'time': 10}}``
        attrs : dict | None
            Optional attributes to write to file
        """
        chunks = chunks or {}
        attrs = attrs or {}
        out = xr.Dataset(
            data_vars={
                feature: (
                    data[feature].dims,
                    data[feature].data,
                    data[feature].attrs,
                )
            },
            coords=data.coords,
            attrs=attrs,
        )
        f_chunks = chunks.get(feature, 'auto')
        logger.info(
            'Writing %s to %s with chunks=%s', feature, out_file, f_chunks
        )
        out = out.chunk(f_chunks)
        out[feature].load().to_netcdf(out_file)
        del out
