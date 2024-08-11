"""Basic objects that can cache rasterized / derived data."""

# netCDF4 has to be imported before h5py
# isort: skip_file
import copy
import itertools
import logging
import os
from typing import Dict, Optional, Union

import netCDF4 as nc4  # noqa
import h5py
import dask
import dask.array as da

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

            Can also include a ``chunks`` key, value with
            a dictionary of dictionaries for each feature (or a single
            dictionary to use for all features). e.g.
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
            )
            os.replace(tmp_file, out_file)
            logger.info('Moved %s to %s', tmp_file, out_file)

    def cache_data(self, cache_kwargs):
        """Cache data to file with file type based on user provided
        cache_pattern.

        Parameters
        ----------
        cache_kwargs : dict
            Can include 'cache_pattern' and 'chunks'. 'chunks' is a dictionary
            with feature keys and a dictionary of chunks as entries, or a
            dictionary of chunks to use for all features. e.g. ``{'u_10m':
            {'time: 5, 'south_north': 10, 'west_east': 10}}`` or ``{'time: 5,
            'south_north': 10, 'west_east': 10}`` 'cache_pattern' must have a
            ``{feature}`` format key.
        """
        cache_pattern = cache_kwargs.get('cache_pattern', None)
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
            for feature, out_file in zip(missing_features, missing_files):
                self._write_single(
                    feature=feature, out_file=out_file, chunks=chunks
                )
            logger.info('Finished writing %s', missing_files)
        return missing_files + cached_files

    @staticmethod
    def parse_chunks(feature, chunks, dims):
        """Parse chunks input to Cacher. Needs to be a dictionary of dimensions
        and chunk values but parsed to a tuple for H5 caching."""
        if isinstance(chunks, dict) and feature in chunks:
            fchunks = chunks.get(feature, {})
        else:
            fchunks = copy.deepcopy(chunks)
        if isinstance(fchunks, dict):
            fchunks = {d: fchunks.get(d, None) for d in dims}
        if isinstance(fchunks, int):
            fchunks = {feature: fchunks}
        if any(chk is None for chk in fchunks):
            fchunks = 'auto'
        return fchunks

    @classmethod
    def get_chunksizes(cls, dset, data, chunks):
        """Get chunksizes after rechunking (could be undetermined before hand
        if ``chunks == 'auto'``) and return rechunked data."""
        data_var = data.coords[dset] if dset in data.coords else data[dset]
        fchunk = cls.parse_chunks(dset, chunks, data_var.dims)
        if fchunk is not None and isinstance(fchunk, dict):
            fchunk = {k: v for k, v in fchunk.items() if k in data_var.dims}
            data_var = data_var.chunk(fchunk)

        data_var = data_var.unify_chunks()
        chunksizes = tuple(d[0] for d in data_var.chunksizes.values())
        chunksizes = chunksizes if chunksizes else None
        return data_var, chunksizes

    @classmethod
    def write_h5(cls, out_file, feature, data, chunks=None):
        """Cache data to h5 file using user provided chunks value.

        Parameters
        ----------
        out_file : str
            Name of file to write. Must have a .h5 extension.
        feature : str
            Name of feature to write to file.
        data : Sup3rDataset | Sup3rX | xr.Dataset
            Data to write to file. Comes from ``self.data``, so an
            ``xr.Dataset`` like object with ``.dims`` and ``.coords``
        chunks : dict | None
            Chunk sizes for coordinate dimensions. e.g.
            ``{'u_10m': {'time': 10, 'south_north': 100, 'west_east': 100}}``
        attrs : dict | None
            Optional attributes to write to file
        """
        if len(data.dims) == 3:
            data = data.transpose(Dimension.TIME, *Dimension.dims_2d())

        chunks = chunks or 'auto'
        attrs = {k: safe_serialize(v) for k, v in data.attrs.items()}
        with h5py.File(out_file, 'w') as f:
            for k, v in attrs.items():
                f.attrs[k] = v
            for dset in [*list(data.coords), feature]:
                data_var, chunksizes = cls.get_chunksizes(dset, data, chunks)

                if dset == Dimension.TIME:
                    data_var = da.asarray(data_var.astype(int).data)
                else:
                    data_var = data_var.data

                dset_name = dset
                if dset in Dimension.coords_2d():
                    dset_name = f'meta/{dset}'
                if dset == Dimension.TIME:
                    dset_name = 'time_index'

                logger.debug(
                    'Adding %s to %s with chunks=%s',
                    dset,
                    out_file,
                    chunksizes,
                )

                d = f.create_dataset(
                    f'/{dset_name}',
                    dtype=data_var.dtype,
                    shape=data_var.shape,
                    chunks=chunksizes,
                )
                da.store(data_var, d)

    @staticmethod
    def get_chunk_slices(chunks, shape):
        """Get slices used to write xarray data to netcdf file in chunks."""
        slices = []
        for i in range(len(shape)):
            slice_ranges = [
                (slice(k, min(k + chunks[i], shape[i])))
                for k in range(0, shape[i], chunks[i])
            ]
            slices.append(slice_ranges)
        return list(itertools.product(*slices))

    @staticmethod
    def write_chunk(out_file, dset, chunk_slice, chunk_data):
        """Add chunk to netcdf file."""
        with nc4.Dataset(out_file, 'a', format='NETCDF4') as ds:
            var = ds.variables[dset]
            var[chunk_slice] = chunk_data

    @classmethod
    def write_netcdf_chunks(cls, out_file, feature, data, chunks=None):
        """Write netcdf chunks with delayed dask tasks."""
        tasks = []
        data_var = data[feature]
        data_var, chunksizes = cls.get_chunksizes(feature, data, chunks)
        for chunk_slice in cls.get_chunk_slices(chunksizes, data_var.shape):
            chunk = data_var.data[chunk_slice]
            tasks.append(
                dask.delayed(cls.write_chunk)(
                    out_file, feature, chunk_slice, chunk
                )
            )
        dask.compute(*tasks)

    @classmethod
    def write_netcdf(cls, out_file, feature, data, chunks=None):
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
        """
        chunks = chunks or 'auto'
        attrs = {k: safe_serialize(v) for k, v in data.attrs.items()}

        with nc4.Dataset(out_file, 'w', format='NETCDF4') as ncfile:
            for dim_name, dim_size in data.sizes.items():
                ncfile.createDimension(dim_name, dim_size)

            for attr_name, attr_value in attrs.items():
                setattr(ncfile, attr_name, attr_value)

            for dset in [*list(data.coords), feature]:
                data_var, chunksizes = cls.get_chunksizes(dset, data, chunks)

                if dset == Dimension.TIME:
                    data_var = data_var.astype(int)

                dout = ncfile.createVariable(
                    dset, data_var.dtype, data_var.dims, chunksizes=chunksizes
                )

                for attr_name, attr_value in data_var.attrs.items():
                    setattr(dout, attr_name, attr_value)

                dout.coordinates = ' '.join(list(data_var.coords))

                logger.debug(
                    'Adding %s to %s with chunks=%s',
                    dset,
                    out_file,
                    chunksizes,
                )

                if dset in data.coords:
                    data_var = data_var.compute()
                    ncfile.variables[dset][:] = data_var.data

        cls.write_netcdf_chunks(out_file, feature, data, chunks=chunks)
