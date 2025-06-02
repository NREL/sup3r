"""Basic objects that can cache rasterized / derived data."""

# netCDF4 has to be imported before h5py
# isort: skip_file
import pandas as pd
import copy
import itertools
import logging
import os
from typing import Dict, Optional, Union, TYPE_CHECKING
import netCDF4 as nc4  # noqa
import h5py
import dask
import dask.array as da
import numpy as np
from warnings import warn
from sup3r.preprocessing.base import Container
from sup3r.preprocessing.names import Dimension
from sup3r.preprocessing.utilities import _mem_check, _lowered
from sup3r.utilities.utilities import safe_cast, safe_serialize
from rex.utilities.utilities import to_records_array

from .utilities import _check_for_cache

if TYPE_CHECKING:
    from sup3r.preprocessing.accessor import Sup3rX
    from sup3r.preprocessing.base import Sup3rDataset


logger = logging.getLogger(__name__)


class Cacher(Container):
    """Base cacher object. Simply writes given data to H5 or NETCDF files. By
    default every feature will be written to a separate file. To write multiple
    features to the same file call :meth:`write_netcdf` or :meth:`write_h5`
    directly"""

    def __init__(
        self,
        data: Union['Sup3rX', 'Sup3rDataset'],
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

            Can also include a ``max_workers`` key and ``chunks`` key.
            ``max_workers`` is an inteeger specifying number of threads to use
            for writing chunks to output files and ``chunks`` is a dictionary
            of dictionaries for each feature (or a single dictionary to use
            for all features). e.g.
            .. code-block:: JSON
                {'cache_pattern': ...,
                    'chunks': {
                        'u_10m': {
                            'time': 20,
                            'south_north': 100,
                            'west_east': 100
                        }
                    }
                }

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
            self.out_files = self.cache_data(**cache_kwargs)

    @classmethod
    def _write_single(
        cls,
        out_file,
        data,
        features='all',
        chunks=None,
        max_workers=None,
        mode='w',
        attrs=None,
        verbose=False,
        overwrite=False,
        keep_dim_order=False,
    ):
        """Write single NETCDF or H5 cache file."""
        if os.path.exists(out_file) and not overwrite:
            logger.info(
                f'{out_file} already exists. Delete if you want to overwrite.'
            )
            return
        if features == 'all':
            features = list(data.data_vars)
        features = features if isinstance(features, list) else [features]
        _, ext = os.path.splitext(out_file)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        tmp_file = out_file + '.tmp'
        logger.info('Writing %s to %s. %s', features, tmp_file, _mem_check())
        if ext == '.h5':
            func = cls.write_h5
        elif ext == '.nc':
            func = cls.write_netcdf
        else:
            msg = (
                'cache_pattern must have either h5 or nc extension. '
                f'Received {ext}.'
            )
            logger.error(msg)
            raise ValueError(msg)
        func(
            out_file=tmp_file,
            data=data,
            features=features,
            chunks=chunks,
            max_workers=max_workers,
            mode=mode,
            attrs=attrs,
            verbose=verbose,
            keep_dim_order=keep_dim_order,
        )
        os.replace(tmp_file, out_file)
        logger.info('Moved %s to %s', tmp_file, out_file)

    def cache_data(
        self,
        cache_pattern,
        chunks=None,
        max_workers=None,
        mode='w',
        attrs=None,
        verbose=False,
        keep_dim_order=False,
    ):
        """Cache data to file with file type based on user provided
        cache_pattern.

        Parameters
        ----------
        cache_pattern : str
            Cache file pattern. Must have a {feature} format key. The extension
            (.h5 or .nc) specifies which format to use for caching.
        chunks : dict | None
            Chunk sizes for coordinate dimensions. e.g.
            ``{'u_10m': {'time': 10, 'south_north': 100, 'west_east': 100}}``
        max_workers : int | None
            Number of workers to use for parallel writing of chunks
        mode : str
            Write mode for ``out_file``. Defaults to write.
        attrs : dict | None
            Optional attributes to write to file. Can specify dataset specific
            attributes by adding a dictionary with the dataset name as a key.
            e.g. {**global_attrs, dset: {...}}
        verbose : bool
            Whether to log progress for each chunk written to output files.
        keep_dim_order : bool
            Whether to keep the original dimension order of the data. If
            ``False`` then the data will be transposed to have the time
            dimension first.
        """
        msg = 'cache_pattern must have {feature} format key.'
        assert '{feature}' in cache_pattern, msg

        cached_files, _, missing_files, missing_features = _check_for_cache(
            features=self.features,
            cache_kwargs={'cache_pattern': cache_pattern},
        )

        if any(cached_files):
            logger.info(
                f'Cache files with pattern {cache_pattern} already exist. '
                'Delete to overwrite.'
            )

        if any(missing_files):
            logger.info('Caching %s to %s', missing_features, missing_files)
            for feature, out_file in zip(missing_features, missing_files):
                self._write_single(
                    data=self.data,
                    features=feature,
                    out_file=out_file,
                    chunks=chunks,
                    max_workers=max_workers,
                    mode=mode,
                    verbose=verbose,
                    attrs=attrs,
                    keep_dim_order=keep_dim_order,
                )
            logger.info('Finished writing %s', missing_files)
        return missing_files + cached_files

    @staticmethod
    def parse_chunks(feature, chunks, dims):
        """Parse chunks input to Cacher. Needs to be a dictionary of dimensions
        and chunk values but parsed to a tuple for H5 caching."""

        if isinstance(chunks, dict) and feature.lower() in _lowered(chunks):
            chunks = {k.lower(): v for k, v in chunks.items()}
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
        """Get chunksizes after rechunking (could be undetermined beforehand
        if ``chunks == 'auto'``) and return rechunked data.

        Parameters
        ----------
        dset : str
            Name of feature to get chunksizes for.
        data : Sup3rX | xr.Dataset
            ``Sup3rX`` or ``xr.Dataset`` containing data to be cached.
        chunks : dict | None | 'auto'
            Dictionary of chunksizes either to use for all features or, if the
            dictionary includes feature keys, feature specific chunksizes. Can
            also be None or 'auto'.
        """
        data_var = data.coords[dset] if dset in data.coords else data[dset]
        fchunk = cls.parse_chunks(dset, chunks, data_var.dims)
        if isinstance(fchunk, dict):
            fchunk = {k: v for k, v in fchunk.items() if k in data_var.dims}

        data_var = data_var.chunk(fchunk)
        data_var = data_var.unify_chunks()

        chunksizes = tuple(d[0] for d in data_var.chunksizes.values())
        chunksizes = chunksizes if chunksizes else None
        if chunksizes is not None:
            chunkmem = np.prod(chunksizes) * data_var.dtype.itemsize / 1e9
            chunkmem = round(chunkmem, 3)
            if chunkmem > 4:
                msg = (
                    'Chunks cannot be larger than 4GB. Given chunksizes %s '
                    'result in %sGB. Will use chunksizes = None'
                )
                logger.warning(msg, chunksizes, chunkmem)
                warn(msg % (chunksizes, chunkmem))
                chunksizes = None
        return data_var, chunksizes

    @classmethod
    def add_coord_meta(cls, out_file, data, meta=None):
        """Add flattened coordinate meta to out_file. This is used for h5
        caching.

        Parameters
        ----------
        out_file : str
            Name of output file.
        data : Sup3rX | xr.Dataset
            Data being written to the given ``out_file``.
        meta : pd.DataFrame | None
            Optional additional meta information to be written to the given
            ``out_file``. If this is None then only coordinate info will be
            included in the meta written to the ``out_file``
        """
        if meta is None or (isinstance(meta, dict) and not meta):
            meta = pd.DataFrame()
        for coord in Dimension.coords_2d():
            if coord in data:
                meta[coord] = data[coord].data.flatten()
        logger.info('Adding coordinate meta to %s', out_file)
        with h5py.File(out_file, 'a') as f:
            meta = to_records_array(meta)
            f.create_dataset(
                '/meta', shape=meta.shape, dtype=meta.dtype, data=meta
            )

    @classmethod
    def write_h5(
        cls,
        out_file,
        data,
        features='all',
        chunks=None,
        max_workers=None,
        mode='w',
        attrs=None,
        verbose=False,  # noqa # pylint: disable=W0613
        keep_dim_order=False,
    ):
        """Cache data to h5 file using user provided chunks value.

        Parameters
        ----------
        out_file : str
            Name of file to write. Must have a .h5 extension.
        data : Sup3rDataset | Sup3rX | xr.Dataset
            Data to write to file. Comes from ``self.data``, so an
            ``xr.Dataset`` like object with ``.dims`` and ``.coords``
        features : str | list
            Name of feature(s) to write to file.
        chunks : dict | None
            Chunk sizes for coordinate dimensions. e.g.
            ``{'u_10m': {'time': 10, 'south_north': 100, 'west_east': 100}}``
        max_workers : int | None
            Number of workers to use for parallel writing of chunks
        mode : str
            Write mode for ``out_file``. Defaults to write.
        attrs : dict | None
            Optional attributes to write to file. Can specify dataset specific
            attributes by adding a dictionary with the dataset name as a key.
            e.g. {**global_attrs, dset: {...}}. Can also include a global meta
            dataframe that will then be added to the coordinate meta.
        verbose : bool
            Dummy arg to match ``write_netcdf`` signature
        keep_dim_order : bool
            Whether to keep the original dimension order of the data. If
            ``False`` then the data will be transposed to have the time
            dimension first.
        """
        if (
            len(data.dims) == 3
            and Dimension.TIME in data.dims
            and not keep_dim_order
        ):
            data = data.transpose(Dimension.TIME, *Dimension.dims_2d())
        if features == 'all':
            features = list(data.data_vars)
        features = features if isinstance(features, list) else [features]
        chunks = chunks or 'auto'
        global_attrs = data.attrs.copy()
        attrs = attrs or {}
        meta = attrs.pop('meta', {})
        global_attrs.update(attrs)
        attrs = {k: safe_cast(v) for k, v in global_attrs.items()}
        with h5py.File(out_file, mode) as f:
            for k, v in attrs.items():
                f.attrs[k] = v

            coord_names = [
                crd for crd in data.coords if crd in Dimension.coords_4d()
            ]

            if Dimension.TIME in data:
                # int64 used explicity to avoid incorrect encoding as int32
                data[Dimension.TIME] = data[Dimension.TIME].astype('int64')

            for dset in [*coord_names, *features]:
                data_var, chunksizes = cls.get_chunksizes(dset, data, chunks)
                data_var = data_var.data

                if not isinstance(data_var, da.core.Array):
                    data_var = da.asarray(data_var)

                dset_name = dset
                if dset == Dimension.TIME:
                    dset_name = 'time_index'

                logger.debug(
                    'Adding %s to %s with chunks=%s and max_workers=%s',
                    dset,
                    out_file,
                    chunksizes,
                    max_workers,
                )

                d = f.create_dataset(
                    f'/{dset_name}',
                    dtype=data_var.dtype,
                    shape=data_var.shape,
                    chunks=chunksizes,
                )
                if max_workers == 1:
                    da.store(data_var, d, scheduler='single-threaded')
                else:
                    da.store(
                        data_var,
                        d,
                        scheduler='threads',
                        num_workers=max_workers,
                    )
        cls.add_coord_meta(out_file=out_file, data=data, meta=meta)

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
    def write_chunk(out_file, dset, chunk_slice, chunk_data, msg=None):
        """Add chunk to netcdf file."""
        if msg is not None:
            logger.debug(msg)
        with nc4.Dataset(out_file, 'a') as ds:
            var = ds.variables[dset]
            var[chunk_slice] = chunk_data

    @classmethod
    def write_netcdf_chunks(
        cls,
        out_file,
        feature,
        data,
        chunks=None,
        max_workers=None,
        verbose=False,
    ):
        """Write netcdf chunks with delayed dask tasks."""
        tasks = []
        data_var = data[feature]
        data_var, chunksizes = cls.get_chunksizes(feature, data, chunks)
        chunksizes = data_var.shape if chunksizes is None else chunksizes
        chunk_slices = cls.get_chunk_slices(chunksizes, data_var.shape)
        logger.info(
            'Adding %s to %s with %s chunks and max_workers=%s. %s',
            feature,
            out_file,
            len(chunk_slices),
            max_workers,
            _mem_check(),
        )
        for i, chunk_slice in enumerate(chunk_slices):
            msg = f'Writing chunk {i + 1} / {len(chunk_slices)} to {out_file}'
            msg = None if not verbose else msg
            chunk = data_var.data[chunk_slice]
            task = dask.delayed(cls.write_chunk)(
                out_file, feature, chunk_slice, chunk, msg
            )
            tasks.append(task)
        if max_workers == 1:
            dask.compute(*tasks, scheduler='single-threaded')
        else:
            dask.compute(*tasks, scheduler='threads', num_workers=max_workers)

    @classmethod
    def write_netcdf(
        cls,
        out_file,
        data,
        features='all',
        chunks=None,
        max_workers=None,
        mode='w',
        attrs=None,
        verbose=False,
        keep_dim_order=False # noqa # pylint: disable=W0613
    ):
        """Cache data to a netcdf file.

        Parameters
        ----------
        out_file : str
            Name of file to write. Must have a ``.nc`` extension.
        data : Sup3rDataset
            Data to write to file. Comes from ``self.data``, so a
            ``Sup3rDataset`` with coords attributes
        features : str | list
            Names of feature(s) to write to file.
        chunks : dict | None
            Chunk sizes for coordinate dimensions. e.g. ``{'south_north': 100,
            'west_east': 100, 'time': 10}`` Can also include dataset specific
            values. e.g. ``{'windspeed': {'south_north': 100, 'west_east': 100,
            'time': 10}}``
        max_workers : int | None
            Number of workers to use for parallel writing of chunks
        mode : str
            Write mode for ``out_file``. Defaults to write.
        attrs : dict | None
            Optional attributes to write to file. Can specify dataset specific
            attributes by adding a dictionary with the dataset name as a key.
            e.g. {**global_attrs, dset: {...}}
        verbose : bool
            Whether to log output after each chunk is written.
        keep_dim_order : bool
            Dummy arg to match ``write_h5`` signature
        """
        chunks = chunks or 'auto'
        global_attrs = data.attrs.copy()
        global_attrs.update(attrs or {})
        attrs = global_attrs.copy()
        if features == 'all':
            features = list(data.data_vars)
        features = features if isinstance(features, list) else [features]
        with nc4.Dataset(out_file, mode, format='NETCDF4') as ncfile:
            for dim_name, dim_size in data.sizes.items():
                ncfile.createDimension(dim_name, dim_size)

            coord_names = [
                crd for crd in data.coords if crd in Dimension.coords_4d()
            ]
            for dset in [*coord_names, *features]:
                data_var, chunksizes = cls.get_chunksizes(dset, data, chunks)

                if dset == Dimension.TIME:
                    data_var = data_var.astype(int)

                dout = ncfile.createVariable(
                    dset, data_var.dtype, data_var.dims, chunksizes=chunksizes
                )
                var_attrs = data_var.attrs.copy()
                var_attrs.update(attrs.pop(dset, {}))
                for attr_name, attr_value in var_attrs.items():
                    dout.setncattr(attr_name, safe_cast(attr_value))

                dout.coordinates = ' '.join(list(coord_names))

                logger.debug(
                    'Adding %s to %s with chunks=%s',
                    dset,
                    out_file,
                    chunksizes,
                )

                if dset in data.coords:
                    ncfile.variables[dset][:] = np.asarray(data_var.data)

            for attr_name, attr_value in attrs.items():
                attr_value = safe_cast(attr_value)
                try:
                    ncfile.setncattr(attr_name, attr_value)
                except Exception as e:
                    msg = (
                        f'Could not write {attr_name} as attribute, '
                        f'serializing with json dumps, '
                        f'received error: "{e}"'
                    )
                    logger.warning(msg)
                    warn(msg)
                    ncfile.setncattr(attr_name, safe_serialize(attr_value))

        for feature in features:
            cls.write_netcdf_chunks(
                out_file=out_file,
                feature=feature,
                data=data,
                chunks=chunks,
                max_workers=max_workers,
                verbose=verbose,
            )

        logger.info('Finished writing %s to %s', features, out_file)
