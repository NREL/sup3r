"""NETCDF file collection.

TODO: Integrate this with Cacher class
"""

import logging
import os

import xarray as xr
from rex.utilities.loggers import init_logger

from sup3r.preprocessing.cachers import Cacher
from sup3r.preprocessing.names import Dimension
from sup3r.utilities.utilities import xr_open_mfdataset

from .base import BaseCollector

logger = logging.getLogger(__name__)


class CollectorNC(BaseCollector):
    """Sup3r NETCDF file collection framework"""

    @classmethod
    def collect(
        cls,
        file_paths,
        out_file,
        features='all',
        log_level=None,
        log_file=None,
        overwrite=True,
        res_kwargs=None,
        cacher_kwargs=None,
    ):
        """Collect data files from a dir to one output file.

        TODO: This assumes that if there is any spatial chunking it is split
        by latitude. This should be generalized to allow for any spatial
        chunking and any dimension. This will either require a new file
        naming scheme with a spatial index for both latitude and
        longitude or checking each chunk to see how they are split.

        Filename requirements:
         - Should end with ".nc"

        Parameters
        ----------
        file_paths : list | str
            Explicit list of str file paths that will be sorted and collected
            or a single string with unix-style /search/patt*ern.nc.
        out_file : str
            File path of final output file.
        features : list | str
            List of dsets to collect. If 'all' then all ``data_vars`` will be
            collected.
        log_level : str | None
            Desired log level, None will not initialize logging.
        log_file : str | None
            Target log file. None logs to stdout.
        write_status : bool
            Flag to write status file once complete if running from pipeline.
        job_name : str
            Job name for status file if running from pipeline.
        overwrite : bool
            Whether to overwrite existing output file
        res_kwargs : dict | None
            Dictionary of kwargs to pass to xarray.open_mfdataset.
        cacher_kwargs : dict | None
            Dictionary of kwargs to pass to Cacher._write_single.
        """
        logger.info(f'Initializing collection for file_paths={file_paths}')

        if log_level is not None:
            init_logger(
                'sup3r.preprocessing', log_file=log_file, log_level=log_level
            )

        if not os.path.exists(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file), exist_ok=True)

        collector = cls(file_paths)
        logger.info(
            'Collecting {} files to {}'.format(len(collector.flist), out_file)
        )
        if overwrite and os.path.exists(out_file):
            logger.info(f'overwrite=True, removing {out_file}.')
            os.remove(out_file)

        spatial_chunks = collector.group_spatial_chunks()

        if not os.path.exists(out_file):
            res_kwargs = res_kwargs or {
                'combine': 'nested',
                'concat_dim': Dimension.TIME,
            }
            for s_idx in spatial_chunks:
                spatial_chunks[s_idx] = xr_open_mfdataset(
                    spatial_chunks[s_idx], **res_kwargs
                )
            out = xr.concat(spatial_chunks.values(), dim=Dimension.SOUTH_NORTH)
            cacher_kwargs = cacher_kwargs or {}
            Cacher._write_single(
                out_file=out_file,
                data=out,
                features=features,
                **cacher_kwargs,
            )

        logger.info('Finished file collection.')

    def group_spatial_chunks(self):
        """Group same spatial chunks together so each entry has same spatial
        footprint but different times"""
        chunks = {}
        for file in self.flist:
            _, s_idx = self.get_chunk_indices(file)
            chunks[s_idx] = [*chunks.get(s_idx, []), file]
        for k, v in chunks.items():
            chunks[k] = sorted(v)
        return chunks
