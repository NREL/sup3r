"""NETCDF file collection.

TODO: Integrate this with Cacher class
"""

import logging
import os

from rex.utilities.loggers import init_logger

from sup3r.preprocessing.cachers import Cacher
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
    ):
        """Collect data files from a dir to one output file.

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

        tmp_file = out_file + '.tmp'
        if not os.path.exists(tmp_file):
            res_kwargs = res_kwargs or {}
            out = xr_open_mfdataset(collector.flist, **res_kwargs)
            Cacher.write_netcdf(tmp_file, data=out, features=features)

        os.replace(tmp_file, out_file)
        logger.info('Moved %s to %s.', tmp_file, out_file)

        logger.info('Finished file collection.')

    def group_spatial_chunks(self):
        """Group same spatial chunks together so each chunk has same spatial
        footprint but different times"""
        chunks = {}
        for file in self.flist:
            s_chunk = file.split('_')[0]
            dirname = os.path.dirname(file)
            s_file = os.path.join(dirname, f's_{s_chunk}.nc')
            chunks[s_file] = [*chunks.get(s_file, []), s_file]
        return chunks
