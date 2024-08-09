"""NETCDF file collection."""

import logging
import os
import time

import xarray as xr
from gaps import Status
from rex.utilities.loggers import init_logger

from sup3r.preprocessing.utilities import _lowered

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
        write_status=False,
        job_name=None,
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
        t0 = time.time()

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
            out = xr.open_mfdataset(collector.flist, **res_kwargs)
            features = list(out.data_vars) if features == 'all' else features
            features = set(features).intersection(_lowered(out.data_vars))
            for feat in features:
                mode = 'a' if os.path.exists(tmp_file) else 'w'
                out[feat].load().to_netcdf(
                    tmp_file, mode=mode, engine='h5netcdf', format='NETCDF4'
                )
                logger.info(f'Finished writing {feat} to {tmp_file}.')

        if write_status and job_name is not None:
            status = {
                'out_dir': os.path.dirname(out_file),
                'fout': out_file,
                'flist': collector.flist,
                'job_status': 'successful',
                'runtime': (time.time() - t0) / 60,
            }
            Status.make_single_job_file(
                os.path.dirname(out_file), 'collect', job_name, status
            )
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
