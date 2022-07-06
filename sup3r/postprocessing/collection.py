# -*- coding: utf-8 -*-
"""H5 file collection."""
from concurrent.futures import as_completed
import logging
import numpy as np
import os
import pandas as pd
import psutil
import time
import glob

from rex.utilities.execution import SpawnProcessPool
from rex.utilities.loggers import init_logger
from rex import Outputs
from rex.utilities.fun_utils import get_fun_call_str

from sup3r.pipeline import Status
from sup3r.postprocessing.file_handling import H5_ATTRS
from sup3r.preprocessing.feature_handling import Feature

logger = logging.getLogger(__name__)


def get_dset_attrs(feature):
    """Get attrributes for output feature

    Parameters
    ----------
    feature : str
        Name of feature to write

    Returns
    -------
    attrs : dict
        Dictionary of attributes for requested dset
    dtype : str
        Data type for requested dset
    """
    attrs = H5_ATTRS[Feature.get_basename(feature)]
    dtype = attrs['dtype']
    return attrs, dtype


class Collector:
    """Sup3r H5 file collection framework"""

    def __init__(self, file_paths):
        """
        Parameters
        ----------
        file_paths : list | str
            Explicit list of str file paths that will be sorted and collected
            or a single string with unix-style /search/patt*ern.h5. Files
            should have non-overlapping time_index dataset and fully
            overlapping meta dataset.
        """
        if not isinstance(file_paths, list):
            file_paths = glob.glob(file_paths)
        self.flist = sorted(file_paths)

    @classmethod
    def get_node_cmd(cls, config):
        """Get a CLI call to collect data.

        Parameters
        ----------
        config : dict
            sup3r collection config with all necessary args and kwargs to
            run data collection.
        """

        import_str = ('from sup3r.postprocessing.collection '
                      'import Collector; from rex import init_logger')

        dc_fun_str = get_fun_call_str(cls.collect, config)

        log_file = config.get('log_file', None)
        log_level = config.get('log_level', 'INFO')
        log_arg_str = '\"sup3r\", '
        log_arg_str += f'log_file=\"{log_file}\", '
        log_arg_str += f'log_level=\"{log_level}\"'

        cmd = (f"python -c \'{import_str};\n"
               f"logger = init_logger({log_arg_str});\n"
               f"{dc_fun_str};\'\n").replace('\\', '/')

        return cmd

    @staticmethod
    def get_slices(final_time_index, final_meta, new_time_index, new_meta):
        """Get index slices where the new ti/meta belong in the final ti/meta.

        Parameters
        ----------
        final_time_index : pd.Datetimeindex
            Time index of the final file that new_time_index is being written
            to.
        final_meta : pd.DataFrame
            Meta data of the final file that new_meta is being written to.
        new_time_index : pd.Datetimeindex
            Chunk time index that is a subset of the final_time_index.
        new_meta : pd.DataFrame
            Chunk meta data that is a subset of the final_meta.

        Returns
        -------
        row_slice : slice
            final_time_index[row_slice] = new_time_index
        col_slice : slice
            final_meta[col_slice] = new_meta
        """

        final_index = final_meta.index
        new_index = new_meta.index

        row_loc = np.where(final_time_index.isin(new_time_index))[0]
        col_loc = np.where(final_index.isin(new_index))[0]

        if not any(row_loc):
            msg = ('Could not find row locations in file collection. '
                   'New time index: {} final time index: {}'
                   .format(new_time_index, final_time_index))
            logger.error(msg)
            raise RuntimeError(msg)

        if not any(col_loc):
            msg = ('Could not find col locations in file collection. '
                   'New index: {} final index: {}'
                   .format(new_index, final_index))
            logger.error(msg)
            raise RuntimeError(msg)

        row_slice = slice(np.min(row_loc), np.max(row_loc) + 1)
        col_slice = slice(np.min(col_loc), np.max(col_loc) + 1)

        return row_slice, col_slice

    @staticmethod
    def get_data(file_path, feature, time_index, meta, scale_factor, dtype):
        """Retreive a data array from a chunked file.

        Parameters
        ----------
        file_path : str
            h5 file to get data from
        feature : str
            dataset to retrieve data from fpath.
        time_index : pd.Datetimeindex
            Time index of the final file.
        final_meta : pd.DataFrame
            Meta data of the final file.
        scale_factor : int | float
            Final destination scale factor after collection. If the data
            retrieval from the files to be collected has a different scale
            factor, the collected data will be rescaled and returned as
            float32.
        dtype : np.dtype
            Final dtype to return data as

        Returns
        -------
        f_data : np.ndarray
            Data array from the fpath cast as input dtype.
        row_slice : slice
            final_time_index[row_slice] = new_time_index
        col_slice : slice
            final_meta[col_slice] = new_meta
        """

        with Outputs(file_path, unscale=False, mode='r') as f:
            f_ti = f.time_index
            f_meta = f.meta
            source_scale_factor = f.attrs[feature].get('scale_factor', 1)

            if feature not in f.dsets:
                e = ('Trying to collect dataset "{}" but cannot find in '
                     'available: {}'.format(feature, f.dsets))
                logger.error(e)
                raise KeyError(e)

            f_data = f[feature][...]

        row_slice, col_slice = Collector.get_slices(time_index, meta,
                                                    f_ti, f_meta)

        if scale_factor != source_scale_factor:
            f_data = f_data.astype(np.float32)
            f_data *= (scale_factor / source_scale_factor)

        if np.issubdtype(dtype, np.integer):
            f_data = np.round(f_data)

        f_data = f_data.astype(dtype)

        return f_data, row_slice, col_slice

    @staticmethod
    def _get_collection_attrs(file_paths, feature, sort=True, sort_key=None):
        """Get important dataset attributes from a file list to be collected.

        Assumes the file list is chunked in time (row chunked).

        Parameters
        ----------
        file_paths : list | str
            Explicit list of str file paths that will be sorted and collected
            or a single string with unix-style /search/patt*ern.h5. Files
            should have non-overlapping time_index dataset and fully
            overlapping meta dataset.
        feature : str
            Dataset name to collect.
        sort : bool
            flag to sort flist to determine meta data order.
        sort_key : None | fun
            Optional sort key to sort flist by (determines how meta is built
            if f_out does not exist).

        Returns
        -------
        time_index : pd.datetimeindex
            Concatenated full size datetime index from the flist that is
            being collected
        meta : pd.DataFrame
            Concatenated full size meta data from the flist that is being
            collected
        shape : tuple
            Output (collected) dataset shape
        dtype : str
            Dataset output (collected on disk) dataset data type.
        """

        if sort:
            file_paths = sorted(file_paths, key=sort_key)

        time_index = None
        meta = []
        for fn in file_paths:
            with Outputs(fn, mode='r') as f:
                meta.append(f.meta)

                if time_index is None:
                    time_index = f.time_index
                else:
                    time_index = time_index.append(f.time_index)

        time_index = time_index.sort_values()
        time_index = time_index.drop_duplicates()
        meta = pd.concat(meta)

        if 'latitude' in meta and 'longitude' in meta:
            meta = meta.drop_duplicates(subset=['latitude', 'longitude'])

        shape = (len(time_index), len(meta))

        with Outputs(file_paths[0], mode='r') as fin:
            dtype = fin.get_dset_properties(feature)[1]

        return time_index, meta, shape, dtype

    @staticmethod
    def _init_collected_h5(f_out, time_index, meta):
        """Initialize the output h5 file to save collected data to.

        Parameters
        ----------
        f_out : str
            Output file path - must not yet exist.
        time_index : pd.datetimeindex
            Full datetime index of collected data.
        meta : pd.DataFrame
            Full meta dataframe collected data.
        """

        with Outputs(f_out, mode='w-') as f:
            logger.info('Initializing collection output file: {}'
                        .format(f_out))
            logger.info('Initializing collection output file with shape {} '
                        'and meta data:\n{}'
                        .format((len(time_index), len(meta)), meta))
            f['time_index'] = time_index
            f['meta'] = meta

    @staticmethod
    def _ensure_dset_in_output(f_out, dset, data=None):
        """Ensure that dset is initialized in f_out and initialize if not.
        Parameters
        ----------
        f_out : str
            Pre-existing H5 file output path
        dset : str
            Dataset name
        data : np.ndarray | None
            Optional data to write to dataset if initializing.
        """

        with Outputs(f_out, mode='a') as f:
            if dset not in f.dsets:
                attrs, dtype = get_dset_attrs(dset)
                logger.info('Initializing dataset "{}" with shape {} and '
                            'dtype {}'.format(dset, f.shape, dtype))
                f._create_dset(dset, f.shape, dtype,
                               attrs=attrs, data=data)

    @staticmethod
    def collect_flist(file_paths, f_out, feature, sort=False, sort_key=None,
                      max_workers=None):
        """Collect a dataset from a file list with data pre-init.

        Collects data that can be chunked in both space and time.

        Parameters
        ----------
        file_paths : list | str
            Explicit list of str file paths that will be sorted and collected
            or a single string with unix-style /search/patt*ern.h5. Files
            should have non-overlapping time_index dataset and fully
            overlapping meta dataset.
        collect_dir : str
            Directory of chunked files (flist).
        f_out : str
            File path of final output file.
        dset : str
            Dataset name to collect.
        sort : bool
            flag to sort flist to determine meta data order.
        sort_key : None | fun
            Optional sort key to sort flist by (determines how meta is built
            if f_out does not exist).
        max_workers : int | None
            Number of workers to use in parallel. 1 runs serial,
            None uses all available.
        """

        time_index, meta, shape, _ = \
            Collector._get_collection_attrs(file_paths, feature, sort=sort,
                                            sort_key=sort_key)

        attrs, final_dtype = get_dset_attrs(feature)
        scale_factor = attrs.get('scale_factor', 1)

        logger.debug('Collecting file list of shape {}: {}'
                     .format(shape, file_paths))

        data = np.zeros(shape, dtype=final_dtype)
        mem = psutil.virtual_memory()
        logger.debug('Initializing output dataset "{0}" in-memory with shape '
                     '{1} and dtype {2}. Current memory usage is '
                     '{3:.3f} GB out of {4:.3f} GB total.'
                     .format(feature, shape, final_dtype,
                             mem.used / 1e9, mem.total / 1e9))

        if max_workers == 1:
            for i, fname in enumerate(file_paths):
                logger.debug('Collecting data from file {} out of {}.'
                             .format(i + 1, len(file_paths)))
                f_data, row_slice, col_slice = Collector.get_data(fname,
                                                                  feature,
                                                                  time_index,
                                                                  meta,
                                                                  scale_factor,
                                                                  final_dtype)
                data[row_slice, col_slice] = f_data
        else:
            logger.info('Running parallel collection on {} workers.'
                        .format(max_workers))

            futures = []
            completed = 0
            loggers = ['sup3r']
            with SpawnProcessPool(loggers=loggers,
                                  max_workers=max_workers) as exe:
                for fname in file_paths:
                    futures.append(exe.submit(Collector.get_data, fname,
                                              feature, time_index, meta,
                                              scale_factor, final_dtype))
                for future in as_completed(futures):
                    completed += 1
                    mem = psutil.virtual_memory()
                    logger.info('Collection futures completed: '
                                '{0} out of {1}. '
                                'Current memory usage is '
                                '{2:.3f} GB out of {3:.3f} GB total.'
                                .format(completed, len(futures),
                                        mem.used / 1e9, mem.total / 1e9))
                    f_data, row_slice, col_slice = future.result()
                    data[row_slice, col_slice] = f_data

        if not os.path.exists(f_out):
            Collector._init_collected_h5(f_out, time_index, meta)
            x_write_slice, y_write_slice = slice(None), slice(None)
        else:
            with Outputs(f_out, 'r') as f:
                target_meta = f.meta
                target_ti = f.time_index
            y_write_slice, x_write_slice = Collector.get_slices(target_ti,
                                                                target_meta,
                                                                time_index,
                                                                meta)
        Collector._ensure_dset_in_output(f_out, feature)
        with Outputs(f_out, mode='a') as f:
            f[feature, y_write_slice, x_write_slice] = data

        logger.debug('Finished writing "{}" for row {} and col {} to: {}'
                     .format(feature, y_write_slice, x_write_slice,
                             os.path.basename(f_out)))

    @classmethod
    def collect(cls, file_paths, f_out, features, n_writes=1,
                max_workers=None, log_level=None, log_file=None,
                write_status=False, job_name=None):
        """Collect data files from a dir to one output file.

        Assumes the file list is chunked in time (row chunked).

        Filename requirements:
         - Should end with ".h5"

        Parameters
        ----------
        file_paths : list | str
            Explicit list of str file paths that will be sorted and collected
            or a single string with unix-style /search/patt*ern.h5. Files
            should have non-overlapping time_index dataset and fully
            overlapping meta dataset.
        f_out : str
            File path of final output file.
        features : list
            List of dsets to collect
        n_writes : None | int
            Number of file list divisions to write per dataset. For example,
            if ghi and dni are being collected and n_writes is set to 2,
            half of the source ghi files will be collected at once and then
            written, then the second half of ghi files, then dni.
        max_workers : int | None
            Number of workers to use in parallel. 1 runs serial,
            None will use all available workers.
        log_level : str | None
            Desired log level, None will not initialize logging.
        log_file : str | None
            Target log file. None logs to stdout.
        write_status : bool
            Flag to write status file once complete if running from pipeline.
        job_name : str
            Job name for status file if running from pipeline.
        """
        t0 = time.time()

        if log_level is not None:
            init_logger('sup3r.preprocessing', log_file=log_file,
                        log_level=log_level)

        logger.info('Collecting data from {} to {}'.format(file_paths, f_out))

        collector = cls(file_paths)
        for _, dset in enumerate(features):
            logger.debug('Collecting dataset "{}".'.format(dset))
            if n_writes > len(collector.flist):
                e = ('Cannot split file list of length {} into '
                     '{} write chunks!'
                     .format(len(collector.flist), n_writes))
                logger.error(e)
                raise ValueError(e)

            if not os.path.exists(f_out):
                time_index, meta, _, _ = \
                    collector._get_collection_attrs(collector.flist, dset)
                collector._init_collected_h5(f_out, time_index, meta)

            flist_chunks = np.array_split(np.array(collector.flist),
                                          n_writes)
            flist_chunks = [fl.tolist() for fl in flist_chunks]
            for j, flist in enumerate(flist_chunks):
                logger.info('Collecting file list chunk {} out of {} '
                            .format(j + 1, len(flist_chunks)))
                collector.collect_flist(flist, f_out, dset,
                                        max_workers=max_workers)

        if write_status and job_name is not None:
            status = {'out_dir': os.path.dirname(f_out),
                      'fout': f_out,
                      'flist': file_paths,
                      'job_status': 'successful',
                      'runtime': (time.time() - t0) / 60}
            Status.make_job_file(os.path.dirname(f_out), 'collect',
                                 job_name, status)

        logger.info('Finished file collection.')
