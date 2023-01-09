# -*- coding: utf-8 -*-
"""H5 file collection."""
from concurrent.futures import as_completed, ThreadPoolExecutor
import logging
import numpy as np
import os
import pandas as pd
import psutil
import time
import glob
from warnings import warn
from scipy.spatial import KDTree

from rex.utilities.loggers import init_logger
from rex.utilities.fun_utils import get_fun_call_str

from sup3r.pipeline import Status
from sup3r.postprocessing.file_handling import RexOutputs, OutputMixIn
from sup3r.utilities import ModuleName
from sup3r.utilities.cli import BaseCLI

logger = logging.getLogger(__name__)


class Collector(OutputMixIn):
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
        self.data = None

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
                      'import Collector;\n'
                      'from rex import init_logger;\n'
                      'import time;\n'
                      'from reV.pipeline.status import Status;\n')

        dc_fun_str = get_fun_call_str(cls.collect, config)

        log_file = config.get('log_file', None)
        log_level = config.get('log_level', 'INFO')
        log_arg_str = (f'"sup3r", log_level="{log_level}"')
        if log_file is not None:
            log_arg_str += f', log_file="{log_file}"'

        cmd = (f"python -c \'{import_str}\n"
               "t0 = time.time();\n"
               f"logger = init_logger({log_arg_str});\n"
               f"{dc_fun_str};\n"
               "t_elap = time.time() - t0;\n"
               )

        cmd = BaseCLI.add_status_cmd(config, ModuleName.DATA_COLLECT, cmd)

        cmd += (";\'\n")

        return cmd.replace('\\', '/')

    @classmethod
    def get_slices(cls, final_time_index, final_meta, new_time_index,
                   new_meta):
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
        col_loc = np.where(final_meta['gid'].isin(new_meta['gid']))[0]

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

        return row_slice, col_loc

    @classmethod
    def get_coordinate_indices(cls, target_meta, full_meta, threshold=1e-4):
        """Get coordindate indices in meta data for given targets

        Parameters
        ----------
        target_meta : pd.DataFrame
            Dataframe of coordinates to find within the full meta
        full_meta : pd.DataFrame
            Dataframe of full set of coordinates for unfiltered dataset
        threshold : float
            Threshold distance for finding target coordinates within full meta
        """
        ll2 = np.vstack((full_meta.latitude.values,
                         full_meta.longitude.values)).T
        tree = KDTree(ll2)

        targets = np.vstack((target_meta.latitude.values,
                             target_meta.longitude.values)).T
        _, indices = tree.query(targets, distance_upper_bound=threshold)
        indices = indices[indices < len(full_meta)]
        return indices

    def get_data(self, file_path, feature, time_index, meta, scale_factor,
                 dtype, threshold=1e-4):
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
        threshold : float
            Threshold distance for finding target coordinates within full meta

        Returns
        -------
        f_data : np.ndarray
            Data array from the fpath cast as input dtype.
        row_slice : slice
            final_time_index[row_slice] = new_time_index
        col_slice : slice
            final_meta[col_slice] = new_meta
        """

        with RexOutputs(file_path, unscale=False, mode='r') as f:
            f_ti = f.time_index
            f_meta = f.meta
            source_scale_factor = f.attrs[feature].get('scale_factor', 1)

            if feature not in f.dsets:
                e = ('Trying to collect dataset "{}" but cannot find in '
                     'available: {}'.format(feature, f.dsets))
                logger.error(e)
                raise KeyError(e)

            mask = self.get_coordinate_indices(meta, f_meta,
                                               threshold=threshold)
            f_meta = f_meta.iloc[mask]
            f_data = f[feature][:, mask]

        if len(mask) == 0:
            msg = ('No target coordinates found in masked meta. '
                   f'Skipping collection for {file_path}.')
            logger.warning(msg)
            warn(msg)

        else:
            row_slice, col_slice = Collector.get_slices(time_index, meta,
                                                        f_ti, f_meta)

            if scale_factor != source_scale_factor:
                f_data = f_data.astype(np.float32)
                f_data *= (scale_factor / source_scale_factor)

            if np.issubdtype(dtype, np.integer):
                f_data = np.round(f_data)

            f_data = f_data.astype(dtype)
            self.data[row_slice, col_slice] = f_data

    @staticmethod
    def _get_file_attrs(file):
        """Get meta data and time index for a single file"""
        with RexOutputs(file, mode='r') as f:
            meta = f.meta
            time_index = f.time_index
        return meta, time_index

    @classmethod
    def _get_collection_attrs_parallel(cls, file_paths, max_workers=None):
        """Get meta data and time index from a file list to be collected.

        Assumes the file list is chunked in time (row chunked).

        Parameters
        ----------
        file_paths : list | str
            Explicit list of str file paths that will be sorted and collected
            or a single string with unix-style /search/patt*ern.h5. Files
            should have non-overlapping time_index dataset and fully
            overlapping meta dataset.
        max_workers : int | None
            Number of workers to use in parallel. 1 runs serial,
            None will use all available workers.

        Returns
        -------
        time_index : pd.datetimeindex
            List of datetime indices for each file that is being collected
        meta : pd.DataFrame
            List of meta data for each files that is being
            collected
        """

        time_index = [None] * len(file_paths)
        meta = [None] * len(file_paths)
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            for i, fn in enumerate(file_paths):
                future = exe.submit(cls._get_file_attrs, fn)
                futures[future] = i

            interval = int(np.ceil(len(futures) / 10))
            for i, future in enumerate(as_completed(futures)):
                if i % interval == 0:
                    mem = psutil.virtual_memory()
                    logger.info('Meta collection futures completed: '
                                '{0} out of {1}. '
                                'Current memory usage is '
                                '{2:.3f} GB out of {3:.3f} GB total.'
                                .format(i + 1, len(futures),
                                        mem.used / 1e9, mem.total / 1e9))
                try:
                    idx = futures[future]
                    meta[idx], time_index[idx] = future.result()
                except Exception as e:
                    msg = ('Falied to get attrs from '
                           f'{file_paths[futures[future]]}')
                    logger.exception(msg)
                    raise RuntimeError(msg) from e
        return meta, time_index

    @classmethod
    def _get_collection_attrs(cls, file_paths, sort=True,
                              sort_key=None, max_workers=None,
                              target_final_meta_file=None, threshold=1e-4):
        """Get important dataset attributes from a file list to be collected.

        Assumes the file list is chunked in time (row chunked).

        Parameters
        ----------
        file_paths : list | str
            Explicit list of str file paths that will be sorted and collected
            or a single string with unix-style /search/patt*ern.h5.
        sort : bool
            flag to sort flist to determine meta data order.
        sort_key : None | fun
            Optional sort key to sort flist by (determines how meta is built
            if out_file does not exist).
        max_workers : int | None
            Number of workers to use in parallel. 1 runs serial,
            None will use all available workers.
        target_final_meta_file : str
            Path to target final meta containing coordinates to keep from the
            full list of coordinates present in the collected meta for the full
            file list.
        threshold : float
            Threshold distance for finding target coordinates within full meta

        Returns
        -------
        time_index : pd.datetimeindex
            Concatenated full size datetime index from the flist that is
            being collected
        target_final_meta : pd.DataFrame
            Concatenated full size meta data from the flist that is being
            collected or provided target meta
        masked_meta : pd.DataFrame
            Concatenated full size meta data from the flist that is being
            collected masked against target_final_meta
        shape : tuple
            Output (collected) dataset shape
        global_attrs : dict
            Global attributes from the first file in file_paths (it's assumed
            that all the files in file_paths have the same global file
            attributes).
        """

        if sort:
            file_paths = sorted(file_paths, key=sort_key)

        logger.info('Getting collection attrs for full dataset')

        if max_workers == 1:
            meta = []
            time_index = None
            for i, fn in enumerate(file_paths):
                with RexOutputs(fn, mode='r') as f:
                    meta.append(f.meta)

                    if time_index is None:
                        time_index = f.time_index
                    else:
                        time_index = time_index.append(f.time_index)
                logger.debug(f'{i+1} / {len(file_paths)} files finished')
        else:
            meta, time_index = cls._get_collection_attrs_parallel(
                file_paths, max_workers=max_workers)
            time_index = pd.DatetimeIndex(np.concatenate(time_index))

        time_index = time_index.sort_values()
        time_index = time_index.drop_duplicates()
        meta = pd.concat(meta)

        if 'latitude' in meta and 'longitude' in meta:
            meta = meta.drop_duplicates(subset=['latitude', 'longitude'])

        meta = meta.sort_values('gid')

        if (target_final_meta_file is not None
                and os.path.exists(target_final_meta_file)):
            target_final_meta = pd.read_csv(target_final_meta_file)
            if 'gid' in target_final_meta.columns:
                target_final_meta = target_final_meta.drop('gid', axis=1)
            mask = cls.get_coordinate_indices(target_final_meta, meta,
                                              threshold=threshold)
            masked_meta = meta.iloc[mask]
            logger.info(f'Masked meta coordinates: {len(masked_meta)}')
            mask = cls.get_coordinate_indices(masked_meta, target_final_meta,
                                              threshold=threshold)
            target_final_meta = target_final_meta.iloc[mask]
            logger.info(f'Target meta coordinates: {len(target_final_meta)}')
        else:
            target_final_meta = masked_meta = meta

        shape = (len(time_index), len(target_final_meta))

        with RexOutputs(file_paths[0], mode='r') as fin:
            global_attrs = fin.global_attrs

        return time_index, target_final_meta, masked_meta, shape, global_attrs

    def _collect_flist(self, feature, subset_masked_meta, time_index, shape,
                       file_paths, out_file, target_masked_meta,
                       max_workers=None):
        """Collect a dataset from a file list without getting attributes first.
        This file list can be a subset of a full file list to be collected.

        Parameters
        ----------
        feature : str
            Dataset name to collect.
        subset_masked_meta : pd.DataFrame
            Meta data containing the list of coordinates present in both the
            given file paths and the target_final_meta. This can be a subset of
            the coordinates present in the full file list. The coordinates
            contained in this dataframe have the same gids as those present in
            the meta for the full file list.
        time_index : pd.datetimeindex
            Concatenated datetime index for the given file paths.
        shape : tuple
            Output (collected) dataset shape
        file_paths : list | str
            File list to be collected. This can be a subset of a full file list
            to be collected.
        out_file : str
            File path of final output file.
        target_masked_meta : pd.DataFrame
            Same as subset_masked_meta but instead for the entire list of files
            to be collected.
        max_workers : int | None
            Number of workers to use in parallel. 1 runs serial,
            None uses all available.
        """
        if len(subset_masked_meta) > 0:
            attrs, final_dtype = self.get_dset_attrs(feature)
            scale_factor = attrs.get('scale_factor', 1)

            logger.debug('Collecting file list of shape {}: {}'
                         .format(shape, file_paths))

            self.data = np.zeros(shape, dtype=final_dtype)
            mem = psutil.virtual_memory()
            logger.debug('Initializing output dataset "{0}" in-memory with '
                         'shape {1} and dtype {2}. Current memory usage is '
                         '{3:.3f} GB out of {4:.3f} GB total.'
                         .format(feature, shape, final_dtype,
                                 mem.used / 1e9, mem.total / 1e9))

            if max_workers == 1:
                for i, fname in enumerate(file_paths):
                    logger.debug('Collecting data from file {} out of {}.'
                                 .format(i + 1, len(file_paths)))
                    self.get_data(fname, feature, time_index,
                                  subset_masked_meta, scale_factor,
                                  final_dtype)
            else:
                logger.info('Running parallel collection on {} workers.'
                            .format(max_workers))

                futures = {}
                completed = 0
                with ThreadPoolExecutor(max_workers=max_workers) as exe:
                    for fname in file_paths:
                        future = exe.submit(self.get_data, fname, feature,
                                            time_index, subset_masked_meta,
                                            scale_factor, final_dtype)
                        futures[future] = fname
                    for future in as_completed(futures):
                        completed += 1
                        mem = psutil.virtual_memory()
                        logger.info('Collection futures completed: '
                                    '{0} out of {1}. '
                                    'Current memory usage is '
                                    '{2:.3f} GB out of {3:.3f} GB total.'
                                    .format(completed, len(futures),
                                            mem.used / 1e9, mem.total / 1e9))
                        try:
                            future.result()
                        except Exception as e:
                            msg = 'Failed to collect data from '
                            msg += f'{futures[future]}'
                            logger.exception(msg)
                            raise RuntimeError(msg) from e
            with RexOutputs(out_file, mode='r') as f:
                target_ti = f.time_index
                y_write_slice, x_write_slice = Collector.get_slices(
                    target_ti, target_masked_meta, time_index,
                    subset_masked_meta)
            Collector._ensure_dset_in_output(out_file, feature)
            with RexOutputs(out_file, mode='a') as f:
                f[feature, y_write_slice, x_write_slice] = self.data

            logger.debug('Finished writing "{}" for row {} and col {} to: {}'
                         .format(feature, y_write_slice, x_write_slice,
                                 os.path.basename(out_file)))
        else:
            msg = ('No target coordinates found in masked meta. Skipping '
                   f'collection for {file_paths}.')
            logger.warning(msg)
            warn(msg)

    @classmethod
    def group_time_chunks(cls, file_paths, n_writes=None):
        """Group files by temporal_chunk_index. Assumes file_paths have a
        suffix format like _{temporal_chunk_index}_{spatial_chunk_index}.h5

        Parameters
        ----------
        file_paths : list
            List of file paths each with a suffix
            _{temporal_chunk_index}_{spatial_chunk_index}.h5
        n_writes : int | None
            Number of writes to use for collection

        Returns
        -------
        file_chunks : list
            List of lists of file paths groups by temporal_chunk_index
        """
        file_split = {}
        for file in file_paths:
            t_chunk = file.split('_')[-2]
            file_split[t_chunk] = file_split.get(t_chunk, []) + [file]
        file_chunks = []
        for files in file_split.values():
            file_chunks.append(files)

        logger.debug(f'Split file list into {len(file_chunks)} chunks '
                     'according to temporal chunk indices')

        if n_writes is not None:
            msg = (f'n_writes ({n_writes}) must be less than or equal '
                   f'to the number of temporal chunks ({len(file_chunks)}).')
            assert n_writes < len(file_chunks), msg
        return file_chunks

    def get_flist_chunks(self, file_paths, n_writes=None, join_times=False):
        """Get file list chunks based on n_writes

        Parameters
        ----------
        file_paths : list
            List of file paths to collect
        n_writes : int | None
            Number of writes to use for collection
        join_times : bool
            Option to split full file list into chunks with each chunk having
            the same temporal_chunk_index. The number of writes will then be
            min(number of temporal chunks, n_writes). This ensures that each
            write has all the spatial chunks for a given time index. Assumes
            file_paths have a suffix format
            _{temporal_chunk_index}_{spatial_chunk_index}.h5.  This is required
            if there are multiple writes and chunks have different time
            indices.

        Returns
        -------
        flist_chunks : list
            List of file list chunks. Used to split collection and writing into
            multiple steps.
        """
        if join_times:
            flist_chunks = self.group_time_chunks(file_paths,
                                                  n_writes=n_writes)
        else:
            flist_chunks = [[f] for f in file_paths]

        if n_writes is not None:
            flist_chunks = np.array_split(flist_chunks, n_writes)
            flist_chunks = [np.concatenate(fp_chunk)
                            for fp_chunk in flist_chunks]
            logger.debug(f'Split file list into {len(flist_chunks)} '
                         f'chunks according to n_writes={n_writes}')
        return flist_chunks

    @classmethod
    def collect(cls, file_paths, out_file, features, max_workers=None,
                log_level=None, log_file=None, write_status=False,
                job_name=None, join_times=False, target_final_meta_file=None,
                n_writes=None, overwrite=True, threshold=1e-4):
        """Collect data files from a dir to one output file.

        Assumes the file list is chunked in time (row chunked).

        Filename requirements:
         - Should end with ".h5"

        Parameters
        ----------
        file_paths : list | str
            Explicit list of str file paths that will be sorted and collected
            or a single string with unix-style /search/patt*ern.h5.
        out_file : str
            File path of final output file.
        features : list
            List of dsets to collect
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
        join_times : bool
            Option to split full file list into chunks with each chunk having
            the same temporal_chunk_index. The number of writes will then be
            min(number of temporal chunks, n_writes). This ensures that each
            write has all the spatial chunks for a given time index. Assumes
            file_paths have a suffix format
            _{temporal_chunk_index}_{spatial_chunk_index}.h5.  This is required
            if there are multiple writes and chunks have different time
            indices.
        target_final_meta_file : str
            Path to target final meta containing coordinates to keep from the
            full file list collected meta. This can be but is not necessarily a
            subset of the full list of coordinates for all files in the file
            list. This is used to remove coordinates from the full file list
            which are not present in the target_final_meta. Either this full
            meta or a subset, depending on which coordinates are present in
            the data to be collected, will be the final meta for the collected
            output files.
        n_writes : int | None
            Number of writes to split full file list into. Must be less than
            or equal to the number of temporal chunks if chunks have different
            time indices.
        overwrite : bool
            Whether to overwrite existing output file
        threshold : float
            Threshold distance for finding target coordinates within full meta
        """
        t0 = time.time()

        if log_level is not None:
            init_logger('sup3r.preprocessing', log_file=log_file,
                        log_level=log_level)

        logger.info(f'Using target_final_meta_file={target_final_meta_file}')

        if not os.path.exists(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file), exist_ok=True)

        collector = cls(file_paths)
        logger.info('Collecting {} files to {}'.format(len(collector.flist),
                                                       out_file))
        if overwrite and os.path.exists(out_file):
            logger.info(f'overwrite=True, removing {out_file}.')
            os.remove(out_file)

        out = collector._get_collection_attrs(
            collector.flist, max_workers=max_workers,
            target_final_meta_file=target_final_meta_file,
            threshold=threshold)
        time_index, target_final_meta, target_masked_meta = out[:3]
        shape, global_attrs = out[3:]

        for _, dset in enumerate(features):
            logger.debug('Collecting dataset "{}".'.format(dset))
            if join_times or n_writes is not None:
                flist_chunks = collector.get_flist_chunks(
                    collector.flist, n_writes=n_writes, join_times=join_times)
            else:
                flist_chunks = [collector.flist]

            if not os.path.exists(out_file):
                collector._init_h5(out_file, time_index, target_final_meta,
                                   global_attrs)

            if len(flist_chunks) == 1:
                collector._collect_flist(dset, target_masked_meta, time_index,
                                         shape, flist_chunks[0], out_file,
                                         target_masked_meta,
                                         max_workers=max_workers)

            else:
                for j, flist in enumerate(flist_chunks):
                    logger.info('Collecting file list chunk {} out of {} '
                                .format(j + 1, len(flist_chunks)))
                    time_index, target_final_meta, masked_meta, shape, _ = \
                        collector._get_collection_attrs(
                            flist,
                            target_final_meta_file=target_final_meta_file,
                            threshold=threshold)
                    collector._collect_flist(dset, masked_meta, time_index,
                                             shape, flist, out_file,
                                             target_masked_meta,
                                             max_workers=max_workers)

        if write_status and job_name is not None:
            status = {'out_dir': os.path.dirname(out_file),
                      'fout': out_file,
                      'flist': collector.flist,
                      'job_status': 'successful',
                      'runtime': (time.time() - t0) / 60}
            Status.make_job_file(os.path.dirname(out_file), 'collect',
                                 job_name, status)

        logger.info('Finished file collection.')
