"""Output handling

author : @bbenton
"""
import json
import logging
import os
from warnings import warn

import xarray as xr

from sup3r.postprocessing.file_handling import H5_ATTRS, RexOutputs
from sup3r.preprocessing.feature_handling import Feature

logger = logging.getLogger(__name__)


class OutputMixIn:
    """Methods used by various Output and Collection classes"""

    @staticmethod
    def get_time_dim_name(filepath):
        """Get the name of the time dimension in the given file

        Parameters
        ----------
        filepath : str
            Path to the file

        Returns
        -------
        time_key : str
            Name of the time dimension in the given file
        """

        handle = xr.open_dataset(filepath)
        valid_vars = set(handle.dims)
        time_key = list({'time', 'Time'}.intersection(valid_vars))
        if len(time_key) > 0:
            return time_key[0]
        else:
            return 'time'

    @staticmethod
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
            Data type for requested dset. Defaults to float32
        """
        feat_base_name = Feature.get_basename(feature)
        if feat_base_name in H5_ATTRS:
            attrs = H5_ATTRS[feat_base_name]
            dtype = attrs.get('dtype', 'float32')
        else:
            attrs = {}
            dtype = 'float32'
            msg = ('Could not find feature "{}" with base name "{}" in '
                   'H5_ATTRS global variable. Writing with float32 and no '
                   'chunking.'.format(feature, feat_base_name))
            logger.warning(msg)
            warn(msg)

        return attrs, dtype

    @staticmethod
    def _init_h5(out_file, time_index, meta, global_attrs):
        """Initialize the output h5 file to save data to.

        Parameters
        ----------
        out_file : str
            Output file path - must not yet exist.
        time_index : pd.datetimeindex
            Full datetime index of final output data.
        meta : pd.DataFrame
            Full meta dataframe for the final output data.
        global_attrs : dict
            Namespace of file-global attributes for the final output data.
        """

        with RexOutputs(out_file, mode='w-') as f:
            logger.info('Initializing output file: {}'
                        .format(out_file))
            logger.info('Initializing output file with shape {} '
                        'and meta data:\n{}'
                        .format((len(time_index), len(meta)), meta))
            f.time_index = time_index
            f.meta = meta
            f.run_attrs = global_attrs

    @classmethod
    def _ensure_dset_in_output(cls, out_file, dset, data=None):
        """Ensure that dset is initialized in out_file and initialize if not.

        Parameters
        ----------
        out_file : str
            Pre-existing H5 file output path
        dset : str
            Dataset name
        data : np.ndarray | None
            Optional data to write to dataset if initializing.
        """

        with RexOutputs(out_file, mode='a') as f:
            if dset not in f.dsets:
                attrs, dtype = cls.get_dset_attrs(dset)
                logger.info('Initializing dataset "{}" with shape {} and '
                            'dtype {}'.format(dset, f.shape, dtype))
                f._create_dset(dset, f.shape, dtype,
                               attrs=attrs, data=data,
                               chunks=attrs.get('chunks', None))

    @classmethod
    def write_data(cls, out_file, dsets, time_index, data_list, meta,
                   global_attrs=None):
        """Write list of datasets to out_file.

        Parameters
        ----------
        out_file : str
            Pre-existing H5 file output path
        dsets : list
            list of datasets to write to out_file
        time_index : pd.DatetimeIndex()
            Pandas datetime index to use for file time_index.
        data_list : list
            List of np.ndarray objects to write to out_file
        meta : pd.DataFrame
            Full meta dataframe for the final output data.
        global_attrs : dict
            Namespace of file-global attributes for the final output data.
        """
        tmp_file = out_file.replace('.h5', '.h5.tmp')
        with RexOutputs(tmp_file, 'w') as fh:
            fh.meta = meta
            fh.time_index = time_index

            for dset, data in zip(dsets, data_list):
                attrs, dtype = cls.get_dset_attrs(dset)
                fh.add_dataset(tmp_file, dset, data, dtype=dtype,
                               attrs=attrs, chunks=attrs['chunks'])
                logger.info(f'Added {dset} to output file {out_file}.')

            if global_attrs is not None:
                attrs = {k: v if isinstance(v, str) else json.dumps(v)
                         for k, v in global_attrs.items()}
                fh.run_attrs = attrs

        os.replace(tmp_file, out_file)
        msg = ('Saved output of size '
               f'{(len(data_list), *data_list[0].shape)} to: {out_file}')
        logger.info(msg)
