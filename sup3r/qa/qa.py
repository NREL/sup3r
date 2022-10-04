# -*- coding: utf-8 -*-
"""sup3r QA module."""
import os
import json
import pandas as pd
import numpy as np
import xarray as xr
import logging
from inspect import signature
from rex import Resource
from rex.utilities.fun_utils import get_fun_call_str
import sup3r.bias.bias_transforms
from sup3r.preprocessing.feature_handling import Feature
from sup3r.postprocessing.file_handling import RexOutputs, H5_ATTRS
from sup3r.utilities import ModuleName
from sup3r.utilities.utilities import (get_input_handler_class,
                                       get_source_type,
                                       spatial_coarsening,
                                       temporal_coarsening,
                                       )


logger = logging.getLogger(__name__)


class Sup3rQa:
    """Class for doing QA on sup3r forward pass outputs."""

    def __init__(self, source_file_paths, out_file_path, s_enhance, t_enhance,
                 temporal_coarsening_method,
                 features=None,
                 temporal_slice=slice(None),
                 target=None,
                 shape=None,
                 raster_file=None,
                 qa_fp=None,
                 bias_correct_method=None,
                 bias_correct_kwargs=None,
                 save_sources=True,
                 time_chunk_size=None,
                 cache_pattern=None,
                 overwrite_cache=False,
                 input_handler=None,
                 max_workers=None,
                 extract_workers=None,
                 compute_workers=None,
                 load_workers=None,
                 ti_workers=None):
        """
        Parameters
        ----------
        source_file_paths : list | str
            A list of low-resolution source files to extract raster data from.
            Each file must have the same number of timesteps. Can also pass a
            string with a unix-style file path which will be passed through
            glob.glob
        out_file_path : str
            A single sup3r-resolved output file (either .nc or .h5) with
            high-resolution data corresponding to the
            source_file_paths * s_enhance * t_enhance
        s_enhance : int
            Factor by which the Sup3rGan model will enhance the spatial
            dimensions of low resolution data
        t_enhance : int
            Factor by which the Sup3rGan model will enhance temporal dimension
            of low resolution data
        temporal_coarsening_method : str
            [subsample, average, total]
            Subsample will take every t_enhance-th time step, average will
            average over t_enhance time steps, total will sum over t_enhance
            time steps
        features : list | dict | None
            Explicit list of features to validate. Can be a list of string
            feature names, a dictionary mapping the sup3r output feature name
            to the source_handler feature name (e.g. {'ghi': 'rsds'}), or None
            for all features found in the out_file_path. The dict can also map
            the special case of one sup3r output windspeed name to two
            source_handler windspeed component names so that each windspeed
            component gets bias corrected separately
            (e.g. {'windspeed_100m': ['U_100m', 'V_100m']})
        temporal_slice : slice | tuple | list
            Slice defining size of full temporal domain. e.g. If we have 5
            files each with 5 time steps then temporal_slice = slice(None) will
            select all 25 time steps. This can also be a tuple / list with
            length 3 that will be interpreted as slice(*temporal_slice)
        target : tuple
            (lat, lon) lower left corner of raster. You should provide
            target+shape or raster_file, or if all three are None the full
            source domain will be used.
        shape : tuple
            (rows, cols) grid size. You should provide target+shape or
            raster_file, or if all three are None the full source domain will
            be used.
        raster_file : str | None
            File for raster_index array for the corresponding target and
            shape. If specified the raster_index will be loaded from the file
            if it exists or written to the file if it does not yet exist.
            If None raster_index will be calculated directly. You should
            provide target+shape or raster_file, or if all three are None the
            full source domain will be used.
        qa_fp : str | None
            Optional filepath to output QA file when you call Sup3rQa.run()
            (only .h5 is supported)
        bias_correct_method : str | None
            Optional bias correction function name that can be imported from
            the sup3r.bias.bias_transforms module. This will transform the
            source data according to some predefined bias correction
            transformation along with the bias_correct_kwargs. As the first
            argument, this method must receive a generic numpy array of data to
            be bias corrected
        bias_correct_kwargs : dict | None
            Optional namespace of kwargs to provide to bias_correct_method.
            If this is provided, it must be a dictionary where each key is a
            feature name and each value is a dictionary of kwargs to correct
            that feature. You can bias correct only certain input features by
            only including those feature names in this dict.
        save_sources : bool
            Flag to save re-coarsened synthetic data and true low-res data to
            qa_fp in addition to the error dataset
        time_chunk_size : int
            Size of chunks to split time dimension into for parallel data
            extraction. If running in serial this can be set to the size
            of the full time index for best performance.
        cache_pattern : str | None
            Pattern for files for saving feature data. e.g.
            file_path_{feature}.pkl Each feature will be saved to a file with
            the feature name replaced in cache_pattern. If not None
            feature arrays will be saved here and not stored in self.data until
            load_cached_data is called. The cache_pattern can also include
            {shape}, {target}, {times} which will help ensure unique cache
            files for complex problems.
        overwrite_cache : bool
            Whether to overwrite cache files storing the computed/extracted
            feature data
        input_handler : str | None
            data handler class to use for input data. Provide a string name to
            match a class in data_handling.py. If None the correct handler will
            be guessed based on file type and time series properties.
        max_workers : int | None
            Providing a value for max workers will be used to set the value of
            extract_workers, compute_workers, output_workers, and
            load_workers.  If max_workers == 1 then all processes will be
            serialized. If None extract_workers, compute_workers, load_workers,
            output_workers will use their own provided
            values.
        extract_workers : int | None
            max number of workers to use for extracting features from source
            data.
        compute_workers : int | None
            max number of workers to use for computing derived features from
            raw features in source data.
        load_workers : int | None
            max number of workers to use for loading cached feature data.
        ti_workers : int | None
            max number of workers to use to get full time index. Useful when
            there are many input files each with a single time step. If this is
            greater than one, time indices for input files will be extracted in
            parallel and then concatenated to get the full time index. If input
            files do not all have time indices or if there are few input files
            this should be set to one.
        """

        logger.info('Initializing Sup3rQa and retrieving source data...')

        if max_workers is not None:
            extract_workers = compute_workers = load_workers = max_workers
            ti_workers = max_workers

        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        self._t_meth = temporal_coarsening_method
        self._out_fp = out_file_path
        self._features = features
        self.qa_fp = qa_fp
        self.save_sources = save_sources
        self.output_handler = self.output_handler_class(self._out_fp)

        self.bias_correct_method = bias_correct_method
        self.bias_correct_kwargs = bias_correct_kwargs or {}

        HandlerClass = get_input_handler_class(source_file_paths,
                                               input_handler)
        self.source_handler = HandlerClass(source_file_paths,
                                           self.source_features,
                                           target=target,
                                           shape=shape,
                                           temporal_slice=temporal_slice,
                                           raster_file=raster_file,
                                           cache_pattern=cache_pattern,
                                           time_chunk_size=time_chunk_size,
                                           overwrite_cache=overwrite_cache,
                                           val_split=0.0,
                                           max_workers=max_workers,
                                           extract_workers=extract_workers,
                                           compute_workers=compute_workers,
                                           ti_workers=ti_workers,
                                           load_workers=load_workers)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if type is not None:
            raise

    def close(self):
        """Close any open file handlers"""
        self.output_handler.close()

    @property
    def meta(self):
        """Get the meta data corresponding to the flattened source low-res data

        Returns
        -------
        pd.DataFrame
        """
        lat_lon = self.source_handler.lat_lon
        meta = pd.DataFrame({'latitude': lat_lon[..., 0].flatten(),
                             'longitude': lat_lon[..., 1].flatten()})
        return meta

    @property
    def lr_shape(self):
        """Get the shape of the source low-res data raster
        (rows, cols, time, features)"""
        return self.source_handler.shape

    @property
    def time_index(self):
        """Get the time index associated with the source low-res data

        Returns
        -------
        pd.DatetimeIndex
        """
        return self.source_handler.time_index

    @property
    def features(self):
        """Get a list of feature names from the output file, excluding meta and
        time index datasets

        Returns
        -------
        list
        """
        # all lower case
        ignore = ('meta', 'time_index', 'times', 'xlat', 'xlong')

        if self._features is None:
            if self.output_type == 'nc':
                features = list(self.output_handler.variables.keys())
            elif self.output_type == 'h5':
                features = self.output_handler.dsets
            features = [f for f in features if f.lower() not in ignore]

        elif isinstance(self._features, (list, tuple)):
            features = self._features

        elif isinstance(self._features, dict):
            features = sorted(self._features.keys())

        return features

    @property
    def source_features(self):
        """Get a list of feature names from the source input file, excluding
        meta and time index datasets. This property considers the features
        input mapping if a dictionary was provided, e.g. if
        features={'ghi': 'rsds'}, this property will return ['rsds']"""
        if isinstance(self._features, dict):
            temp = [self._features[f] for f in self.features]
            source_features = []
            for feat in temp:
                if isinstance(feat, (list, tuple)):
                    source_features += list(feat)
                else:
                    source_features.append(feat)
        else:
            source_features = self.features

        return source_features

    @property
    def output_type(self):
        """Get output data type

        Returns
        -------
        output_type
            e.g. 'nc' or 'h5'
        """
        ftype = get_source_type(self._out_fp)
        if ftype not in ('nc', 'h5'):
            msg = 'Did not recognize output file type: {}'.format(self._out_fp)
            logger.error(msg)
            raise TypeError(msg)
        return ftype

    @property
    def output_handler_class(self):
        """Get the output handler class.

        Returns
        -------
        HandlerClass : rex.Resource | xr.open_dataset
        """
        if self.output_type == 'nc':
            return xr.open_dataset
        elif self.output_type == 'h5':
            return Resource

    def bias_correct_source_data(self, data, feature):
        """Bias correct data using a method defined by the bias_correct_method
        input to ForwardPassStrategy

        Parameters
        ----------
        data : np.ndarray
            Any source data to be bias corrected, with the feature channel in
            the last axis.

        Returns
        -------
        data : np.ndarray
            Data corrected by the bias_correct_method ready for input to the
            forward pass through the generative model.
        """
        method = self.bias_correct_method
        kwargs = self.bias_correct_kwargs
        if method is not None:
            method = getattr(sup3r.bias.bias_transforms, method)
            logger.info('Running bias correction with: {}'.format(method))
            feature_kwargs = kwargs[feature]
            if 'time_index' in signature(method).parameters:
                feature_kwargs['time_index'] = self.time_index
            logger.debug('Bias correcting feature "{}" using function: {} '
                         'with kwargs: {}'
                         .format(feature, method, feature_kwargs))

            data = method(data, **feature_kwargs)

        return data

    def get_source_dset(self, idf, feature):
        """Get source low res input data including optional bias correction

        Parameters
        ----------
        idf : int
            Feature index in axis=-1 of the source data handler.
        feature : str
            Feature name

        Returns
        -------
        data_true : np.array
            Low-res source input data including optional bias correction
        """

        height = str(Feature.get_height(feature))
        uv_comps = np.sum([name.startswith(('U_', 'V_'))
                           for name in self.source_features
                           if height in name])

        if 'windspeed' in feature and uv_comps >= 2:
            u_feat = feature.replace('windspeed_', 'U_')
            v_feat = feature.replace('windspeed_', 'V_')
            logger.info('For sup3r output feature "{}", retrieving u/v '
                        'components "{}" and "{}"'
                        .format(feature, u_feat, v_feat))
            u_idf = self.source_handler.features.index(u_feat)
            v_idf = self.source_handler.features.index(v_feat)
            u_true = self.source_handler.data[..., u_idf]
            v_true = self.source_handler.data[..., v_idf]
            u_true = self.bias_correct_source_data(u_true, u_feat)
            v_true = self.bias_correct_source_data(v_true, v_feat)
            data_true = np.hypot(u_true, v_true)
        else:
            data_true = self.source_handler.data[..., idf]
            data_true = self.bias_correct_source_data(data_true, feature)

        return data_true

    def get_dset_out(self, name):
        """Get an output dataset from the forward pass output file.

        Parameters
        ----------
        name : str
            Name of the output dataset to retrieve. Must be found in the
            features property and the forward pass output file.

        Returns
        -------
        out : np.ndarray
            A copy of the high-resolution output data as a numpy
            array of shape (spatial_1, spatial_2, temporal)
        """

        logger.debug('Getting sup3r output dataset "{}"'.format(name))
        data = self.output_handler[name]
        if self.output_type == 'nc':
            data = data.values
        elif self.output_type == 'h5':
            shape = (len(self.time_index) * self.t_enhance,
                     int(self.lr_shape[0] * self.s_enhance),
                     int(self.lr_shape[1] * self.s_enhance))
            data = data.reshape(shape)

        # data always needs to be converted from (t, s1, s2) -> (s1, s2, t)
        data = np.transpose(data, axes=(1, 2, 0))

        return data

    def coarsen_data(self, data):
        """Re-coarsen a high-resolution synthetic output dataset

        Parameters
        ----------
        data : np.ndarray
            A copy of the high-resolution output data as a numpy
            array of shape (spatial_1, spatial_2, temporal)

        Returns
        -------
        data : np.ndarray
            A spatiotemporally coarsened copy of the input dataset, still with
            shape (spatial_1, spatial_2, temporal)
        """

        data = spatial_coarsening(data, s_enhance=self.s_enhance,
                                  obs_axis=False)

        # t_coarse needs shape to be 5D: (obs, s1, s2, t, f)
        data = np.expand_dims(data, axis=0)
        data = np.expand_dims(data, axis=4)
        data = temporal_coarsening(data, t_enhance=self.t_enhance,
                                   method=self._t_meth)
        data = data[0]
        data = data[..., 0]

        return data

    @classmethod
    def get_node_cmd(cls, config):
        """Get a CLI call to initialize Sup3rQa and execute the Sup3rQa.run()
        method based on an input config

        Parameters
        ----------
        config : dict
            sup3r QA config with all necessary args and kwargs to
            initialize Sup3rQa and execute Sup3rQa.run()
        """
        import_str = 'import time;\n'
        import_str += 'from reV.pipeline.status import Status;\n'
        import_str += 'from rex import init_logger;\n'
        import_str += 'from sup3r.qa.qa import Sup3rQa;\n'

        qa_init_str = get_fun_call_str(cls, config)

        log_file = config.get('log_file', None)
        log_level = config.get('log_level', 'INFO')

        log_arg_str = (f'"sup3r", log_level="{log_level}"')
        if log_file is not None:
            log_arg_str += f', log_file="{log_file}"'

        cmd = (f"python -c \'{import_str}\n"
               "t0 = time.time();\n"
               f"logger = init_logger({log_arg_str});\n"
               f"qa = {qa_init_str};\n"
               "qa.run();\n"
               "t_elap = time.time() - t0;\n")

        job_name = config.get('job_name', None)
        if job_name is not None:
            status_dir = config.get('status_dir', None)
            status_file_arg_str = f'"{status_dir}", '
            status_file_arg_str += f'module="{ModuleName.QA}", '
            status_file_arg_str += f'job_name="{job_name}", '
            status_file_arg_str += 'attrs=job_attrs'

            cmd += ('job_attrs = {};\n'.format(json.dumps(config)
                                               .replace("null", "None")
                                               .replace("false", "False")
                                               .replace("true", "True")))
            cmd += 'job_attrs.update({"job_status": "successful"});\n'
            cmd += 'job_attrs.update({"time": t_elap});\n'
            cmd += f'Status.make_job_file({status_file_arg_str})'

        cmd += (";\'\n")

        return cmd.replace('\\', '/')

    def export(self, qa_fp, data, dset_name, dset_suffix=''):
        """Export error dictionary to h5 file.

        Parameters
        ----------
        qa_fp : str | None
            Optional filepath to output QA file (only .h5 is supported)
        data : np.ndarray
            An array with shape (space1, space2, time) that represents the
            re-coarsened synthetic data minus the source true low-res data, or
            another dataset of the same shape to be written to disk
        dset_name : str
            Base dataset name to save data to
        dset_suffix : str
            Optional suffix to append to dset_name with an underscore before
            saving.
        """

        if not os.path.exists(qa_fp):
            logger.info('Initializing qa output file: "{}"'.format(qa_fp))
            with RexOutputs(qa_fp, mode='w') as f:
                f.meta = self.meta
                f.time_index = self.time_index

        shape = (len(self.time_index), len(self.meta))
        attrs = H5_ATTRS.get(Feature.get_basename(dset_name), {})

        # dont scale the re-coarsened data or diffs
        attrs['scale_factor'] = 1
        attrs['dtype'] = 'float32'

        if dset_suffix:
            dset_name = dset_name + '_' + dset_suffix

        logger.info('Adding dataset "{}" to output file.'.format(dset_name))

        # transpose and flatten to typical h5 (time, space) dimensions
        data = np.transpose(data, axes=(2, 0, 1)).reshape(shape)

        RexOutputs.add_dataset(qa_fp, dset_name, data,
                               dtype=attrs['dtype'],
                               chunks=attrs.get('chunks', None),
                               attrs=attrs)

    def run(self):
        """Go through all datasets and get the error for the re-coarsened
        synthetic minus the true low-res source data.

        Returns
        -------
        errors : dict
            Dictionary of errors, where keys are the feature names, and each
            value is an array with shape (space1, space2, time) that represents
            the re-coarsened synthetic data minus the source true low-res data
        """

        errors = {}
        for idf, feature in enumerate(self.features):
            logger.info('Running QA on dataset {} of {}: "{}"'
                        .format(idf + 1, len(self.features), feature))
            data_syn = self.get_dset_out(feature)
            data_syn = self.coarsen_data(data_syn)
            data_true = self.get_source_dset(idf, feature)

            if data_syn.shape != data_true.shape:
                msg = ('Sup3rQa failed while trying to inspect the "{}" '
                       'feature. The source low-res data had shape {} '
                       'while the re-coarsened synthetic data had shape {}.'
                       .format(feature, data_true.shape, data_syn.shape))
                logger.error(msg)
                raise RuntimeError(msg)

            feature_diff = data_syn - data_true
            errors[feature] = feature_diff

            if self.qa_fp is not None:
                self.export(self.qa_fp, feature_diff, feature, 'error')
                if self.save_sources:
                    self.export(self.qa_fp, data_syn, feature, 'synthetic')
                    self.export(self.qa_fp, data_true, feature, 'true')

        logger.info('Finished Sup3rQa run method.')

        return errors
