# -*- coding: utf-8 -*-
"""sup3r WindStats module."""
import json
import pandas as pd
import numpy as np
import xarray as xr
import os
import pickle
import logging
from rex import MultiFileResourceX
from rex.utilities.fun_utils import get_fun_call_str
from sup3r.utilities import ModuleName
from sup3r.utilities.utilities import (get_input_handler_class,
                                       get_source_type,
                                       transform_rotate_wind)
from sup3r.qa.utilities import (ws_ramp_rate_dist, tke_series,
                                velocity_gradient_dist, vorticity_dist,
                                tke_wavenumber_spectrum,
                                tke_frequency_spectrum, st_interp)


logger = logging.getLogger(__name__)


class Sup3rWindStats:
    """Class for doing statistical QA on sup3r forward pass wind outputs."""

    def __init__(self, source_file_paths, out_file_path, s_enhance, t_enhance,
                 heights, temporal_slice=slice(None), target=None,
                 shape=None, raster_file=None, qa_fp=None,
                 time_chunk_size=None, cache_pattern=None,
                 overwrite_cache=False, input_handler=None, max_workers=None,
                 extract_workers=None, compute_workers=None,
                 load_workers=None, ti_workers=None, get_interp=False,
                 get_hr=True, get_lr=True, include_stats=None, max_values=None,
                 ramp_rate_t_step=1):
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
        heights : list
            Heights for which to compute wind stats. e.g. [10, 40, 80, 100]
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
            Optional filepath to output QA file when you call
            Sup3rWindStats.run()
            (only .pkl is supported)
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
        get_interp : bool
            Whether to include interpolated baseline stats in output
        get_hr : bool
            Whether to include high resolution stats in output
        get_lr : bool
            Whether to include low resolution stats in output
        include_stats : list | None
            List of stats to include in output. e.g. ['ws_ramp_rate',
            'velocity_grad', 'vorticity', 'tke_avg_k', 'tke_avg_f', 'tke_ts']
        max_values : dict | None
            Dictionary of max values to keep for stats. e.g.
            {'ws_ramp_rate_max': 10, 'v_grad_max': 14, 'vorticity_max': 7}
        ramp_rate_t_step : int | list
            Number of time steps to use for ramp rate calculation. If low res
            data is hourly then t_step=1 will calculate the hourly ramp rate.
            Can provide a list of different t_step values.
        """

        logger.info('Initializing Sup3rWindStats and retrieving source data')

        if max_workers is not None:
            extract_workers = compute_workers = load_workers = max_workers
            ti_workers = max_workers

        self.max_values = max_values or {}
        self.ws_ramp_rate_max = self.max_values.get('ws_ramp_rate_max', 10)
        self.v_grad_max = self.max_values.get('v_grad_max', 7)
        self.vorticity_max = self.max_values.get('vorticity_max', 14)
        self.ramp_rate_t_step = (ramp_rate_t_step
                                 if isinstance(ramp_rate_t_step, list)
                                 else [ramp_rate_t_step])
        self.include_stats = include_stats or ['ws_ramp_rate', 'velocity_grad',
                                               'vorticity', 'tke_avg_k',
                                               'tke_avg_f', 'tke_ts',
                                               'mean_ws_ramp_rate']

        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        self.heights = heights if isinstance(heights, list) else [heights]
        self._out_fp = (out_file_path if isinstance(out_file_path, list)
                        else [out_file_path])
        self.qa_fp = qa_fp
        self.output_handler = self.output_handler_class(self._out_fp)
        self._hr_lat_lon = None
        self.get_interp = get_interp
        self.get_hr = get_hr
        self.get_lr = get_lr
        self.cache_pattern = cache_pattern

        HandlerClass = get_input_handler_class(source_file_paths,
                                               input_handler)
        if self.get_lr:
            self.source_handler = HandlerClass(source_file_paths,
                                               self.features,
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
                                               load_workers=load_workers,
                                               ti_workers=ti_workers)
            self.source_handler.load_cached_data()
        if self.get_hr and self.get_lr:
            self.lr_t_slice, self.hr_t_slice = self.time_overlap_slices()
        else:
            self.lr_t_slice = slice(None)
            self.hr_t_slice = slice(None)

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
    def hr_shape(self):
        """Shape of output data"""
        shape = (self.lr_shape[0] * self.s_enhance,
                 self.lr_shape[1] * self.s_enhance,
                 self.lr_shape[2] * self.t_enhance)
        return shape

    @property
    def hr_lat_lon(self):
        """Get lat/lon for output data"""
        if self._hr_lat_lon is None:
            meta = self.output_handler.meta
            lats = meta.latitude.values
            lons = meta.longitude.values
            lat_lon = np.dstack((lats.reshape(self.hr_shape[:-1]),
                                 lons.reshape(self.hr_shape[:-1])))
        return lat_lon

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
        return self.source_handler.data[..., self.lr_t_slice, 0].shape

    @property
    def lr_time_index(self):
        """Get the time index associated with the source low-res data

        Returns
        -------
        pd.DatetimeIndex
        """
        return self.source_handler.time_index.tz_localize(None)

    @property
    def hr_time_index(self):
        """Get the time index associated with the high-res data

        Returns
        -------
        pd.DatetimeIndex
        """
        if self.output_type == 'nc':
            raise NotImplementedError('Netcdf output not yet supported')
        return self.output_handler.time_index.tz_localize(None)

    def time_overlap_slices(self):
        """Get slices for temporal overlap of low and high resolution data

        Returns
        -------
        lr_slice : slice
            Slice for overlap of low resolution data with high resolution data
        hr_slice : slice
            Slice for overlap of high resolution data with low resolution data
        """
        min_time = np.max((self.lr_time_index[0], self.hr_time_index[0]))
        max_time = np.min((self.lr_time_index[-1], self.hr_time_index[-1]))

        lr_start = next(i for i, t in enumerate(self.lr_time_index)
                        if t >= min_time)
        hr_start = next(i for i, t in enumerate(self.hr_time_index)
                        if t >= min_time)

        lr_end = next(len(self.lr_time_index) - i
                      for i, t in enumerate(self.lr_time_index[::-1])
                      if t <= max_time) + 1
        hr_end = next(len(self.hr_time_index) - i
                      for i, t in enumerate(self.hr_time_index[::-1])
                      if t <= max_time) + 1

        return slice(lr_start, lr_end), slice(hr_start, hr_end)

    @property
    def features(self):
        """Get a list of requested wind feature names

        Returns
        -------
        list
        """

        # all lower case
        features = []
        for height in self.heights:
            features.append(f'U_{height}m')
            features.append(f'V_{height}m')

        return features

    def feature_indices(self, height):
        """Indices for U/V of given height

        Parameters
        ----------
        height : int
            Height in meters for requested U/V fields

        Returns
        -------
        uidx : int
            Index for U_{height}m
        vidx : int
            Index for V_{height}m
        """
        uidx = self.features.index(f'U_{height}m')
        vidx = self.features.index(f'V_{height}m')
        return uidx, vidx

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
            return MultiFileResourceX

    def get_uv_out(self, height):
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

        logger.debug('Getting sup3r u/v data ({}m)'.format(height))
        t_steps = len(self.hr_time_index[self.hr_t_slice])
        shape = f'{self.hr_shape[0]}x{self.hr_shape[1]}x{t_steps}'

        u_file = v_file = None
        if self.cache_pattern is not None:
            tmp_file = self.cache_pattern.replace('{shape}', f'{shape}')
            u_file = tmp_file.replace('{feature}', f'u_{height}m')
            v_file = tmp_file.replace('{feature}', f'v_{height}m')

            if os.path.exists(u_file) and os.path.exists(v_file):
                u = self.load_cache(u_file)
                v = self.load_cache(v_file)
                return u, v

        if self.output_type == 'nc':
            raise NotImplementedError('Netcdf output not yet supported')
        elif self.output_type == 'h5':
            ws_f = f'windspeed_{height}m'
            wd_f = f'winddirection_{height}m'
            logger.info('Extracting data from output handler for '
                        f'time_slice={self.hr_t_slice}')
            ws = self.output_handler[ws_f, self.hr_t_slice, :]
            wd = self.output_handler[wd_f, self.hr_t_slice, :]
            ws = ws.reshape((-1, self.hr_shape[0], self.hr_shape[1]))
            ws = np.transpose(ws, axis=(1, 2, 0))
            wd = wd.reshape((-1, self.hr_shape[0], self.hr_shape[1]))
            wd = np.transpose(wd, axis=(1, 2, 0))
            logger.info(f'Transforming ws / wd to u / v for height={height}m '
                        f'with shape={ws.shape}')
            u, v = transform_rotate_wind(ws, wd, self.hr_lat_lon)
            if u_file is not None:
                self.save_cache(u, u_file)
            if v_file is not None:
                self.save_cache(v, v_file)

        return u, v

    @classmethod
    def save_cache(cls, array, file_name):
        """Save data to cache file

        Parameters
        ----------
        array : ndarray
            Wind field data
        file_name : str
            Path to cache file
        """
        logger.info(f'Saving data to {file_name}')
        with open(file_name, 'wb') as f:
            pickle.dump(array, f, protocol=4)

    @classmethod
    def load_cache(cls, file_name):
        """Load data from cache file

        Parameters
        ----------
        file_name : str
            Path to cache file

        Returns
        -------
        array : ndarray
            Wind field data
        """
        logger.info(f'Loading data from {file_name}')
        with open(file_name, 'rb') as f:
            arr = pickle.load(f)
        return arr

    @classmethod
    def export(cls, qa_fp, data):
        """Export stats dictionary to pkl file.

        Parameters
        ----------
        qa_fp : str | None
            Optional filepath to output QA file (only .h5 is supported)
        data : dict
            A dictionary with stats for low and high resolution wind fields
        """

        if not os.path.exists(qa_fp):
            logger.info('Initializing qa output file: "{}"'.format(qa_fp))
            with open(qa_fp, 'wb') as f:
                pickle.dump(data, f)
        else:
            logger.info(f'{qa_fp} already exists.')

    @classmethod
    def get_node_cmd(cls, config):
        """Get a CLI call to initialize Sup3rWindStats and execute the
        Sup3rWindStats.run() method based on an input config

        Parameters
        ----------
        config : dict
            sup3r wind stats config with all necessary args and kwargs to
            initialize Sup3rWindStats and execute Sup3rWindStats.run()
        """
        import_str = 'import time;\n'
        import_str += 'from reV.pipeline.status import Status;\n'
        import_str += 'from rex import init_logger;\n'
        import_str += 'from sup3r.qa.stats import Sup3rWindStats;\n'

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
            status_file_arg_str += f'module="{ModuleName.WIND_STATS}", '
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

    def get_spectra_stats(self, u, v):
        """Compute spectra based statistics

        Parameters
        ----------
        u: ndarray
            Longitudinal velocity component
            (lat, lon, temporal)
        v : ndarray
            Latitudinal velocity component
            (lat, lon, temporal)

        Returns
        -------
        stats : dict
            Dictionary of spectra stats for wind fields
        """

        stats_dict = {}
        if 'tke_ts' in self.include_stats:
            logger.info('Computing mean kinetic energy spectrum series.')
            stats_dict['tke_ts'] = tke_series(u, v)

        if 'tke_avg_k' in self.include_stats:
            logger.info('Computing time averaged kinetic energy'
                        ' wavenumber spectrum.')
            stats_dict['tke_avg_k'] = tke_wavenumber_spectrum(
                np.mean(u, axis=-1), np.mean(v, axis=-1))

        if 'tke_avg_f' in self.include_stats:
            logger.info('Computing time averaged kinetic energy'
                        ' frequency spectrum.')
            stats_dict['tke_avg_f'] = tke_frequency_spectrum(u, v)

        return stats_dict

    def get_ramp_rate_stats(self, u, v, res='high'):
        """Compute statistics for ramp rates

        Parameters
        ----------
        u: ndarray
            Longitudinal velocity component
            (lat, lon, temporal)
        v : ndarray
            Latitudinal velocity component
            (lat, lon, temporal)
        res : str
            Resolution of input fields. If this is 'high' then the ramp rate
            time step needs to be multipled by the temporal enhancement factor

        Returns
        -------
        stats : dict
            Dictionary of ramp rate stats for wind fields
        """

        stats_dict = {}
        if 'ws_ramp_rate' in self.include_stats:
            for i, time in enumerate(self.ramp_rate_t_step):
                freq = time * (self.lr_time_index[1] - self.lr_time_index[0])
                if res == 'high':
                    time = time * self.t_enhance
                logger.info(f'Computing ramp rate pdf ({freq}).')
                out = ws_ramp_rate_dist(u, v, diff_max=self.ws_ramp_rate_max,
                                        t_steps=time)
                stats_dict[f'ramp_rate_{self.ramp_rate_t_step[i]}'] = out
        if 'mean_ws_ramp_rate' in self.include_stats:
            for i, time in enumerate(self.ramp_rate_t_step):
                freq = time * (self.lr_time_index[1] - self.lr_time_index[0])
                if res == 'high':
                    time = time * self.t_enhance
                logger.info(f'Computing mean ramp rate pdf ({freq}).')
                out = ws_ramp_rate_dist(np.mean(u, axis=(0, 1)),
                                        np.mean(v, axis=(0, 1)),
                                        diff_max=self.ws_ramp_rate_max,
                                        t_steps=time)
                stats_dict[f'mean_ramp_rate_{self.ramp_rate_t_step[i]}'] = out
        return stats_dict

    def get_wind_stats(self, u, v, res='high'):
        """Get stats for wind fields

        Parameters
        ----------
        u: ndarray
            Longitudinal velocity component
            (lat, lon, temporal)
        v : ndarray
            Latitudinal velocity component
            (lat, lon, temporal)
        res : str
            Resolution of input fields. If this is 'high' then the ramp rate
            time step needs to be multipled by the temporal enhancement factor

        Returns
        -------
        stats : dict
            Dictionary of stats for wind fields
        """
        stats_dict = self.get_spectra_stats(u, v)

        if 'velocity_grad' in self.include_stats:
            logger.info('Computing velocity gradient pdf.')
            stats_dict['velocity_grad'] = velocity_gradient_dist(
                u, diff_max=self.v_grad_max)

        if 'vorticity' in self.include_stats:
            logger.info('Computing vorticity pdf.')
            stats_dict['vorticity'] = vorticity_dist(
                u, v, diff_max=self.vorticity_max)

        out = self.get_ramp_rate_stats(u, v, res=res)
        stats_dict.update(out)
        return stats_dict

    def get_height_stats(self, height):
        """Get stats for high and low resolution wind fields

        Parameters
        ----------
        height : int
            Height in meters for requested U/V fields

        Returns
        -------
        low_res : dict
            Dictionary of stats for low resolution wind fields
        high_res : dict
            Dictionary of stats for high resolution wind fields
        interp : dict
            Dictionary of stats for spatiotemporally interpolated wind fields
        """
        low_res = {}
        if self.get_lr:
            uidx, vidx = self.feature_indices(height)
            u_lr = self.source_handler.data[..., self.lr_t_slice, uidx]
            v_lr = self.source_handler.data[..., self.lr_t_slice, vidx]
            logger.info(f'Getting low res stats for height={height}m')
            low_res = self.get_wind_stats(u_lr, v_lr, res='low')

        high_res = {}
        if self.get_hr:
            u_hr, v_hr = self.get_uv_out(height)
            logger.info(f'Getting high res stats for height={height}m')
            high_res = self.get_wind_stats(u_hr, v_hr, res='high')

        interp = {}
        if self.get_interp:
            logger.info(f'Interpolating low res U for height={height}')
            u_itp = st_interp(u_lr, self.s_enhance, self.t_enhance)
            logger.info(f'Interpolating low res V for height={height}')
            v_itp = st_interp(v_lr, self.s_enhance, self.t_enhance)
            logger.info('Getting interpolated baseline stats for '
                        f'height={height}m')
            interp = self.get_wind_stats(u_itp, v_itp, res='high')
        return low_res, high_res, interp

    def run(self):
        """Go through all datasets and get the dictionary of wind field
        statistics.

        Returns
        -------
        stats : dict
            Dictionary of statistics, where keys are lr/hr/interp appended with
            the height of the corresponding wind field. Values are dictionaries
            of statistics, such as velocity_gradient, vorticity, ws_ramp_rate,
            etc
        """

        stats = {}
        for idf, height in enumerate(self.heights):
            logger.info('Running WindStats on height {} of {} ({}m)'
                        .format(idf + 1, len(self.heights), height))
            lr_stats, hr_stats, interp = self.get_height_stats(height)

            if self.get_lr:
                stats[f'lr_{height}m'] = lr_stats
            if self.get_hr:
                stats[f'hr_{height}m'] = hr_stats
            if self.get_interp:
                stats[f'interp_{height}m'] = interp

        if self.qa_fp is not None:
            self.export(self.qa_fp, stats)
        logger.info('Finished Sup3rWindStats run method.')

        return stats
