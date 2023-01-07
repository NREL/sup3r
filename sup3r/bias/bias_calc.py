# -*- coding: utf-8 -*-
"""Utilities to calculate the bias correction factors for biased data that is
going to be fed into the sup3r downscaling models. This is typically used to
bias correct GCM data vs. some historical record like the WTK or NSRDB."""
import os
import h5py
import json
import logging
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.stats import ks_2samp
from scipy.ndimage.filters import gaussian_filter
from concurrent.futures import ProcessPoolExecutor, as_completed
import rex
from rex.utilities.fun_utils import get_fun_call_str
from sup3r.utilities import ModuleName, VERSION_RECORD
from sup3r.utilities.utilities import nn_fill_array
from sup3r.utilities.cli import BaseCLI
import sup3r.preprocessing.data_handling


logger = logging.getLogger(__name__)


class DataRetrievalBase:
    """Base class to handle data retrieval for the biased data and the
    baseline data
    """

    def __init__(self, base_fps, bias_fps, base_dset, bias_feature,
                 target, shape,
                 base_handler='Resource', bias_handler='DataHandlerNCforCC',
                 bias_handler_kwargs=None, decimals=None):
        """
        Parameters
        ----------
        base_fps : list | str
            One or more baseline .h5 filepaths representing non-biased data to
            use to correct the biased dataset. This is typically several years
            of WTK or NSRDB files.
        bias_fps : list | str
            One or more biased .nc or .h5 filepaths representing the biased
            data to be corrected based on the baseline data. This is typically
            several years of GCM .nc files.
        base_dset : str
            A single dataset from the base_fps to retrieve. In the case of wind
            components, this can be U_100m or V_100m which will retrieve
            windspeed and winddirection and derive the U/V component.
        bias_feature : str
            This is the biased feature from bias_fps to retrieve. This should
            be a single feature name corresponding to base_dset
        target : tuple
            (lat, lon) lower left corner of raster to retrieve from bias_fps.
        shape : tuple
            (rows, cols) grid size to retrieve from bias_fps.
        base_handler : str
            Name of rex resource handler class to be retrieved from the rex
            library.
        bias_handler : str
            Name of the bias data handler class to be retrieved from the
            sup3r.preprocessing.data_handling library.
        bias_handler_kwargs : dict | None
            Optional kwargs to send to the initialization of the bias_handler
            class
        decimals : int | None
            Option to round bias and base data to this number of
            decimals, this gets passed to np.around(). If decimals
            is negative, it specifies the number of positions to
            the left of the decimal point.
        """

        logger.info('Initializing DataRetrievalBase for base dset "{}" '
                    'correcting biased dataset(s): {}'
                    .format(base_dset, bias_feature))
        self.base_fps = base_fps
        self.bias_fps = bias_fps
        self.base_dset = base_dset
        self.bias_feature = bias_feature
        self.target = target
        self.shape = shape
        self.decimals = decimals
        bias_handler_kwargs = bias_handler_kwargs or {}

        if isinstance(self.base_fps, str):
            self.base_fps = [self.base_fps]
        if isinstance(self.bias_fps, str):
            self.bias_fps = [self.bias_fps]

        self.base_handler = getattr(rex, base_handler)
        self.bias_handler = getattr(sup3r.preprocessing.data_handling,
                                    bias_handler)

        with self.base_handler(self.base_fps[0]) as res:
            self.base_meta = res.meta
            self.base_tree = KDTree(self.base_meta[['latitude', 'longitude']])

        self.bias_dh = self.bias_handler(self.bias_fps, [self.bias_feature],
                                         target=self.target, shape=self.shape,
                                         val_split=0.0, **bias_handler_kwargs)

        lats = self.bias_dh.lat_lon[..., 0].flatten()
        lons = self.bias_dh.lat_lon[..., 1].flatten()
        self.bias_meta = pd.DataFrame({'latitude': lats, 'longitude': lons})
        self.bias_ti = self.bias_dh.time_index

        raster_shape = self.bias_dh.lat_lon[..., 0].shape
        self.bias_tree = KDTree(self.bias_meta[['latitude', 'longitude']])
        self.bias_gid_raster = np.arange(lats.size)
        self.bias_gid_raster = self.bias_gid_raster.reshape(raster_shape)
        logger.info('Finished initializing DataRetrievalBase.')

    @property
    def meta(self):
        """Get a meta data dictionary on how these bias factors were calculated
        """
        meta = {'base_fps': self.base_fps,
                'bias_fps': self.bias_fps,
                'base_dset': self.base_dset,
                'bias_feature': self.bias_feature,
                'target': self.target,
                'shape': self.shape,
                'class': str(self.__class__),
                'version_record': VERSION_RECORD,
                }
        return meta

    @staticmethod
    def compare_dists(base_data, bias_data, adder=0, scalar=1):
        """Compare two distributions using the two-sample Kolmogorov-Smirnov.
        When the output is minimized, the two distributions are similar.

        Parameters
        ----------
        base_data : np.ndarray
            1D array of base data observations.
        bias_data : np.ndarray
            1D array of biased data observations.
        adder : float
            Factor to adjust the biased data before comparing distributions:
            bias_data * scalar + adder
        scalar : float
            Factor to adjust the biased data before comparing distributions:
            bias_data * scalar + adder

        Returns
        -------
        out : float
            KS test statistic
        """
        out = ks_2samp(base_data, bias_data * scalar + adder)
        return out.statistic

    @classmethod
    def get_node_cmd(cls, config):
        """Get a CLI call to call cls.run() on a single node based on an input
        config.

        Parameters
        ----------
        config : dict
            sup3r bias calc config with all necessary args and kwargs to
            initialize the class and call run() on a single node.
        """
        import_str = 'import time;\n'
        import_str += 'from reV.pipeline.status import Status;\n'
        import_str += 'from rex import init_logger;\n'
        import_str += f'from sup3r.bias.bias_calc import {cls.__name__};\n'

        if not hasattr(cls, 'run'):
            msg = ('I can only get you a node command for subclasses of '
                   'DataRetrievalBase with a run() method.')
            logger.error(msg)
            raise NotImplementedError(msg)

        # pylint: disable=E1101
        init_str = get_fun_call_str(cls, config)
        fun_str = get_fun_call_str(cls.run, config)
        fun_str = fun_str.partition('.')[-1]
        fun_str = 'bc.' + fun_str

        log_file = config.get('log_file', None)
        log_level = config.get('log_level', 'INFO')
        log_arg_str = (f'"sup3r", log_level="{log_level}"')
        if log_file is not None:
            log_arg_str += f', log_file="{log_file}"'

        cmd = (f"python -c \'{import_str}\n"
               "t0 = time.time();\n"
               f"logger = init_logger({log_arg_str});\n"
               f"bc = {init_str};\n"
               f"{fun_str};\n"
               "t_elap = time.time() - t0;\n")

        cmd = BaseCLI.add_status_cmd(config, ModuleName.BIAS_CALC, cmd)
        cmd += (";\'\n")

        return cmd.replace('\\', '/')

    def get_bias_gid(self, coord):
        """Get the bias gid from a coordinate.

        Parameters
        ----------
        coord : tuple
            (lat, lon) to get data for.

        Returns
        -------
        bias_gid : int
            gid of the data to retrieve in the bias data source raster data.
            The gids for this data source are the enumerated indices of the
            flattened coordinate array.
        d : float
            Distance in decimal degrees from coord to bias gid
        """
        d, i = self.bias_tree.query(coord)
        bias_gid = self.bias_gid_raster.flatten()[i]
        return bias_gid, d

    def get_base_gid(self, bias_gid, knn):
        """Get one or more base gid(s) corresponding to a bias gid.

        Parameters
        ----------
        bias_gid : int
            gid of the data to retrieve in the bias data source raster data.
            The gids for this data source are the enumerated indices of the
            flattened coordinate array.
        knn : int
            Number of nearest neighbors to aggregate from the base data when
            comparing to a single site from the bias data.

        Returns
        -------
        dist : np.ndarray
            Array of nearest neighbor distances with length == knn
        base_gid : np.ndarray
            Array of base gids that are the nearest neighbors of bias_gid with
            length == knn
        """
        coord = self.bias_meta.loc[bias_gid, ['latitude', 'longitude']]
        dist, base_gid = self.base_tree.query(coord, k=knn)
        return dist, base_gid

    def get_data_pair(self, coord, knn, daily_reduction='avg'):
        """Get base and bias data observations based on a single bias gid.

        Parameters
        ----------
        coord : tuple
            (lat, lon) to get data for.
        knn : int
            Number of nearest neighbors to aggregate from the base data when
            comparing to a single site from the bias data.
        daily_reduction : None | str
            Option to do a reduction of the hourly+ source base data to daily
            data. Can be None (no reduction, keep source time frequency), "avg"
            (daily average), "max" (daily max), or "min" (daily min)

        Returns
        -------
        base_data : np.ndarray
            1D array of base data spatially averaged across the base_gid input
            and possibly daily-averaged or min/max'd as well.
        bias_data : np.ndarray
            1D array of temporal data at the requested gid.
        base_dist : np.ndarray
            Array of nearest neighbor distances from coord to the base data
            sites with length == knn
        bias_dist : Float
            Nearest neighbor distance from coord to the bias data site
        """
        bias_gid, bias_dist = self.get_bias_gid(coord)
        base_dist, base_gid = self.get_base_gid(bias_gid, knn)
        bias_data = self.get_bias_data(bias_gid)
        base_data = self.get_base_data(self.base_fps, self.base_dset, base_gid,
                                       self.base_handler,
                                       daily_reduction=daily_reduction,
                                       decimals=self.decimals)
        base_data = base_data[0]
        return base_data, bias_data, base_dist, bias_dist

    def get_bias_data(self, bias_gid):
        """Get data from the biased data source for a single gid

        Parameters
        ----------
        bias_gid : int
            gid of the data to retrieve in the bias data source raster data.
            The gids for this data source are the enumerated indices of the
            flattened coordinate array.

        Returns
        -------
        bias_data : np.ndarray
            1D array of temporal data at the requested gid.
        """
        idx = np.where(self.bias_gid_raster == bias_gid)
        bias_data = self.bias_dh.data[idx][0]

        if bias_data.shape[-1] == 1:
            bias_data = bias_data[:, 0]
        else:
            msg = ('Found a weird number of feature channels for the bias '
                   'data retrieval: {}. Need just one channel'
                   .format(bias_data.shape))
            logger.error(msg)
            raise RuntimeError(msg)

        if self.decimals is not None:
            bias_data = np.around(bias_data, decimals=self.decimals)

        return bias_data

    @classmethod
    def get_base_data(cls, base_fps, base_dset, base_gid, base_handler,
                      daily_reduction='avg', decimals=None):
        """Get data from the baseline data source, possibly for many high-res
        base gids corresponding to a single coarse low-res bias gid.

        Parameters
        ----------
        base_fps : list | str
            One or more baseline .h5 filepaths representing non-biased data to
            use to correct the biased dataset. This is typically several years
            of WTK or NSRDB files.
        base_dset : str
            A single dataset from the base_fps to retrieve.
        base_gid : int | np.ndarray
            One or more spatial gids to retrieve from base_fps. The data will
            be spatially averaged across all of these sites.
        base_handler : rex.Resource
            A rex data handler similar to rex.Resource
        daily_reduction : None | str
            Option to do a reduction of the hourly+ source base data to daily
            data. Can be None (no reduction, keep source time frequency), "avg"
            (daily average), "max" (daily max), or "min" (daily min)
        decimals : int | None
            Option to round bias and base data to this number of
            decimals, this gets passed to np.around(). If decimals
            is negative, it specifies the number of positions to
            the left of the decimal point.

        Returns
        -------
        out : np.ndarray
            1D array of base data spatially averaged across the base_gid input
            and possibly daily-averaged or min/max'd as well.
        out_ti : pd.DatetimeIndex
            DatetimeIndex object of datetimes corresponding to the
            output data.
        """

        out = []
        out_ti = []
        for fp in base_fps:
            with base_handler(fp) as res:
                base_ti = res.time_index

                base_data, base_cs_ghi = cls._read_base_data(res, base_dset,
                                                             base_gid)
                if daily_reduction is not None:
                    base_data = cls._reduce_base_data(base_ti, base_data,
                                                      base_cs_ghi, base_dset,
                                                      daily_reduction)
                    base_ti = np.array(sorted(set(base_ti.date)))

            out.append(base_data)
            out_ti.append(base_ti)

        out = np.hstack(out)

        if decimals is not None:
            out = np.around(out, decimals=decimals)

        return out, pd.DatetimeIndex(np.hstack(out_ti))

    @staticmethod
    def _read_base_data(res, base_dset, base_gid):
        """Read baseline data from the resource handler with extra logic for
        special datasets (e.g. u/v wind components or clearsky_ratio)

        Parameters
        ----------
        res : rex.Resource
            rex Resource handler that is an open file handler of the base
            file(s)
        base_dset : str
            A single dataset from the base_fps to retrieve.
        base_gid : int | np.ndarray
            One or more spatial gids to retrieve from base_fps. The data will
            be spatially averaged across all of these sites.

        Returns
        -------
        base_data : np.ndarray
            1D array of base data spatially averaged across the base_gid input
        base_cs_ghi : np.ndarray | None
            If base_dset == "clearsky_ratio", the base_data array is GHI and
            this base_cs_ghi is clearsky GHI. Otherwise this is None
        """

        base_cs_ghi = None

        if base_dset.startswith(('U_', 'V_')):
            dset_ws = base_dset.replace('U_', 'windspeed_')
            dset_ws = dset_ws.replace('V_', 'windspeed_')
            dset_wd = dset_ws.replace('speed', 'direction')
            base_ws = res[dset_ws, :, base_gid]
            base_wd = res[dset_wd, :, base_gid]

            if base_dset.startswith('U_'):
                base_data = -base_ws * np.sin(np.radians(base_wd))
            else:
                base_data = -base_ws * np.cos(np.radians(base_wd))

        elif base_dset == 'clearsky_ratio':
            base_data = res['ghi', :, base_gid]
            base_cs_ghi = res['clearsky_ghi', :, base_gid]

        else:
            base_data = res[base_dset, :, base_gid]

        if len(base_data.shape) == 2:
            base_data = base_data.mean(axis=1)
            if base_cs_ghi is not None:
                base_cs_ghi = base_cs_ghi.mean(axis=1)

        return base_data, base_cs_ghi

    @staticmethod
    def _reduce_base_data(base_ti, base_data, base_cs_ghi, base_dset,
                          daily_reduction):
        """Reduce the base timeseries data using some sort of daily reduction
        function.

        Parameters
        ----------
        base_ti : pd.DatetimeIndex
            Time index associated with base_data
        base_data : np.ndarray
            1D array of base data spatially averaged across the base_gid input
        base_cs_ghi : np.ndarray | None
            If base_dset == "clearsky_ratio", the base_data array is GHI and
            this base_cs_ghi is clearsky GHI. Otherwise this is None
        base_dset : str
            A single dataset from the base_fps to retrieve.
        daily_reduction : str
            Option to do a reduction of the hourly+ source base data to daily
            data. Can be None (no reduction, keep source time frequency), "avg"
            (daily average), "max" (daily max), or "min" (daily min)

        Returns
        -------
        base_data : np.ndarray
            1D array of base data spatially averaged across the base_gid input
            and possibly daily-averaged or min/max'd as well.
        """

        if daily_reduction is None:
            return base_data

        slices = [np.where(base_ti.date == date)
                  for date in sorted(set(base_ti.date))]

        if base_dset == 'clearsky_ratio' and daily_reduction.lower() == 'avg':
            base_data = np.array([base_data[s0].sum() / base_cs_ghi[s0].sum()
                                  for s0 in slices])

        elif daily_reduction.lower() == 'avg':
            base_data = np.array([base_data[s0].mean() for s0 in slices])

        elif daily_reduction.lower() == 'max':
            base_data = np.array([base_data[s0].max() for s0 in slices])

        elif daily_reduction.lower() == 'min':
            base_data = np.array([base_data[s0].min() for s0 in slices])

        return base_data


class LinearCorrection(DataRetrievalBase):
    """Calculate linear correction *scalar +adder factors to bias correct data

    This calculation operates on single bias sites for the full time series of
    available data (no season bias correction)
    """

    # size of the time dimension, 1 is no time-based bias correction
    NT = 1

    @staticmethod
    def get_linear_correction(bias_data, base_data, bias_feature, base_dset):
        """Get the linear correction factors based on 1D bias and base datasets

        Parameters
        ----------
        bias_data : np.ndarray
            1D array of biased data observations.
        base_data : np.ndarray
            1D array of base data observations.
        bias_feature : str
            This is the biased feature from bias_fps to retrieve. This should
            be a single feature name corresponding to base_dset
        base_dset : str
            A single dataset from the base_fps to retrieve. In the case of wind
            components, this can be U_100m or V_100m which will retrieve
            windspeed and winddirection and derive the U/V component.

        Returns
        -------
        out : dict
            Dictionary of values defining the mean/std of the bias + base
            data and the scalar + adder factors to correct the biased data
            like: bias_data * scalar + adder
        """

        bias_std = bias_data.std()
        if bias_std == 0:
            bias_std = base_data.std()

        scalar = base_data.std() / bias_std
        adder = base_data.mean() - bias_data.mean() * scalar

        out = {f'bias_{bias_feature}_mean': bias_data.mean(),
               f'bias_{bias_feature}_std': bias_std,
               f'base_{base_dset}_mean': base_data.mean(),
               f'base_{base_dset}_std': base_data.std(),
               f'{bias_feature}_scalar': scalar,
               f'{bias_feature}_adder': adder,
               }

        return out

    # pylint: disable=W0613
    @classmethod
    def _run_single(cls, bias_data, base_fps, bias_feature, base_dset,
                    base_gid, base_handler, daily_reduction, bias_ti,
                    decimals):
        """Find the nominal scalar + adder combination to bias correct data
        at a single site"""

        base_data, _ = cls.get_base_data(base_fps, base_dset,
                                         base_gid, base_handler,
                                         daily_reduction=daily_reduction,
                                         decimals=decimals)

        out = cls.get_linear_correction(bias_data, base_data,
                                        bias_feature, base_dset)
        return out

    def fill_extend(self, out, smooth_extend):
        """Fill data extending beyond the base meta data extent by doing a
        nearest neighbor gap fill.

        Parameters
        ----------
        out : dict
            Dictionary of values defining the mean/std of the bias + base
            data and the scalar + adder factors to correct the biased data
            like: bias_data * scalar + adder. Each value is of shape
            (lat, lon, time).
        smooth_extend : float
            Option to smooth the scalar/adder data outside of the spatial
            domain set by the threshold input. This alleviates the weird seams
            far from the domain of interest. This value is the standard
            deviation for the gaussian_filter kernel

        Returns
        -------
        out : dict
            Dictionary of values defining the mean/std of the bias + base
            data and the scalar + adder factors to correct the biased data
            like: bias_data * scalar + adder. Each value is of shape
            (lat, lon, time).
        """
        for key, arr in out.items():
            nan_mask = np.isnan(arr[..., 0])
            for idt in range(self.NT):
                arr[..., idt] = nn_fill_array(arr[..., idt])
                if smooth_extend > 0:
                    arr_smooth = gaussian_filter(arr[..., idt],
                                                 smooth_extend,
                                                 mode='nearest')
                    out[key][nan_mask, idt] = arr_smooth[nan_mask]
        return out

    def write_outputs(self, fp_out, out):
        """Write outputs to an .h5 file.

        Parameters
        ----------
        fp_out : str | None
            Optional .h5 output file to write scalar and adder arrays.
        out : dict
            Dictionary of values defining the mean/std of the bias + base
            data and the scalar + adder factors to correct the biased data
            like: bias_data * scalar + adder. Each value is of shape
            (lat, lon, time).
        """

        if fp_out is not None:
            if not os.path.exists(os.path.dirname(fp_out)):
                os.makedirs(os.path.dirname(fp_out), exist_ok=True)

            with h5py.File(fp_out, 'w') as f:
                # pylint: disable=E1136
                lat = self.bias_dh.lat_lon[..., 0]
                lon = self.bias_dh.lat_lon[..., 1]
                f.create_dataset('latitude', data=lat)
                f.create_dataset('longitude', data=lon)
                for dset, data in out.items():
                    f.create_dataset(dset, data=data)

                for k, v in self.meta.items():
                    f.attrs[k] = json.dumps(v)

                logger.info('Wrote scalar adder factors to file: {}'
                            .format(fp_out))

    def run(self, knn, threshold=0.6, fp_out=None, max_workers=None,
            daily_reduction='avg', fill_extend=True, smooth_extend=0):
        """Run linear correction factor calculations for every site in the bias
        dataset

        Parameters
        ----------
        knn : int
            Number of nearest neighbors to aggregate from the base data when
            comparing to a single site from the bias data.
        threshold : float
            If the bias data coordinate is on average further from the base
            data coordinates than this threshold, no bias correction factors
            will be calculated directly and will just be filled from nearest
            neighbor (if fill_extend=True, else it will be nan).
        fp_out : str | None
            Optional .h5 output file to write scalar and adder arrays.
        max_workers : int
            Number of workers to run in parallel. 1 is serial and None is all
            available.
        daily_reduction : None | str
            Option to do a reduction of the hourly+ source base data to daily
            data. Can be None (no reduction, keep source time frequency), "avg"
            (daily average), "max" (daily max), or "min" (daily min)
        fill_extend : bool
            Flag to fill data past threshold using spatial nearest neighbor. If
            False, the extended domain will be left as NaN.
        smooth_extend : float
            Option to smooth the scalar/adder data outside of the spatial
            domain set by the threshold input. This alleviates the weird seams
            far from the domain of interest. This value is the standard
            deviation for the gaussian_filter kernel

        Returns
        -------
        out : dict
            Dictionary of values defining the mean/std of the bias + base
            data and the scalar + adder factors to correct the biased data
            like: bias_data * scalar + adder. Each value is of shape
            (lat, lon, time).
        """
        logger.debug('Starting linear correction calculation...')

        keys = [f'{self.bias_feature}_scalar',
                f'{self.bias_feature}_adder',
                f'bias_{self.bias_feature}_mean',
                f'bias_{self.bias_feature}_std',
                f'base_{self.base_dset}_mean',
                f'base_{self.base_dset}_std',
                ]
        out = {k: np.full(self.bias_gid_raster.shape + (self.NT,),
                          np.nan, np.float32)
               for k in keys}

        logger.info('Initialized scalar / adder with shape: {}'
                    .format(self.bias_gid_raster.shape))

        if max_workers == 1:
            logger.debug('Running serial calculation.')
            for i, (bias_gid, row) in enumerate(self.bias_meta.iterrows()):
                raster_loc = np.where(self.bias_gid_raster == bias_gid)
                coord = row[['latitude', 'longitude']]
                dist, base_gid = self.base_tree.query(coord, k=knn)

                if np.mean(dist) < threshold:
                    bias_data = self.get_bias_data(bias_gid)
                    single_out = self._run_single(bias_data, self.base_fps,
                                                  self.bias_feature,
                                                  self.base_dset, base_gid,
                                                  self.base_handler,
                                                  daily_reduction,
                                                  self.bias_ti, self.decimals)
                    for key, arr in single_out.items():
                        out[key][raster_loc] = arr

                logger.info('Completed bias calculations for {} out of {} '
                            'sites'.format(i + 1, len(self.bias_meta)))

        else:
            logger.debug('Running parallel calculation with {} workers.'
                         .format(max_workers))
            with ProcessPoolExecutor(max_workers=max_workers) as exe:
                futures = {}
                for bias_gid, bias_row in self.bias_meta.iterrows():
                    raster_loc = np.where(self.bias_gid_raster == bias_gid)
                    coord = bias_row[['latitude', 'longitude']]
                    dist, base_gid = self.base_tree.query(coord, k=knn)

                    if np.mean(dist) < threshold:
                        bias_data = self.get_bias_data(bias_gid)

                        future = exe.submit(self._run_single, bias_data,
                                            self.base_fps, self.bias_feature,
                                            self.base_dset, base_gid,
                                            self.base_handler, daily_reduction,
                                            self.bias_ti, self.decimals)
                        futures[future] = raster_loc

                logger.debug('Finished launching futures.')
                for i, future in enumerate(as_completed(futures)):
                    raster_loc = futures[future]
                    single_out = future.result()
                    for key, arr in single_out.items():
                        out[key][raster_loc] = arr

                    logger.info('Completed bias calculations for {} out of {} '
                                'sites'.format(i + 1, len(futures)))

        logger.info('Finished calculating bias correction factors.')

        if fill_extend:
            out = self.fill_extend(out, smooth_extend)

        self.write_outputs(fp_out, out)

        return out


class MonthlyLinearCorrection(LinearCorrection):
    """Calculate linear correction *scalar +adder factors to bias correct data

    This calculation operates on single bias sites on a montly basis
    """

    # size of the time dimension, 12 is monthly bias correction
    NT = 12

    @classmethod
    def _run_single(cls, bias_data, base_fps, bias_feature, base_dset,
                    base_gid, base_handler, daily_reduction, bias_ti,
                    decimals):
        """Find the nominal scalar + adder combination to bias correct data
        at a single site"""

        base_data, base_ti = cls.get_base_data(base_fps, base_dset,
                                               base_gid, base_handler,
                                               daily_reduction=daily_reduction,
                                               decimals=decimals)

        base_arr = np.full(cls.NT, np.nan, dtype=np.float32)
        out = {f'bias_{bias_feature}_mean': base_arr.copy(),
               f'bias_{bias_feature}_std': base_arr.copy(),
               f'base_{base_dset}_mean': base_arr.copy(),
               f'base_{base_dset}_std': base_arr.copy(),
               f'{bias_feature}_scalar': base_arr.copy(),
               f'{bias_feature}_adder': base_arr.copy(),
               }

        for month in range(1, 13):
            bias_mask = bias_ti.month == month
            base_mask = base_ti.month == month

            if any(bias_mask) and any(base_mask):
                mout = cls.get_linear_correction(bias_data[bias_mask],
                                                 base_data[base_mask],
                                                 bias_feature,
                                                 base_dset)
                for k, v in mout.items():
                    out[k][month - 1] = v

        return out
