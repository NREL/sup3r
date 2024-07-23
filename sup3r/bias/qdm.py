"""QDM related methods to correct and bias and trend

Procedures to apply Quantile Delta Method correction and derived methods such
as PresRat.
"""

import copy
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
from rex.utilities.bc_utils import (
    QuantileDeltaMapping,
    sample_q_invlog,
    sample_q_linear,
    sample_q_log,
)
from typing import Optional

from sup3r.preprocessing.data_handling.base import DataHandler
from sup3r.utilities.utilities import expand_paths
from .bias_calc import DataRetrievalBase
from .mixins import FillAndSmoothMixin, ZeroRateMixin

logger = logging.getLogger(__name__)


class QuantileDeltaMappingCorrection(FillAndSmoothMixin, DataRetrievalBase):
    """Estimate probability distributions required by Quantile Delta Mapping

    The main purpose of this class is to estimate the probability
    distributions required by Quantile Delta Mapping (QDM) ([Cannon2015]_)
    technique. Therefore, the name 'Correction' can be misleading since it is
    not the correction *per se*, but that was used to keep consistency within
    this module.

    The QDM technique corrects bias and trend by comparing the data
    distributions of three datasets: a historical reference, a biased
    reference, and a biased target to correct (in Cannon et. al. (2015)
    called: historical observed, historical modeled, and future modeled
    respectively). Those three probability distributions provided here
    can be, for instance, used by
    :func:`~sup3r.bias.bias_transforms.local_qdm_bc` to actually correct
    a dataset.
    """

    # Default to ~monthly scale centered on every ~15 days
    NT = 24
    """Number of times to calculate QDM parameters in a year"""
    WINDOW_SIZE = 30
    """Window width in days"""

    def __init__(self,
                 base_fps,
                 bias_fps,
                 bias_fut_fps,
                 base_dset,
                 bias_feature,
                 distance_upper_bound=None,
                 target=None,
                 shape=None,
                 base_handler='Resource',
                 bias_handler='DataHandlerNCforCC',
                 base_handler_kwargs=None,
                 bias_handler_kwargs=None,
                 decimals=None,
                 match_zero_rate=False,
                 n_quantiles=101,
                 dist='empirical',
                 relative=None,
                 sampling='linear',
                 log_base=10,
                 ):
        """
        Parameters
        ----------
        base_fps : list | str
            One or more baseline .h5 filepaths representing non-biased data to
            use
            to correct the biased dataset (observed historical in Cannon et.
            al. (2015)). This is typically several years of WTK or NSRDB files.
        bias_fps : list | str
            One or more biased .nc or .h5 filepaths representing the biased
            data
            to be compared with the baseline data (modeled historical in Cannon
            et. al. (2015)). This is typically several years of GCM .nc files.
        bias_fut_fps : list | str
            Consistent data to `bias_fps` but for a different time period
            (modeled
            future in Cannon et. al. (2015)). This is the dataset that would be
            corrected, while `bias_fsp` is used to provide a transformation map
            with the baseline data.
        base_dset : str
            A single dataset from the base_fps to retrieve. In the case of wind
            components, this can be U_100m or V_100m which will retrieve
            windspeed and winddirection and derive the U/V component.
        bias_feature : str
            This is the biased feature from bias_fps to retrieve. This should
            be a single feature name corresponding to base_dset
        distance_upper_bound : float
            Upper bound on the nearest neighbor distance in decimal degrees.
            This should be the approximate resolution of the low-resolution
            bias data. None (default) will calculate this based on the median
            distance between points in bias_fps
        target : tuple
            (lat, lon) lower left corner of raster to retrieve from bias_fps.
            If None then the lower left corner of the full domain will be used.
        shape : tuple
            (rows, cols) grid size to retrieve from bias_fps. If None then the
            full domain shape will be used.
        base_handler : str
            Name of rex resource handler or sup3r.preprocessing.data_handling
            class to be retrieved from the rex/sup3r library. If a
            sup3r.preprocessing.data_handling class is used, all data will be
            loaded in this class' initialization and the subsequent bias
            calculation will be done in serial
        bias_handler : str
            Name of the bias data handler class to be retrieved from the
            sup3r.preprocessing.data_handling library.
        base_handler_kwargs : dict | None
            Optional kwargs to send to the initialization of the base_handler
            class
        bias_handler_kwargs : dict | None
            Optional kwargs to send to the initialization of the bias_handler
            class
        decimals : int | None
            Option to round bias and base data to this number of
            decimals, this gets passed to np.around(). If decimals
            is negative, it specifies the number of positions to
            the left of the decimal point.
        match_zero_rate : bool
            Option to fix the frequency of zero values in the biased data. The
            lowest percentile of values in the biased data will be set to zero
            to match the percentile of zeros in the base data. If
            SkillAssessment is being run and this is True, the distributions
            will not be mean-centered. This helps resolve the issue where
            global climate models produce too many days with small
            precipitation totals e.g., the "drizzle problem" [Polade2014]_.
        dist : str, default="empirical",
            Define the type of distribution, which can be "empirical" or any
            parametric distribution defined in "scipy".
        n_quantiles : int, default=101
            Defines the number of quantiles (between 0 and 1) for an empirical
            distribution.
        sampling : str, default="linear",
            Defines how the quantiles are sampled. For instance, 'linear' will
            result in a linearly spaced quantiles. Other options are: 'log'
            and 'invlog'.
        log_base : int or float, default=10
            Log base value if sampling is "log" or "invlog".

        Attributes
        ----------
        NT : int
            Number of times to calculate QDM parameters equally distributed
            along a year. For instance, `NT=1` results in a single set of
            parameters while `NT=12` is approximately every month.
        WINDOW_SIZE : int
            Total time window period to be considered for each time QDM is
            calculated. For instance, `WINDOW_SIZE=30` with `NT=12` would
            result in approximately monthly estimates.

        See Also
        --------
        sup3r.bias.bias_transforms.local_qdm_bc :
            Bias correction using QDM.
        sup3r.preprocessing.data_handling.DataHandler :
            Bias correction using QDM directly from a derived handler.
        rex.utilities.bc_utils.QuantileDeltaMapping
            Quantile Delta Mapping method and support functions. Since
            :mod:`rex.utilities.bc_utils` is used here, the arguments
            ``dist``, ``n_quantiles``, ``sampling``, and ``log_base``
            must be consitent with that package/module.

        Notes
        -----
        One way of using this class is by saving the distributions definitions
        obtained here with the method :meth:`.write_outputs` and then use that
        file with :func:`~sup3r.bias.bias_transforms.local_qdm_bc` or through
        a derived :class:`~sup3r.preprocessing.data_handling.base.DataHandler`.
        **ATTENTION**, be careful handling that file of parameters. There is
        no checking process and one could missuse the correction estimated for
        the wrong dataset.

        References
        ----------
        .. [Cannon2015] Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015).
           Bias correction of GCM precipitation by quantile mapping: how well
           do methods preserve changes in quantiles and extremes?. Journal of
           Climate, 28(17), 6938-6959.
        """

        self.n_quantiles = n_quantiles
        self.dist = dist
        self.relative = relative
        self.sampling = sampling
        self.log_base = log_base

        super().__init__(base_fps=base_fps,
                         bias_fps=bias_fps,
                         base_dset=base_dset,
                         bias_feature=bias_feature,
                         distance_upper_bound=distance_upper_bound,
                         target=target,
                         shape=shape,
                         base_handler=base_handler,
                         bias_handler=bias_handler,
                         base_handler_kwargs=base_handler_kwargs,
                         bias_handler_kwargs=bias_handler_kwargs,
                         decimals=decimals,
                         match_zero_rate=match_zero_rate,
                         )

        self.bias_fut_fps = bias_fut_fps

        self.bias_fut_fps = expand_paths(self.bias_fut_fps)

        self.bias_fut_dh = self.bias_handler(self.bias_fut_fps,
                                             [self.bias_feature],
                                             target=self.target,
                                             shape=self.shape,
                                             val_split=0.0,
                                             **self.bias_handler_kwargs,
                                             )

    def _init_out(self):
        """Initialize output arrays `self.out`

        Three datasets are saved here with information to reconstruct the
        probability distributions for the three datasets (see class
        documentation).
        """
        keys = [f'bias_{self.bias_feature}_params',
                f'bias_fut_{self.bias_feature}_params',
                f'base_{self.base_dset}_params',
                ]
        shape = (*self.bias_gid_raster.shape, self.NT, self.n_quantiles)
        arr = np.full(shape, np.nan, np.float32)
        self.out = {k: arr.copy() for k in keys}

        self.time_window_center = self._window_center(self.NT)

    @staticmethod
    def _window_center(ntimes: int):
        """A sequence of equally spaced `ntimes` day of year

        This is used to identify the center of moving windows to apply filters
        and masks. For instance, if `ntimes` equal to 12, it would return
        12 equal periods, i.e. approximately the months' center time.

        This is conveniently shifted one half time interval so that December
        31st would be closest to the last interval of the year instead of the
        first.

        Leap years are neglected, but if processed here, the doy 366 is
        included in the relevant windows as an extra data, which would
        cause a negligible error.

        Parameters
        ----------
        ntimes : int
            Number of intervals in one year. Choose 12 for approximately
            monthly time scales.

        Returns
        -------
        np.ndarray :
            Sequence of days of the year equally spaced and shifted by half
            window size, thus `ntimes`=12 results in approximately [15, 45,
            ...]. It includes the fraction of a day, thus 15.5 is equivalent
            to January 15th, 12:00h.
        """
        assert ntimes > 0, "Requires a positive number of intervals"

        dt = 365 / ntimes
        return np.arange(dt / 2, 366, dt)

    # pylint: disable=W0613
    @classmethod
    def _run_single(cls,
                    bias_data,
                    bias_fut_data,
                    base_fps,
                    bias_feature,
                    base_dset,
                    base_gid,
                    base_handler,
                    daily_reduction,
                    *,
                    bias_ti,
                    bias_fut_ti,
                    decimals,
                    dist,
                    relative,
                    sampling,
                    n_samples,
                    log_base,
                    base_dh_inst=None,
                    ):
        """Estimate probability distributions at a single site"""

        base_data, base_ti = cls.get_base_data(base_fps,
                                               base_dset,
                                               base_gid,
                                               base_handler,
                                               daily_reduction=daily_reduction,
                                               decimals=decimals,
                                               base_dh_inst=base_dh_inst)

        window_size = cls.WINDOW_SIZE or 365 / cls.NT
        window_center = cls._window_center(cls.NT)

        template = np.full((cls.NT, n_samples), np.nan, np.float32)
        out = {}

        for nt, idt in enumerate(window_center):
            base_idx = cls.window_mask(base_ti.day_of_year, idt, window_size)
            bias_idx = cls.window_mask(bias_ti.day_of_year, idt, window_size)
            bias_fut_idx = cls.window_mask(bias_fut_ti.day_of_year,
                                           idt,
                                           window_size)

            if any(base_idx) and any(bias_idx) and any(bias_fut_idx):
                tmp = cls.get_qdm_params(bias_data[bias_idx],
                                         bias_fut_data[bias_fut_idx],
                                         base_data[base_idx],
                                         bias_feature,
                                         base_dset,
                                         sampling,
                                         n_samples,
                                         log_base)
                for k, v in tmp.items():
                    if k not in out:
                        out[k] = template.copy()
                    out[k][(nt), :] = v

        return out

    @staticmethod
    def get_qdm_params(bias_data,
                       bias_fut_data,
                       base_data,
                       bias_feature,
                       base_dset,
                       sampling,
                       n_samples,
                       log_base,
                       ):
        """Get quantiles' cut point for given datasets

        Estimate the quantiles' cut points for each of the three given
        datasets. Lacking a good analytical approximation, such as one of
        the parametric distributions, those quantiles can be used to
        approximate the statistical distribution of those datasets.

        Parameters
        ----------
        bias_data : np.ndarray
            1D array of biased data observations.
        bias_fut_data : np.ndarray
            1D array of biased data observations.
        base_data : np.ndarray
            1D array of base data observations.
        bias_feature : str
            This is the biased feature from bias_fps to retrieve. This should
            be a single feature name corresponding to base_dset.
        base_dset : str
            A single dataset from the base_fps to retrieve. In the case of wind
            components, this can be U_100m or V_100m which will retrieve
            windspeed and winddirection and derive the U/V component.
        sampling : str
            Defines how the quantiles are sampled. For instance, 'linear' will
            result in a linearly spaced quantiles. Other options are: 'log'
            and 'invlog'.
        n_samples : int
            Number of points to sample between 0 and 1, i.e. number of
            quantiles.
        log_base : int | float
            Log base value.

        Returns
        -------
        out : dict
            Dictionary of the quantiles' cut points. Note that to make sense
            of those cut point values, one need to know the given arguments
            such as `log_base`. For instance, the sequence [-1, 0, 2] are,
            if sampling was linear, the minimum, median, and maximum values
            respectively. The expected keys are "bias_{bias_feature}_params",
            "bias_fut_{bias_feature}_params", and "base_{base_dset}_params".

        See Also
        --------
        rex.utilities.bc_utils : Sampling scales, such as `sample_q_linear()`
        """

        if sampling == 'linear':
            quantiles = sample_q_linear(n_samples)
        elif sampling == 'log':
            quantiles = sample_q_log(n_samples, log_base)
        elif sampling == 'invlog':
            quantiles = sample_q_invlog(n_samples, log_base)
        else:
            msg = ('sampling option must be linear, log, or invlog, but '
                   'received: {}'.format(sampling)
                   )
            logger.error(msg)
            raise KeyError(msg)

        out = {
            f'bias_{bias_feature}_params': np.quantile(bias_data, quantiles),
            f'bias_fut_{bias_feature}_params': np.quantile(bias_fut_data,
                                                           quantiles),
            f'base_{base_dset}_params': np.quantile(base_data, quantiles),
        }

        return out

    def write_outputs(self, fp_out, out=None):
        """Write outputs to an .h5 file.

        Parameters
        ----------
        fp_out : str | None
            An HDF5 filename to write the estimated statistical distributions.
        out : dict, optional
            A dictionary with the three statistical distribution parameters.
            If not given, it uses :attr:`.out`.
        """

        out = out or self.out

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
                f.attrs["dist"] = self.dist
                f.attrs["sampling"] = self.sampling
                f.attrs["log_base"] = self.log_base
                f.attrs["base_fps"] = self.base_fps
                f.attrs["bias_fps"] = self.bias_fps
                f.attrs["bias_fut_fps"] = self.bias_fut_fps
                f.attrs["time_window_center"] = self.time_window_center
                logger.info(
                    'Wrote quantiles to file: {}'.format(fp_out))

    def run(self,
            fp_out=None,
            max_workers=None,
            daily_reduction='avg',
            fill_extend=True,
            smooth_extend=0,
            smooth_interior=0,
            ):
        """Estimate the statistical distributions for each location

        Parameters
        ----------
        fp_out : str | None
            Optional .h5 output file to write scalar and adder arrays.
        max_workers : int, optional
            Number of workers to run in parallel. 1 is serial and None is all
            available.
        daily_reduction : None | str
            Option to do a reduction of the hourly+ source base data to daily
            data. Can be None (no reduction, keep source time frequency), "avg"
            (daily average), "max" (daily max), "min" (daily min),
            "sum" (daily sum/total)

        Returns
        -------
        out : dict
            Dictionary with parameters defining the statistical distributions
            for each of the three given datasets. Each value has dimensions
            (lat, lon, n-parameters).
        """

        logger.debug('Calculate CDF parameters for QDM')

        logger.info('Initialized params with shape: {}'
                    .format(self.bias_gid_raster.shape))
        self.bad_bias_gids = []

        # sup3r DataHandler opening base files will load all data in parallel
        # during the init and should not be passed in parallel to workers
        if isinstance(self.base_dh, DataHandler):
            max_workers = 1

        if max_workers == 1:
            logger.debug('Running serial calculation.')
            for i, bias_gid in enumerate(self.bias_meta.index):
                raster_loc = np.where(self.bias_gid_raster == bias_gid)
                _, base_gid = self.get_base_gid(bias_gid)

                if not base_gid.any():
                    self.bad_bias_gids.append(bias_gid)
                    logger.debug(f'No base data for bias_gid: {bias_gid}. '
                                 'Adding it to bad_bias_gids')
                else:
                    bias_data = self.get_bias_data(bias_gid)
                    bias_fut_data = self.get_bias_data(bias_gid,
                                                       self.bias_fut_dh)
                    single_out = self._run_single(
                        bias_data,
                        bias_fut_data,
                        self.base_fps,
                        self.bias_feature,
                        self.base_dset,
                        base_gid,
                        self.base_handler,
                        daily_reduction,
                        bias_ti=self.bias_fut_dh.time_index,
                        bias_fut_ti=self.bias_fut_dh.time_index,
                        decimals=self.decimals,
                        dist=self.dist,
                        relative=self.relative,
                        sampling=self.sampling,
                        n_samples=self.n_quantiles,
                        log_base=self.log_base,
                        base_dh_inst=self.base_dh,
                    )
                    for key, arr in single_out.items():
                        self.out[key][raster_loc] = arr

                logger.info('Completed bias calculations for {} out of {} '
                            'sites'.format(i + 1, len(self.bias_meta)))

        else:
            logger.debug(
                'Running parallel calculation with {} workers.'.format(
                    max_workers))
            with ProcessPoolExecutor(max_workers=max_workers) as exe:
                futures = {}
                for bias_gid in self.bias_meta.index:
                    raster_loc = np.where(self.bias_gid_raster == bias_gid)
                    _, base_gid = self.get_base_gid(bias_gid)

                    if not base_gid.any():
                        self.bad_bias_gids.append(bias_gid)
                    else:
                        bias_data = self.get_bias_data(bias_gid)
                        bias_fut_data = self.get_bias_data(bias_gid,
                                                           self.bias_fut_dh)
                        future = exe.submit(
                            self._run_single,
                            bias_data,
                            bias_fut_data,
                            self.base_fps,
                            self.bias_feature,
                            self.base_dset,
                            base_gid,
                            self.base_handler,
                            daily_reduction,
                            bias_ti=self.bias_fut_dh.time_index,
                            bias_fut_ti=self.bias_fut_dh.time_index,
                            decimals=self.decimals,
                            dist=self.dist,
                            relative=self.relative,
                            sampling=self.sampling,
                            n_samples=self.n_quantiles,
                            log_base=self.log_base,
                        )
                        futures[future] = raster_loc

                logger.debug('Finished launching futures.')
                for i, future in enumerate(as_completed(futures)):
                    raster_loc = futures[future]
                    single_out = future.result()
                    for key, arr in single_out.items():
                        self.out[key][raster_loc] = arr

                    logger.info('Completed bias calculations for {} out of {} '
                                'sites'.format(i + 1, len(futures)))

        logger.info('Finished calculating bias correction factors.')

        self.out = self.fill_and_smooth(self.out, fill_extend, smooth_extend,
                                        smooth_interior)

        self.write_outputs(fp_out, self.out)

        return copy.deepcopy(self.out)

    @staticmethod
    def window_mask(doy, d0, window_size):
        """An index of elements within a given time window

        Create an index of days of the year within the target time window. It
        only considers the day of the year (doy), hence, it is limited to time
        scales smaller than annual.

        Parameters
        ----------
        doy : np.ndarray
            An unordered array of days of year, i.e. January 1st is 1.
        d0 : int
            Center point of the target window [day of year].
        window_size : float
            Total size of the target window, i.e. the window covers half this
            value on each side of d0. Note that it has the same units of doy,
            thus it is equal to the number of points only if doy is daily.

        Returns
        -------
        np.array
            An boolean array with the same shape of the given `doy`, where
            True means that position is within the target window.

        Notes
        -----
        Leap years have the day 366 included in the output index, but a
        precise shift is not calculated, resulting in a negligible error
        in large datasets.
        """
        d_start = d0 - window_size / 2
        d_end = d0 + window_size / 2

        if d_start < 0:
            idx = (doy > 365 + d_start) | (doy < d_end)
        elif d_end > 365:
            idx = (doy > d_start) | (doy < d_end - 365)
        else:
            idx = (doy > d_start) & (doy < d_end)

        return idx


class PresRat(ZeroRateMixin, QuantileDeltaMappingCorrection):
    """PresRat bias correction method (precipitation)

    The PresRat correction [Pierce2015]_ is defined as the combination of
    three steps:
    * Use the model-predicted change ratio (with the CDFs);
    * The treatment of zero-precipitation days (with the fraction of dry days);
    * The final correction factor (K) to preserve the mean (ratio between both
      estimated means);

    To keep consistency with the full sup3r pipeline, PresRat was implemented
    as follows:

    1) Define zero rate from observations (oh)

    Using the historical observations, estimate the zero rate precipitation
    for each gridpoint. It is expected a long time series here, such as
    decadal or longer. A threshold larger than zero is an option here.

    The result is a 2D (space) `zero_rate` (non-dimensional).

    2) Find the threshold for each gridpoint (mh)

    Using the zero rate from the previous step, identify the magnitude
    threshold for each gridpoint that satisfies that dry days rate.

    Note that Pierce (2015) impose `tau` >= 0.01 mm/day for precipitation.

    The result is a 2D (space) threshold `tau` with the same dimensions
    of the data been corrected. For instance, it could be mm/day for
    precipitation.

    3) Define `Z_fg` using `tau` (mf)

    The `tau` that was defined with the *modeled historical*, is now
    used as a threshold on *modeled future* before any correction to define
    the equivalent zero rate in the future.

    The result is a 2D (space) rate (non-dimensional)

    4) Estimate `tau_fut` using `Z_fg`

    Since sup3r process data in smaller chunks, it wouldn't be possible to
    apply the rate `Z_fg` directly. To address that, all *modeled future*
    data is corrected with QDM, and applying `Z_fg` it is defined the
    `tau_fut`.

    References
    ----------
    .. [Pierce2015] Pierce, D. W., Cayan, D. R., Maurer, E. P., Abatzoglou, J.
       T., & Hegewisch, K. C. (2015). Improved bias correction techniques for
       hydrological simulations of climate change. Journal of Hydrometeorology,
       16(6), 2421-2442.
    """
    def _init_out(self):
        super()._init_out()

        shape = (*self.bias_gid_raster.shape, 1)
        self.out[f'{self.base_dset}_zero_rate'] = np.full(shape,
                                                          np.nan,
                                                          np.float32)
        self.out[f'{self.bias_feature}_tau_fut'] = np.full(shape,
                                                           np.nan,
                                                           np.float32)
        shape = (*self.bias_gid_raster.shape, self.NT)
        self.out[f'{self.bias_feature}_k_factor'] = np.full(
            shape, np.nan, np.float32)

    # pylint: disable=W0613
    @classmethod
    def _run_single(cls,
                    bias_data,
                    bias_fut_data,
                    base_fps,
                    bias_feature,
                    base_dset,
                    base_gid,
                    base_handler,
                    daily_reduction,
                    *,
                    bias_ti,
                    bias_fut_ti,
                    decimals,
                    dist,
                    relative,
                    sampling,
                    n_samples,
                    log_base,
                    zero_rate_threshold,
                    base_dh_inst=None,
                    ):
        """Estimate probability distributions at a single site

        TODO! This should be refactored. There is too much redundancy in
        the code. Let's make it work first, and optimize later.
        """
        base_data, base_ti = cls.get_base_data(
            base_fps,
            base_dset,
            base_gid,
            base_handler,
            daily_reduction=daily_reduction,
            decimals=decimals,
            base_dh_inst=base_dh_inst,
        )

        window_size = cls.WINDOW_SIZE or 365 / cls.NT
        window_center = cls._window_center(cls.NT)

        template = np.full((cls.NT, n_samples), np.nan, np.float32)
        out = {}
        corrected_fut_data = np.full_like(bias_fut_data, np.nan)
        for nt, t in enumerate(window_center):
            # Define indices for which data goes in the current time window
            base_idx = cls.window_mask(base_ti.day_of_year, t, window_size)
            bias_idx = cls.window_mask(bias_ti.day_of_year, t, window_size)
            bias_fut_idx = cls.window_mask(bias_fut_ti.day_of_year,
                                           t,
                                           window_size)

            if any(base_idx) and any(bias_idx) and any(bias_fut_idx):
                tmp = cls.get_qdm_params(bias_data[bias_idx],
                                         bias_fut_data[bias_fut_idx],
                                         base_data[base_idx],
                                         bias_feature,
                                         base_dset,
                                         sampling,
                                         n_samples,
                                         log_base)
                for k, v in tmp.items():
                    if k not in out:
                        out[k] = template.copy()
                    out[k][(nt), :] = v

            QDM = QuantileDeltaMapping(
                out[f'base_{base_dset}_params'][nt],
                out[f'bias_{bias_feature}_params'][nt],
                out[f'bias_fut_{bias_feature}_params'][nt],
                dist=dist,
                relative=relative,
                sampling=sampling,
                log_base=log_base
            )
            subset = bias_fut_data[bias_fut_idx]
            corrected_fut_data[bias_fut_idx] = QDM(subset).squeeze()

        # Step 1: Define zero rate from observations
        assert base_data.ndim == 1
        obs_zero_rate = cls.zero_precipitation_rate(
            base_data, zero_rate_threshold)
        out[f'{base_dset}_zero_rate'] = obs_zero_rate

        # Step 2: Find tau for each grid point

        # Removed NaN handling, thus reinforce finite-only data.
        assert np.isfinite(bias_data).all(), "Unexpected invalid values"
        assert bias_data.ndim == 1, "Assumed bias_data to be 1D"
        n_threshold = round(obs_zero_rate * bias_data.size)
        n_threshold = min(n_threshold, bias_data.size - 1)
        tau = np.sort(bias_data)[n_threshold]
        # Pierce (2015) imposes 0.01 mm/day
        # tau = max(tau, 0.01)

        # Step 3: Find Z_gf as the zero rate in mf
        assert np.isfinite(bias_fut_data).all(), "Unexpected invalid values"
        z_fg = (bias_fut_data < tau).astype('i').sum() / bias_fut_data.size

        # Step 4: Estimate tau_fut with corrected mf
        tau_fut = np.sort(corrected_fut_data)[round(
            z_fg * corrected_fut_data.size)]

        out[f'{bias_feature}_tau_fut'] = tau_fut

        # ---- K factor ----

        k = np.full(cls.NT, np.nan, np.float32)
        for nt, t in enumerate(window_center):
            base_idx = cls.window_mask(base_ti.day_of_year, t, window_size)
            bias_idx = cls.window_mask(bias_ti.day_of_year, t, window_size)
            bias_fut_idx = cls.window_mask(bias_fut_ti.day_of_year,
                                           t,
                                           window_size)

            oh = base_data[base_idx].mean()
            mh = bias_data[bias_idx].mean()
            mf = bias_fut_data[bias_fut_idx].mean()
            mf_unbiased = corrected_fut_data[bias_fut_idx].mean()

            x = mf / mh
            x_hat = mf_unbiased / oh
            k[nt] = x / x_hat

        out[f'{bias_feature}_k_factor'] = k

        return out

    def run(
        self,
        fp_out=None,
        max_workers=None,
        daily_reduction='avg',
        fill_extend=True,
        smooth_extend=0,
        smooth_interior=0,
        zero_rate_threshold=0.0,
    ):
        """Estimate the required information for PresRat correction

        Parameters
        ----------
        fp_out : str | None
            Optional .h5 output file to write scalar and adder arrays.
        max_workers : int, optional
            Number of workers to run in parallel. 1 is serial and None is all
            available.
        daily_reduction : None | str
            Option to do a reduction of the hourly+ source base data to daily
            data. Can be None (no reduction, keep source time frequency), "avg"
            (daily average), "max" (daily max), "min" (daily min),
            "sum" (daily sum/total)
        fill_extend : bool
            Whether to fill data extending beyond the base meta data with
            nearest neighbor values.
        smooth_extend : float
            Option to smooth the scalar/adder data outside of the spatial
            domain set by the threshold input. This alleviates the weird seams
            far from the domain of interest. This value is the standard
            deviation for the gaussian_filter kernel
        smooth_interior : float
            Value to use to smooth the scalar/adder data inside of the spatial
            domain set by the threshold input. This can reduce the effect of
            extreme values within aggregations over large number of pixels.
            This value is the standard deviation for the gaussian_filter
            kernel.
        zero_rate_threshold : float, default=0.0
            Threshold value used to determine the zero rate in the observed
            historical dataset. For instance, 0.01 means that anything less
            than that will be considered negligible, hence equal to zero.

        Returns
        -------
        out : dict
            Dictionary with parameters defining the statistical distributions
            for each of the three given datasets. Each value has dimensions
            (lat, lon, n-parameters).
        """
        logger.debug('Calculate CDF parameters for QDM')

        logger.info(
            'Initialized params with shape: {}'.format(
                self.bias_gid_raster.shape
            )
        )
        self.bad_bias_gids = []

        # sup3r DataHandler opening base files will load all data in parallel
        # during the init and should not be passed in parallel to workers
        if isinstance(self.base_dh, DataHandler):
            max_workers = 1

        if max_workers == 1:
            logger.debug('Running serial calculation.')
            for i, bias_gid in enumerate(self.bias_meta.index):
                raster_loc = np.where(self.bias_gid_raster == bias_gid)
                _, base_gid = self.get_base_gid(bias_gid)

                if not base_gid.any():
                    self.bad_bias_gids.append(bias_gid)
                    logger.debug(
                        f'No base data for bias_gid: {bias_gid}. '
                        'Adding it to bad_bias_gids'
                    )
                else:
                    bias_data = self.get_bias_data(bias_gid)
                    bias_fut_data = self.get_bias_data(
                        bias_gid, self.bias_fut_dh
                    )
                    single_out = self._run_single(
                        bias_data,
                        bias_fut_data,
                        self.base_fps,
                        self.bias_feature,
                        self.base_dset,
                        base_gid,
                        self.base_handler,
                        daily_reduction,
                        bias_ti=self.bias_fut_dh.time_index,
                        bias_fut_ti=self.bias_fut_dh.time_index,
                        decimals=self.decimals,
                        dist=self.dist,
                        relative=self.relative,
                        sampling=self.sampling,
                        n_samples=self.n_quantiles,
                        log_base=self.log_base,
                        base_dh_inst=self.base_dh,
                        zero_rate_threshold=zero_rate_threshold,
                    )
                    for key, arr in single_out.items():
                        self.out[key][raster_loc] = arr

                logger.info(
                    'Completed bias calculations for {} out of {} '
                    'sites'.format(i + 1, len(self.bias_meta))
                )

        else:
            logger.debug(
                'Running parallel calculation with {} workers.'.format(
                    max_workers
                )
            )
            with ProcessPoolExecutor(max_workers=max_workers) as exe:
                futures = {}
                for bias_gid in self.bias_meta.index:
                    raster_loc = np.where(self.bias_gid_raster == bias_gid)
                    _, base_gid = self.get_base_gid(bias_gid)

                    if not base_gid.any():
                        self.bad_bias_gids.append(bias_gid)
                    else:
                        bias_data = self.get_bias_data(bias_gid)
                        bias_fut_data = self.get_bias_data(
                            bias_gid, self.bias_fut_dh
                        )
                        future = exe.submit(
                            self._run_single,
                            bias_data,
                            bias_fut_data,
                            self.base_fps,
                            self.bias_feature,
                            self.base_dset,
                            base_gid,
                            self.base_handler,
                            daily_reduction,
                            bias_ti=self.bias_fut_dh.time_index,
                            bias_fut_ti=self.bias_fut_dh.time_index,
                            decimals=self.decimals,
                            dist=self.dist,
                            relative=self.relative,
                            sampling=self.sampling,
                            n_samples=self.n_quantiles,
                            log_base=self.log_base,
                            zero_rate_threshold=zero_rate_threshold,
                        )
                        futures[future] = raster_loc

                logger.debug('Finished launching futures.')
                for i, future in enumerate(as_completed(futures)):
                    raster_loc = futures[future]
                    single_out = future.result()
                    for key, arr in single_out.items():
                        self.out[key][raster_loc] = arr

                    logger.info(
                        'Completed bias calculations for {} out of {} '
                        'sites'.format(i + 1, len(futures))
                    )

        logger.info('Finished calculating bias correction factors.')

        self.out = self.fill_and_smooth(
            self.out, fill_extend, smooth_extend, smooth_interior
        )

        extra_attrs = {
            'zero_rate_threshold': zero_rate_threshold,
            'time_window_center': self.time_window_center,
        }
        self.write_outputs(fp_out,
                           self.out,
                           extra_attrs=extra_attrs,
                           )

        return copy.deepcopy(self.out)

    def write_outputs(self, fp_out: str,
                      out: dict = None,
                      extra_attrs: Optional[dict] = None):
        """Write outputs to an .h5 file.

        Parameters
        ----------
        fp_out : str | None
            An HDF5 filename to write the estimated statistical distributions.
        out : dict, optional
            A dictionary with the three statistical distribution parameters.
            If not given, it uses :attr:`.out`.
        extra_attrs: dict, optional
            Extra attributes to be exported together with the dataset.

        Examples
        --------
        >>> mycalc = PresRat(...)
        >>> mycalc.write_outputs(fp_out="myfile.h5", out=mydictdataset,
        ...   extra_attrs={'zero_rate_threshold': 0.01})
        """

        out = out or self.out

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
                f.attrs['dist'] = self.dist
                f.attrs['sampling'] = self.sampling
                f.attrs['log_base'] = self.log_base
                f.attrs['base_fps'] = self.base_fps
                f.attrs['bias_fps'] = self.bias_fps
                f.attrs['bias_fut_fps'] = self.bias_fut_fps
                if extra_attrs is not None:
                    for a, v in extra_attrs.items():
                        f.attrs[a] = v
                logger.info('Wrote quantiles to file: {}'.format(fp_out))
