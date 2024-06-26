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
    sample_q_invlog,
    sample_q_linear,
    sample_q_log,
)

from sup3r.preprocessing.data_handling.base import DataHandler
from sup3r.utilities.utilities import expand_paths
from .bias_calc import DataRetrievalBase
from .mixins import FillAndSmoothMixin

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
                 dist="empirical",
                 sampling="linear",
                 log_base=10):
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
                         match_zero_rate=match_zero_rate)

        self.bias_fut_fps = bias_fut_fps

        self.bias_fut_fps = expand_paths(self.bias_fut_fps)

        self.bias_fut_dh = self.bias_handler(self.bias_fut_fps,
                                             [self.bias_feature],
                                             target=self.target,
                                             shape=self.shape,
                                             val_split=0.0,
                                             **self.bias_handler_kwargs)

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
        shape = (*self.bias_gid_raster.shape, self.n_quantiles)
        arr = np.full(shape, np.nan, np.float32)
        self.out = {k: arr.copy() for k in keys}

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
                    decimals,
                    sampling,
                    n_samples,
                    log_base,
                    base_dh_inst=None,
                    ):
        """Estimate probability distributions at a single site"""

        base_data, _ = cls.get_base_data(base_fps,
                                         base_dset,
                                         base_gid,
                                         base_handler,
                                         daily_reduction=daily_reduction,
                                         decimals=decimals,
                                         base_dh_inst=base_dh_inst)

        out = cls.get_qdm_params(bias_data,
                                 bias_fut_data,
                                 base_data,
                                 bias_feature,
                                 base_dset,
                                 sampling,
                                 n_samples,
                                 log_base)
        return out

    @staticmethod
    def get_qdm_params(bias_data,
                       bias_fut_data,
                       base_data,
                       bias_feature,
                       base_dset,
                       sampling,
                       n_samples,
                       log_base):
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
                   'received: {}'.format(sampling))
            logger.error(msg)
            raise KeyError(msg)

        out = {
            f'bias_{bias_feature}_params': np.quantile(bias_data, quantiles),
            f'bias_fut_{bias_feature}_params': np.quantile(bias_fut_data,
                                                           quantiles),
            f'base_{base_dset}_params': np.quantile(base_data, quantiles)}

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
                logger.info(
                    'Wrote quantiles to file: {}'.format(fp_out))

    def run(self,
            fp_out=None,
            max_workers=None,
            daily_reduction='avg',
            fill_extend=True,
            smooth_extend=0,
            smooth_interior=0):
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
                    logger.debug(f"No base data for bias_gid: {bias_gid}. "
                                 "Adding it to bad_bias_gids")
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
                        self.decimals,
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
                            self.decimals,
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
