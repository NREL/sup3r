"""Bias correction class interface."""

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from sup3r.preprocessing import DataHandler

logger = logging.getLogger(__name__)


class AbstractBiasCorrection(ABC):
    """Minimal interface for bias correction classes"""

    @abstractmethod
    def _get_run_kwargs(self, **run_single_kwargs):
        """Get dictionary of kwarg dictionaries to use for calls to
        ``_run_single``. Each key-value pair is a bias_gid with the associated
        ``_run_single`` kwargs dict for that gid"""

    def _run(
        self,
        out,
        max_workers=None,
        fill_extend=True,
        smooth_extend=0,
        smooth_interior=0,
        **run_single_kwargs,
    ):
        """Run correction factor calculations for every site in the bias
        dataset

        Parameters
        ----------
        out : dict
            Dictionary of arrays to fill with bias correction factors.
        max_workers : int
            Number of workers to run in parallel. 1 is serial and None is all
            available.
        daily_reduction : None | str
            Option to do a reduction of the hourly+ source base data to daily
            data. Can be None (no reduction, keep source time frequency), "avg"
            (daily average), "max" (daily max), "min" (daily min),
            "sum" (daily sum/total)
        fill_extend : bool
            Flag to fill data past distance_upper_bound using spatial nearest
            neighbor. If False, the extended domain will be left as NaN.
        smooth_extend : float
            Option to smooth the scalar/adder data outside of the spatial
            domain set by the distance_upper_bound input. This alleviates the
            weird seams far from the domain of interest. This value is the
            standard deviation for the gaussian_filter kernel
        smooth_interior : float
            Option to smooth the scalar/adder data within the valid spatial
            domain.  This can reduce the affect of extreme values within
            aggregations over large number of pixels.
        run_single_kwargs: dict
            Additional kwargs that get sent to ``_run_single`` e.g.
            daily_reduction='avg', zero_rate_threshold=1.157e-7

        Returns
        -------
        out : dict
            Dictionary of values defining the mean/std of the bias + base data
            and correction factors to correct the biased data like: bias_data *
            scalar + adder. Each value is of shape (lat, lon, time).
        """
        self.bad_bias_gids = []

        task_kwargs = self._get_run_kwargs(**run_single_kwargs)

        # sup3r DataHandler opening base files will load all data in parallel
        # during the init and should not be passed in parallel to workers
        if isinstance(self.base_dh, DataHandler):
            max_workers = 1

        if max_workers == 1:
            logger.debug('Running serial calculation.')
            results = {
                bias_gid: self._run_single(**kwargs, base_dh_inst=self.base_dh)
                for bias_gid, kwargs in task_kwargs.items()
            }
        else:
            logger.info(
                'Running parallel calculation with %s workers.', max_workers
            )
            results = {}
            with ProcessPoolExecutor(max_workers=max_workers) as exe:
                futures = {
                    exe.submit(self._run_single, **kwargs): bias_gid
                    for bias_gid, kwargs in task_kwargs.items()
                }
                for future in as_completed(futures):
                    bias_gid = futures[future]
                    results[bias_gid] = future.result()

        for i, (bias_gid, single_out) in enumerate(results.items()):
            raster_loc = np.where(self.bias_gid_raster == bias_gid)
            for key, arr in single_out.items():
                out[key][raster_loc] = arr
            logger.info(
                'Completed bias calculations for %s out of %s sites',
                i + 1,
                len(results),
            )

        logger.info('Finished calculating bias correction factors.')

        return self.fill_and_smooth(
            out, fill_extend, smooth_extend, smooth_interior
        )

    @abstractmethod
    def run(
        self,
        fp_out=None,
        max_workers=None,
        daily_reduction='avg',
        fill_extend=True,
        smooth_extend=0,
        smooth_interior=0,
    ):
        """Run correction factor calculations for every site in the bias
        dataset"""

    @classmethod
    @abstractmethod
    def _run_single(
        cls,
        bias_data,
        base_fps,
        bias_feature,
        base_dset,
        base_gid,
        base_handler,
        daily_reduction,
        bias_ti,
        decimals,
        base_dh_inst=None,
        match_zero_rate=False,
    ):
        """Run correction factor calculations for a single site"""
