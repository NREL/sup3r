"""Mixin classes to support bias calculation"""

import logging

import numpy as np
from scipy.ndimage import gaussian_filter

from sup3r.utilities.utilities import nn_fill_array

logger = logging.getLogger(__name__)


class FillAndSmoothMixin():
    """Fill and extend parameters for calibration on missing positions"""
    def fill_and_smooth(self,
                        out,
                        fill_extend=True,
                        smooth_extend=0,
                        smooth_interior=0):
        """For a given set of parameters, fill and extend missing positions

        Fill data extending beyond the base meta data extent by doing a
        nearest neighbor gap fill. Smooth interior and extended region with
        given smoothing values.
        Interior smoothing can reduce the affect of extreme values
        within aggregations over large number of pixels.
        The interior is assumed to be defined by the region without nan values.
        The extended region is assumed to be the region with nan values.

        Parameters
        ----------
        out : dict
            Dictionary of values defining the mean/std of the bias + base
            data and the scalar + adder factors to correct the biased data
            like: bias_data * scalar + adder. Each value is of shape
            (lat, lon, time).
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

        Returns
        -------
        out : dict
            Dictionary of values defining the mean/std of the bias + base
            data and the scalar + adder factors to correct the biased data
            like: bias_data * scalar + adder. Each value is of shape
            (lat, lon, time).
        """
        if len(self.bad_bias_gids) > 0:
            logger.info('Found {} bias gids that are out of bounds: {}'
                        .format(len(self.bad_bias_gids), self.bad_bias_gids))

        for key, arr in out.items():
            nan_mask = np.isnan(arr[..., 0])
            for idt in range(arr.shape[-1]):

                arr_smooth = arr[..., idt]

                needs_fill = (np.isnan(arr_smooth).any()
                              and fill_extend) or smooth_interior > 0

                if needs_fill:
                    logger.info('Filling NaN values outside of valid spatial '
                                'extent for dataset "{}" for timestep {}'
                                .format(key, idt))
                    arr_smooth = nn_fill_array(arr_smooth)

                arr_smooth_int = arr_smooth_ext = arr_smooth

                if smooth_extend > 0:
                    arr_smooth_ext = gaussian_filter(arr_smooth_ext,
                                                     smooth_extend,
                                                     mode='nearest')

                if smooth_interior > 0:
                    arr_smooth_int = gaussian_filter(arr_smooth_int,
                                                     smooth_interior,
                                                     mode='nearest')

                out[key][nan_mask, idt] = arr_smooth_ext[nan_mask]
                out[key][~nan_mask, idt] = arr_smooth_int[~nan_mask]

        return out
