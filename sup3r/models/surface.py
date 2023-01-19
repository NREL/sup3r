# -*- coding: utf-8 -*-
"""Special models for surface meteorological data."""
import os
import json
import logging
from inspect import signature
from fnmatch import fnmatch
import numpy as np
from PIL import Image
from sklearn import linear_model
from warnings import warn

from sup3r.models.linear import LinearInterp
from sup3r.utilities.utilities import spatial_coarsening

logger = logging.getLogger(__name__)


class SurfaceSpatialMetModel(LinearInterp):
    """Model to spatially downscale daily-average near-surface temperature,
    relative humidity, and pressure

    Note that this model can also operate on temperature_*m,
    relativehumidity_*m, and pressure_*m datasets that are not
    strictly at the surface but could be near hub height.
    """

    TEMP_LAPSE = 6.5 / 1000
    """Temperature lapse rate: change in degrees C/K per meter"""

    PRES_DIV = 44307.69231
    """Air pressure scaling equation divisor variable:
    101325 * (1 - (1 - topo / PRES_DIV)**PRES_EXP)"""

    PRES_EXP = 5.25328
    """Air pressure scaling equation exponent variable:
    101325 * (1 - (1 - topo / PRES_DIV)**PRES_EXP)"""

    W_DELTA_TEMP = -3.99242830
    """Weight for the delta-temperature feature for the relative humidity
    linear regression model."""

    W_DELTA_TOPO = -0.01736911
    """Weight for the delta-topography feature for the relative humidity linear
    regression model."""

    def __init__(self, features, s_enhance, noise_adders=None,
                 temp_lapse=None, w_delta_temp=None, w_delta_topo=None,
                 pres_div=None, pres_exp=None, interp_method='LANCZOS',
                 fix_bias=True):
        """
        Parameters
        ----------
        features : list
            List of feature names that this model will operate on for both
            input and output. This must match the feature axis ordering in the
            array input to generate(). Typically this is a list containing:
            temperature_*m, relativehumidity_*m, and pressure_*m. The list can
            contain multiple instances of each variable at different heights.
            relativehumidity_*m entries must have corresponding temperature_*m
            entires at the same hub height.
        s_enhance : int
            Integer factor by which the spatial axes are to be enhanced.
        noise_adders : float | list | None
            Option to add gaussian noise to spatial model output. Noise will be
            normally distributed with mean of 0 and standard deviation =
            noise_adders. noise_adders can be a single value or a list
            corresponding to the features list. None is no noise. The addition
            of noise has been shown to help downstream temporal-only models
            produce diurnal cycles in regions where there is minimal change in
            topography. A noise_adders around 0.07C (temperature) and 0.1%
            (relative humidity) have been shown to be effective. This is
            unnecessary if daily min/max temperatures are provided as low res
            training features.
        temp_lapse : None | float
            Temperature lapse rate: change in degrees C/K per meter. Defaults
            to the cls.TEMP_LAPSE attribute.
        w_delta_temp : None | float
            Weight for the delta-temperature feature for the relative humidity
            linear regression model. Defaults to the cls.W_DELTA_TEMP
            attribute.
        w_delta_topo : None | float
            Weight for the delta-topography feature for the relative humidity
            linear regression model. Defaults to the cls.W_DELTA_TOPO
            attribute.
        pres_div : None | float
            Divisor factor in the pressure scale height equation. Defaults to
            the cls.PRES_DIV attribute.
        pres_div : None | float
            Exponential factor in the pressure scale height equation. Defaults
            to the cls.PRES_EXP attribute.
        interp_method : str
            Name of the interpolation method to use from PIL.Image.Resampling
            (NEAREST, BILINEAR, BICUBIC, LANCZOS)
            LANCZOS is default and has been tested to work best for
            SurfaceSpatialMetModel.
        fix_bias : bool
            Some local bias can be introduced by the bilinear interp + lapse
            rate, this flag will attempt to correct that bias by using the
            low-resolution deviation from the input data
        """

        self._features = features
        self._s_enhance = s_enhance
        self._noise_adders = noise_adders
        self._temp_lapse = temp_lapse or self.TEMP_LAPSE
        self._w_delta_temp = w_delta_temp or self.W_DELTA_TEMP
        self._w_delta_topo = w_delta_topo or self.W_DELTA_TOPO
        self._pres_div = pres_div or self.PRES_DIV
        self._pres_exp = pres_exp or self.PRES_EXP
        self._fix_bias = fix_bias
        self._interp_method = getattr(Image.Resampling, interp_method)

        if isinstance(self._noise_adders, (int, float)):
            self._noise_adders = [self._noise_adders] * len(self._features)

    def __len__(self):
        """Get number of model steps (match interface of MultiStepGan)"""
        return 1

    @staticmethod
    def _get_s_enhance(topo_lr, topo_hr):
        """Get the spatial enhancement factor given low-res and high-res
        spatial rasters.

        Parameters
        ----------
        topo_lr : np.ndarray
            low-resolution surface elevation data in meters with shape
            (lat, lon)
        topo_hr : np.ndarray
            High-resolution surface elevation data in meters with shape
            (lat, lon)

        Returns
        -------
        s_enhance : int
            Integer factor by which the spatial axes are to be enhanced.
        """
        assert len(topo_lr.shape) == 2, 'topo_lr must be 2D'
        assert len(topo_hr.shape) == 2, 'topo_hr must be 2D'

        se0 = topo_hr.shape[0] / topo_lr.shape[0]
        se1 = topo_hr.shape[1] / topo_lr.shape[1]

        assert se0 % 1 == 0, f'Bad calculated s_enhance on axis 0: {se0}'
        assert se1 % 1 == 0, f'Bad calculated s_enhance on axis 1: {se1}'
        assert se0 == se1, 'Calculated s_enhance does not match along axis'

        return int(se0)

    @property
    def feature_inds_temp(self):
        """Get the feature index values for the temperature features."""
        inds = [i for i, name in enumerate(self._features)
                if fnmatch(name, 'temperature_*')]
        return inds

    @property
    def feature_inds_pres(self):
        """Get the feature index values for the pressure features."""
        inds = [i for i, name in enumerate(self._features)
                if fnmatch(name, 'pressure_*')]
        return inds

    @property
    def feature_inds_rh(self):
        """Get the feature index values for the relative humidity features."""
        inds = [i for i, name in enumerate(self._features)
                if fnmatch(name, 'relativehumidity_*')]
        return inds

    def _get_temp_rh_ind(self, idf_rh):
        """Get the feature index value for the temperature feature
        corresponding to a relative humidity feature at the same hub height.

        Parameters
        ----------
        idf_rh : int
            Index in the feature list for a relativehumidity_*m feature

        Returns
        -------
        idf_temp : int
            Index in the feature list for a temperature_*m feature with the
            same hub height as the idf_rh input.
        """
        name_rh = self._features[idf_rh]
        hh_suffix = name_rh.split('_')[-1]
        idf_temp = None
        for i in self.feature_inds_temp:
            same_hh = self._features[i].endswith(hh_suffix)
            not_minmax = not any(mm in name_rh for mm in ('_min_', '_max_'))
            both_mins = '_min_' in name_rh and '_min_' in self._features[i]
            both_maxs = '_max_' in name_rh and '_max_' in self._features[i]

            if same_hh and (not_minmax or both_mins or both_maxs):
                idf_temp = i
                break

        if idf_temp is None:
            msg = ('Could not find temperature feature corresponding to '
                   '"{}" in feature list: {}'.format(name_rh, self._features))
            logger.error(msg)
            raise KeyError(msg)

        return idf_temp

    def _fix_downscaled_bias(self, single_lr, single_hr,
                             method=Image.Resampling.LANCZOS):
        """Fix any bias introduced by the spatial downscaling with lapse rate.

        Parameters
        ----------
        single_lr : np.ndarray
            Single timestep raster data with shape
            (lat, lon) matching the low-resolution input data.
        single_hr : np.ndarray
            Single timestep downscaled raster data with shape
            (lat, lon) matching the high-resolution input data.
        method : Image.Resampling.LANCZOS
            An Image.Resampling method (NEAREST, BILINEAR, BICUBIC, LANCZOS).
            NEAREST enforces zero bias but makes slightly more spatial seams.

        Returns
        -------
        single_hr : np.ndarray
            Single timestep downscaled raster data with shape
            (lat, lon) matching the high-resolution input data.
        """

        re_coarse = spatial_coarsening(np.expand_dims(single_hr, axis=-1),
                                       s_enhance=self._s_enhance,
                                       obs_axis=False)[..., 0]
        bias = re_coarse - single_lr
        bc = self.downscale_arr(bias, s_enhance=self._s_enhance, method=method)
        single_hr -= bc
        return single_hr

    @staticmethod
    def downscale_arr(arr, s_enhance, method=Image.Resampling.LANCZOS):
        """Downscale a 2D array of data Image.resize() method

        Parameters
        ----------
        arr : np.ndarray
            2D raster data, typically spatial daily average data with shape
            (lat, lon)
        s_enhance : int
            Integer factor by which the spatial axes are to be enhanced.
        method : Image.Resampling.LANCZOS
            An Image.Resampling method (NEAREST, BILINEAR, BICUBIC, LANCZOS).
            LANCZOS is default and has been tested to work best for
            SurfaceSpatialMetModel.
        """
        im = Image.fromarray(arr)
        im = im.resize((arr.shape[1] * s_enhance, arr.shape[0] * s_enhance),
                       resample=method)
        out = np.array(im)
        return out

    def downscale_temp(self, single_lr_temp, topo_lr, topo_hr):
        """Downscale temperature raster data at a single observation.

        This model uses a simple lapse rate that adjusts temperature as a
        function of elevation. The process is as follows:
            - add a scale factor to the low-res temperature field:
              topo_lr * TEMP_LAPSE
            - perform bilinear interpolation of the scaled temperature field.
            - subtract the scale factor from the high-res temperature field:
              topo_hr * TEMP_LAPSE

        Parameters
        ----------
        single_lr_temp : np.ndarray
            Single timestep temperature (deg C) raster data with shape
            (lat, lon) matching the low-resolution input data.
        topo_lr : np.ndarray
            low-resolution surface elevation data in meters with shape
            (lat, lon)
        topo_hr : np.ndarray
            High-resolution surface elevation data in meters with shape
            (lat, lon)

        Returns
        -------
        hi_res_temp : np.ndarray
            Single timestep temperature (deg C) raster data with shape
            (lat, lon) matching the high-resolution output data.
        """

        assert len(single_lr_temp.shape) == 2, 'Bad shape for single_lr_temp'
        assert len(topo_lr.shape) == 2, 'Bad shape for topo_lr'
        assert len(topo_hr.shape) == 2, 'Bad shape for topo_hr'

        lower_data = single_lr_temp.copy() + topo_lr * self._temp_lapse
        hi_res_temp = self.downscale_arr(lower_data, self._s_enhance,
                                         method=self._interp_method)
        hi_res_temp -= topo_hr * self._temp_lapse

        if self._fix_bias:
            hi_res_temp = self._fix_downscaled_bias(single_lr_temp,
                                                    hi_res_temp,
                                                    method=self._interp_method)

        return hi_res_temp

    def downscale_rh(self, single_lr_rh, single_lr_temp, single_hr_temp,
                     topo_lr, topo_hr):
        """Downscale relative humidity raster data at a single observation.

        Here's a description of the humidity scaling model:
            - Take low-resolution and high-resolution daily average
              temperature, relative humidity, and topography data.
            - Downscale all low-resolution variables using a bilinear image
              enhancement
            - The target model output is the difference between the
              interpolated low-res relative humidity and the true high-res
              relative humidity. Calculate this difference as a linear function
              of the same difference in the temperature and topography fields.
            - Use this linear regression model to calculate high-res relative
              humidity fields when provided low-res input of relative humidity
              along with low-res/high-res pairs of temperature and topography.

        Parameters
        ----------
        single_lr_rh : np.ndarray
            Single timestep relative humidity (%) raster data with shape
            (lat, lon) matching the low-resolution input data.
        single_lr_temp : np.ndarray
            Single timestep temperature (deg C) raster data with shape
            (lat, lon) matching the low-resolution input data.
        single_hr_temp : np.ndarray
            Single timestep temperature (deg C) raster data with shape
            (lat, lon) matching the high-resolution output data.
        topo_lr : np.ndarray
            low-resolution surface elevation data in meters with shape
            (lat, lon)
        topo_hr : np.ndarray
            High-resolution surface elevation data in meters with shape
            (lat, lon)

        Returns
        -------
        hi_res_rh : np.ndarray
            Single timestep relative humidity (%) raster data with shape
            (lat, lon) matching the high-resolution output data.
        """

        assert len(single_lr_rh.shape) == 2, 'Bad shape for single_lr_rh'
        assert len(single_hr_temp.shape) == 2, 'Bad shape for single_hr_temp'
        assert len(topo_lr.shape) == 2, 'Bad shape for topo_lr'
        assert len(topo_hr.shape) == 2, 'Bad shape for topo_hr'

        interp_rh = self.downscale_arr(single_lr_rh, self._s_enhance,
                                       method=self._interp_method)
        interp_temp = self.downscale_arr(single_lr_temp, self._s_enhance,
                                         method=self._interp_method)
        interp_topo = self.downscale_arr(topo_lr, self._s_enhance,
                                         method=self._interp_method)

        delta_temp = single_hr_temp - interp_temp
        delta_topo = topo_hr - interp_topo

        hi_res_rh = (interp_rh
                     + self._w_delta_temp * delta_temp
                     + self._w_delta_topo * delta_topo)

        if self._fix_bias:
            hi_res_rh = self._fix_downscaled_bias(single_lr_rh, hi_res_rh,
                                                  method=self._interp_method)

        return hi_res_rh

    def downscale_pres(self, single_lr_pres, topo_lr, topo_hr):
        """Downscale pressure raster data at a single observation.

        This model uses a simple exponential scale height that adjusts
        temperature as a function of elevation. The process is as follows:
            - add a scale factor to the low-res pressure field:
              101325 * (1 - (1 - topo_lr / PRES_DIV)**PRES_EXP)
            - perform bilinear interpolation of the scaled pressure field.
            - subtract the scale factor from the high-res pressure field:
              101325 * (1 - (1 - topo_hr / PRES_DIV)**PRES_EXP)

        Parameters
        ----------
        single_lr_pres : np.ndarray
            Single timestep pressure (Pa) raster data with shape
            (lat, lon) matching the low-resolution input data.
        topo_lr : np.ndarray
            low-resolution surface elevation data in meters with shape
            (lat, lon)
        topo_hr : np.ndarray
            High-resolution surface elevation data in meters with shape
            (lat, lon)

        Returns
        -------
        hi_res_pres : np.ndarray
            Single timestep pressure (Pa) raster data with shape
            (lat, lon) matching the high-resolution output data.
        """

        if np.max(single_lr_pres) < 10000:
            msg = ('Pressure data appears to not be in Pa with min/mean/max: '
                   '{:.1f}/{:.1f}/{:.1f}'
                   .format(single_lr_pres.min(), single_lr_pres.mean(),
                           single_lr_pres.max()))
            logger.warning(msg)
            warn(msg)

        const = 101325 * (1 - (1 - topo_lr / self._pres_div)**self._pres_exp)
        lr_pres_adj = single_lr_pres.copy() + const

        if np.min(lr_pres_adj) < 0.0:
            msg = ('Spatial interpolation of surface pressure '
                   'resulted in negative values. Incorrectly '
                   'scaled/unscaled values or incorrect units are '
                   'the most likely causes. All pressure data should be '
                   'in Pascals.')
            logger.error(msg)
            raise ValueError(msg)

        hi_res_pres = self.downscale_arr(lr_pres_adj, self._s_enhance,
                                         method=self._interp_method)

        const = 101325 * (1 - (1 - topo_hr / self._pres_div)**self._pres_exp)
        hi_res_pres -= const

        if self._fix_bias:
            hi_res_pres = self._fix_downscaled_bias(single_lr_pres,
                                                    hi_res_pres,
                                                    method=self._interp_method)

        if np.min(hi_res_pres) < 0.0:
            msg = ('Spatial interpolation of surface pressure '
                   'resulted in negative values. Incorrectly '
                   'scaled/unscaled values or incorrect units are '
                   'the most likely causes.')
            logger.error(msg)
            raise ValueError(msg)

        return hi_res_pres

    # pylint: disable=unused-argument
    def generate(self, low_res, norm_in=False, un_norm_out=False,
                 exogenous_data=None):
        """Use the generator model to generate high res data from low res
        input. This is the public generate function.

        Parameters
        ----------
        low_res : np.ndarray
            Low-resolution spatial input data, a 4D array of shape:
            (n_obs, spatial_1, spatial_2, n_features), Where the feature
            channel can include temperature_*m, relativehumidity_*m, and/or
            pressure_*m
        norm_in : bool
            This doesnt do anything for this SurfaceSpatialMetModel, but is
            kept to keep the same interface as Sup3rGan
        un_norm_out : bool
            This doesnt do anything for this SurfaceSpatialMetModel, but is
            kept to keep the same interface as Sup3rGan
        exogenous_data : list
            For the SurfaceSpatialMetModel, this must be a 2-entry list where
            the first entry is a 2D (lat, lon) array of low-resolution surface
            elevation data in meters (must match spatial_1, spatial_2 from
            low_res), and the second entry is a 2D (lat, lon) array of
            high-resolution surface elevation data in meters.

        Returns
        -------
        hi_res : ndarray
            high-resolution spatial output data, a 4D array of shape:
            (n_obs, spatial_1, spatial_2, n_features), Where the feature
            channel can include temperature_*m, relativehumidity_*m, and/or
            pressure_*m
        """

        msg = ('exogenous_data is of a bad type {}!'
               .format(type(exogenous_data)))
        assert isinstance(exogenous_data, (list, tuple)), msg
        msg = ('exogenous_data is of a bad length {}!'
               .format(len(exogenous_data)))
        assert len(exogenous_data) == 2, msg

        topo_lr = exogenous_data[0]
        topo_hr = exogenous_data[1]
        logger.debug('SurfaceSpatialMetModel received low/high res topo '
                     'shapes of {} and {}'
                     .format(topo_lr.shape, topo_hr.shape))

        msg = ('topo_lr has a bad shape {} that doesnt match the low res '
               'data shape {}'.format(topo_lr.shape, topo_hr.shape))
        assert isinstance(topo_lr, np.ndarray), msg
        assert isinstance(topo_hr, np.ndarray), msg
        assert len(topo_lr.shape) == 2, msg
        assert len(topo_hr.shape) == 2, msg
        assert topo_lr.shape[0] == low_res.shape[1], msg
        assert topo_lr.shape[1] == low_res.shape[2], msg
        s_enhance = self._get_s_enhance(topo_lr, topo_hr)
        msg = ('Topo shapes of {} and {} did not match desired spatial '
               'enhancement of {}'
               .format(topo_lr.shape, topo_hr.shape, self._s_enhance))
        assert self._s_enhance == s_enhance, msg

        hr_shape = (len(low_res),
                    int(low_res.shape[1] * self._s_enhance),
                    int(low_res.shape[2] * self._s_enhance),
                    len(self.output_features))
        logger.debug('SurfaceSpatialMetModel with s_enhance of {} '
                     'downscaling low-res shape {} to high-res shape {}'
                     .format(self._s_enhance, low_res.shape, hr_shape))

        hi_res = np.zeros(hr_shape, dtype=np.float32)
        for iobs in range(len(low_res)):
            for idf_temp in self.feature_inds_temp:
                _tmp = self.downscale_temp(low_res[iobs, :, :, idf_temp],
                                           topo_lr, topo_hr)
                hi_res[iobs, :, :, idf_temp] = _tmp

            for idf_pres in self.feature_inds_pres:
                _tmp = self.downscale_pres(low_res[iobs, :, :, idf_pres],
                                           topo_lr, topo_hr)
                hi_res[iobs, :, :, idf_pres] = _tmp

            for idf_rh in self.feature_inds_rh:
                idf_temp = self._get_temp_rh_ind(idf_rh)
                _tmp = self.downscale_rh(low_res[iobs, :, :, idf_rh],
                                         low_res[iobs, :, :, idf_temp],
                                         hi_res[iobs, :, :, idf_temp],
                                         topo_lr, topo_hr)
                hi_res[iobs, :, :, idf_rh] = _tmp

        if self._noise_adders is not None:
            for idf, stdev in enumerate(self._noise_adders):
                if stdev is not None:
                    noise = np.random.normal(0, stdev, hi_res.shape[:-1])
                    hi_res[..., idf] += noise

        return hi_res

    @property
    def meta(self):
        """Get meta data dictionary that defines the model params"""
        return {'temp_lapse_rate': self._temp_lapse,
                's_enhance': self._s_enhance,
                't_enhance': 1,
                'noise_adders': self._noise_adders,
                'weight_for_delta_temp': self._w_delta_temp,
                'weight_for_delta_topo': self._w_delta_topo,
                'pressure_divisor': self._pres_div,
                'pressure_exponent': self._pres_exp,
                'training_features': self.training_features,
                'output_features': self.output_features,
                'interp_method': str(self._interp_method),
                'fix_bias': self._fix_bias,
                'class': self.__class__.__name__,
                }

    def train(self, true_hr_temp, true_hr_rh, true_hr_topo):
        """This method trains the relative humidity linear model. The
        temperature and surface lapse rate models are parameterizations taken
        from the NSRDB and are not trained.

        Parameters
        ----------
        true_hr_temp : np.ndarray
            True high-resolution daily average temperature data in a 3D array
            of shape (lat, lon, n_days)
        true_hr_rh : np.ndarray
            True high-resolution daily average relative humidity data in a 3D
            array of shape (lat, lon, n_days)
        true_hr_topo : np.ndarray
            High-resolution surface elevation data in meters with shape
            (lat, lon)

        Returns
        -------
        w_delta_temp : float
            Weight for the delta-temperature feature for the relative humidity
            linear regression model.
        w_delta_topo : float
            Weight for the delta-topography feature for the relative humidity
            linear regression model.
        """

        assert len(true_hr_temp.shape) == 3, 'Bad true_hr_temp shape'
        assert len(true_hr_rh.shape) == 3, 'Bad true_hr_rh shape'
        assert len(true_hr_topo.shape) == 2, 'Bad true_hr_topo shape'

        true_hr_topo = np.expand_dims(true_hr_topo, axis=-1)
        true_hr_topo = np.repeat(true_hr_topo, true_hr_temp.shape[-1], axis=-1)

        true_lr_temp = spatial_coarsening(true_hr_temp,
                                          s_enhance=self._s_enhance,
                                          obs_axis=False)
        true_lr_rh = spatial_coarsening(true_hr_rh,
                                        s_enhance=self._s_enhance,
                                        obs_axis=False)
        true_lr_topo = spatial_coarsening(true_hr_topo,
                                          s_enhance=self._s_enhance,
                                          obs_axis=False)

        interp_hr_temp = np.full(true_hr_temp.shape, np.nan, dtype=np.float32)
        interp_hr_rh = np.full(true_hr_rh.shape, np.nan, dtype=np.float32)
        interp_hr_topo = np.full(true_hr_topo.shape, np.nan, dtype=np.float32)

        for i in range(interp_hr_temp.shape[-1]):
            interp_hr_temp[..., i] = self.downscale_arr(true_lr_temp[..., i],
                                                        self._s_enhance)
            interp_hr_rh[..., i] = self.downscale_arr(true_lr_rh[..., i],
                                                      self._s_enhance)
            interp_hr_topo[..., i] = self.downscale_arr(true_lr_topo[..., i],
                                                        self._s_enhance)

        x1 = true_hr_temp - interp_hr_temp
        x2 = true_hr_topo - interp_hr_topo
        x = np.vstack((x1.flatten(), x2.flatten())).T
        y = (true_hr_rh - interp_hr_rh).flatten()

        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        if np.abs(regr.intercept_) > 1e-6:
            msg = ('Relative humidity linear model should have an intercept '
                   'of zero but the model fit an intercept of {}'
                   .format(regr.intercept_))
            logger.warning(msg)
            warn(msg)

        w_delta_temp, w_delta_topo = regr.coef_[0], regr.coef_[1]

        return w_delta_temp, w_delta_topo
