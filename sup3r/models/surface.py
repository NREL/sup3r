# -*- coding: utf-8 -*-
"""
Special models for surface meteorological data.
"""
import logging
import numpy as np
from PIL import Image
from warnings import warn
from sup3r.models.abstract import AbstractSup3rGan

logger = logging.getLogger(__name__)


class SurfaceSpatialMetModel(AbstractSup3rGan):
    """Model to spatially downscale daily-average near-surface temperature and
    humidity
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

    def __init__(self, s_enhance, temp_lapse=None, w_delta_temp=None,
                 w_delta_topo=None, pres_div=None, pres_exp=None):
        """
        Parameters
        ----------
        s_enhance : int
            Integer factor by which the spatial axes are to be enhanced.
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
        """

        self._s_enhance = s_enhance
        self._temp_lapse = temp_lapse or self.TEMP_LAPSE
        self._w_delta_temp = w_delta_temp or self.W_DELTA_TEMP
        self._w_delta_topo = w_delta_topo or self.W_DELTA_TOPO
        self._pres_div = pres_div or self.PRES_DIV
        self._pres_exp = pres_div or self.PRES_EXP

    def __len__(self):
        """Get number of model steps (mimic MultiStepGan)"""
        return 1

    @classmethod
    def load(cls, s_enhance, verbose=False, **kwargs):
        """Load the GAN with its sub-networks from a previously saved-to output
        directory.

        Parameters
        ----------
        s_enhance : int
            Integer factor by which the spatial axes are to be enhanced.
        verbose : bool
            Flag to log information about the loaded model.
        kwargs : None | dict
            Optional kwargs to initialize SurfaceSpatialMetModel

        Returns
        -------
        out : SurfaceSpatialMetModel
            Returns an initialized SurfaceSpatialMetModel
        """

        model = cls(s_enhance, **kwargs)

        if verbose:
            logger.info('Loading SurfaceSpatialMetModel with meta data: {}'
                        .format(model.meta))

        return model

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

    @staticmethod
    def downscale_arr(arr, s_enhance, method=Image.Resampling.BILINEAR):
        """Downscale a 2D array of data Image.resize() method

        Parameters
        ----------
        arr : np.ndarray
            2D raster data, typically spatial daily average data with shape
            (lat, lon)
        s_enhance : int
            Integer factor by which the spatial axes are to be enhanced.
        method : Image.Resampling.BILINEAR
            An Image.Resampling method (NEAREST, BILINEAR, BICUBIC, LANCZOS).
            BILINEAR is default and has been tested to work best for
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

        lower_data = single_lr_temp + topo_lr * self._temp_lapse
        hi_res_temp = self.downscale_arr(lower_data, self._s_enhance)
        hi_res_temp -= topo_hr * self._temp_lapse

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

        interp_rh = self.downscale_arr(single_lr_rh, self._s_enhance)
        interp_temp = self.downscale_arr(single_lr_temp, self._s_enhance)
        interp_topo = self.downscale_arr(topo_lr, self._s_enhance)

        delta_temp = single_hr_temp - interp_temp
        delta_topo = topo_hr - interp_topo

        hi_res_rh = (interp_rh
                     + self._w_delta_temp * delta_temp
                     + self._w_delta_topo * delta_topo)

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
        single_lr_pres += const

        if np.min(single_lr_pres) < 0.0:
            msg = ('Spatial interpolation of surface pressure '
                   'resulted in negative values. Incorrectly '
                   'scaled/unscaled values or incorrect units are '
                   'the most likely causes.')
            logger.error(msg)
            raise ValueError(msg)

        hi_res_pres = self.downscale_arr(single_lr_pres, self._s_enhance)

        const = 101325 * (1 - (1 - topo_hr / self._pres_div)**self._pres_exp)
        hi_res_pres -= const

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
            (n_obs, spatial_1, spatial_2, 2), Where the feature channel is:
            [temperature_2m, relativehumidity_2m]
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
            (n_obs, spatial_1, spatial_2, 2), Where the feature channel is:
            [temperature_2m, relativehumidity_2m]
        """

        assert isinstance(exogenous_data, (list, tuple))
        assert len(exogenous_data) == 2

        topo_lr = exogenous_data[0]
        topo_hr = exogenous_data[1]
        logger.debug('SurfaceSpatialMetModel received low/high res topo '
                     'shapes of {} and {}'
                     .format(topo_lr.shape, topo_hr.shape))

        assert isinstance(topo_lr, np.ndarray)
        assert isinstance(topo_hr, np.ndarray)
        assert len(topo_lr.shape) == 2
        assert len(topo_hr.shape) == 2
        assert topo_lr.shape[0] == low_res.shape[1]
        assert topo_lr.shape[1] == low_res.shape[2]
        s_enhance = self._get_s_enhance(topo_lr, topo_hr)
        msg = ('Topo shapes of {} and {} did not match desired spatial '
               'enhancement of {}'
               .format(topo_lr.shape, topo_hr.shape, self._s_enhance))
        assert self._s_enhance == s_enhance, msg

        hr_shape = (len(low_res),
                    int(low_res.shape[1] * self._s_enhance),
                    int(low_res.shape[2] * self._s_enhance),
                    2)
        logger.debug('SurfaceSpatialMetModel with s_enhance of {} '
                     'downscaling low-res shape {} to high-res shape {}'
                     .format(self._s_enhance, low_res.shape, hr_shape))

        hi_res = np.zeros(hr_shape, dtype=np.float32)
        for iobs in range(len(low_res)):
            hi_res[iobs, :, :, 0] = self.downscale_temp(low_res[iobs, :, :, 0],
                                                        topo_lr, topo_hr)

            hi_res[iobs, :, :, 1] = self.downscale_rh(low_res[iobs, :, :, 1],
                                                      low_res[iobs, :, :, 0],
                                                      hi_res[iobs, :, :, 0],
                                                      topo_lr, topo_hr)

        return hi_res

    @property
    def meta(self):
        """Get meta data dictionary that defines the model params"""
        return {'temp_lapse_rate': self._temp_lapse,
                's_enhance': self._s_enhance,
                't_enhance': 1,
                'weight_for_delta_temp': self._w_delta_temp,
                'weight_for_delta_topo': self._w_delta_topo,
                'pressure_divisor': self._pres_div,
                'pressure_exponent': self._pres_exp,
                'training_features': self.training_features,
                'output_features': self.output_features,
                'class': self.__class__.__name__,
                }

    @property
    def training_features(self):
        """Get the list of input feature names that the generative model was
        trained on.

        Note that topography needs to be passed into generate() as an exogenous
        data input.
        """
        return ['temperature_2m', 'relativehumidity_2m', 'topography']

    @property
    def output_features(self):
        """Get the list of output feature names that the generative model
        outputs"""
        return ['temperature_2m', 'relativehumidity_2m']
