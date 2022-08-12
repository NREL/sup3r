# -*- coding: utf-8 -*-
"""
Special models for surface meteorological data.
"""
import numpy as np
from PIL import Image
from sup3r.models.abstract import AbstractSup3rGan


class SurfaceSpatialMetModel(AbstractSup3rGan):
    """Model to spatially downscale daily-average near-surface temperature and
    humidity
    """

    # Temperature lapse rate: change in degrees C/K per meter
    TEMP_LAPSE = 6.5 / 1000

    # Weights for the relative humidity linear regression model.
    W_DELTA_TEMP = -3.99242830
    W_DELTA_TOPO = -0.01736911

    @classmethod
    def load(cls, model_dir=None, verbose=False):
        """Load the GAN with its sub-networks from a previously saved-to output
        directory.

        Parameters
        ----------
        model_dir
            Directory to load GAN model files from.
        verbose : bool
            Flag to log information about the loaded model.

        Returns
        -------
        out : SurfaceSpatialMetModel
            Returns a pretrained gan model that was previously saved to
            model_dir
        """

        if verbose:
            logger.info('Loading SurfaceSpatialMetModel with temp lapse {}, '
                        'w_delta_temp {}, and w_delta_topo'
                        .format(cls.TEMP_LAPSE, cls.W_DELTA_TEMP,
                                cls.W_DELTA_TOPO))

        return cls()

    @staticmethod
    def get_s_enhance(topo_lr, topo_hr):
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

    @classmethod
    def downscale_temp(cls, single_lr_temp, topo_lr, topo_hr, s_enhance):
        """Downscale temperature raster data at a single observation.

        This model uses a simple lapse rate that adjusts temperature as a
        function of elevation.

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
        s_enhance : int
            Integer factor by which the spatial axes are to be enhanced.

        Returns
        -------
        hi_res_temp : np.ndarray
            Single timestep temperature (deg C) raster data with shape
            (lat, lon) matching the high-resolution output data.
        """

        assert len(single_lr_temp.shape) == 2, 'Bad shape for single_lr_temp'
        assert len(topo_lr.shape) == 2, 'Bad shape for topo_lr'
        assert len(topo_hr.shape) == 2, 'Bad shape for topo_hr'

        lower_data = single_lr_temp + topo_lr * cls.TEMP_LAPSE
        hi_res_temp = cls.downscale_arr(lower_data, s_enhance)
        hi_res_temp -= topo_hr * cls.TEMP_LAPSE

        return hi_res_temp

    @classmethod
    def downscale_rh(cls, single_lr_rh, single_lr_temp, single_hr_temp,
                     topo_lr, topo_hr, s_enhance):
        """Downscale relative humidity raster data at a single observation.

        This model is based on the following process:
            - Take low-resolution and high-resolution daily average
              temperature, relative humidity, and topography data.
            - Downscale all low-resolution variables using a bilinear image
              enhancement
            - Calculate the difference from the interpolated low-res relative
              humidity to the high-res as a linear function of the difference
              at the same location for temperature and topography.
            - Use this linear regression model to calculate high-res relative
              humidity fields when provided low-res/high-res pairs of
              temperature and topography and a low-res input of relative
              humidity.

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
        s_enhance : int
            Integer factor by which the spatial axes are to be enhanced.

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

        interp_rh = cls.downscale_arr(single_lr_rh, s_enhance)
        interp_temp = cls.downscale_arr(single_lr_temp, s_enhance)
        interp_topo = cls.downscale_arr(topo_lr, s_enhance)

        delta_temp = single_hr_temp - interp_temp
        delta_topo = topo_hr - interp_topo

        hi_res_rh = (interp_rh
                     + cls.W_DELTA_TEMP * delta_temp
                     + cls.W_DELTA_TOPO * delta_topo)

        return hi_res_rh

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
        assert isinstance(topo_lr, np.ndarray)
        assert isinstance(topo_hr, np.ndarray)
        assert len(topo_lr.shape) == 2
        assert len(topo_hr.shape) == 2
        assert topo_lr.shape[0] == low_res.shape[1]
        assert topo_lr.shape[1] == low_res.shape[2]

        s_enhance = self.get_s_enhance(topo_lr, topo_hr)
        hr_shape = (len(low_res),
                    int(low_res.shape[1] * s_enhance),
                    int(low_res.shape[2] * s_enhance),
                    2)

        hi_res = np.zeros(hr_shape, dtype=np.float32)
        for iobs in range(len(low_res)):
            hi_res[iobs, :, :, 0] = self.downscale_temp(low_res[iobs, :, :, 0],
                                                        topo_lr, topo_hr,
                                                        s_enhance)

            hi_res[iobs, :, :, 1] = self.downscale_rh(low_res[iobs, :, :, 1],
                                                      low_res[iobs, :, :, 0],
                                                      hi_res[iobs, :, :, 0],
                                                      topo_lr, topo_hr,
                                                      s_enhance)

        return hi_res

    @property
    def meta(self):
        """Get meta data dictionary that defines the model params"""
        return {'temp_lapse_rate': self.TEMP_LAPSE,
                'weight_for_delta_temp': self.W_DELTA_TEMP,
                'weight_for_delta_topo': self.W_DELTA_TOPO,
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
