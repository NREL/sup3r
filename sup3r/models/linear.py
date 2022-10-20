# -*- coding: utf-8 -*-
"""Simple models for super resolution such as linear interp models."""
import numpy as np
import logging
from sup3r.utilities.utilities import st_interp
from sup3r.models.abstract import AbstractSup3rGan

logger = logging.getLogger(__name__)


class LinearInterp(AbstractSup3rGan):
    """Simple model to do linear interpolation on the spatial and temporal axes
    """

    def __init__(self, features, s_enhance, t_enhance, t_centered=False):
        """
        Parameters
        ----------
        features : list
            List of feature names that this model will operate on for both
            input and output. This must match the feature axis ordering in the
            array input to generate().
        s_enhance : int
            Integer factor by which the spatial axes is to be enhanced.
        t_enhance : int
            Integer factor by which the temporal axes is to be enhanced.
        t_centered : bool
            Flag to switch time axis from time-beginning (Default, e.g.
            interpolate 00:00 01:00 to 00:00 00:30 01:00 01:30) to
            time-centered (e.g. interp 01:00 02:00 to 00:45 01:15 01:45 02:15)
        """

        self._features = features
        self._s_enhance = s_enhance
        self._t_enhance = t_enhance
        self._t_centered = t_centered

    @classmethod
    def load(cls, features, s_enhance, t_enhance, t_centered=False,
             verbose=False):
        """Load the GAN with its sub-networks from a previously saved-to output
        directory.

        Parameters
        ----------
        features : list
            List of feature names that this model will operate on for both
            input and output. This must match the feature axis ordering in the
            array input to generate().
        s_enhance : int
            Integer factor by which the spatial axes is to be enhanced.
        t_enhance : int
            Integer factor by which the temporal axes is to be enhanced.
        t_centered : bool
            Flag to switch time axis from time-beginning (Default, e.g.
            interpolate 00:00 01:00 to 00:00 00:30 01:00 01:30) to
            time-centered (e.g. interp 01:00 02:00 to 00:45 01:15 01:45 02:15)
        verbose : bool
            Flag to log information about the loaded model.

        Returns
        -------
        out : LinearInterp
            Returns an initialized LinearInterp model
        """

        model = cls(features, s_enhance, t_enhance, t_centered=t_centered)

        if verbose:
            logger.info('Loading LinearInterp with meta data: {}'
                        .format(model.meta))

        return model

    @property
    def meta(self):
        """Get meta data dictionary that defines the model params"""
        return {'s_enhance': self._s_enhance,
                't_enhance': self._t_enhance,
                't_centered': self._t_centered,
                'training_features': self.training_features,
                'output_features': self.output_features,
                'class': self.__class__.__name__,
                }

    @property
    def training_features(self):
        """Get the list of input feature names that the generative model was
        trained on.
        """
        return self._features

    @property
    def output_features(self):
        """Get the list of output feature names that the generative model
        outputs"""
        return self._features

    # pylint: disable=unused-argument
    def generate(self, low_res, norm_in=False, un_norm_out=False,
                 exogenous_data=None):
        """Use the generator model to generate high res data from low res
        input. This is the public generate function.

        Parameters
        ----------
        low_res : np.ndarray
            Low-resolution spatiotemporal input data, a 5D array of shape:
            (n_obs, spatial_1, spatial_2, temporal, n_features)
        norm_in : bool
            This doesnt do anything for this LinearInterp, but is
            kept to keep the same interface as Sup3rGan
        un_norm_out : bool
            This doesnt do anything for this LinearInterp, but is
            kept to keep the same interface as Sup3rGan
        exogenous_data : list
            This doesnt do anything for this LinearInterp, but is
            kept to keep the same interface as Sup3rGan

        Returns
        -------
        hi_res : ndarray
            high-resolution spatial output data, a 5D array of shape:
            (n_obs, spatial_1, spatial_2, temporal, n_features)
        """

        hr_shape = (len(low_res),
                    int(low_res.shape[1] * self._s_enhance),
                    int(low_res.shape[2] * self._s_enhance),
                    int(low_res.shape[3] * self._t_enhance),
                    len(self.output_features))
        logger.debug('LinearInterp model with s_enhance of {} '
                     'and t_enhance of {} '
                     'downscaling low-res shape {} to high-res shape {}'
                     .format(self._s_enhance, self._t_enhance,
                             low_res.shape, hr_shape))

        hi_res = np.zeros(hr_shape, dtype=np.float32)

        for iobs in range(len(low_res)):
            for idf in range(low_res.shape[-1]):
                hi_res[iobs, ..., idf] = st_interp(low_res[iobs, ..., idf],
                                                   self.s_enhance,
                                                   self.t_enhance,
                                                   t_centered=self._t_centered)

        return hi_res
