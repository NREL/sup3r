# -*- coding: utf-8 -*-
"""Sup3r model software"""
import logging
import numpy as np
from warnings import warn

from sup3r.models.base import Sup3rGan


logger = logging.getLogger(__name__)


class MultiStepGan:
    """Multi-Step GAN, which is really just an abstraction layer on top of one
    or more Sup3rGan models that will perform their forward passes in
    serial."""

    def __init__(self, model_dirs):
        """
        Parameters
        ----------
        model_dirs : list | tuple
            An ordered list/tuple of one or more directories containing trained
            + saved Sup3rGan models created using the Sup3rGan.save() method.
        """
        self._models = [Sup3rGan.load(model_dir) for model_dir in model_dirs]
        self._models = tuple(self._models)
        self._all_same_norm_stats = self._norm_stats_same()

    def _norm_stats_same(self):
        """Determine whether or not the normalization stats for the models are
        the same or not.

        Returns
        -------
        all_same : bool
            True if all the norm stats for all models are the same
        """

        all_means = [model.means for model in self.models]
        nones = [m is None for m in all_means]

        if any(nones) and not all(nones):
            m_all_same = False
        elif all(nones):
            m_all_same = True
        else:
            m_all_same = True
            m0 = all_means[0]
            for m1 in all_means[1:]:
                if m0.shape != m1.shape or not np.allclose(m0, m1):
                    m_all_same = False
                    break

        all_stdevs = [model.stdevs for model in self.models]
        nones = [m is None for m in all_stdevs]

        if any(nones) and not all(nones):
            s_all_same = False
        elif all(nones):
            s_all_same = True
        else:
            s_all_same = True
            s0 = all_stdevs[0]
            for s1 in all_stdevs[1:]:
                if s0.shape != s1.shape or not np.allclose(s0, s1):
                    s_all_same = False
                    break

        return m_all_same and s_all_same

    @property
    def models(self):
        """Get an ordered tuple of the Sup3rGan models that are part of this
        MultiStepGan
        """
        return self._models

    @property
    def means(self):
        """Get the data normalization mean values. This is either
        a 1D np.ndarray if all the models have the same means values or a
        tuple of means from all models if they are not all the same.

        Returns
        -------
        tuple | np.ndarray
        """
        if self._all_same_norm_stats:
            return self.models[0].means
        else:
            return tuple(model.means for model in self.models)

    @property
    def stdevs(self):
        """Get the data normalization standard deviation values. This is either
        a 1D np.ndarray if all the models have the same stdevs values or a
        tuple of stdevs from all models if they are not all the same.

        Returns
        -------
        tuple | np.ndarray
        """
        if self._all_same_norm_stats:
            return self.models[0].stdevs
        else:
            return tuple(model.stdevs for model in self.models)

    @staticmethod
    def seed(s=0):
        """
        Set the random seed for reproducible results.

        Parameters
        ----------
        s : int
            Random seed
        """
        Sup3rGan.seed(s=s)

    def _normalize_input(self, low_res):
        """Normalize an input array before being passed to the first model in
        the MultiStepGan

        Parameters
        ----------
        low_res : np.ndarray
            Low-resolution input data, usually a 4D or 5D array of shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)

        Returns
        -------
        low_res : np.ndarray
            Same array shape as input but with the data normalized based on the
            1st model's means/stdevs
        """

        means = self.models[0].means
        stdevs = self.models[0].stdevs

        if means is not None:
            low_res = low_res.copy()
            for i, (m, s) in enumerate(zip(means, stdevs)):
                low_res[..., i] -= m
                if s > 0:
                    low_res[..., i] /= s
                else:
                    msg = ('Standard deviation is zero for '
                           f'{self.training_features[i]}')
                    logger.warning(msg)
                    warn(msg)

        return low_res

    def _unnormalize_output(self, hi_res):
        """Un-normalize an output array before being passed out of the
        MultiStepGan

        Parameters
        ----------
        hi_res : np.ndarray
            Synthetically generated high-resolution data with mean=0 and
            stdev=1. Usually a 4D or 5D array with shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)

        Returns
        -------
        hi_res : np.ndarray
            Synthetically generated high-resolution data un-normalized to
            physical units with mean != 0 and stdev != 1
        """

        means = self.models[-1].means
        stdevs = self.models[-1].stdevs

        if means is not None:
            for i, feature in enumerate(self.models[-1].training_features):
                if feature in self.models[-1].output_features:
                    m = means[i]
                    s = stdevs[i]
                    j = self.models[-1].output_features.index(feature)
                    hi_res[..., j] = (hi_res[..., j] * s) + m

        return hi_res

    def generate(self, low_res, norm_in=True, un_norm_out=True):
        """Use the generator model to generate high res data from low res
        input. This is the public generate function.

        Parameters
        ----------
        low_res : np.ndarray
            Low-resolution input data, usually a 4D or 5D array of shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)
        norm_in : bool
            Flag to normalize low_res input data if the self.means,
            self.stdevs attributes are available. The generator should always
            received normalized data with mean=0 stdev=1.
        un_norm_out : bool
           Flag to un-normalize synthetically generated output data to physical
           units

        Returns
        -------
        hi_res : ndarray
            Synthetically generated high-resolution data, usually a 4D or 5D
            array with shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)
        """

        if norm_in:
            low_res = self._normalize_input(low_res)

        hi_res = low_res.copy()
        for i, model in enumerate(self.models):

            i_norm_in = False
            if not self._all_same_norm_stats and model != self.models[0]:
                i_norm_in = True

            i_un_norm_out = False
            if not self._all_same_norm_stats and model != self.models[-1]:
                i_un_norm_out = True

            try:
                hi_res = model.generate(hi_res, norm_in=i_norm_in,
                                        un_norm_out=i_un_norm_out)
                logger.debug('Data output from model #{} of {} has shape {}'
                             .format(i + 1, len(self.models), hi_res.shape))
            except Exception as e:
                msg = ('Could not run model #{} of {} "{}" '
                       'on tensor of shape {}'
                       .format(i + 1, len(self.models), model, hi_res.shape))
                logger.error(msg)
                raise RuntimeError(msg) from e

        if un_norm_out:
            hi_res = self._unnormalize_output(hi_res)

        return hi_res

    @property
    def version_record(self):
        """Get a tuple of version records from all models

        Returns
        -------
        tuple
        """
        return tuple(model.version_record for model in self.models)

    @property
    def meta(self):
        """Get a tuple of meta data dictionaries for all models

        Returns
        -------
        tuple
        """
        return tuple(model.meta for model in self.models)

    @property
    def training_features(self):
        """Get the list of input feature names that the first generative model
        in this MultiStepGan requires as input."""
        return self.models[0].meta.get('training_features', None)

    @property
    def output_features(self):
        """Get the list of output feature names that the last generative model
        in this MultiStepGan outputs."""
        return self.models[-1].meta.get('output_features', None)

    @property
    def model_params(self):
        """Get a tuple of model parameters for all models

        Returns
        -------
        tuple
        """
        return tuple(model.model_params for model in self.models)


class SpatialFirstGan(MultiStepGan):
    """A two-step GAN where the first step is a spatial-only enhancement on a
    4D tensor and the second step is a spatiotemporal enhancement on a 5D
    tensor.

    NOTE: The low res input to the spatial enhancement should be a 4D tensor of
    the shape (temporal, spatial_1, spatial_2, features) where temporal
    (usually the observation index) is a series of sequential timesteps that
    will be transposed to a 5D tensor of shape
    (1, spatial_1, spatial_2, temporal, features) tensor and then fed to the
    2nd-step spatiotemporal GAN.
    """

    def __init__(self, model_dirs):
        """
        Parameters
        ----------
        model_dirs : list | tuple
            An ordered list/tuple of one or more directories containing trained
            + saved Sup3rGan models created using the Sup3rGan.save() method.
        """
        msg = ('SpatialFirstGan can only have two steps: a spatial-only GAN, '
               'and then a spatiotemporal GAN, but received {} model inputs.'
               .format(len(model_dirs)))
        assert len(model_dirs) == 2, msg
        super().__init__(model_dirs)

    def generate(self, low_res, norm_in=True, un_norm_out=True):
        """Use the generator model to generate high res data from low res
        input. This is the public generate function.

        Parameters
        ----------
        low_res : np.ndarray
            Low-resolution input data to the 1st step spatial GAN, which is a
            4D array of shape: (temporal, spatial_1, spatial_2, n_features)
        norm_in : bool
            Flag to normalize low_res input data if the self.means,
            self.stdevs attributes are available. The generator should always
            received normalized data with mean=0 stdev=1.
        un_norm_out : bool
           Flag to un-normalize synthetically generated output data to physical
           units

        Returns
        -------
        hi_res : ndarray
            Synthetically generated high-resolution data output from the 2nd
            step spatiotemporal GAN with a 5D array shape:
            (1, spatial_1, spatial_2, n_temporal, n_features)
        """

        try:
            hi_res = self.models[0].generate(low_res, norm_in=norm_in,
                                             un_norm_out=True)
        except Exception as e:
            msg = ('Could not run the 1st step spatial-only GAN on input '
                   'shape {}'.format(low_res.shape))
            logger.error(msg)
            raise RuntimeError(msg) from e

        logger.debug('Data output from the 1st step spatial-only '
                     'enhancement has shape {}'.format(hi_res.shape))
        hi_res = np.transpose(hi_res, axes=(1, 2, 0, 3))
        hi_res = np.expand_dims(hi_res, axis=0)
        logger.debug('Data from the 1st step spatial-only enhancement has '
                     'been reshaped to {}'.format(hi_res.shape))

        try:
            hi_res = self.models[1].generate(hi_res, norm_in=True,
                                             un_norm_out=un_norm_out)
        except Exception as e:
            msg = ('Could not run the 2nd step spatiotemporal GAN on input '
                   'shape {}'.format(low_res.shape))
            logger.error(msg)
            raise RuntimeError(msg) from e

        return hi_res
