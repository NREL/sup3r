# -*- coding: utf-8 -*-
"""Sup3r model software"""
import logging
import numpy as np
from warnings import warn

from sup3r.models.abstract import AbstractSup3rGan
from sup3r.models.base import Sup3rGan


logger = logging.getLogger(__name__)


class MultiStepGan(AbstractSup3rGan):
    """Multi-Step GAN, which is really just an abstraction layer on top of one
    or more Sup3rGan models that will perform their forward passes in
    serial."""

    def __init__(self, models):
        """
        Parameters
        ----------
        models : list | tuple
            An ordered list/tuple of one or more trained Sup3rGan models
        """
        self._models = tuple(models)
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

    @classmethod
    def load(cls, model_dirs, verbose=True):
        """Load the GANs with its sub-networks from a previously saved-to
        output directory.

        Parameters
        ----------
        model_dirs : list | tuple
            An ordered list/tuple of one or more directories containing trained
            + saved Sup3rGan models created using the Sup3rGan.save() method.
        verbose : bool
            Flag to log information about the loaded model.

        Returns
        -------
        out : MultiStepGan
            Returns a pretrained gan model that was previously saved to
            model_dirs
        """
        models = [Sup3rGan.load(model_dir, verbose=verbose)
                  for model_dir in model_dirs]
        return cls(models)

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


class SpatialThenTemporalGan(AbstractSup3rGan):
    """A two-step GAN where the first step is a spatial-only enhancement on a
    4D tensor and the second step is a (spatio)temporal enhancement on a 5D
    tensor.

    NOTE: The low res input to the spatial enhancement should be a 4D tensor of
    the shape (temporal, spatial_1, spatial_2, features) where temporal
    (usually the observation index) is a series of sequential timesteps that
    will be transposed to a 5D tensor of shape
    (1, spatial_1, spatial_2, temporal, features) tensor and then fed to the
    2nd-step (spatio)temporal GAN.
    """

    def __init__(self, spatial_models, temporal_models):
        """
        Parameters
        ----------
        spatial_models : MultiStepGan
            A loaded MultiStepGan object representing the one or more spatial
            super resolution steps in this composite SpatialThenTemporalGan
            model
        temporal_models : MultiStepGan
            A loaded MultiStepGan object representing the one or more
            (spatio)temporal super resolution steps in this composite
            SpatialThenTemporalGan model
        """

        self._spatial_models = spatial_models
        self._temporal_models = temporal_models

    @property
    def spatial_models(self):
        """Get the MultiStepGan object for the spatial-only model(s)

        Returns
        -------
        MultiStepGan
        """
        return self._spatial_models

    @property
    def temporal_models(self):
        """Get the MultiStepGan object for the (spatio)temporal model(s)

        Returns
        -------
        MultiStepGan
        """
        return self._temporal_models

    @property
    def meta(self):
        """Get a tuple of meta data dictionaries for all models

        Returns
        -------
        tuple
        """
        return self.spatial_models.meta + self.temporal_models.meta

    @property
    def training_features(self):
        """Get the list of input feature names that the first spatial
        generative model in this SpatialThenTemporalGan requires as input."""
        return self.spatial_models.training_features

    @property
    def output_features(self):
        """Get the list of output feature names that the last spatiotemporal
        generative model in this SpatialThenTemporalGan outputs."""
        return self.temporal_models.output_features

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
            step (spatio)temporal GAN with a 5D array shape:
            (1, spatial_1, spatial_2, n_temporal, n_features)
        """

        logger.debug('Data input to the 1st step spatial-only '
                     'enhancement has shape {}'.format(low_res.shape))
        try:
            hi_res = self.spatial_models.generate(low_res, norm_in=norm_in,
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
            hi_res = self.temporal_models.generate(hi_res, norm_in=True,
                                                   un_norm_out=un_norm_out)
        except Exception as e:
            msg = ('Could not run the 2nd step (spatio)temporal GAN on input '
                   'shape {}'.format(low_res.shape))
            logger.error(msg)
            raise RuntimeError(msg) from e

        logger.debug('Final multistep GAN output has shape: {}'
                     .format(hi_res.shape))

        return hi_res

    @classmethod
    def load(cls, spatial_model_dirs, temporal_model_dirs, verbose=True):
        """Load the GANs with its sub-networks from a previously saved-to
        output directory.

        Parameters
        ----------
        spatial_model_dirs : str | list | tuple
            An ordered list/tuple of one or more directories containing trained
            + saved Sup3rGan models created using the Sup3rGan.save() method.
            This must contain only spatial models that input/output 4D
            tensors.
        temporal_model_dirs : str | list | tuple
            An ordered list/tuple of one or more directories containing trained
            + saved Sup3rGan models created using the Sup3rGan.save() method.
            This must contain only (spatio)temporal models that input/output 5D
            tensors.
        verbose : bool
            Flag to log information about the loaded model.

        Returns
        -------
        out : MultiStepGan
            Returns a pretrained gan model that was previously saved to
            model_dirs
        """
        if isinstance(spatial_model_dirs, str):
            spatial_model_dirs = [spatial_model_dirs]
        if isinstance(temporal_model_dirs, str):
            temporal_model_dirs = [temporal_model_dirs]

        s_models = MultiStepGan.load(spatial_model_dirs, verbose=verbose)
        t_models = MultiStepGan.load(temporal_model_dirs, verbose=verbose)

        return cls(s_models, t_models)
