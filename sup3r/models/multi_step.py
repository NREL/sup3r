# -*- coding: utf-8 -*-
"""Sup3r multi step model frameworks"""
import os
import json
import logging
import numpy as np
from phygnn.layers.custom_layers import Sup3rAdder, Sup3rConcat

# pylint: disable=cyclic-import
import sup3r.models
from sup3r.models.abstract import AbstractInterface
from sup3r.models.base import Sup3rGan


logger = logging.getLogger(__name__)


class MultiStepGan(AbstractInterface):
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

    def __len__(self):
        """Get number of model steps"""
        return len(self._models)

    @staticmethod
    def _needs_hr_exo(model):
        """Determine whether or not the sup3r model needs hi-res exogenous data

        Parameters
        ----------
        model : Sup3rGan | WindGan
            Sup3r GAN model based on Sup3rGan with a .generator attribute

        Returns
        -------
        needs_hr_exo : bool
            True if the model requires high-resolution exogenous data,
            typically because of the use of Sup3rAdder or Sup3rConcat layers.
        """
        return (hasattr(model, 'generator')
                and any(isinstance(layer, (Sup3rAdder, Sup3rConcat))
                for layer in model.generator.layers))

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

        models = []

        if isinstance(model_dirs, str):
            model_dirs = [model_dirs]

        for model_dir in model_dirs:
            fp_params = os.path.join(model_dir, 'model_params.json')
            assert os.path.exists(fp_params), f'Could not find: {fp_params}'
            with open(fp_params, 'r') as f:
                params = json.load(f)

            meta = params.get('meta', {'class': 'Sup3rGan'})
            class_name = meta.get('class', 'Sup3rGan')
            Sup3rClass = getattr(sup3r.models, class_name)
            models.append(Sup3rClass.load(model_dir, verbose=verbose))

        return cls(models)

    @property
    def models(self):
        """Get an ordered tuple of the Sup3rGan models that are part of this
        MultiStepGan
        """
        return self._models

    @property
    def means(self):
        """Get the data normalization mean values. This is a tuple of means
        from all models.

        Returns
        -------
        tuple | np.ndarray
        """
        return tuple(model.means for model in self.models)

    @property
    def stdevs(self):
        """Get the data normalization standard deviation values. This is a
        tuple of stdevs from all models.

        Returns
        -------
        tuple | np.ndarray
        """
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

    def generate(self, low_res, norm_in=True, un_norm_out=True,
                 exogenous_data=None):
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
        exogenous_data : list
            List of arrays of exogenous_data with length equal to the
            number of model steps. e.g. If we want to include topography as
            an exogenous feature in a spatial + temporal multistep model then
            we need to provide a list of length=2 with topography at the low
            spatial resolution and at the high resolution. If we include more
            than one exogenous feature the ordering must be consistent.
            Each array in the list has 3D or 4D shape:
            (spatial_1, spatial_2, n_features)
            (spatial_1, spatial_2, n_temporal, n_features)

        Returns
        -------
        hi_res : ndarray
            Synthetically generated high-resolution data, usually a 4D or 5D
            array with shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)
        """

        exo_data = ([None] * len(self.models) if not exogenous_data
                    else exogenous_data)

        hi_res = low_res.copy()
        for i, model in enumerate(self.models):

            # pylint: disable=R1719
            i_norm_in = False if (i == 0 and not norm_in) else True
            i_un_norm_out = (False
                             if (i + 1 == len(self.models) and not un_norm_out)
                             else True)

            i_exo_data = exo_data[i]
            if self._needs_hr_exo(model):
                i_exo_data = [exo_data[i], exo_data[i + 1]]

            try:
                logger.debug('Data input to model #{} of {} has shape {}'
                             .format(i + 1, len(self.models), hi_res.shape))
                hi_res = model.generate(hi_res, norm_in=i_norm_in,
                                        un_norm_out=i_un_norm_out,
                                        exogenous_data=i_exo_data)
                logger.debug('Data output from model #{} of {} has shape {}'
                             .format(i + 1, len(self.models), hi_res.shape))
            except Exception as e:
                msg = ('Could not run model #{} of {} "{}" '
                       'on tensor of shape {}'
                       .format(i + 1, len(self.models), model, hi_res.shape))
                logger.exception(msg)
                raise RuntimeError(msg) from e

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


class SpatialThenTemporalBase(MultiStepGan):
    """A base class for spatial-then-temporal or temporal-then-spatial multi
    step GANs
    """

    def __init__(self, spatial_models, temporal_models):
        """
        Parameters
        ----------
        spatial_models : MultiStepGan
            A loaded MultiStepGan object representing the one or more spatial
            super resolution steps in this composite SpatialThenTemporal model
        temporal_models : MultiStepGan
            A loaded MultiStepGan object representing the single temporal
            enhancement model in this composite SpatialThenTemporal model
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


class SpatialThenTemporalGan(SpatialThenTemporalBase):
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

    @property
    def models(self):
        """Get an ordered tuple of the Sup3rGan models that are part of this
        MultiStepGan
        """
        if isinstance(self.spatial_models, MultiStepGan):
            spatial_models = self.spatial_models.models
        else:
            spatial_models = [self.spatial_models]
        if isinstance(self.temporal_models, MultiStepGan):
            temporal_models = self.temporal_models.models
        else:
            temporal_models = [self.temporal_models]

        return (*spatial_models, *temporal_models)

    @property
    def meta(self):
        """Get a tuple of meta data dictionaries for all models

        Returns
        -------
        tuple
        """
        if isinstance(self.spatial_models, MultiStepGan):
            spatial_models = self.spatial_models.meta
        else:
            spatial_models = [self.spatial_models.meta]
        if isinstance(self.temporal_models, MultiStepGan):
            temporal_models = self.temporal_models.meta
        else:
            temporal_models = [self.temporal_models.meta]
        return (*spatial_models, *temporal_models)

    @property
    def training_features(self):
        """Get the list of input feature names that the first spatial
        generative model in this SpatialThenTemporalGan model requires as
        input."""
        return self.spatial_models.training_features

    @property
    def output_features(self):
        """Get the list of output feature names that the last spatiotemporal
        interpolation model in this SpatialThenTemporalGan model outputs."""
        return self.temporal_models.output_features

    def generate(self, low_res, norm_in=True, un_norm_out=True,
                 exogenous_data=None):
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
        exogenous_data : list
            List of arrays of exogenous_data with length equal to the
            number of model steps. e.g. If we want to include topography as
            an exogenous feature in a spatial + temporal multistep model then
            we need to provide a list of length=2 with topography at the low
            spatial resolution and at the high resolution. If we include more
            than one exogenous feature the ordering must be consistent.
            Each array in the list has 3D or 4D shape:
            (spatial_1, spatial_2, n_features)
            (temporal, spatial_1, spatial_2, n_features)

        Returns
        -------
        hi_res : ndarray
            Synthetically generated high-resolution data output from the 2nd
            step (spatio)temporal GAN with a 5D array shape:
            (1, spatial_1, spatial_2, n_temporal, n_features)
        """
        logger.debug('Data input to the 1st step spatial-only '
                     'enhancement has shape {}'.format(low_res.shape))
        t_exogenous = None
        if exogenous_data is not None:
            t_exogenous = exogenous_data[len(self.spatial_models):]

        try:
            hi_res = self.spatial_models.generate(
                low_res, norm_in=norm_in, un_norm_out=True,
                exogenous_data=exogenous_data)
        except Exception as e:
            msg = ('Could not run the 1st step spatial-only GAN on input '
                   'shape {}'.format(low_res.shape))
            logger.exception(msg)
            raise RuntimeError(msg) from e

        logger.debug('Data output from the 1st step spatial-only '
                     'enhancement has shape {}'.format(hi_res.shape))
        hi_res = np.transpose(hi_res, axes=(1, 2, 0, 3))
        hi_res = np.expand_dims(hi_res, axis=0)
        logger.debug('Data from the 1st step spatial-only enhancement has '
                     'been reshaped to {}'.format(hi_res.shape))

        try:
            hi_res = self.temporal_models.generate(
                hi_res, norm_in=True, un_norm_out=un_norm_out,
                exogenous_data=t_exogenous)
        except Exception as e:
            msg = ('Could not run the 2nd step (spatio)temporal GAN on input '
                   'shape {}'.format(low_res.shape))
            logger.exception(msg)
            raise RuntimeError(msg) from e

        logger.debug('Final multistep GAN output has shape: {}'
                     .format(hi_res.shape))

        return hi_res


class TemporalThenSpatialGan(SpatialThenTemporalBase):
    """A two-step GAN where the first step is a spatiotemporal enhancement on a
    5D tensor and the second step is a spatial enhancement on a 4D tensor.
    """

    @property
    def models(self):
        """Get an ordered tuple of the Sup3rGan models that are part of this
        MultiStepGan
        """
        if isinstance(self.spatial_models, MultiStepGan):
            spatial_models = self.spatial_models.models
        else:
            spatial_models = [self.spatial_models]
        if isinstance(self.temporal_models, MultiStepGan):
            temporal_models = self.temporal_models.models
        else:
            temporal_models = [self.temporal_models]

        return (*temporal_models, *spatial_models)

    @property
    def meta(self):
        """Get a tuple of meta data dictionaries for all models

        Returns
        -------
        tuple
        """
        if isinstance(self.spatial_models, MultiStepGan):
            spatial_models = self.spatial_models.meta
        else:
            spatial_models = [self.spatial_models.meta]
        if isinstance(self.temporal_models, MultiStepGan):
            temporal_models = self.temporal_models.meta
        else:
            temporal_models = [self.temporal_models.meta]

        return (*temporal_models, *spatial_models)

    @property
    def training_features(self):
        """Get the list of input feature names that the first temporal
        generative model in this TemporalThenSpatialGan model requires as
        input."""
        return self.temporal_models.training_features

    @property
    def output_features(self):
        """Get the list of output feature names that the last spatial
        interpolation model in this TemporalThenSpatialGan model outputs."""
        return self.spatial_models.output_features

    def generate(self, low_res, norm_in=True, un_norm_out=True,
                 exogenous_data=None):
        """Use the generator model to generate high res data from low res
        input. This is the public generate function.

        Parameters
        ----------
        low_res : np.ndarray
            Low-resolution input data, a 5D array of shape:
            (1, spatial_1, spatial_2, n_temporal, n_features)
        norm_in : bool
            Flag to normalize low_res input data if the self.means,
            self.stdevs attributes are available. The generator should always
            received normalized data with mean=0 stdev=1.
        un_norm_out : bool
           Flag to un-normalize synthetically generated output data to physical
           units
        exogenous_data : list
            List of arrays of exogenous_data with length equal to the
            number of model steps. e.g. If we want to include topography as
            an exogenous feature in a temporal + spatial multistep model then
            we need to provide a list of length=2 with topography at the low
            spatial resolution and at the high resolution. If we include more
            than one exogenous feature the ordering must be consistent.
            Each array in the list has 3D or 4D shape:
            (spatial_1, spatial_2, n_features)
            (temporal, spatial_1, spatial_2, n_features)

        Returns
        -------
        hi_res : ndarray
            Synthetically generated high-resolution data output from the 2nd
            step (spatio)temporal GAN with a 5D array shape:
            (1, spatial_1, spatial_2, n_temporal, n_features)
        """
        logger.debug('Data input to the 1st step (spatio)temporal '
                     'enhancement has shape {}'.format(low_res.shape))
        s_exogenous = None
        if exogenous_data is not None:
            s_exogenous = exogenous_data[len(self.temporal_models):]

        assert low_res.shape[0] == 1, 'Low res input can only have 1 obs!'

        try:
            hi_res = self.temporal_models.generate(
                low_res, norm_in=norm_in, un_norm_out=True,
                exogenous_data=exogenous_data)
        except Exception as e:
            msg = ('Could not run the 1st step (spatio)temporal GAN on input '
                   'shape {}'.format(low_res.shape))
            logger.exception(msg)
            raise RuntimeError(msg) from e

        logger.debug('Data output from the 1st step (spatio)temporal '
                     'enhancement has shape {}'.format(hi_res.shape))
        hi_res = np.transpose(hi_res[0], axes=(2, 0, 1, 3))
        logger.debug('Data from the 1st step (spatio)temporal enhancement has '
                     'been reshaped to {}'.format(hi_res.shape))

        try:
            hi_res = self.spatial_models.generate(
                hi_res, norm_in=True, un_norm_out=un_norm_out,
                exogenous_data=s_exogenous)
        except Exception as e:
            msg = ('Could not run the 2nd step spatial GAN on input '
                   'shape {}'.format(low_res.shape))
            logger.exception(msg)
            raise RuntimeError(msg) from e

        hi_res = np.transpose(hi_res, axes=(1, 2, 0, 3))
        hi_res = np.expand_dims(hi_res, axis=0)

        logger.debug('Final multistep GAN output has shape: {}'
                     .format(hi_res.shape))

        return hi_res


class MultiStepSurfaceMetGan(SpatialThenTemporalGan):
    """A two-step GAN where the first step is a spatial-only enhancement on a
    4D tensor of near-surface temperature and relative humidity data, and the
    second step is a (spatio)temporal enhancement on a 5D tensor.

    NOTE: no inputs are needed for the first spatial-only surface meteorology
    model. The spatial enhancement is determined by the low and high res
    topography inputs in the exogenous_data kwargs in the
    MultiStepSurfaceMetGan.generate() method.

    NOTE: The low res input to the spatial enhancement should be a 4D tensor of
    the shape (temporal, spatial_1, spatial_2, features) where temporal
    (usually the observation index) is a series of sequential timesteps that
    will be transposed to a 5D tensor of shape
    (1, spatial_1, spatial_2, temporal, features) tensor and then fed to the
    2nd-step (spatio)temporal GAN.
    """

    def generate(self, low_res, norm_in=True, un_norm_out=True,
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
            Flag to normalize low_res input data if the self.means,
            self.stdevs attributes are available. The generator should always
            received normalized data with mean=0 stdev=1.
        un_norm_out : bool
           Flag to un-normalize synthetically generated output data to physical
           units
        exogenous_data : list
            For the MultiStepSurfaceMetGan model, this must be a 2-entry list
            where the first entry is a 2D (lat, lon) array of low-resolution
            surface elevation data in meters (must match spatial_1, spatial_2
            from low_res), and the second entry is a 2D (lat, lon) array of
            high-resolution surface elevation data in meters.

        Returns
        -------
        hi_res : ndarray
            Synthetically generated high-resolution data output from the 2nd
            step (spatio)temporal GAN with a 5D array shape:
            (1, spatial_1, spatial_2, n_temporal, n_features), Where the
            feature channel can include temperature_*m, relativehumidity_*m,
            and/or pressure_*m
        """
        logger.debug('Data input to the 1st step spatial-only '
                     'enhancement has shape {}'.format(low_res.shape))

        msg = ('MultiStepSurfaceMetGan needs exogenous_data  with two '
               'entries for low and high res topography inputs.')
        assert exogenous_data is not None, msg
        assert isinstance(exogenous_data, (list, tuple)), msg
        exogenous_data = [d for d in exogenous_data if d is not None]
        assert len(exogenous_data) == 2, msg

        # SurfaceSpatialMetModel needs a 2D array for exo topography input
        for i, i_exo in enumerate(exogenous_data):
            if len(i_exo.shape) == 3:
                exogenous_data[i] = i_exo[:, :, 0]
            elif len(i_exo.shape) == 4:
                exogenous_data[i] = i_exo[0, :, :, 0]
            elif len(i_exo.shape) == 5:
                exogenous_data[i] = i_exo[0, :, :, 0, 0]

        try:
            hi_res = self.spatial_models.generate(
                low_res, exogenous_data=exogenous_data)
        except Exception as e:
            msg = ('Could not run the 1st step spatial-only GAN on input '
                   'shape {}'.format(low_res.shape))
            logger.exception(msg)
            raise RuntimeError(msg) from e

        logger.debug('Data output from the 1st step spatial-only '
                     'enhancement has shape {}'.format(hi_res.shape))
        hi_res = np.transpose(hi_res, axes=(1, 2, 0, 3))
        hi_res = np.expand_dims(hi_res, axis=0)
        logger.debug('Data from the 1st step spatial-only enhancement has '
                     'been reshaped to {}'.format(hi_res.shape))

        try:
            hi_res = self.temporal_models.generate(
                hi_res, norm_in=True, un_norm_out=un_norm_out)
        except Exception as e:
            msg = ('Could not run the 2nd step (spatio)temporal GAN on input '
                   'shape {}'.format(low_res.shape))
            logger.exception(msg)
            raise RuntimeError(msg) from e

        logger.debug('Final multistep GAN output has shape: {}'
                     .format(hi_res.shape))

        return hi_res

    @classmethod
    def load(cls, surface_model_class='SurfaceSpatialMetModel',
             temporal_model_class='MultiStepGan',
             surface_model_kwargs=None, temporal_model_kwargs=None,
             verbose=True):
        """Load the GANs with its sub-networks from a previously saved-to
        output directory.

        Parameters
        ----------
        surface_model_class : str
            Name of surface model class to be retrieved from sup3r.models, this
            is typically "SurfaceSpatialMetModel"
        temporal_model_class : str
            Name of temporal model class to be retrieved from sup3r.models,
            this is typically "Sup3rGan"
        surface_model_kwargs : None | dict
            Optional additional kwargs to surface_model_class.load()
        temporal_model_kwargs : None | dict
            Optional additional kwargs to temporal_model_class.load()
        verbose : bool
            Flag to log information about the loaded model.

        Returns
        -------
        out : MultiStepGan
            Returns a pretrained gan model that was previously saved to
            model_dirs
        """

        if surface_model_kwargs is None:
            surface_model_kwargs = {}

        if temporal_model_kwargs is None:
            temporal_model_kwargs = {}

        SpatialModelClass = getattr(sup3r.models, surface_model_class)
        s_models = SpatialModelClass.load(verbose=verbose,
                                          **surface_model_kwargs)

        TemporalModelClass = getattr(sup3r.models, temporal_model_class)
        t_models = TemporalModelClass.load(verbose=verbose,
                                           **temporal_model_kwargs)

        return cls(s_models, t_models)


class SolarMultiStepGan(SpatialThenTemporalGan):
    """Special multi step model for solar clearsky ratio super resolution.

    This model takes in two parallel models for wind-only and solar-only
    spatial super resolutions, then combines them into a 3-channel
    high-spatial-resolution input (clearsky_ratio, U_200m, V_200m) for a solar
    temporal super resolution model.
    """

    def __init__(self, spatial_solar_models, spatial_wind_models,
                 temporal_solar_models, t_enhance=None, temporal_pad=0):
        """
        Parameters
        ----------
        spatial_solar_models : MultiStepGan
            A loaded MultiStepGan object representing the one or more spatial
            super resolution steps in this composite SpatialThenTemporalGan
            model that inputs and outputs clearsky_ratio
        spatial_wind_models : MultiStepGan
            A loaded MultiStepGan object representing the one or more spatial
            super resolution steps in this composite SpatialThenTemporalGan
            model that inputs and outputs wind u/v features and must include
            U_200m + V_200m as output features.
        temporal_solar_models : MultiStepGan
            A loaded MultiStepGan object representing the one or more
            (spatio)temporal super resolution steps in this composite
            SolarMultiStepGan model. This is the final step in the custom solar
            downscaling methodology.
        t_enhance : int | None
            Optional argument to fix or update the temporal enhancement of the
            model. This can be used with temporal_pad to manipulate the output
            shape to match whatever padded shape the sup3r forward pass module
            expects.
        temporal_pad : int
            Optional reflected padding of the generated output array.
        """

        self._spatial_solar_models = spatial_solar_models
        self._spatial_wind_models = spatial_wind_models
        self._temporal_solar_models = temporal_solar_models
        self._t_enhance = t_enhance
        self._temporal_pad = temporal_pad

        self.preflight()

        if self._t_enhance is not None:
            msg = ('Can only update t_enhance for a '
                   'single temporal solar model.')
            assert len(self.temporal_solar_models) == 1, msg
            model = self.temporal_solar_models.models[0]
            model.meta['t_enhance'] = self._t_enhance

    def preflight(self):
        """Run some preflight checks to make sure the loaded models can work
        together."""

        s_enh = [model.s_enhance for model in self.spatial_solar_models.models]
        w_enh = [model.s_enhance for model in self.spatial_wind_models.models]
        msg = ('Solar and wind spatial enhancements must be equivalent but '
               'received models that do spatial enhancements of '
               '{} (solar) and {} (wind)'.format(s_enh, w_enh))
        assert np.product(s_enh) == np.product(w_enh), msg

        s_t_feat = self.spatial_solar_models.training_features
        s_o_feat = self.spatial_solar_models.output_features
        msg = ('Solar spatial enhancement models need to take '
               '"clearsky_ratio" as the only input and output feature but '
               'received models that need {} and output {}'
               .format(s_t_feat, s_o_feat))
        assert s_t_feat == ['clearsky_ratio'], msg
        assert s_o_feat == ['clearsky_ratio'], msg

        temp_solar_feats = self.temporal_solar_models.training_features
        msg = ('Input feature 0 for the temporal_solar_models should be '
               '"clearsky_ratio" but received: {}'
               .format(temp_solar_feats))
        assert temp_solar_feats[0] == 'clearsky_ratio', msg

        spatial_out_features = (self.spatial_wind_models.output_features
                                + self.spatial_solar_models.output_features)
        missing = [fn for fn in temp_solar_feats if fn not in
                   spatial_out_features]
        msg = ('Solar temporal model needs features {} that were not '
               'found in the solar + wind model output feature list {}'
               .format(missing, spatial_out_features))
        assert not any(missing), msg

    @property
    def spatial_models(self):
        """Alias for spatial_solar_models to preserve SpatialThenTemporalGan
        interface."""
        return self.spatial_solar_models

    @property
    def temporal_models(self):
        """Alias for temporal_solar_models to preserve SpatialThenTemporalGan
        interface."""
        return self.temporal_solar_models

    @property
    def spatial_solar_models(self):
        """Get the MultiStepGan object for the spatial-only solar model(s)

        Returns
        -------
        MultiStepGan
        """
        return self._spatial_solar_models

    @property
    def spatial_wind_models(self):
        """Get the MultiStepGan object for the spatial-only wind model(s)

        Returns
        -------
        MultiStepGan
        """
        return self._spatial_wind_models

    @property
    def temporal_solar_models(self):
        """Get the MultiStepGan object for the (spatio)temporal model(s)

        Returns
        -------
        MultiStepGan
        """
        return self._temporal_solar_models

    @property
    def meta(self):
        """Get a tuple of meta data dictionaries for all models

        Returns
        -------
        tuple
        """
        return (self.spatial_solar_models.meta + self.spatial_wind_models.meta
                + self.temporal_solar_models.meta)

    @property
    def training_features(self):
        """Get the list of input feature names that the first spatial
        generative models in this SolarMultiStepGan requires as input.
        This includes the solar + wind training features."""
        return (self.spatial_solar_models.training_features
                + self.spatial_wind_models.training_features)

    @property
    def output_features(self):
        """Get the list of output feature names that the last solar
        spatiotemporal generative model in this SolarMultiStepGan outputs."""
        return self.temporal_solar_models.output_features

    @property
    def idf_wind(self):
        """Get an array of feature indices for the subset of features required
        for the spatial_wind_models. This excludes topography which is assumed
        to be provided as exogenous_data."""
        return np.array([self.training_features.index(fn) for fn in
                         self.spatial_wind_models.training_features
                         if fn != 'topography'])

    @property
    def idf_wind_out(self):
        """Get an array of spatial_wind_models output feature indices that are
        required for input to the temporal_solar_models. Typically this is the
        indices of U_200m + V_200m from the output features of
        spatial_wind_models"""
        temporal_solar_features = self.temporal_solar_models.training_features
        return np.array([self.spatial_wind_models.output_features.index(fn)
                         for fn in temporal_solar_features[1:]])

    @property
    def idf_solar(self):
        """Get an array of feature indices for the subset of features required
        for the spatial_solar_models. This excludes topography which is assumed
        to be provided as exogenous_data."""
        return np.array([self.training_features.index(fn) for fn in
                         self.spatial_solar_models.training_features
                         if fn != 'topography'])

    def generate(self, low_res, norm_in=True, un_norm_out=True,
                 exogenous_data=None):
        """Use the generator model to generate high res data from low res
        input. This is the public generate function.

        Parameters
        ----------
        low_res : np.ndarray
            Low-resolution input data to the 1st step spatial GAN, which is a
            4D array of shape: (temporal, spatial_1, spatial_2, n_features).
            This should include all of the self.training_features which is a
            concatenation of both the solar and wind spatial model features.
            The topography feature might be removed from this input and present
            in the exogenous_data input.
        norm_in : bool
            Flag to normalize low_res input data if the self.means,
            self.stdevs attributes are available. The generator should always
            received normalized data with mean=0 stdev=1.
        un_norm_out : bool
           Flag to un-normalize synthetically generated output data to physical
           units
        exogenous_data : list
            List of arrays of exogenous_data with length equal to the
            number of model steps. e.g. If we want to include topography as
            an exogenous feature in a spatial + temporal multistep model then
            we need to provide a list of length=2 with topography at the low
            spatial resolution and at the high resolution. If we include more
            than one exogenous feature the ordering must be consistent.
            Each array in the list has 3D or 4D shape:
            (spatial_1, spatial_2, n_features)
            (temporal, spatial_1, spatial_2, n_features)
            It's assumed that the spatial_solar_models do not require
            exogenous_data and only use clearsky_ratio.

        Returns
        -------
        hi_res : ndarray
            Synthetically generated high-resolution data output from the 2nd
            step (spatio)temporal GAN with a 5D array shape:
            (1, spatial_1, spatial_2, n_temporal, n_features)
        """

        logger.debug('Data input to the SolarMultiStepGan has shape {} which '
                     'will be split up for solar- and wind-only features.'
                     .format(low_res.shape))
        t_exogenous = None
        if exogenous_data is not None:
            t_exogenous = exogenous_data[len(self.spatial_wind_models):]

        try:
            hi_res_wind = self.spatial_wind_models.generate(
                low_res[..., self.idf_wind],
                norm_in=norm_in, un_norm_out=True,
                exogenous_data=exogenous_data)
        except Exception as e:
            msg = ('Could not run the 1st step spatial-wind-only GAN on '
                   'input shape {}'.format(low_res.shape))
            logger.exception(msg)
            raise RuntimeError(msg) from e

        try:
            hi_res_solar = self.spatial_solar_models.generate(
                low_res[..., self.idf_solar],
                norm_in=norm_in, un_norm_out=True)
        except Exception as e:
            msg = ('Could not run the 1st step spatial-solar-only GAN on '
                   'input shape {}'.format(low_res.shape))
            logger.exception(msg)
            raise RuntimeError(msg) from e

        logger.debug('Data output from the 1st step spatial enhancement has '
                     'shape {} (solar) and shape {} (wind)'
                     .format(hi_res_solar.shape, hi_res_wind.shape))

        hi_res = (hi_res_solar, hi_res_wind[..., self.idf_wind_out])
        hi_res = np.concatenate(hi_res, axis=3)

        logger.debug('Data output from the concatenated solar + wind 1st step '
                     'spatial-only enhancement has shape {}'
                     .format(hi_res.shape))
        hi_res = np.transpose(hi_res, axes=(1, 2, 0, 3))
        hi_res = np.expand_dims(hi_res, axis=0)
        logger.debug('Data from the concatenated solar + wind 1st step '
                     'spatial-only enhancement has been reshaped to {}'
                     .format(hi_res.shape))

        try:
            hi_res = self.temporal_solar_models.generate(
                hi_res, norm_in=True, un_norm_out=un_norm_out,
                exogenous_data=t_exogenous)
        except Exception as e:
            msg = ('Could not run the 2nd step (spatio)temporal solar GAN on '
                   'input shape {}'.format(low_res.shape))
            logger.exception(msg)
            raise RuntimeError(msg) from e

        hi_res = self.temporal_pad(hi_res)

        logger.debug('Final SolarMultiStepGan output has shape: {}'
                     .format(hi_res.shape))

        return hi_res

    def temporal_pad(self, hi_res, mode='reflect'):
        """Optionally add temporal padding to the 5D generated output array

        Parameters
        ----------
        hi_res : ndarray
            Synthetically generated high-resolution data output from the 2nd
            step (spatio)temporal GAN with a 5D array shape:
            (1, spatial_1, spatial_2, n_temporal, n_features)
        mode : str
            Padding mode for np.pad()

        Returns
        -------
        hi_res : ndarray
            Synthetically generated high-resolution data output from the 2nd
            step (spatio)temporal GAN with a 5D array shape:
            (1, spatial_1, spatial_2, n_temporal, n_features)
            With the temporal axis padded with self._temporal_pad on either
            side.
        """
        if self._temporal_pad > 0:
            pad_width = ((0, 0), (0, 0), (0, 0),
                         (self._temporal_pad, self._temporal_pad),
                         (0, 0))
            hi_res = np.pad(hi_res, pad_width, mode=mode)
        return hi_res

    @classmethod
    def load(cls, spatial_solar_model_dirs, spatial_wind_model_dirs,
             temporal_solar_model_dirs, t_enhance=None, temporal_pad=0,
             verbose=True):
        """Load the GANs with its sub-networks from a previously saved-to
        output directory.

        Parameters
        ----------
        spatial_solar_model_dirs : str | list | tuple
            An ordered list/tuple of one or more directories containing trained
            + saved Sup3rGan models created using the Sup3rGan.save() method.
            This must contain only spatial solar models that input/output 4D
            tensors.
        spatial_wind_model_dirs : str | list | tuple
            An ordered list/tuple of one or more directories containing trained
            + saved Sup3rGan models created using the Sup3rGan.save() method.
            This must contain only spatial wind models that input/output 4D
            tensors.
        temporal_solar_model_dirs : str | list | tuple
            An ordered list/tuple of one or more directories containing trained
            + saved Sup3rGan models created using the Sup3rGan.save() method.
            This must contain only (spatio)temporal solar models that
            input/output 5D tensors that are the concatenated output of the
            spatial_solar_models and the spatial_wind_models.
        t_enhance : int | None
            Optional argument to fix or update the temporal enhancement of the
            model. This can be used with temporal_pad to manipulate the output
            shape to match whatever padded shape the sup3r forward pass module
            expects.
        temporal_pad : int
            Optional reflected padding of the generated output array.
        verbose : bool
            Flag to log information about the loaded model.

        Returns
        -------
        out : SolarMultiStepGan
            Returns a pretrained gan model that was previously saved to
            model_dirs
        """
        if isinstance(spatial_solar_model_dirs, str):
            spatial_solar_model_dirs = [spatial_solar_model_dirs]
        if isinstance(spatial_wind_model_dirs, str):
            spatial_wind_model_dirs = [spatial_wind_model_dirs]
        if isinstance(temporal_solar_model_dirs, str):
            temporal_solar_model_dirs = [temporal_solar_model_dirs]

        ssm = MultiStepGan.load(spatial_solar_model_dirs, verbose=verbose)
        swm = MultiStepGan.load(spatial_wind_model_dirs, verbose=verbose)
        tsm = MultiStepGan.load(temporal_solar_model_dirs, verbose=verbose)

        return cls(ssm, swm, tsm, t_enhance=t_enhance,
                   temporal_pad=temporal_pad)
