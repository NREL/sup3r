# -*- coding: utf-8 -*-
"""Sup3r multi step model frameworks"""
import json
import logging
import os

import numpy as np

# pylint: disable=cyclic-import
import sup3r.models
from sup3r.models.abstract import AbstractInterface
from sup3r.models.base import Sup3rGan
from sup3r.preprocessing.data_handling.exogenous_data_handling import ExoData

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
            with open(fp_params) as f:
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

    def _transpose_model_input(self, model, hi_res):
        """Transpose input data according to mdel input dimensions.

        NOTE: If hi_res.shape == 4, it is assumed that the dimensions have the
              ordering (n_obs, spatial_1, spatial_2, features)

              If hi_res.shape == 5, it is assumed that the dimensions have the
              ordering (1, spatial_1, spatial_2, temporal, features)

        Parameters
        ----------
        model : Sup3rGan
            A single step model with the attribute model.input_dims
        hi_res : ndarray
            Synthetically generated high-resolution data, usually a 4D or 5D
            array with shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)

        Returns
        -------
        hi_res : ndarray
            Synthetically generated high-resolution data transposed according
            to the number of model input dimensions
        """
        if model.input_dims == 5 and len(hi_res.shape) == 4:
            hi_res = np.transpose(
                hi_res, axes=(1, 2, 0, 3))[np.newaxis]
        elif model.input_dims == 4 and len(hi_res.shape) == 5:
            msg = ('Recieved 5D input data with shape '
                   f'({hi_res.shape}) to a 4D model.')
            assert hi_res.shape[0] == 1, msg
            hi_res = np.transpose(hi_res[0], axes=(2, 0, 1, 3))
        else:
            msg = ('Recieved input data with shape '
                   f'{hi_res.shape} to a {model.input_dims}D model.')
            assert model.input_dims == len(hi_res.shape), msg
        return hi_res

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
        exogenous_data : ExoData
            class:`ExoData` object, which is a special dictionary containing
            exogenous data for each model step and info about how to use the
            data at each step.

        Returns
        -------
        hi_res : ndarray
            Synthetically generated high-resolution data, usually a 4D or 5D
            array with shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)
        """
        if (isinstance(exogenous_data, dict)
                and not isinstance(exogenous_data, ExoData)):
            exogenous_data = ExoData(exogenous_data)

        hi_res = low_res.copy()
        for i, model in enumerate(self.models):
            # pylint: disable=R1719
            i_norm_in = False if (i == 0 and not norm_in) else True
            i_un_norm_out = (False
                             if (i + 1 == len(self.models) and not un_norm_out)
                             else True)

            i_exo_data = (None if exogenous_data is None
                          else exogenous_data.get_model_step_exo(i))

            try:
                hi_res = self._transpose_model_input(model, hi_res)
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


class MultiStepSurfaceMetGan(MultiStepGan):
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
        exogenous_data : dict
            For the MultiStepSurfaceMetGan, this must be a nested dictionary
            with a main 'topography' key and two entries for
            exogenous_data['topography']['steps']. The first entry includes a
            2D (lat, lon) array of low-resolution surface elevation data in
            meters (must match spatial_1, spatial_2 from low_res), and the
            second entry includes a 2D (lat, lon) array of high-resolution
            surface elevation data in meters. e.g.
            {'topography': {
                'steps': [
                    {'model': 0, 'combine_type': 'input', 'data': lr_topo},
                    {'model': 0, 'combine_type': 'output', 'data': hr_topo'}]}}

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

        msg = ('MultiStepSurfaceMetGan needs exogenous_data with two '
               'topography steps, for low and high res topography inputs.')
        exo_check = (exogenous_data is not None
                     and len(exogenous_data['topography']['steps']) == 2)
        assert exo_check, msg

        return super().generate(low_res, norm_in, un_norm_out, exogenous_data)

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

        s_models = getattr(s_models, 'models', [s_models])
        t_models = getattr(t_models, 'models', [t_models])
        return cls([*s_models, *t_models])


class SolarMultiStepGan(MultiStepGan):
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
            super resolution steps in this composite MultiStepGan
            model that inputs and outputs clearsky_ratio
        spatial_wind_models : MultiStepGan
            A loaded MultiStepGan object representing the one or more spatial
            super resolution steps in this composite MultiStepGan
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
        exogenous_data : ExoData
            class:`ExoData` object with data arrays for each exogenous data
            step. Each array has 3D or 4D shape:
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
        if exogenous_data is not None:
            s_exo, t_exo = exogenous_data.split_exo_dict(
                split_step=len(self.spatial_solar_models))
        else:
            s_exo = t_exo = None

        try:
            hi_res_wind = self.spatial_wind_models.generate(
                low_res[..., self.idf_wind],
                norm_in=norm_in, un_norm_out=True,
                exogenous_data=s_exo)
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
                exogenous_data=t_exo)
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
