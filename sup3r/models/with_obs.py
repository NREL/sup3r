"""Sup3r model with training on observation data."""

import logging

import numpy as np
import tensorflow as tf
from phygnn.layers.custom_layers import Sup3rConcatObs
from tensorflow.keras.losses import MeanAbsoluteError

from sup3r.utilities.utilities import RANDOM_GENERATOR

from .base import Sup3rGan

logger = logging.getLogger(__name__)


class Sup3rGanWithObs(Sup3rGan):
    """Sup3r GAN model which includes mid network observation fixing. This
    model is useful for when production runs will be over a domain for which
    observation data is available."""

    def __init__(self, *args, obs_frac=None, loss_obs_weight=None, **kwargs):
        """
        Initialize the Sup3rGanWithObs model.

        Parameters
        ----------
        args : list
            Positional args for ``Sup3rGan`` parent class.
        obs_frac : dict
            Fraction of the batch that should be "fixed" with observations.
            Should include ``spatial`` key and optionally ``time`` key if this
            is a spatiotemporal model. The values should correspond roughly to
            the fraction of the production domain for which observations are
            available (spatial) and the fraction of the full time period that
            these cover. For each batch a spatial frac will be selected by
            uniformly selecting from the range ``(0, obs_frac['spatial'])``
        loss_obs_weight : float
            Value used to weight observation locations in extra content loss
            term. e.g. The new content loss will include ``obs_loss_weight *
            MAE(hi_res_gen[~obs_mask], hi_res_true[~obs_mask])``
        kwargs : dict
            Keyword arguments for the ``Sup3rGan`` parent class.
        """
        self.obs_frac = {} if obs_frac is None else obs_frac
        self.loss_obs_weight = loss_obs_weight
        super().__init__(*args, **kwargs)

    @property
    def obs_features(self):
        """Get list of exogenous observation feature names the model uses.
        These come from the names of the ``Sup3rObs`` layers."""
        # pylint: disable=E1101
        features = []
        if hasattr(self, '_gen'):
            for layer in self._gen.layers:
                check = isinstance(layer, Sup3rConcatObs)
                check = check and layer.name not in features
                if check:
                    features.append(layer.name)
        return features

    @tf.function
    def _get_loss_obs_comparison(self, hi_res_true, hi_res_gen, obs_mask):
        """Get loss for observation locations and for non observation
        locations."""

        hr_true = [
            hi_res_true[..., self.hr_out_features.index(f)]
            for f in self.obs_features
        ]
        hr_true = tf.stack(hr_true, axis=-1)
        hr_gen = [
            hi_res_gen[..., self.hr_out_features.index(f)]
            for f in self.obs_features
        ]
        hr_gen = tf.stack(hr_gen, axis=-1)

        loss_obs = MeanAbsoluteError()(hr_true[~obs_mask], hr_gen[~obs_mask])
        loss_non_obs = MeanAbsoluteError()(hr_true[obs_mask], hr_gen[obs_mask])
        return loss_obs, loss_non_obs

    def _get_obs_mask(self, hi_res, spatial_frac=None, time_frac=None):
        """Define observation mask for the current batch. This is done
        with a spatial mask and a temporal mask since often observation data
        might be very sparse spatially but cover most of the full time period
        for those locations."""
        spatial_frac = (
            self.obs_frac['spatial'] if spatial_frac is None else spatial_frac
        )
        obs_mask = RANDOM_GENERATOR.choice(
            [True, False],
            size=hi_res.shape[1:3],
            p=[1 - spatial_frac, spatial_frac],
        )
        if self.is_5d:
            time_frac = (
                self.obs_frac['time'] if time_frac is None else time_frac
            )
            sp_mask = obs_mask.copy()
            obs_mask = RANDOM_GENERATOR.choice(
                [True, False],
                size=hi_res.shape[1:-1],
                p=[1 - time_frac, time_frac],
            )
            obs_mask[sp_mask] = True
        return np.repeat(obs_mask[None, ...], hi_res.shape[0], axis=0)

    def init_weights(self, lr_shape, hr_shape, device=None):
        """Initialize the generator and discriminator weights with device
        placement.

        Parameters
        ----------
        lr_shape : tuple
            Shape of one batch of low res input data for sup3r resolution. Note
            that the batch size (axis=0) must be included, but the actual batch
            size doesnt really matter.
        hr_shape : tuple
            Shape of one batch of high res input data for sup3r resolution.
            Note that the batch size (axis=0) must be included, but the actual
            batch size doesnt really matter.
        device : str | None
            Option to place model weights on a device. If None,
            self.default_device will be used.
        """

        if not self.generator_weights:
            if device is None:
                device = self.default_device

            logger.info(
                'Initializing model weights on device "{}"'.format(device)
            )
            low_res = np.ones(lr_shape).astype(np.float32)
            hi_res = np.ones(hr_shape).astype(np.float32)

            hr_exo_shape = hr_shape[:-1] + (1,)
            hr_exo = np.ones(hr_exo_shape).astype(np.float32)

            with tf.device(device):
                hr_exo_data = {}
                for feature in self.hr_exo_features + self.obs_features:
                    hr_exo_data[feature] = hr_exo
                _ = self._tf_generate(low_res, hr_exo_data)
                _ = self._tf_discriminate(hi_res)

    @property
    def model_params(self):
        """
        Model parameters, used to save model to disc

        Returns
        -------
        dict
        """
        params = super().model_params
        params['obs_frac'] = self.obs_frac
        params['loss_obs_weight'] = self.loss_obs_weight
        return params

    def get_hr_exo_input(self, hi_res_true):
        """Mask high res data to act as sparse observation data. Add this to
        the standard high res exo input"""
        exo_data = super().get_hr_exo_input(hi_res_true)
        spatial_frac = RANDOM_GENERATOR.uniform(
            low=0, high=self.obs_frac['spatial']
        )
        time_frac = self.obs_frac.get('time', None)
        obs_mask = self._get_obs_mask(hi_res_true, spatial_frac, time_frac)
        for feature in self.obs_features:
            # obs_features can include a _obs suffix to avoid name conflict
            # with fully gridded exo features
            f_idx = self.hr_out_features.index(feature.replace('_obs', ''))
            exo_data[feature] = tf.where(
                obs_mask, np.nan, hi_res_true[..., f_idx]
            )[..., None]
        exo_data['mask'] = obs_mask
        return exo_data

    def _get_hr_exo_and_loss(
        self,
        low_res,
        hi_res_true,
        **calc_loss_kwargs,
    ):
        """Get high-resolution exogenous data, generate synthetic output, and
        compute loss. Includes artificially masking hi res data to act as
        sparse observation data."""
        out = super()._get_hr_exo_and_loss(
            low_res, hi_res_true, **calc_loss_kwargs
        )
        loss, loss_details, hi_res_gen, hi_res_exo = out
        loss_obs, loss_non_obs = self._get_loss_obs_comparison(
            hi_res_true,
            hi_res_gen,
            hi_res_exo['mask'],
        )
        loss_update = {
            'loss_obs': loss_obs,
            'loss_non_obs': loss_non_obs,
        }
        if self.loss_obs_weight is not None and calc_loss_kwargs['train_gen']:
            loss_obs *= self.loss_obs_weight
            loss += loss_obs
            loss_update['loss_gen'] = loss
            loss_update['loss_gen_content'] = (
                loss_details['loss_gen_content'] + loss_obs
            )
        loss_details.update(loss_update)
        return loss, loss_details, hi_res_gen, hi_res_exo
