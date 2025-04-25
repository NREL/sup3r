"""Sup3r model with training on observation data."""

import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError

from sup3r.utilities.utilities import RANDOM_GENERATOR

from .base import Sup3rGan

logger = logging.getLogger(__name__)


class Sup3rGanWithObs(Sup3rGan):
    """Sup3r GAN model which includes mid network observation fusion. This
    model is useful for when production runs will be over a domain for which
    observation data is available."""

    def __init__(
        self,
        *args,
        onshore_obs_frac=None,
        offshore_obs_frac=None,
        loss_obs_weight=None,
        **kwargs,
    ):
        """
        Initialize the Sup3rGanWithObs model.

        Parameters
        ----------
        args : list
            Positional args for ``Sup3rGan`` parent class.
        onshore_obs_frac : Dict[List] | Dict[float]
            Fraction of the batch that should be treated as onshore
            observations. Should include ``spatial`` key and optionally
            ``time`` key if this is a spatiotemporal model. The values should
            correspond roughly to the fraction of the production domain for
            which onshore observations are available (spatial) and the fraction
            of the full time period that these cover. The values can be either
            a list (for a lower and upper bound, respectively) or a single
            float. For each batch a spatial frac will be selected by either
            sampling uniformly between this lower and upper bound or just
            using a single float.
        offshore_obs_frac : dict
            Same as ``onshore_obs_frac`` but for offshore observations.
            Offshore observations are frequently sparser than onshore
            observations.
        loss_obs_weight : float
            Value used to weight observation locations in extra content loss
            term. e.g. The new content loss will include ``obs_loss_weight *
            MAE(hi_res_gen[~obs_mask], hi_res_true[~obs_mask])``
        kwargs : dict
            Keyword arguments for the ``Sup3rGan`` parent class.
        """
        self.onshore_obs_frac = (
            {} if onshore_obs_frac is None else onshore_obs_frac
        )
        self.offshore_obs_frac = (
            {} if offshore_obs_frac is None else offshore_obs_frac
        )
        self.loss_obs_weight = loss_obs_weight
        super().__init__(*args, **kwargs)

    @tf.function
    def _get_loss_obs_comparison(self, hi_res_true, hi_res_gen, obs_mask):
        """Get loss for observation locations and for non observation
        locations."""
        hr_true = hi_res_true[..., : len(self.hr_out_features)]
        loss_obs = MeanAbsoluteError()(
            hr_true[~obs_mask], hi_res_gen[~obs_mask]
        )
        loss_non_obs = MeanAbsoluteError()(
            hr_true[obs_mask], hi_res_gen[obs_mask]
        )
        return loss_obs, loss_non_obs

    def _get_obs_mask(self, hi_res, spatial_frac, time_frac=None):
        """Get observation mask for a given spatial and temporal obs
        fraction."""
        obs_mask = RANDOM_GENERATOR.choice(
            [True, False],
            size=hi_res.shape[1:3],
            p=[1 - spatial_frac, spatial_frac],
        )
        if self.is_5d:
            sp_mask = obs_mask.copy()
            obs_mask = RANDOM_GENERATOR.choice(
                [True, False],
                size=hi_res.shape[1:-1],
                p=[1 - time_frac, time_frac],
            )
            obs_mask[sp_mask] = True
        return np.repeat(obs_mask[None, ...], hi_res.shape[0], axis=0)

    def get_obs_mask(self, hi_res):
        """Define observation mask for the current batch. This is done
        with a spatial mask and a temporal mask since often observation data
        might be very sparse spatially but cover most of the full time period
        for those locations. This is also divided between onshore and offshore
        regions"""
        on_sf = self.onshore_obs_frac['spatial']
        if not isinstance(on_sf, (list, tuple)):
            on_sf = [on_sf, on_sf]
        on_sf = max(RANDOM_GENERATOR.uniform(*on_sf), 0)
        on_tf = self.onshore_obs_frac.get('time', None)
        off_tf = self.offshore_obs_frac.get('time', None)
        obs_mask = self._get_obs_mask(hi_res, on_sf, on_tf)
        if 'topography' in self.hr_exo_features and self.offshore_obs_frac:
            topo_idx = len(self.hr_out_features) + self.hr_exo_features.index(
                'topography'
            )
            topo = hi_res[..., topo_idx]
            off_sf = self.offshore_obs_frac['spatial']
            if not isinstance(off_sf, (list, tuple)):
                off_sf = [off_sf, off_sf]
            off_sf = max(RANDOM_GENERATOR.uniform(*off_sf), 0)
            offshore_mask = self._get_obs_mask(hi_res, off_sf, off_tf)
            obs_mask = tf.where(topo > 0, obs_mask, offshore_mask)
        return obs_mask

    @property
    def model_params(self):
        """
        Model parameters, used to save model to disc

        Returns
        -------
        dict
        """
        params = super().model_params
        params['onshore_obs_frac'] = self.onshore_obs_frac
        params['offshore_obs_frac'] = self.offshore_obs_frac
        params['loss_obs_weight'] = self.loss_obs_weight
        return params

    def get_hr_exo_input(self, hi_res_true):
        """Mask high res data to act as sparse observation data. Add this to
        the standard high res exo input"""
        exo_data = super().get_hr_exo_input(hi_res_true)
        obs_mask = self.get_obs_mask(hi_res_true)
        for feature in self.obs_features:
            # obs_features can include a _obs suffix to avoid name conflict
            # with fully gridded exo features
            f_idx = self.hr_out_features.index(feature.replace('_obs', ''))
            tmp = tf.where(obs_mask, np.nan, hi_res_true[..., f_idx])
            exo_data[feature] = tmp[..., None]
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

        if calc_loss_kwargs.get('train_gen', True):
            loss_obs, loss_non_obs = self._get_loss_obs_comparison(
                hi_res_true,
                hi_res_gen,
                hi_res_exo['mask'],
            )
            loss_update = {
                'loss_obs': loss_obs,
                'loss_non_obs': loss_non_obs,
                'obs_frac': np.sum(~hi_res_exo['mask'])
                / np.size(hi_res_exo['mask']),
            }
            if self.loss_obs_weight is not None:
                loss_obs *= self.loss_obs_weight
                loss += loss_obs
                loss_update['loss_gen'] = loss
                loss_update['loss_gen_content'] = (
                    loss_details['loss_gen_content'] + loss_obs
                )
            loss_details.update(loss_update)
        return loss, loss_details, hi_res_gen, hi_res_exo
